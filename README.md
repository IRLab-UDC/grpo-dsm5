# Zero-Shot Clinical Reasoning in Mental Health with GRPO

This repository contains the code for the paper "Zero-Shot Clinical Reasoning in Mental Health with Group Relative Policy Optimization" by Bao et al. (2025).

## Overview

We fine-tune Large Language Models (LLMs) using Group Relative Policy Optimization (GRPO) to generate both accurate symptom labels and reasoning aligned with DSM-5 diagnostic criteria for depression detection from social media posts.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- Dependencies listed in `requirements.txt`

## Repository Structure

```
grpo-dsm5/
├── train_sft.py              # Supervised fine-tuning
├── train_grpo.py             # GRPO reinforcement learning
├── inference.py              # Inference script
├── evaluate.py               # Evaluation and metrics
├── requirements.txt          # Python dependencies
├── secrets.yaml.example      # Template for API tokens
├── scripts/
│   ├── setup.sh              # Environment setup
│   ├── train_pipeline.sh     # Automated SFT+GRPO training
│   └── run_batch_inference.sh # Batch inference runner
└── README.md                  # This file
```

## Installation

### Quick Setup

```bash
bash scripts/setup.sh
```

This creates a virtual environment, installs dependencies, and prepares the environment for training/inference.

### Manual Setup

```bash
pip install -r requirements.txt
```

### Authentication (Optional)

For HuggingFace models and datasets, configure your API tokens:

1. Copy the example secrets file:
   ```bash
   cp secrets.yaml.example secrets.yaml
   ```

2. Edit `secrets.yaml` with your tokens:
   ```yaml
   HF_TOKEN: "hf_your_actual_token_here"
   OPENAI_API_KEY: "sk_your_actual_key_here"  # if using OpenAI
   ```

## Dataset

This code uses the **ReDSM5** dataset, which contains Reddit posts annotated by a licensed psychologist for nine DSM-5 depression symptoms. The dataset is available on HuggingFace:

- Dataset: `irlab-udc/redsm5`
- Contains: 1,484 posts with sentence-level annotations
- Symptoms: DEPRESSED_MOOD, ANHEDONIA, APPETITE_CHANGE, SLEEP_ISSUES, PSYCHOMOTOR, FATIGUE, WORTHLESSNESS, COGNITIVE_ISSUES, SUICIDAL_THOUGHTS

**Note:** This repository only provides the code. You need to download the dataset separately from HuggingFace.

## Usage

### 1. Supervised Fine-Tuning (SFT)

First, fine-tune a base model using supervised learning:

```bash
python train_sft.py \
    --model unsloth/gemma-3-4b-it \
    --dataset-path irlab-udc/redsm5 \
    --improved-reasonings-file improved_reasonings.jsonl \
    --output-dir saved_model_sft \
    --max-steps 200 \
    --learning-rate 2e-4
```

**Arguments:**
- `--model`: Base model to fine-tune (HuggingFace model name)
- `--dataset-path`: HuggingFace dataset path
- `--improved-reasonings-file`: Path to JSONL file with improved clinical reasonings
- `--output-dir`: Directory to save the fine-tuned model
- `--max-steps`: Maximum training steps
- `--learning-rate`: Learning rate for training
- `--batch-size`: Per-device training batch size (default: 2)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 4)
- `--lora-r`: LoRA rank (default: 8)

### 2. Group Relative Policy Optimization (GRPO)

After SFT, apply GRPO reinforcement learning to refine the model:

```bash
python train_grpo.py \
    --model-path saved_model_sft \
    --dataset-path irlab-udc/redsm5 \
    --improved-reasonings-file improved_reasonings.jsonl \
    --output-dir saved_model_grpo \
    --max-steps 500 \
    --learning-rate 5e-6
```

**Arguments:**
- `--model-path`: Path to SFT-trained model checkpoint
- `--output-dir`: Directory to save the GRPO-trained model
- `--max-steps`: Maximum training steps
- `--learning-rate`: Learning rate for GRPO (typically lower than SFT)
- `--num-generations`: Number of generations per prompt (default: 4)
- `--batch-size`: Per-device training batch size (default: 1)

### 3. Inference

Run inference on the test set:

```bash
python inference.py \
    --model your-model-name \
    --base-url http://localhost:11434/v1 \
    --dataset-path irlab-udc/redsm5 \
    --improved-reasonings-file improved_reasonings.jsonl \
    --output-file inference_results.jsonl \
    --n-shots 0
```

**Arguments:**
- `--model`: Model name or path (works with Ollama or any OpenAI-compatible API)
- `--base-url`: API base URL (default: Ollama local server)
- `--api-key`: API key (default: "ollama")
- `--output-file`: Output JSONL file for results
- `--n-shots`: Number of few-shot examples (0 for zero-shot)
- `--min-per-class`: Minimum examples per class in few-shot selection
- `--max-examples`: Maximum number of test examples (None for all)

**Note:** For inference, you can use:
- Ollama for local inference
- HuggingFace models via an OpenAI-compatible API
- Any OpenAI-compatible endpoint

### 4. Evaluation

Evaluate model predictions and generate reports:

```bash
python evaluate.py \
    --input inference_results.jsonl \
    --output eval_outputs
```

**Arguments:**
- `--input`: Path to JSONL with inference results
- `--output`: Output directory for reports and plots
- `--truth-field`: JSON field for ground truth labels (default: "gold_all_symptoms")
- `--pred-field`: JSON field for predicted labels (default: "predicted_symptoms")
- `--top-k`: Number of lowest-F1 labels to plot (default: 30)
- `--dpi`: Figure DPI for saved images (default: 160)
- `--raw-confusions`: Show raw counts instead of percentages in confusion matrices

The evaluation generates:
- `eval_summary.txt`: Human-readable summary with metrics
- `per_label_metrics.csv`: Per-label precision, recall, F1, support
- `confusions_per_label.png`: Grid of 2×2 confusion matrices per label
- `per_label_f1.png`: Bar plot of lowest-K F1 scores
- `confusion_matrix_all_labels.png`: Heatmap of all-labels confusion matrix

## Example Workflows

### Complete Training Pipeline

Use the automated training script:

```bash
bash scripts/train_pipeline.sh unsloth/gemma-3-4b-it saved_models
```

This runs both SFT and GRPO training stages automatically.

### Manual Step-by-Step

```bash
# 1. Train with SFT
python train_sft.py \
    --model unsloth/gemma-3-4b-it \
    --output-dir models/gemma-4b-sft

# 2. Apply GRPO
python train_grpo.py \
    --model-path models/gemma-4b-sft \
    --output-dir models/gemma-4b-grpo

# 3. Run inference (assuming model is loaded in Ollama)
python inference.py \
    --model gemma-4b-grpo \
    --output-file results/gemma-4b-grpo.jsonl

# 4. Evaluate
python evaluate.py \
    --input results/gemma-4b-grpo.jsonl \
    --output results/eval_gemma-4b-grpo
```

### Batch Inference

Run inference on multiple configurations:

```bash
# Set Ollama server details (optional, defaults to localhost:11434)
export OLLAMA_HOST=localhost
export OLLAMA_PORT=11434

# Run inference with zero-shot, reasoning mode
bash scripts/run_batch_inference.sh gemma3:4b 1 0

# Run inference with 20-shot, no reasoning mode
bash scripts/run_batch_inference.sh gemma3:4b 0 20
```

## Key Results

- GRPO improves symptom detection over SFT alone, with relative gains exceeding 10% for mid-size models
- Weighted F1 scores above 0.60 for fine-tuned models
- Generating clinical reasoning alongside predictions improves classification across all model scales (0.09-0.39 F1 points)
- Largest gains for initially difficult symptoms requiring deeper contextual understanding (Fatigue, Cognitive Issues, Suicidal Thoughts)

## Citation

If you use this code, please cite:

```bibtex
@article{bao2025grpo,
  title={Zero-Shot Clinical Reasoning in Mental Health with Group Relative Policy Optimization},
  author={Bao, Eliseo and Perez, Anxo and Parapar, Javier},
  year={2025}
}
```

## Acknowledgments

This work was supported by the project PID2022-137061OB-C21 (MCIN/AEI/10.13039/501100011033, Ministerio de Ciencia e Innovación, ERDF); the Consellería de Educación, Universidade e Formación Profesional, Spain (grant number ED481A-2024-079); and the European Regional Development Fund, which supports the CITIC Research Center.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Eliseo Bao: eliseo.bao@udc.es
- IRLab, CITIC, Universidade da Coruña, Spain

## Ethical Considerations

- This work uses publicly available, anonymized social media data from Reddit
- The ReDSM5 dataset was collected in accordance with Reddit's terms of service
- No personally identifiable information was collected or retained
- These systems should function as screening tools for human review, not autonomous diagnostic instruments
- Professional mental health assessment requires consideration of factors beyond isolated social media posts

## Disclaimer

This research is for academic purposes only. The models and code should not be used for clinical diagnosis without proper validation and human oversight. Depression assessment requires professional clinical judgment considering symptom duration, functional impairment, medical history, and contextual factors that isolated social media posts cannot capture.
