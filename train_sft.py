import json
import os
import argparse
from functools import partial
from collections import defaultdict

from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig


SYSTEM_PROMPT = (
    "You are a mental-health text reviewer. "
    "You are given social media posts written by third parties (not you, not the user). "
    "Your task is to classify them strictly for DSM-5 depression symptoms. "
    "These symptoms are to be referenced using the following standardized tags: "
    "DEPRESSED_MOOD, ANHEDONIA, APPETITE_CHANGE, SLEEP_ISSUES, PSYCHOMOTOR, "
    "FATIGUE, WORTHLESSNESS, COGNITIVE_ISSUES, SUICIDAL_THOUGHTS."
)

HUMAN_INSTRUCTIONS = (
    "Given the following social media post, identify and classify any DSM-5 depression "
    "symptoms strictly based on explicit and unambiguous evidence in the text.\n\n"
    "Follow this 4-step method:\n"
    "1) QUOTE: Extract exact, minimal evidence snippets in quotes.\n"
    "2) MAP: For each snippet, map it to the single best DSM-5 symptom tag from the allowed set "
    "and name the DSM-5 Criterion A it reflects.\n"
    "3) JUSTIFY: Brief, clinician-style justification explaining why the text satisfies that tag "
    "(no speculation; no could indicate). Address negations or ambiguity explicitly.\n"
    "4) DEDUPE: If multiple snippets support the same tag, include the tag only once in the verdict "
    "but keep all relevant evidence in the reasoning.\n\n"
    "Strict inclusion rules:\n"
    "- Select only symptoms that are explicitly and unambiguously present in the text. "
    "Do NOT infer unstated symptoms, causes, diagnoses, or severity.\n"
    "- Ignore general psychoeducation or third-person statements unless the author clearly reports "
    "their own symptoms (current or past).\n"
    "- Respect negations and uncertainty; if the text says a symptom is absent, do not label it.\n"
    "- Duration is NOT required to tag a symptom, but if timing is stated, mention it in the reasoning.\n\n"
    "Output format:\n"
    "- Return only a JSON object with two fields: 'reasoning' (a concise paragraph that follows the "
    "QUOTE → MAP → JUSTIFY pattern) and 'verdict' (de-duplicated list with at least one symptom tag).\n\n"
    "Be concise, clinical, and literal. No treatment advice or prognosis."
)


def load_improved_reasonings(improved_reasonings_file):
    improved_reasonings = {}
    if os.path.exists(improved_reasonings_file):
        with open(improved_reasonings_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                post_id = data.get("post_id")
                reasoning = data.get("analysis", "")
                if post_id and reasoning:
                    improved_reasonings[post_id] = reasoning
    return improved_reasonings


def load_and_prepare_dataset(args):
    annotations = load_dataset(args.dataset_path, data_files="redsm5_annotations.csv")["train"]
    posts = load_dataset(args.dataset_path, data_files="redsm5_posts.csv")["train"]

    improved_reasonings = load_improved_reasonings(args.improved_reasonings_file)

    annotations_lookup = defaultdict(list)
    for a in annotations:
        if a.get("status") == 1 and a.get("DSM5_symptom") != "SPECIAL_CASE":
            rename_map = {
                "DSM5_symptom": "symptom",
                "explanation": "explanation",
                "sentence_text": "sentence",
                "sentence_id": "sentence_id",
            }
            annotations_lookup[a["post_id"]].append(
                {rename_map[k]: v for k, v in a.items() if k in rename_map}
            )

    def map_fn(s):
        post_id = s["post_id"]
        related_annotations = annotations_lookup.get(post_id, [])
        all_symptoms = list(set([a["symptom"] for a in related_annotations]))

        improved_reasoning = improved_reasonings.get(post_id, "")

        return {
            "conversation": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{HUMAN_INSTRUCTIONS}\n\nPost:\n\n\"\"\"{s['text']}\"\"\"",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"reasoning": improved_reasoning, "verdict": all_symptoms},
                        separators=(", ", ": "),
                    ),
                },
            ],
            "improved_reasoning": improved_reasoning,
            "annotations": related_annotations,
            "all_symptoms": all_symptoms,
            "post_id": post_id,
            "post_text": s["text"],
        }

    mapped_dataset = posts.map(map_fn)
    mapped_dataset = mapped_dataset.filter(lambda ex: len(ex["annotations"]) > 0)
    mapped_dataset = mapped_dataset.filter(lambda ex: ex["improved_reasoning"] != "")

    split = mapped_dataset.train_test_split(train_size=args.train_size, seed=args.seed)
    return split


def formatting_prompts_func_sft(examples, tokenizer):
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in examples["conversation"]
    ]
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Train SFT model for DSM-5 symptom classification")
    parser.add_argument("--model", type=str, default="unsloth/gemma-3-4b-it",
                        help="Base model to fine-tune")
    parser.add_argument("--dataset-path", type=str, default="irlab-udc/redsm5",
                        help="HuggingFace dataset path")
    parser.add_argument("--improved-reasonings-file", type=str, default="improved_reasonings.jsonl",
                        help="Path to improved reasonings JSONL file")
    parser.add_argument("--output-dir", type=str, default="saved_model_sft",
                        help="Output directory for saved model")
    parser.add_argument("--train-size", type=float, default=0.8,
                        help="Training split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank")
    
    args = parser.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    model.gradient_checkpointing_enable()

    chat_template = "gemma-3" if "gemma" in args.model.lower() else None
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    dataset = load_and_prepare_dataset(args)
    dataset_sft = dataset["train"].map(
        partial(formatting_prompts_func_sft, tokenizer=tokenizer), batched=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_sft,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            output_dir=args.output_dir,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
