import json
import os
import re
import argparse
from functools import partial
from collections import defaultdict

from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from Levenshtein import distance as levenshtein_distance
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams


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
            "answer": json.dumps(
                {"reasoning": improved_reasoning, "verdict": all_symptoms},
                separators=(", ", ": "),
            ),
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


def formatting_prompts_func_grpo(examples, tokenizer):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        for convo in examples["conversation"]
    ]
    return {
        "prompt": texts,
        "answer": examples["answer"],
        "post_id": examples["post_id"],
        "post_text": examples["post_text"],
        "all_symptoms": examples["all_symptoms"],
        "annotations": examples["annotations"],
        "improved_reasoning": examples["improved_reasoning"],
    }


def _extract_json_block(text: str):
    if text is None:
        return None, "", ""

    fence_regexes = [
        re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL),
    ]
    for rgx in fence_regexes:
        m = rgx.search(text)
        if m:
            return m.group(1).strip(), text[:m.start()].strip(), text[m.end():].strip()

    start = text.find("{")
    if start == -1:
        return None, "", ""
    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        return text[start:end+1].strip(), text[:start].strip(), text[end+1:].strip()
    return None, "", ""


def _to_tag_set(value):
    if isinstance(value, list):
        try:
            return {str(x).strip().lower() for x in value if str(x).strip() != ""}
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return {str(x).strip().lower() for x in parsed if str(x).strip() != ""}
            except Exception:
                return None
    return None


def _jaccard(a, b) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return len(inter) / len(union)


def reward_verdict(prompts, completions, answer, max_reward: float = 3.0,
                   wrong_penalty: float = 1.0, **kwargs):
    responses = []
    for completion in completions:
        if isinstance(completion, str):
            responses.append(completion)
        elif (isinstance(completion, list) and completion and 
              isinstance(completion[0], dict) and "content" in completion[0]):
            responses.append(completion[0]["content"])
        else:
            responses.append(str(completion))

    extracted_verdicts = []
    for r in responses:
        js, _, _ = _extract_json_block(r)
        if js is None:
            extracted_verdicts.append(None)
            continue
        try:
            obj = json.loads(js)
            verdict = obj.get("verdict")
            if not isinstance(obj, dict) or not isinstance(verdict, list):
                extracted_verdicts.append(None)
                continue
            tag_set = _to_tag_set(verdict)
            extracted_verdicts.append(tag_set)
        except Exception:
            extracted_verdicts.append(None)

    scores = []
    for guess, true_answer in zip(extracted_verdicts, answer):
        try:
            gold_obj = json.loads(true_answer)
        except Exception:
            scores.append(0.0)
            continue

        gold_verdict = gold_obj.get("verdict")
        gold_tags = _to_tag_set(gold_verdict) if isinstance(gold_verdict, list) else None

        if gold_tags is None:
            scores.append(0.0)
            continue

        if guess is None or len(guess) == 0:
            scores.append(float(-wrong_penalty))
            continue

        J = _jaccard(guess, gold_tags)
        if J == 0.0:
            score = -float(wrong_penalty)
        else:
            score = float(max_reward * J)
        scores.append(score)

    return scores


def reward_reasoning_similarity(prompts, completions, answer, max_reward: float = 3.0,
                                 missing_penalty: float = 1.0, **kwargs):
    def _prep(s: str) -> str:
        if not isinstance(s, str):
            return ""
        t = s[:2000]
        t = re.sub(r"\s+", " ", t).strip()
        return t.casefold()

    scores = []
    for i, completion in enumerate(completions):
        if isinstance(completion, str):
            resp = completion
        elif (isinstance(completion, list) and completion and 
              isinstance(completion[0], dict) and "content" in completion[0]):
            resp = completion[0]["content"]
        else:
            resp = str(completion)

        js, _, _ = _extract_json_block(resp)
        if js is None:
            scores.append(float(-missing_penalty))
            continue

        try:
            obj = json.loads(js)
        except Exception:
            scores.append(float(-missing_penalty))
            continue

        pred_reason = obj.get("reasoning", "")
        try:
            gold_obj = json.loads(answer[i])
        except Exception:
            scores.append(0.0)
            continue
        gold_reason = gold_obj.get("reasoning", "")

        pred_reason_p = _prep(pred_reason)
        gold_reason_p = _prep(gold_reason)

        if not pred_reason_p or not gold_reason_p:
            scores.append(float(-missing_penalty))
            continue

        dist = levenshtein_distance(pred_reason_p, gold_reason_p)
        denom = max(len(pred_reason_p), len(gold_reason_p))
        sim = 1.0 - (dist / denom) if denom > 0 else 1.0
        sim = max(0.0, min(1.0, sim))

        score = max_reward * sim
        scores.append(float(score))

    return scores


def main():
    parser = argparse.ArgumentParser(description="Train GRPO model for DSM-5 symptom classification")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to SFT-trained model checkpoint")
    parser.add_argument("--dataset-path", type=str, default="irlab-udc/redsm5",
                        help="HuggingFace dataset path")
    parser.add_argument("--improved-reasonings-file", type=str, default="improved_reasonings.jsonl",
                        help="Path to improved reasonings JSONL file")
    parser.add_argument("--output-dir", type=str, default="saved_model_grpo",
                        help="Output directory for saved model")
    parser.add_argument("--train-size", type=float, default=0.8,
                        help="Training split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device training batch size")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of generations per prompt")
    
    args = parser.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    chat_template = "gemma-3" if "gemma" in args.model_path.lower() else None
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    dataset = load_and_prepare_dataset(args)
    dataset_grpo = dataset["train"].map(
        partial(formatting_prompts_func_grpo, tokenizer=tokenizer), batched=True
    )

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=8192,
        max_completion_length=4096,
        max_steps=args.max_steps,
        save_steps=100,
        report_to="none",
        output_dir=args.output_dir,
    )

    trainer_grpo = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_verdict, reward_reasoning_similarity],
        args=training_args,
        train_dataset=dataset_grpo,
    )
    trainer_grpo.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
