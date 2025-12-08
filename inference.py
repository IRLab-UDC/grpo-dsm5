import os
import json
import argparse
import random
from typing import List, Set
from collections import defaultdict

from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal


DSM5Symptom = (
    Annotated[Literal["DEPRESSED_MOOD"], 
              Field(description="Depressed mood most of the day, as indicated by subjective report or observation by others.")]
    | Annotated[Literal["ANHEDONIA"],
                Field(description="Markedly diminished interest or pleasure in all, or almost all, activities most of the day.")]
    | Annotated[Literal["APPETITE_CHANGE"],
                Field(description="Significant weight loss when not dieting, weight gain, or a marked decrease or increase in appetite.")]
    | Annotated[Literal["SLEEP_ISSUES"],
                Field(description="Insomnia or hypersomnia.")]
    | Annotated[Literal["PSYCHOMOTOR"],
                Field(description="Psychomotor agitation or retardation, observable by others.")]
    | Annotated[Literal["FATIGUE"],
                Field(description="Fatigue or loss of energy.")]
    | Annotated[Literal["WORTHLESSNESS"],
                Field(description="Feelings of worthlessness or excessive or inappropriate guilt.")]
    | Annotated[Literal["COGNITIVE_ISSUES"],
                Field(description="Diminished ability to think or concentrate, or indecisiveness.")]
    | Annotated[Literal["SUICIDAL_THOUGHTS"],
                Field(description="Recurrent thoughts of death, recurrent suicidal ideation, or a suicide attempt or specific plan for suicide.")]
)


class DSM5Classification(BaseModel):
    reasoning: str = Field(..., description="A step-by-step explanation of how the text was analyzed in relation to DSM-5 depression criteria.")
    verdict: List[DSM5Symptom] = Field(..., description="List of DSM-5 depression symptoms explicitly present in the text.", min_items=1)


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
    "- Return a JSON object with exactly two fields:\n"
    "  - 'reasoning': a concise paragraph that follows the QUOTE → MAP → JUSTIFY pattern.\n"
    "  - 'verdict': a de-duplicated list of selected tags (DSM5Symptom). The list must have at least one item.\n\n"
    "Be concise, clinical, and literal. No treatment advice or prognosis."
)

ALL_LABELS = [
    "DEPRESSED_MOOD", "ANHEDONIA", "APPETITE_CHANGE", "SLEEP_ISSUES",
    "PSYCHOMOTOR", "FATIGUE", "WORTHLESSNESS", "COGNITIVE_ISSUES", "SUICIDAL_THOUGHTS"
]


def load_and_prepare_dataset(dataset_path, train_size=0.8, seed=42):
    annotations = load_dataset(dataset_path, data_files="redsm5_annotations.csv")["train"]
    posts = load_dataset(dataset_path, data_files="redsm5_posts.csv")["train"]

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
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{HUMAN_INSTRUCTIONS}\nPost:\n\"{s['text']}\""},
            ],
            "annotations": related_annotations,
            "all_symptoms": list(set([a["symptom"] for a in related_annotations])),
        }

    mapped_dataset = posts.map(map_fn)
    mapped_dataset = mapped_dataset.filter(lambda ex: len(ex["annotations"]) > 0)

    split = mapped_dataset.train_test_split(train_size=train_size, seed=seed)
    return {"train": split["train"], "test": split["test"]}


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


def stratified_few_shot_indices(train_dataset, n_shots: int, min_per_class: int = 2, seed: int = 42):
    rng = random.Random(seed)

    label_to_idxs = defaultdict(list)
    for i, ex in enumerate(train_dataset):
        for lab in ex.get("all_symptoms", []):
            label_to_idxs[lab].append(i)

    selected: Set[int] = set()
    per_label_need = {}
    for lab in ALL_LABELS:
        have = label_to_idxs.get(lab, [])
        per_label_need[lab] = min(max(min_per_class, 0), len(have))
        idxs = have[:]
        rng.shuffle(idxs)
        take = []
        for idx in idxs:
            if idx not in selected:
                take.append(idx)
            if len(take) == per_label_need[lab]:
                break
        selected.update(take)

    if len(selected) < n_shots:
        all_idxs = list(range(len(train_dataset)))
        remaining = [i for i in all_idxs if i not in selected]
        rng.shuffle(remaining)
        need = n_shots - len(selected)
        selected.update(remaining[:need])

    selected_list = list(selected)

    if len(selected_list) > n_shots:
        def labels_of(i):
            return set(train_dataset[i].get("all_symptoms", []))

        coverage = defaultdict(int)
        for i in selected_list:
            for lab in labels_of(i):
                coverage[lab] += 1

        candidates = selected_list[:]
        rng.shuffle(candidates)
        kept = set(selected_list)

        def would_break_min(idx):
            for lab in labels_of(idx):
                if coverage[lab] - 1 < per_label_need[lab]:
                    return True
            return False

        for idx in candidates:
            if len(kept) <= n_shots:
                break
            if not would_break_min(idx):
                kept.remove(idx)
                for lab in labels_of(idx):
                    coverage[lab] -= 1

        selected_list = list(kept)
        if len(selected_list) > n_shots:
            rng.shuffle(selected_list)
            selected_list = selected_list[:n_shots]

    rng.shuffle(selected_list)
    return selected_list


def generate_response(client, prompt, model_name, response_format):
    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=prompt,
            response_format=response_format,
            timeout=15
        )

        message = completion.choices[0].message
        if message.parsed:
            return message
        elif message.refusal:
            print("Refusal:", message.refusal)
            return None
    except Exception as e:
        print("Error during completion:", e)
    return None


def run_inference(client, datasets, improved_reasonings, args):
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]

    with open(args.output_file, "w", encoding="utf-8") as f:
        for idx, example in enumerate(tqdm(test_dataset, desc=f"Inference ({args.model})")):
            if args.max_examples and idx >= args.max_examples:
                break

            conversation = example["prompt"].copy()

            if args.n_shots > 0 and len(train_dataset) > 0:
                indices = stratified_few_shot_indices(
                    train_dataset=train_dataset,
                    n_shots=args.n_shots,
                    min_per_class=args.min_per_class,
                    seed=args.seed,
                )

                if len(indices) > 1:
                    samples = train_dataset.select(indices[:-1])
                    for shot in samples:
                        conversation.insert(1, {"role": "user", "content": f"Post:\n\"{shot['text']}\""})
                        conversation.insert(2, {
                            "role": "assistant",
                            "content": json.dumps({
                                "reasoning": improved_reasonings.get(shot["post_id"], ""),
                                "verdict": shot["all_symptoms"],
                            })
                        })

                last_idx = indices[-1]
                samples = train_dataset.select([last_idx])
                for shot in samples:
                    conversation.insert(1, {
                        "role": "user",
                        "content": f"{HUMAN_INSTRUCTIONS}\nPost:\n\"{shot['text']}\""
                    })
                    conversation.insert(2, {
                        "role": "assistant",
                        "content": json.dumps({
                            "reasoning": improved_reasonings.get(shot["post_id"], ""),
                            "verdict": shot["all_symptoms"],
                        })
                    })

            message = generate_response(client, conversation, args.model, DSM5Classification)

            predicted_reasoning = (
                getattr(getattr(message, "parsed", None), "reasoning", "")
                if message is not None else ""
            )
            predicted_symptoms = (
                list(dict.fromkeys(getattr(getattr(message, "parsed", None), "verdict", [])))
                if message is not None else []
            )
            raw_response = getattr(message, "content", "Error during completion or refusal.")

            result = {
                "post_id": example["post_id"],
                "post_text": example["text"],
                "prompt_messages": conversation,
                "raw_response": raw_response,
                "gold_reasoning": improved_reasonings.get(example["post_id"], ""),
                "gold_all_symptoms": example["all_symptoms"],
                "predicted_reasoning": predicted_reasoning,
                "predicted_symptoms": predicted_symptoms,
            }

            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
            f.flush()


def main():
    parser = argparse.ArgumentParser(description="Run inference for DSM-5 symptom classification")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path (for Ollama or HuggingFace)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1",
                        help="API base URL (default: Ollama)")
    parser.add_argument("--api-key", type=str, default="ollama",
                        help="API key")
    parser.add_argument("--dataset-path", type=str, default="irlab-udc/redsm5",
                        help="HuggingFace dataset path")
    parser.add_argument("--improved-reasonings-file", type=str, default="improved_reasonings.jsonl",
                        help="Path to improved reasonings JSONL file")
    parser.add_argument("--output-file", type=str, default="inference_results.jsonl",
                        help="Output JSONL file for results")
    parser.add_argument("--n-shots", type=int, default=0,
                        help="Number of few-shot examples (0 for zero-shot)")
    parser.add_argument("--min-per-class", type=int, default=2,
                        help="Minimum examples per class in few-shot selection")
    parser.add_argument("--train-size", type=float, default=0.8,
                        help="Training split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum number of test examples (None for all)")
    
    args = parser.parse_args()

    random.seed(args.seed)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)

    datasets = load_and_prepare_dataset(args.dataset_path, train_size=args.train_size, seed=args.seed)

    improved_reasonings = load_improved_reasonings(args.improved_reasonings_file)

    run_inference(client, datasets, improved_reasonings, args)


if __name__ == "__main__":
    main()
