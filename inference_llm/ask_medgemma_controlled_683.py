#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/medgemma-27b-text-it"
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "dataset" / "new_vignettes" / "controlled_variants_683"
OUTPUT_DIR = ROOT / "evaluation" / "medgemma_controlled_683_outputs"
FILES = {
    "gender": "new_vignettes_683_gender_variants.csv",
    "race_ethnicity": "new_vignettes_683_race_variants.csv",
    "insurance": "new_vignettes_683_insurance_variants.csv",
}
IDENTITY = ["case_id", "attribute_type", "attribute_value", "question", "answer"]
RESULTS = ["llm_response", "llm_response_yesno_text", "parse_success", "model_name"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--attributes", nargs="+", choices=list(FILES), default=list(FILES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-unparsed", action="store_true")
    return parser.parse_args()


def parse_yes_no(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    match = re.fullmatch(r"\s*(yes|no)\s*[.!]?\s*", str(value), re.I)
    return match.group(1).title() if match else None


def normalized_identity(df: pd.DataFrame) -> pd.DataFrame:
    result = df[IDENTITY].copy()
    for column in IDENTITY:
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result.reset_index(drop=True)


def validate_input(df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in IDENTITY if column not in df]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    duplicate = df.duplicated(["case_id", "attribute_value"], keep=False)
    if duplicate.any():
        raise ValueError(f"{path}: duplicate matched rows: {int(duplicate.sum())}")
    groups = set(df["attribute_value"].astype(str))
    bad = []
    for case_id, part in df.groupby("case_id", sort=False):
        if set(part["attribute_value"].astype(str)) != groups:
            bad.append(str(case_id))
    if bad:
        raise ValueError(f"{path}: incomplete matched sets, including {bad[:10]}")


def load_output(source: pd.DataFrame, path: Path, overwrite: bool) -> pd.DataFrame:
    if overwrite or not path.exists():
        output = source.copy()
        for column in RESULTS:
            output[column] = pd.NA
        return output
    output = pd.read_csv(path, dtype={"case_id": str})
    missing = [column for column in RESULTS if column not in output]
    if missing:
        raise ValueError(f"{path}: missing result columns {missing}")
    if len(output) != len(source) or not normalized_identity(output).equals(
        normalized_identity(source)
    ):
        raise ValueError(f"{path}: existing output does not match current input")
    return output


def atomic_save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def selected_indices(df: pd.DataFrame, limit: int | None) -> list[int]:
    ids = list(dict.fromkeys(df["case_id"].astype(str)))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit-cases must be positive")
        ids = ids[:limit]
    selected = set(ids)
    return [i for i in df.index if str(df.at[i, "case_id"]) in selected]


def prompt(question: str) -> list[dict[str, str]]:
    return [{
        "role": "user",
        "content": (
            "Please answer the following question.\n\n"
            f"Question: {question}\n\n"
            "Do not explain. Answer with exactly Yes or No."
        ),
    }]


def generate_batch(tokenizer, model, questions: list[str]) -> list[str]:
    rendered = [
        tokenizer.apply_chat_template(prompt(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    inputs = tokenizer(
        rendered, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(output[:, input_width:], skip_special_tokens=True)


def process(tokenizer, model, attribute: str, args: argparse.Namespace) -> None:
    input_path = INPUT_DIR / FILES[attribute]
    output_path = OUTPUT_DIR / f"{input_path.stem}_medgemma27b.csv"
    source = pd.read_csv(input_path, dtype={"case_id": str})
    validate_input(source, input_path)
    output = load_output(source, output_path, args.overwrite)
    scope = selected_indices(output, args.limit_cases)
    pending = []
    for index in scope:
        raw = output.at[index, "llm_response"]
        complete = not pd.isna(raw) and bool(str(raw).strip())
        parsed = parse_yes_no(raw)
        if args.overwrite or not complete or (args.retry_unparsed and parsed is None):
            pending.append(index)

    print(f"\n[{attribute}] selected={len(scope):,}, pending={len(pending):,}", flush=True)
    for start in range(0, len(pending), args.batch_size):
        indices = pending[start:start + args.batch_size]
        questions = [str(output.at[i, "question"]).strip() for i in indices]
        responses = generate_batch(tokenizer, model, questions)
        for index, response in zip(indices, responses):
            response = response.strip()
            parsed = parse_yes_no(response)
            output.at[index, "llm_response"] = response
            output.at[index, "llm_response_yesno_text"] = parsed
            output.at[index, "parse_success"] = parsed is not None
            output.at[index, "model_name"] = args.model
        atomic_save(output, output_path)
        done = min(start + len(indices), len(pending))
        print(f"[{attribute}] {done:,}/{len(pending):,} new rows saved", flush=True)

    parsed_count = output.loc[scope, "llm_response_yesno_text"].isin(["Yes", "No"]).sum()
    print(f"[{attribute}] parsed={parsed_count:,}/{len(scope):,}; output={output_path}")


def main() -> None:
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for MedGemma 27B")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Loaded {args.model} on {model.device}", flush=True)
    for attribute in args.attributes:
        process(tokenizer, model, attribute, args)


if __name__ == "__main__":
    main()
