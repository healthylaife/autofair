#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_NAME = "gpt-5-2025-08-07"
MAX_RETRIES = 5
SAVE_EVERY = 1

AUTOFAIR_ROOT = Path(__file__).resolve().parents[1]
INPUT_FOLDER = (
    AUTOFAIR_ROOT / "dataset" / "new_vignettes" / "controlled_variants_683"
)
OUTPUT_FOLDER = AUTOFAIR_ROOT / "evaluation" / "gpt5_controlled_683_outputs"

INPUT_FILES = {
    "gender": "new_vignettes_683_gender_variants.csv",
    "race_ethnicity": "new_vignettes_683_race_variants.csv",
    "insurance": "new_vignettes_683_insurance_variants.csv",
}

IDENTITY_COLUMNS = [
    "case_id",
    "attribute_type",
    "attribute_value",
    "question",
    "answer",
]

RESULT_COLUMNS = [
    "llm_response",
    "llm_response_yesno_text",
    "openai_response_id",
    "openai_model",
    "parse_success",
    "api_error",
]


class FatalAPIError(RuntimeError):
    """An error for which continuing would only waste retries."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable GPT-5 inference on complete matched sets."
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Process the first N complete case_id sets per attribute.",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        choices=list(INPUT_FILES),
        default=list(INPUT_FILES),
        help="Attributes to process (default: all three).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs and repeat calls in the selected scope.",
    )
    parser.add_argument(
        "--retry-unparsed",
        action="store_true",
        help="Repeat completed responses that could not be parsed as Yes/No.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"OpenAI model ID (default: {MODEL_NAME}).",
    )
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return (
        "Please answer the following question.\n\n"
        f"Question: {question}\n\n"
        "Do not explain. Answer with exactly Yes or No.\n\n"
        "Answer:"
    )


def normalize_yes_no(text: Any) -> str | None:
    """Parse only a standalone Yes or No, allowing surrounding punctuation."""
    if text is None or pd.isna(text):
        return None
    match = re.fullmatch(r"\s*(yes|no)\s*[.!]?\s*", str(text), re.IGNORECASE)
    return match.group(1).title() if match else None


def atomic_save(df: pd.DataFrame, path: Path) -> None:
    """Write without leaving a half-written output if the process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def normalized_identity(df: pd.DataFrame) -> pd.DataFrame:
    result = df[IDENTITY_COLUMNS].copy()
    for column in IDENTITY_COLUMNS:
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result.reset_index(drop=True)


def validate_input(input_df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in IDENTITY_COLUMNS if column not in input_df]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    if input_df["case_id"].isna().any() or input_df["question"].isna().any():
        raise ValueError(f"Missing case_id or question in {path}")
    duplicated = input_df.duplicated(["case_id", "attribute_value"])
    if duplicated.any():
        raise ValueError(
            f"Duplicate case_id/attribute_value rows in {path}: "
            f"{int(duplicated.sum())}"
        )

    # Every case in a given attribute file must contain the same complete set.
    expected_groups = frozenset(input_df["attribute_value"].astype(str))
    bad_cases = []
    for case_id, case_df in input_df.groupby("case_id", sort=False):
        actual = frozenset(case_df["attribute_value"].astype(str))
        if actual != expected_groups:
            bad_cases.append(str(case_id))
    if bad_cases:
        raise ValueError(
            f"Incomplete matched sets in {path}; first cases: {bad_cases[:10]}"
        )


def load_or_initialize_output(
    input_df: pd.DataFrame, output_path: Path, overwrite: bool
) -> pd.DataFrame:
    if overwrite or not output_path.exists():
        output_df = input_df.copy()
        for column in RESULT_COLUMNS:
            output_df[column] = pd.NA
        return output_df

    output_df = pd.read_csv(output_path, dtype={"case_id": str})
    missing = [column for column in RESULT_COLUMNS if column not in output_df]
    if missing:
        raise KeyError(f"Existing output lacks result columns {missing}: {output_path}")
    if len(output_df) != len(input_df):
        raise ValueError(
            f"Existing output has {len(output_df)} rows but current input has "
            f"{len(input_df)} rows: {output_path}"
        )
    if not normalized_identity(output_df).equals(normalized_identity(input_df)):
        raise ValueError(
            "Existing output does not correspond exactly to the current input. "
            f"Move or rename it before continuing: {output_path}"
        )
    return output_df


def selected_indices(df: pd.DataFrame, limit_cases: int | None) -> list[int]:
    case_ids = list(dict.fromkeys(df["case_id"].astype(str)))
    if limit_cases is not None:
        if limit_cases < 1:
            raise ValueError("--limit-cases must be at least 1")
        case_ids = case_ids[:limit_cases]
    selected = set(case_ids)
    return [index for index in df.index if str(df.at[index, "case_id"]) in selected]


def error_details(error: Exception) -> tuple[int | None, str, str]:
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    code = getattr(error, "code", None)
    if not code and isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict):
            code = nested.get("code") or nested.get("type")
    return status, str(code or ""), str(error)


def is_fatal_api_error(error: Exception) -> bool:
    status, code, message = error_details(error)
    combined = f"{code} {message}".lower()
    fatal_tokens = (
        "credit_balance_exhausted",
        "insufficient_quota",
        "billing",
        "invalid_api_key",
        "authentication",
        "permission_denied",
        "model_not_found",
    )
    return status in {401, 403} or any(token in combined for token in fatal_tokens)


def inference_gpt5(client: Any, question: str, model: str) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=build_prompt(question),
        reasoning={"effort": "minimal"},
        text={"verbosity": "low"},
        max_output_tokens=20,
        store=False,
    )
    raw_response = (response.output_text or "").strip()
    parsed = normalize_yes_no(raw_response)
    return {
        "llm_response": raw_response,
        "llm_response_yesno_text": parsed,
        "openai_response_id": response.id,
        "openai_model": response.model,
        "parse_success": parsed is not None,
        "api_error": pd.NA,
    }


def is_completed(row: pd.Series, retry_unparsed: bool) -> bool:
    if pd.isna(row.get("llm_response")) or not str(row["llm_response"]).strip():
        return False
    if retry_unparsed and normalize_yes_no(row.get("llm_response")) is None:
        return False
    return True


def process_file(
    client: Any,
    attribute: str,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    input_df = pd.read_csv(input_path, dtype={"case_id": str})
    validate_input(input_df, input_path)
    output_df = load_or_initialize_output(input_df, output_path, args.overwrite)
    indices = selected_indices(output_df, args.limit_cases)

    print(f"\n[{attribute}] input rows: {len(input_df):,}")
    print(f"[{attribute}] selected rows: {len(indices):,}")
    print(f"[{attribute}] output: {output_path}")

    completed = 0
    skipped = 0
    for position, index in enumerate(indices, start=1):
        row = output_df.loc[index]
        if not args.overwrite and is_completed(row, args.retry_unparsed):
            skipped += 1
            continue

        question = str(row["question"]).strip()
        case_id = str(row["case_id"])
        group = str(row["attribute_value"])

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = inference_gpt5(client, question, args.model)
                for column, value in result.items():
                    output_df.at[index, column] = value
                completed += 1
                print(
                    f"[{attribute}] {position}/{len(indices)} "
                    f"{case_id} | {group} -> {result['llm_response']!r}"
                )
                break
            except Exception as error:
                output_df.at[index, "api_error"] = str(error)
                atomic_save(output_df, output_path)
                if is_fatal_api_error(error):
                    raise FatalAPIError(
                        f"Stopping on non-retryable API error at {case_id} / "
                        f"{group}: {error}"
                    ) from error
                print(
                    f"[{attribute}] attempt {attempt}/{MAX_RETRIES} failed "
                    f"for {case_id} / {group}: {error}",
                    file=sys.stderr,
                )
                if attempt == MAX_RETRIES:
                    break
                time.sleep(min(2**attempt, 30))

        if completed % SAVE_EVERY == 0 or position == len(indices):
            atomic_save(output_df, output_path)

    atomic_save(output_df, output_path)
    selected_df = output_df.loc[indices]
    parsed = selected_df["llm_response_yesno_text"].isin(["Yes", "No"]).sum()
    remaining = sum(not is_completed(row, False) for _, row in selected_df.iterrows())
    print(
        f"[{attribute}] new calls: {completed:,}; already complete: {skipped:,}; "
        f"parsed selected rows: {parsed:,}/{len(indices):,}; remaining: {remaining:,}"
    )


def main() -> int:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI
    except ImportError:
        print("Missing dependency. Install it with: pip install openai", file=sys.stderr)
        return 2

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    try:
        for attribute in args.attributes:
            filename = INPUT_FILES[attribute]
            input_path = INPUT_FOLDER / filename
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            output_path = OUTPUT_FOLDER / f"{Path(filename).stem}_gpt5.csv"
            process_file(client, attribute, input_path, output_path, args)
    except (FatalAPIError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
