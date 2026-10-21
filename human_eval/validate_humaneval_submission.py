"""Validate HumanEval Excel workbook submissions."""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from scripts.release_support import ANONYMOUS_ID_PATTERN, _is_identifying_key

NAME_PATTERN = re.compile(r"^Human Eval (?P<language>\w+)_(?P<country>\w+)$", re.IGNORECASE)
REQUIRED_COLUMNS = ["Annotator ID", "Score (0-3)", "Error Type"]
OTHER_ERROR_VALUES = ["0.4 - Other (add note)", "2.4 - Other (add note)"]


def load_files(input_path: Path) -> list[dict[str, object]]:
    """Load matching workbooks from an input directory."""
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    dataframes: list[dict[str, object]] = []
    skipped: list[str] = []
    for workbook in sorted(input_path.glob("*.xlsx")):
        match = NAME_PATTERN.match(workbook.stem)
        if not match:
            skipped.append(workbook.name)
            continue

        try:
            sheets = pd.read_excel(workbook, sheet_name=None)
        except Exception as error:
            logging.error("Could not read %s: %s", workbook, error)
            raise ValueError(f"Could not read workbook: {workbook}") from error
        dataframes.append(
            {
                "language": match.group("language"),
                "country": match.group("country"),
                "sheets": sheets,
            }
        )

    print(f"Loaded {len(dataframes)} matching file(s).")
    if skipped:
        print(f"Skipped {len(skipped)} non-matching file(s): {', '.join(skipped)}")
    return dataframes


# Check anonymous annotator IDs, scores, and error types for missing values.
def check_nulls_sheet1(dfs: list[dict[str, object]]) -> list[dict[str, object]]:
    issues = []

    for entry in dfs:
        language = entry["language"]
        country = entry["country"]
        sheets = entry["sheets"]
        sheet_name = next(iter(sheets))
        df = sheets[sheet_name]
        tag = f"{language}_{country}"

        logging.info("-" * 80)
        logging.info(f"Checking {tag} / {sheet_name} Main")

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                issues.append(
                    {
                        "language": language,
                        "country": country,
                        "sheet": sheet_name,
                        "column": col,
                        "problem": "column missing",
                        "row_count": None,
                    }
                )
                logging.error(f"[{tag} / {sheet_name}] Missing column: '{col}'")
                continue

            null_mask = df[col].isnull()
            null_count = null_mask.sum()
            if null_count > 0:
                row_indices = df.index[null_mask].tolist()
                issues.append(
                    {
                        "language": language,
                        "country": country,
                        "sheet": sheet_name,
                        "column": col,
                        "problem": "null values",
                        "row_count": int(null_count),
                        "row_indices": row_indices,
                    }
                )
                logging.error(
                    f"[{tag} / {sheet_name}] '{col}' has {null_count} null(s) at rows {row_indices}"
                )

    if not issues:
        print("[sheet 1] No issues found.")
    return issues


# Check anonymous annotator IDs, scores, and error types for missing values.
def check_nulls_sheet2(dfs: list[dict[str, object]]) -> list[dict[str, object]]:
    issues = []

    for entry in dfs:
        language = entry["language"]
        country = entry["country"]
        sheets = entry["sheets"]
        sheet_names = list(sheets)
        tag = f"{language}_{country}"
        # check that Extra sheet exists
        if len(sheet_names) < 2:
            issues.append(
                {
                    "language": language,
                    "country": country,
                    "sheet": None,
                    "column": None,
                    "problem": "second sheet missing",
                    "row_count": None,
                }
            )
            continue

        sheet_name = sheet_names[1]  # second tab
        df = sheets[sheet_name]
        logging.info("-" * 80)
        logging.info(f"Checking {language}_{country} / {sheet_name}")
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                issues.append(
                    {
                        "language": language,
                        "country": country,
                        "sheet": sheet_name,
                        "column": col,
                        "problem": "column missing",
                        "row_count": None,
                    }
                )
                logging.error(f"[{tag} / {sheet_name}] Missing column: '{col}'")
                continue

            null_mask = df[col].isnull()
            null_count = null_mask.sum()
            if null_count > 0:
                row_indices = df.index[null_mask].tolist()
                issues.append(
                    {
                        "language": language,
                        "country": country,
                        "sheet": sheet_name,
                        "column": col,
                        "problem": "null values",
                        "row_count": int(null_count),
                        "row_indices": df.index[null_mask].tolist(),
                    }
                )
                logging.error(
                    f"[{tag} / {sheet_name}] '{col}' has {null_count} null(s) at rows {row_indices}"
                )

    if not issues:
        print("[sheet 2] No issues found.")
    return issues


# 3. Check that the Validator Note exists for rows in which have Other (add note) as Error Type.
def check_other_notes(
    dfs: list[dict[str, object]], sheet_index: int = 0
) -> list[dict[str, object]]:
    """
    sheet_index=0 -> main sheet, sheet_index=1 -> extra sheet.
    """
    issues = []

    for entry in dfs:
        language = entry["language"]
        country = entry["country"]
        sheets = entry["sheets"]
        sheet_names = list(sheets)
        if len(sheet_names) <= sheet_index:
            issues.append(
                {
                    "language": language,
                    "country": country,
                    "sheet": None,
                    "column": "Validator Note",
                    "problem": f"sheet index {sheet_index} missing",
                    "row_count": None,
                }
            )
            continue

        sheet_name = sheet_names[sheet_index]
        df = sheets[sheet_name]

        if "Error Type" not in df.columns:
            issues.append(
                {
                    "language": language,
                    "country": country,
                    "sheet": sheet_name,
                    "column": "Error Type",
                    "problem": "column missing",
                    "row_count": None,
                }
            )
            continue

        if "Validator Note" not in df.columns:
            issues.append(
                {
                    "language": language,
                    "country": country,
                    "sheet": sheet_name,
                    "column": "Validator Note",
                    "problem": "column missing",
                    "row_count": None,
                }
            )
            continue

        other_mask = df["Error Type"].isin(OTHER_ERROR_VALUES)

        # print(df["Error Type"].value_counts())
        # print(df["Validator Note"].value_counts())

        empty_note_mask = df["Validator Note"].isnull() | (
            df["Validator Note"].astype(str).str.strip() == ""
        )
        bad_mask = other_mask & empty_note_mask

        bad_count = bad_mask.sum()
        if bad_count > 0:
            row_indices = df.index[bad_mask].tolist()
            issues.append(
                {
                    "language": language,
                    "country": country,
                    "sheet": sheet_name,
                    "column": "Validator Note",
                    "problem": "missing note for 'Other' error type",
                    "row_count": int(bad_count),
                    "row_indices": row_indices,
                }
            )
            # Keep validation read-only; issue details are returned to the caller.
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if not issues:
        print(f"[Other-note check, sheet_index={sheet_index}] No issues found.")
    return issues


def main() -> int:
    """Run every validation check and return a conventional process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_directory",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "HumanEval",
        help="Directory containing Human Eval <language>_<country>.xlsx files.",
    )
    args = parser.parse_args()
    if not args.input_directory.is_dir():
        print(f"Error: Input directory does not exist: {args.input_directory}")
        return 2
    try:
        data = load_files(args.input_directory)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    if not data:
        print("Error: no matching HumanEval workbooks found.")
        return 2
    for entry in data:
        for sheet_name, frame in entry["sheets"].items():
            if any(
                _is_identifying_key(column) or str(column).lower() == "name" for column in frame
            ):
                print(f"Error: identity columns are not accepted ({sheet_name}).")
                return 1
            if (
                "Annotator ID" in frame
                and not frame["Annotator ID"]
                .dropna()
                .astype(str)
                .str.fullmatch(ANONYMOUS_ID_PATTERN)
                .all()
            ):
                print(f"Error: invalid anonymous annotator IDs ({sheet_name}).")
                return 1

    issues = [
        *check_nulls_sheet1(data),
        *check_nulls_sheet2(data),
        *check_other_notes(data, sheet_index=0),
        *check_other_notes(data, sheet_index=1),
    ]
    if issues:
        print(f"Validation failed: {len(issues)} issue(s) found.")
        for issue in issues:
            print(issue)
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
