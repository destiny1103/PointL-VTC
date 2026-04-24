import argparse
import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd


DETAIL_COLUMNS = [
    "config_log",
    "dataset",
    "Best_AUC",
    "Aug_Edges",
    "ACC",
    "L1_1",
    "L5_5",
    "L1_5",
    "num_augmentations",
    "augment_prob_threshold",
    "log_mtime",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental evaluation for StructAug output logs."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/panjiale/panjiale_data/TPC2data/AugOutput",
        help="Root directory containing StructAug output folders (timestamp folders).",
    )
    parser.add_argument(
        "--detail_csv",
        type=str,
        default="structaug_eval_details.csv",
        help="Detail cache CSV file name (saved under base_path by default).",
    )
    parser.add_argument(
        "--summary_xlsx",
        type=str,
        default="structaug_eval_summary.xlsx",
        help="Summary Excel file name (saved under base_path by default).",
    )
    parser.add_argument(
        "--force_reparse",
        action="store_true",
        help="Reparse all logs regardless of cache.",
    )
    return parser.parse_args()


def extract_last_float(content: str, pattern: str) -> Optional[float]:
    matches = re.findall(pattern, content)
    if not matches:
        return None
    return float(matches[-1])


def extract_last_int(content: str, pattern: str, default: int = 0) -> int:
    matches = re.findall(pattern, content)
    if not matches:
        return default
    return int(matches[-1])


def parse_training_log(file_path: str) -> Optional[Dict[str, Optional[float]]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {file_path}: {e}")
        return None

    try:
        best_auc = extract_last_float(content, r"Best Test AUC:\s*([\d\.]+)")

        edge_matches = re.findall(r"Added\s*(\d+)\s*new edges", content)
        total_aug_edges = sum(int(v) for v in edge_matches) if edge_matches else 0

        acc = extract_last_float(content, r"Overall Augmentation Accuracy:\s*([\d\.]+)")

        l1 = extract_last_int(content, r"Total Label-1 edges:\s*(\d+)", default=0)
        l5 = extract_last_int(content, r"Total Label-5 edges:\s*(\d+)", default=0)
        l1_5 = extract_last_int(
            content,
            r"Total \(Label-1 \+ Label-5\):\s*(\d+)",
            default=0,
        )

        return {
            "Best_AUC": best_auc,
            "Aug_Edges": total_aug_edges,
            "ACC": acc,
            "L1_1": l1,
            "L5_5": l5,
            "L1_5": l1_5,
        }
    except Exception as e:
        print(f"[WARN] Failed to parse {file_path}: {e}")
        return None


def parse_main_log(main_log_path: str) -> Dict[str, Optional[float]]:
    params = {
        "num_augmentations": None,
        "augment_prob_threshold": None,
    }

    if not os.path.exists(main_log_path):
        return params

    try:
        with open(main_log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read main log {main_log_path}: {e}")
        return params

    num_aug_match = re.search(
        r"^\s*num_augmentations:\s*([+-]?\d+)",
        content,
        flags=re.MULTILINE,
    )
    prob_match = re.search(
        r"^\s*augment_prob_threshold:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        content,
        flags=re.MULTILINE,
    )

    if num_aug_match:
        params["num_augmentations"] = int(num_aug_match.group(1))
    if prob_match:
        params["augment_prob_threshold"] = float(prob_match.group(1))

    return params


def format_mean_std(mean_val: float, std_val: float, digits: int) -> str:
    if pd.isna(mean_val):
        return "N/A"
    std_val = 0.0 if pd.isna(std_val) else std_val
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


def build_summary(details_df: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame(
            columns=[
                "Log_Folder",
                "num_augmentations",
                "augment_prob_threshold",
                "n_datasets",
                "Test AUC",
                "Aug Edges",
                "ACC",
                "1-5",
                "1-1",
                "5-5",
            ]
        )

    grouped = (
        details_df.groupby("config_log", dropna=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            num_augmentations=("num_augmentations", "first"),
            augment_prob_threshold=("augment_prob_threshold", "first"),
            AUC_mean=("Best_AUC", "mean"),
            AUC_std=("Best_AUC", "std"),
            AugEdges_mean=("Aug_Edges", "mean"),
            AugEdges_std=("Aug_Edges", "std"),
            ACC_mean=("ACC", "mean"),
            ACC_std=("ACC", "std"),
            L15_mean=("L1_5", "mean"),
            L15_std=("L1_5", "std"),
            L11_mean=("L1_1", "mean"),
            L11_std=("L1_1", "std"),
            L55_mean=("L5_5", "mean"),
            L55_std=("L5_5", "std"),
        )
        .reset_index()
        .rename(columns={"config_log": "Log_Folder"})
    )

    summary = pd.DataFrame()
    summary["Log_Folder"] = grouped["Log_Folder"]
    summary["num_augmentations"] = grouped["num_augmentations"]
    summary["augment_prob_threshold"] = grouped["augment_prob_threshold"]
    summary["n_datasets"] = grouped["n_datasets"]
    summary["Test AUC"] = grouped.apply(
        lambda x: format_mean_std(x["AUC_mean"], x["AUC_std"], 4), axis=1
    )
    summary["Aug Edges"] = grouped.apply(
        lambda x: format_mean_std(x["AugEdges_mean"], x["AugEdges_std"], 1), axis=1
    )
    summary["ACC"] = grouped.apply(
        lambda x: format_mean_std(x["ACC_mean"], x["ACC_std"], 4), axis=1
    )
    summary["1-5"] = grouped.apply(
        lambda x: format_mean_std(x["L15_mean"], x["L15_std"], 2), axis=1
    )
    summary["1-1"] = grouped.apply(
        lambda x: format_mean_std(x["L11_mean"], x["L11_std"], 2), axis=1
    )
    summary["5-5"] = grouped.apply(
        lambda x: format_mean_std(x["L55_mean"], x["L55_std"], 2), axis=1
    )
    return summary.sort_values("Log_Folder").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    base_path = os.path.abspath(args.base_path)
    detail_csv = (
        args.detail_csv
        if os.path.isabs(args.detail_csv)
        else os.path.join(base_path, args.detail_csv)
    )
    summary_xlsx = (
        args.summary_xlsx
        if os.path.isabs(args.summary_xlsx)
        else os.path.join(base_path, args.summary_xlsx)
    )

    if not os.path.exists(base_path):
        print(f"[ERROR] base_path does not exist: {base_path}")
        return

    if os.path.exists(detail_csv):
        try:
            details_df = pd.read_csv(detail_csv)
        except Exception as e:
            print(f"[WARN] Failed to read existing cache, rebuild from scratch: {e}")
            details_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    else:
        details_df = pd.DataFrame(columns=DETAIL_COLUMNS)

    for col in DETAIL_COLUMNS:
        if col not in details_df.columns:
            details_df[col] = np.nan
    details_df = details_df[DETAIL_COLUMNS]

    cached_keys = set()
    if not args.force_reparse and not details_df.empty:
        cached_keys = {
            (str(row["config_log"]), str(row["dataset"]))
            for _, row in details_df[["config_log", "dataset"]].dropna().iterrows()
        }

    config_dirs = [
        d
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    config_dirs.sort()

    new_rows = []
    skipped_cached = 0
    missing_logs = 0

    for config_folder in config_dirs:
        config_path = os.path.join(base_path, config_folder)
        main_log_path = os.path.join(config_path, "main_log.txt")
        params = parse_main_log(main_log_path)

        for dataset_folder in os.listdir(config_path):
            dataset_path = os.path.join(config_path, dataset_folder)
            if not os.path.isdir(dataset_path):
                continue

            key = (config_folder, dataset_folder)
            if key in cached_keys:
                skipped_cached += 1
                continue

            log_file = os.path.join(dataset_path, "training_log.txt")
            if not os.path.exists(log_file):
                missing_logs += 1
                continue

            metrics = parse_training_log(log_file)
            if not metrics:
                continue

            row = {
                "config_log": config_folder,
                "dataset": dataset_folder,
                **metrics,
                "num_augmentations": params["num_augmentations"],
                "augment_prob_threshold": params["augment_prob_threshold"],
                "log_mtime": os.path.getmtime(log_file),
                "updated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            }
            new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df = new_df.reindex(columns=DETAIL_COLUMNS)
        if details_df.empty:
            details_df = new_df.copy()
        else:
            details_df = pd.concat([details_df, new_df], ignore_index=True)

    if not details_df.empty:
        details_df = details_df.sort_values(["config_log", "dataset", "updated_at"])
        details_df = details_df.drop_duplicates(
            subset=["config_log", "dataset"], keep="last"
        ).reset_index(drop=True)

    # Refresh params for all rows from current main_log where possible.
    for config_folder in config_dirs:
        main_log_path = os.path.join(base_path, config_folder, "main_log.txt")
        params = parse_main_log(main_log_path)
        mask = details_df["config_log"] == config_folder
        if mask.any():
            if params["num_augmentations"] is not None:
                details_df.loc[mask, "num_augmentations"] = params["num_augmentations"]
            if params["augment_prob_threshold"] is not None:
                details_df.loc[mask, "augment_prob_threshold"] = params[
                    "augment_prob_threshold"
                ]

    details_df = details_df.sort_values(["config_log", "dataset"]).reset_index(drop=True)
    details_df.to_csv(detail_csv, index=False)

    summary_df = build_summary(details_df)

    excel_saved = False
    try:
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            details_df.to_excel(writer, sheet_name="details", index=False)
        excel_saved = True
    except ModuleNotFoundError:
        summary_csv = os.path.splitext(summary_xlsx)[0] + ".csv"
        details_csv = os.path.splitext(summary_xlsx)[0] + "_details.csv"
        summary_df.to_csv(summary_csv, index=False)
        details_df.to_csv(details_csv, index=False)
        print("[WARN] openpyxl is not installed, Excel output skipped.")
        print(f"[WARN] Saved CSV instead: {summary_csv}")
        print(f"[WARN] Saved CSV instead: {details_csv}")
        print("[HINT] Install openpyxl with: pip install openpyxl")

    print("=== StructAug Log Evaluation Complete ===")
    print(f"base_path: {base_path}")
    print(f"detail cache: {detail_csv}")
    if excel_saved:
        print(f"summary xlsx: {summary_xlsx}")
    else:
        print(f"summary xlsx: skipped (openpyxl missing)")
    print(f"newly parsed rows: {len(new_rows)}")
    print(f"skipped cached rows: {skipped_cached}")
    print(f"missing training_log.txt folders: {missing_logs}")
    print(f"total rows in cache: {len(details_df)}")


if __name__ == "__main__":
    main()
