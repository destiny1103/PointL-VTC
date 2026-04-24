import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill missing sample results in a target RepOutput run from a source run."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/panjiale/panjiale_data/TPC1data",
        help="Base directory containing input/ and RepOutput/",
    )
    parser.add_argument(
        "--source_rep",
        type=str,
        default="20260420_150645",
        help="Source RepOutput run name",
    )
    parser.add_argument(
        "--target_rep",
        type=str,
        default="20260419_235744",
        help="Target RepOutput run name to be supplemented",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="features_best.txt",
        help="Result file used to determine whether a sample has been completed",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned operations without copying",
    )
    return parser.parse_args()


def copy_sample_dir(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    else:
        shutil.copytree(src_dir, dst_dir)


def main():
    args = parse_args()

    input_dir = os.path.join(args.base_dir, "input")
    rep_root = os.path.join(args.base_dir, "RepOutput")
    source_root = os.path.join(rep_root, args.source_rep)
    target_root = os.path.join(rep_root, args.target_rep)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"source rep directory not found: {source_root}")
    if not os.path.isdir(target_root):
        raise FileNotFoundError(f"target rep directory not found: {target_root}")

    sample_names = sorted(
        n for n in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, n))
    )

    to_fill = []
    source_missing = []

    for name in sample_names:
        target_result = os.path.join(target_root, name, args.result_file)
        source_result = os.path.join(source_root, name, args.result_file)

        if not os.path.exists(target_result):
            if os.path.exists(source_result):
                to_fill.append(name)
            else:
                source_missing.append(name)

    print(f"source: {source_root}")
    print(f"target: {target_root}")
    print(f"samples in input: {len(sample_names)}")
    print(f"target missing and fillable: {len(to_fill)}")
    print(f"target missing but source also missing: {len(source_missing)}")

    if source_missing:
        print("\nMissing in both source and target:")
        for name in source_missing:
            print(f"  {name}")

    if not to_fill:
        print("\nNothing to fill.")
        return

    print("\nWill fill these samples:")
    for name in to_fill:
        print(f"  {name}")

    if args.dry_run:
        print("\nDry run enabled. No files copied.")
        return

    copied = 0
    failed = 0
    for name in to_fill:
        src_sample_dir = os.path.join(source_root, name)
        dst_sample_dir = os.path.join(target_root, name)

        try:
            if not os.path.isdir(src_sample_dir):
                print(f"[FAIL] source sample dir missing: {name}")
                failed += 1
                continue
            copy_sample_dir(src_sample_dir, dst_sample_dir)
            copied += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failed += 1

    print(f"\nCopy done. copied={copied}, failed={failed}")

    # Verify remaining missing in target
    remaining_missing = []
    for name in sample_names:
        target_result = os.path.join(target_root, name, args.result_file)
        if not os.path.exists(target_result):
            remaining_missing.append(name)

    print(f"Remaining missing in target: {len(remaining_missing)}")
    if remaining_missing:
        for name in remaining_missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()
