import os
import argparse


DEFAULT_BASE_DIR = "/home/panjiale/panjiale_data/TPC1data"

# DEFAULT_REP_LIST = [
#     "20260418_161642",
#     "20260418_162005",
#     "20260418_162054",
#     "20260419_235744",
#     "20260419_004033",
#     "20260419_004056",
#     "20260419_004131",
#     "20260419_014121",
#     "20260419_103304",
# ]

DEFAULT_REP_LIST = [
    "20260419_235744",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Only check whether required files exist")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing input/ and RepOutput/",
    )
    parser.add_argument(
        "--rep_list",
        type=str,
        default=",".join(DEFAULT_REP_LIST),
        help="Comma-separated RepOutput timestamps",
    )
    return parser.parse_args()


def parse_rep_list(rep_list_str):
    return [item.strip() for item in rep_list_str.split(",") if item.strip()]


def check_one_sample(name, base_input, rep_base):
    raw_dir = os.path.join(base_input, name, "raw")
    rep_dir = os.path.join(rep_base, name)

    feature_path = os.path.join(raw_dir, "feature.txt")
    label_path = os.path.join(raw_dir, "label.txt")
    rep_feature_path = os.path.join(rep_dir, "features_best.txt")

    missing_files = []

    if not os.path.exists(feature_path):
        missing_files.append(feature_path)

    if not os.path.exists(label_path):
        missing_files.append(label_path)

    if not os.path.exists(rep_feature_path):
        missing_files.append(rep_feature_path)

    if missing_files:
        return {
            "name": name,
            "status": "missing",
            "missing_files": missing_files,
        }
    else:
        return {
            "name": name,
            "status": "ok",
            "missing_files": [],
        }


def main():
    args = parse_args()
    rep_list = parse_rep_list(args.rep_list)

    base_input = os.path.join(args.base_dir, "input")
    rep_base_root = os.path.join(args.base_dir, "RepOutput")
    out_root = os.path.join(args.base_dir, "RepTest")

    os.makedirs(out_root, exist_ok=True)

    if not os.path.isdir(base_input):
        print(f"input目录不存在: {base_input}")
        return

    names = [
        n for n in os.listdir(base_input)
        if os.path.isdir(os.path.join(base_input, n))
    ]
    names.sort()

    print(f"base_input: {base_input}")
    print(f"样本目录数: {len(names)}")

    for rep_name in rep_list:
        print("\n" + "=" * 80)
        print(f"当前检查 rep: {rep_name}")
        print("=" * 80)

        rep_base = os.path.join(rep_base_root, rep_name)
        report_path = os.path.join(out_root, f"{rep_name}_file_check.txt")

        if not os.path.isdir(rep_base):
            print(f"[SKIP] rep目录不存在: {rep_base}")
            continue

        ok_count = 0
        missing_count = 0
        lines = []

        for name in names:
            result = check_one_sample(
                name=name,
                base_input=base_input,
                rep_base=rep_base,
            )

            if result["status"] == "ok":
                ok_count += 1
            else:
                missing_count += 1
                lines.append(f"[MISSING] {name}")
                for path in result["missing_files"]:
                    lines.append(f"  {path}")
                lines.append("")

                print(f"[MISSING] {name}")
                for path in result["missing_files"]:
                    print(f"  {path}")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"rep_name: {rep_name}\n")
            f.write(f"总样本数: {len(names)}\n")
            f.write(f"完整: {ok_count}\n")
            f.write(f"缺失: {missing_count}\n\n")

            if lines:
                f.write("\n".join(lines))
            else:
                f.write("所有样本文件都存在。\n")

        print(f"\n===== {rep_name} 检查结果 =====")
        print(f"完整: {ok_count}")
        print(f"缺失: {missing_count}")
        print(f"报告已保存到: {report_path}")


if __name__ == "__main__":
    main()