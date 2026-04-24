import os
import shutil

BASE_DIR = "/home/panjiale/panjiale_data/TPC1data"
INPUT_DIR = os.path.join(BASE_DIR, "input")
INPUT2_DIR = os.path.join(BASE_DIR, "inputtest")
REP_OUTPUT_DIR = os.path.join(BASE_DIR, "RepOutput")

# REP_LIST = [
#     "20260418_161642",
#     "20260418_162005",
#     "20260418_162054",
#     "20260419_004033",
#     "20260419_004056",
#     "20260419_004131",
#     "20260419_014121",
# ]
REP_LIST = [
    "20260419_235744",
]

RESULT_FILE = "features_best.txt"

all_names = sorted([
    n for n in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, n))
])
print(f"共找到样本: {len(all_names)} 个\n")

missing_per_rep = {}  # rep_name -> list of missing names

for rep_name in REP_LIST:
    rep_base = os.path.join(REP_OUTPUT_DIR, rep_name)
    missing = []
    if not os.path.isdir(rep_base):
        print(f"[WARNING] rep 目录不存在: {rep_base}")
        missing = all_names[:]  # 全部视为缺失
    else:
        for name in all_names:
            result_path = os.path.join(rep_base, name, RESULT_FILE)
            if not os.path.exists(result_path):
                missing.append(name)
    missing_per_rep[rep_name] = missing
    print(f"rep {rep_name}: 缺失 {len(missing)}/{len(all_names)} 个样本")

# 按每个 rep 分别打印缺失样本明细
print("\n--- 各 rep 缺失样本明细 ---")
for rep_name in REP_LIST:
    missing = sorted(missing_per_rep.get(rep_name, []))
    print(f"\n[{rep_name}] 缺失 {len(missing)} 个样本")
    if not missing:
        print("  (无缺失)")
        continue
    for name in missing:
        print(f"  {name}")

all_missing_names = set()
for names in missing_per_rep.values():
    all_missing_names.update(names)

print(f"\n所有 rep 中合计缺失（去重）: {len(all_missing_names)} 个样本")

# 打印详情
print("\n--- 缺失样本列表 ---")
for name in sorted(all_missing_names):
    missing_in = [r for r, ml in missing_per_rep.items() if name in ml]
    print(f"  {name}  (缺失于: {', '.join(missing_in)})")

# 将缺失样本的原始 input 数据复制到 input2
print(f"\n--- 开始复制到 {INPUT2_DIR} ---")
os.makedirs(INPUT2_DIR, exist_ok=True)

copied = 0
skipped = 0
for name in sorted(all_missing_names):
    src = os.path.join(INPUT_DIR, name)
    dst = os.path.join(INPUT2_DIR, name)
    if not os.path.exists(src):
        print(f"[SKIP] input 中不存在: {name}")
        skipped += 1
        continue
    if os.path.exists(dst):
        print(f"[SKIP] input2 中已存在: {name}")
        skipped += 1
        continue
    shutil.copytree(src, dst)
    copied += 1

print(f"\n完成: 复制 {copied} 个, 跳过 {skipped} 个")
print(f"input2 路径: {INPUT2_DIR}")
