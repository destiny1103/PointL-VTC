import os
import argparse

# =========================
# 强制单线程（必须在 numpy / sklearn 导入前设置）
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

warnings.filterwarnings("ignore", message="Graph is not fully connected", category=UserWarning)

DEFAULT_BASE_DIR = "/home/panjiale/panjiale_data/TPC2data"

DEFAULT_REP_LIST = [
    "20260422_095347",
]


DEFAULT_RAW_COLS = [0, 1, 2, 4, 29, 30, 32, 33, 34]




def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RepLearn outputs with clustering metrics")
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
        help="Comma-separated RepOutput timestamps, e.g. 20260408_114623,20260408_114746",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="Number of clusters used by all clustering methods",
    )
    return parser.parse_args()


def parse_rep_list(rep_list_str):
    return [item.strip() for item in rep_list_str.split(",") if item.strip()]


# =========================
# eva
# =========================
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    l2 = list(set(y_pred))

    ind = 0
    if len(l1) != len(l2):
        for i in l1:
            if i not in l2 and ind < len(y_pred):
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    if len(l1) != len(l2):
        raise ValueError("类别数不一致，无法计算 cluster_acc")

    cost = np.zeros((len(l1), len(l2)), dtype=int)
    for i, c1 in enumerate(l1):
        idx = [j for j, e in enumerate(y_true) if e == c1]
        for j, c2 in enumerate(l2):
            cost[i][j] = sum(y_pred[k] == c2 for k in idx)

    m = Munkres()
    indexes = m.compute((-cost).tolist())

    new_pred = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        new_pred[y_pred == c2] = c

    acc = metrics.accuracy_score(y_true, new_pred)
    f1 = metrics.f1_score(y_true, new_pred, average="macro")
    return acc, f1


def eva(y_true, y_pred):
    acc, f1 = cluster_acc(y_true.copy(), y_pred.copy())
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    return acc, nmi, ari, f1


# =========================
# 读取
# =========================
def read_numeric_table(path):
    try:
        df = pd.read_csv(path, header=None, sep=",", engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    return pd.read_csv(path, header=None, sep=r"[\s,]+", engine="python")


def read_feature_file(path, use_cols=None):
    df = read_numeric_table(path)
    if use_cols is not None:
        df = df.iloc[:, use_cols]
    return df.astype(float).values


def read_label_file(path):
    arr = np.loadtxt(path)
    return np.array(arr).reshape(-1).astype(int)


# =========================
# 聚类
# =========================
def run_all_cluster_safe(X, n_clusters=3):
    X = np.asarray(X, dtype=np.float64)  # 只保证数值类型
    n_samples = len(X)

    if n_samples < n_clusters:
        raise ValueError(f"样本数小于{n_clusters}，无法做{n_clusters}类聚类")

    results = {}

    methods = {
        "gmm": lambda Z: GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied",
            n_init=20,
            random_state=0
        ).fit(Z).predict(Z),

        "hierarchical": lambda Z: AgglomerativeClustering(
            n_clusters=n_clusters
        ).fit_predict(Z),

        "kmeans": lambda Z: KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=0,
            algorithm="lloyd"
        ).fit_predict(Z),

        "spectral": lambda Z: SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(10, len(Z) - 1),
            assign_labels="discretize",
            random_state=0
        ).fit_predict(Z)
    }

    for method_name, func in methods.items():
        try:
            results[method_name] = func(X)
        except Exception as e:
            print(f"  方法失败 {method_name}: {e}")

    return results


# =========================
# 单个数据处理
# =========================
def process_one_with_rep(name, base_input, rep_base, out_result_dir, raw_cols, n_clusters=3):
    raw_dir = os.path.join(base_input, name, "raw")
    rep_dir = os.path.join(rep_base, name)

    feature_path = os.path.join(raw_dir, "feature.txt")
    label_path = os.path.join(raw_dir, "label.txt")
    rep_feature_path = os.path.join(rep_dir, "features_best.txt")

    if not (
        os.path.exists(feature_path)
        and os.path.exists(label_path)
        and os.path.exists(rep_feature_path)
    ):
        return {
            "name": name,
            "status": "skip",
            "reason": "缺少输入文件",
            "metrics": {}
        }

    try:
        X_raw = read_feature_file(feature_path, raw_cols)
        X_rep = read_feature_file(rep_feature_path)
        y = read_label_file(label_path)

        if len(y) != len(X_raw) or len(y) != len(X_rep):
            return {
                "name": name,
                "status": "skip",
                "reason": f"行数不一致 raw={len(X_raw)} rep={len(X_rep)} label={len(y)}",
                "metrics": {}
            }

        raw_res = run_all_cluster_safe(X_raw, n_clusters=n_clusters)
        rep_res = run_all_cluster_safe(X_rep, n_clusters=n_clusters)

        out_txt = os.path.join(out_result_dir, f"{name}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("TrueLabel\n")
            f.write(" ".join(map(str, y.tolist())) + "\n\n")

            for method, pred in raw_res.items():
                f.write(f"raw_{method}\n")
                f.write(" ".join(map(str, pred.tolist())) + "\n\n")

            for method, pred in rep_res.items():
                f.write(f"rep_{method}\n")
                f.write(" ".join(map(str, pred.tolist())) + "\n\n")

        metrics_dict = {}

        for method, pred in raw_res.items():
            try:
                metrics_dict[("raw", method)] = eva(y, pred)
            except Exception as e:
                print(f"{name} raw_{method} 评估失败: {e}")

        for method, pred in rep_res.items():
            try:
                metrics_dict[("rep", method)] = eva(y, pred)
            except Exception as e:
                print(f"{name} rep_{method} 评估失败: {e}")

        return {
            "name": name,
            "status": "ok",
            "reason": "",
            "metrics": metrics_dict
        }

    except Exception as e:
        return {
            "name": name,
            "status": "fail",
            "reason": str(e),
            "metrics": {}
        }


# =========================
# 主程序：单线程，遍历全部 rep
# =========================
def main():
    args = parse_args()
    rep_list = parse_rep_list(args.rep_list)
    base_input = os.path.join(args.base_dir, "input")
    rep_base_root = os.path.join(args.base_dir, "RepOutput")
    out_root = os.path.join(args.base_dir, "RepTest")

    os.makedirs(out_root, exist_ok=True)

    names = [
        n for n in os.listdir(base_input)
        if os.path.isdir(os.path.join(base_input, n))
    ]
    names.sort()

    print(f"base_input: {base_input}")
    print(f"样本目录数: {len(names)}")

    for rep_name in rep_list:
        print("\n" + "=" * 80)
        print(f"当前处理 rep: {rep_name}")
        print("=" * 80)

        rep_base = os.path.join(rep_base_root, rep_name)
        out_result_dir = os.path.join(out_root, f"{rep_name}_cluster")
        summary_path = os.path.join(out_root, f"{rep_name}_summary.txt")

        if not os.path.isdir(rep_base):
            print(f"跳过，rep目录不存在: {rep_base}")
            continue

        os.makedirs(out_result_dir, exist_ok=True)

        all_metrics = {}
        ok_count = 0
        skip_count = 0
        fail_count = 0

        for name in tqdm(names, desc=rep_name, ncols=100):
            result = process_one_with_rep(
                name=name,
                base_input=base_input,
                rep_base=rep_base,
                out_result_dir=out_result_dir,
                raw_cols=DEFAULT_RAW_COLS,
                n_clusters=args.n_clusters,
            )

            if result["status"] == "ok":
                ok_count += 1
                for key, score in result["metrics"].items():
                    all_metrics.setdefault(key, []).append(score)

            elif result["status"] == "skip":
                skip_count += 1
                print(f"[SKIP] {name}: {result['reason']}")

            else:
                fail_count += 1
                print(f"[FAIL] {name}: {result['reason']}")

        summary_lines = []
        for key in sorted(all_metrics.keys()):
            values = np.array(all_metrics[key], dtype=float)
            mean = values.mean(axis=0)
            std = values.std(axis=0)

            line = (
                f"{key[0]}_{key[1]} | "
                f"ACC {mean[0]:.4f}±{std[0]:.4f} | "
                f"NMI {mean[1]:.4f}±{std[1]:.4f} | "
                f"ARI {mean[2]:.4f}±{std[2]:.4f} | "
                f"F1 {mean[3]:.4f}±{std[3]:.4f}"
            )
            summary_lines.append(line)

        with open(summary_path, "w", encoding="utf-8") as f:
            for line in summary_lines:
                print(line)
                f.write(line + "\n")

        print(f"\n===== {rep_name} 统计 =====")
        print(f"成功: {ok_count}")
        print(f"跳过: {skip_count}")
        print(f"失败: {fail_count}")
        print(f"cluster输出目录: {out_result_dir}")
        print(f"summary 已保存到: {summary_path}")


if __name__ == "__main__":
    main()
