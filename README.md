# PointL-VTC

Structure-Guided Graph Representation Learning for Point-Level Vessel Trajectory Clustering.

---

## Overview

PointL-VTC is a two-stage framework for point-level AIS trajectory clustering:

- Stage 1 (`StructAug.py`): structure augmentation on trajectory-point graphs.
- Stage 2 (`RepLearn.py`): representation learning and clustering based on original/augmented graph inputs.

The project supports two dataset variants designed for baseline and robustness evaluation:

- `TPC1data`: single-trajectory graph samples.
- `TPC2data`: dual-trajectory graph samples.

---

## Repository Structure

```text
PointL-VTC/
├── StructAug.py                  # Stage-1: structure augmentation
├── RepLearn.py                   # Stage-2: representation learning
├── data/
│   ├── TPC1data/
│   └── TPC2data/
├── src/
│   ├── dataset.py                # Dataset loading and split
│   ├── gnn_model.py              # Encoders and edge augmentation helper
│   ├── model.py                  # Link predictors for StructAug
│   ├── train2Aug.py              # Stage-1 training/test
│   ├── train2Rep.py              # Stage-2 training strategies
│   ├── utils.py                  # Preprocessing and utility functions
│   └── clustering.py             # Clustering metrics
└── eval/                         # Evaluation and result checking scripts
```

---

## AIS-TPC Graph Dataset

### 1. Dataset Description
Based on AIS vessel trajectory data, two training datasets are constructed for **GNN trajectory-point clustering**: **TPC1data** and **TPC2data**.  
Both datasets share identical modeling pipelines, feature compositions, and output formats. The primary difference lies in the number of trajectories included in each graph, which is designed to evaluate the **robustness** and **generalization capability** of the proposed method under varying trajectory complexity.

- **TPC1data: Single-Trajectory Graph Dataset (Baseline Evaluation)**
- **TPC2data: Dual-Trajectory Graph Dataset (Robustness Evaluation)**

### 2. Dataset Variants and Usage

#### TPC1data (Baseline Evaluation)
- **Single-trajectory graph construction**: each sample consists of **one** vessel trajectory segment.
- Primarily used to validate the baseline performance of the proposed method in **single-trajectory scenarios**, such as clustering accuracy and representation quality.

#### TPC2data (Robustness Evaluation)
- **Dual-trajectory graph construction**: each sample consists of **two** distinct vessel trajectory segments.
- Primarily used to evaluate whether the model can maintain stable node representations and clustering performance in **dual-trajectory co-graph settings**, where trajectory interference may exist.

> **Notation**  
> `TPC{n}data` denotes a dataset variant where each sample is constructed from `{n}` trajectories  
> (`n = 1` corresponds to TPC1data, and `n = 2` corresponds to TPC2data).

### 3. Sample Definition and Graph Construction

For any `TPC{n}data` variant, each sample is constructed as follows:

- Each sample consists of `{n}` distinct vessel trajectories or trajectory segments;
- Every `{n}` trajectories are modeled as an **undirected weighted graph**;
- **Nodes** represent trajectory points (AIS points);
- **Edges** are formed by combining the following two relationships:  
  1. **Spatial neighborhood (KNN)**: edges constructed based on spatial proximity between trajectory points;  
  2. **Temporal adjacency (Sequential)**: edges connecting temporally consecutive trajectory points to preserve local temporal structure;

### 4. File Organization and Output Format

Each sample is stored in an independent directory with the following structure:

```text
TPC{n}data/
└── input/
    └── <trajectory_id>/
        ├── raw/
        │   ├── feature.txt
        │   ├── graph.txt
        │   └── label.txt
        └── shp/
            ├── point.shp
            └── edge.shp
```

In this repository, the datasets are placed under:

- `data/TPC1data/`
- `data/TPC2data/`

---

## Environment

The project dependencies are provided in [`requirements.txt`](requirements.txt).

Recommended setup:

```bash
conda create -n env4tc python=3.10.19 -y
conda activate env4tc
pip install -r requirements.txt
```

## Quick Start

### Stage 1: Structure Augmentation

```bash
python StructAug.py --custom_root ./data/TPC1data/input
```

For TPC2data:

```bash
python StructAug.py --custom_root ./data/TPC2data/input
```

Stage-1 outputs are stored under:

- `.../AugOutput/<timestamp>/<dataset_name>/augmented_graph.txt`
- `.../AugOutput/<timestamp>/<dataset_name>/node_embeddings.txt`

### Stage 2: Representation Learning

```bash
python RepLearn.py --base_dir ./data/TPC1data --aug_timestamp <STRUCTAUG_TIMESTAMP>
```

For TPC2data:

```bash
python RepLearn.py --base_dir ./data/TPC2data --aug_timestamp <STRUCTAUG_TIMESTAMP>
```

Stage-2 outputs are stored under:

- `.../RepOutput/<timestamp>/<dataset_name>/features_best.txt`
- `.../RepOutput/<timestamp>/<dataset_name>/labels_best.txt`
- `.../RepOutput/<timestamp>/<dataset_name>/history.txt`

---

## Evaluation Utilities

The `eval/` folder provides helper scripts to:

- check missing outputs,
- evaluate representation logs,
- evaluate structure augmentation logs,
- fill/copy missing experiment artifacts.

---

## Notes

- `StructAug.py` and `RepLearn.py` are the main entry points.
- Core training logic is implemented in `src/train2Aug.py` and `src/train2Rep.py`.
- Dataset loaders and graph preprocessing are centralized in `src/dataset.py` and `src/utils.py`.

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
