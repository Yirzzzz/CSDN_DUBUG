# -*- coding: utf-8 -*-
"""
dataset_cleaner.py  (no hallucination / no rebalance)

功能：
1) 读取错误问答 JSON（[{ "question": "...", "answers": [...] }, ...]）
2) 规范化文本
3) 精确去重
4) 近似去重（TF-IDF + 余弦，相似度阈值可调）
5) 每个步骤产出：被移除样本 JSON、阶段性数据 JSON、统计信息 stats.json

用法示例：
    python dataset_cleaner.py --in errors.json --outdir cleaned_out --neardup-th 0.90

依赖：
    Python 3.8+
    numpy, pandas, scikit-learn
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# --------------------------
# I/O 工具
# --------------------------

def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是列表（list）。")
    # 统一确保键存在
    norm = []
    for i, x in enumerate(data):
        q = (x.get("question") or "").strip()
        a = x.get("answers")
        if a is None:
            a = []
        elif not isinstance(a, list):
            a = [a]
        norm.append({"question": q, "answers": a, "_idx": i})
    return norm


def write_json(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 去掉内部索引字段
    out = []
    for x in data:
        y = {k: v for k, v in x.items() if k not in {"_idx", "_norm_q"}}
        out.append(y)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def write_stats(path: str, stats: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# --------------------------
# 预处理 / 规范化
# --------------------------

def normalize_question(q: str) -> str:
    """轻量规范化：去首尾空白，统一空白，去多余标点空格，英文转小写。"""
    if not isinstance(q, str):
        q = str(q)
    q_strip = q.strip().lower()
    q_strip = re.sub(r"\s+", " ", q_strip)
    q_strip = re.sub(r"\s*([:;,.!?(){}[\]/\\-])\s*", r" \1 ", q_strip)
    q_strip = re.sub(r"\s+", " ", q_strip).strip()
    return q_strip


def step_normalize(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = deepcopy(items)
    for x in out:
        x["_norm_q"] = normalize_question(x["question"])
    return out


# --------------------------
# 精确去重
# --------------------------

def step_exact_dedup(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen = set()
    kept, removed = [], []
    for x in items:
        key = x["_norm_q"]
        if key in seen:
            removed.append(x)
        else:
            seen.add(key)
            kept.append(x)
    return kept, removed


# --------------------------
# 近似去重（TF-IDF + 余弦）
# --------------------------

def step_near_dedup(items: List[Dict[str, Any]], sim_th: float = 0.9, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    基于字符 n-gram 的 TF-IDF（3-5 元）+ 余弦半径邻居进行连通分量归并。
    在每个簇中保留首个样本，其余记为近似重复。
    """
    if len(items) <= 1:
        return items, []

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform([x["_norm_q"] for x in items])

    # 余弦距离 = 1 - 余弦相似度；需要半径 = 1 - sim_th
    radius = 1.0 - sim_th
    nn = NearestNeighbors(radius=radius, metric="cosine", algorithm="brute")
    nn.fit(X)
    # 查询半径邻居
    nbrs = nn.radius_neighbors(X, radius=radius, return_distance=False)

    # 并查集 / 连通分量
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, neigh in enumerate(nbrs):
        for j in neigh:
            if i == j:
                continue
            union(i, j)

    # 收集簇
    comp = defaultdict(list)
    for i in range(len(items)):
        comp[find(i)].append(i)

    kept_idx = set()
    removed = []
    for _, idxs in comp.items():
        idxs = sorted(idxs)
        kept_idx.add(idxs[0])  # 保留簇中最早出现的一个
        for j in idxs[1:]:
            removed.append(items[j])

    kept = [items[i] for i in range(len(items)) if i in kept_idx]
    return kept, removed


# --------------------------
# 主流程
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入 JSON 文件路径")
    ap.add_argument("--outdir", required=True, help="输出目录（会创建多份 JSON/统计）")
    ap.add_argument("--neardup-th", type=float, default=0.90, help="近似去重相似度阈值，默认 0.90")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（用于可复现性）")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = args.outdir
    paths = {
        "stage": os.path.join(outdir, "stages"),
        "removed": os.path.join(outdir, "removed"),
        "report": os.path.join(outdir, "report"),
        "final": os.path.join(outdir, "final"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    report = {}

    # 读取
    raw = read_json(args.inp)
    report["input_count"] = len(raw)

    # 0) 规范化
    data = step_normalize(raw)
    write_json(os.path.join(paths["stage"], "00_normalized.json"), data)

    # 1) 精确去重
    d1_kept, d1_removed = step_exact_dedup(data)
    write_json(os.path.join(paths["stage"], "01_exact_dedup.json"), d1_kept)
    write_json(os.path.join(paths["removed"], "01_exact_dedup_removed.json"), d1_removed)
    report["exact_dedup_removed"] = len(d1_removed)

    # 2) 近似去重
    d2_kept, d2_removed = step_near_dedup(d1_kept, sim_th=args.neardup_th, random_seed=args.seed)
    write_json(os.path.join(paths["stage"], "02_near_dedup.json"), d2_kept)
    write_json(os.path.join(paths["removed"], "02_near_dedup_removed.json"), d2_removed)
    report["near_dedup_removed"] = len(d2_removed)

    # 最终数据（近似去重后）
    write_json(os.path.join(paths["final"], "final_dataset.json"), d2_kept)

    # 报告
    report["final_count"] = len(d2_kept)
    write_stats(os.path.join(paths["report"], "stats.json"), report)

    # 样例切片
    def write_sample(name: str, items: List[Dict[str, Any]], k: int = 5):
        samp = items[:k]
        write_json(os.path.join(paths["report"], f"sample_{name}.json"), samp)

    write_sample("01_exact_dedup_removed", d1_removed)
    write_sample("02_near_dedup_removed", d2_removed)
    write_sample("final_dataset", d2_kept)

    print("=== 清洗完成（仅去重）===")
    print(f"输入样本数: {report['input_count']}")
    print(f"精确去重移除: {report['exact_dedup_removed']}")
    print(f"近似去重移除: {report['near_dedup_removed']}")
    print(f"最终样本数: {report['final_count']}")
    print("阶段与报告目录：")
    for k, v in paths.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
