from __future__ import annotations

import csv
import gc
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

CONFIG: dict[str, Any] = {
    "tree_json_path": str(BASE_DIR / "mora_v4.1_0406.json"),
    "similarity_json_path": str(
        BASE_DIR / "total_formal_all_sentence_adjusted_distance_aggressive_llm.json"
    ),
    "similarity_json_fallbacks": [
        str(BASE_DIR / "total_formal_all_sentence_adjusted_distance_aggres.json"),
        str(BASE_DIR / "total_formal_all_sentence_adjusted_distance_aggressive.json"),
    ],
    "output_dir": str(BASE_DIR),
    "vgae_output_path": str(BASE_DIR / "vgae_output.pt"),
    "node_vectors_csv_path": str(BASE_DIR / "node_vectors.csv"),
    "version_similarity_csv_path": str(BASE_DIR / "version_similarity_matrix.csv"),
    "tsne_png_path": str(BASE_DIR / "tsne_visualization.png"),
    "chapter_tsne_png_path": str(BASE_DIR / "chapter_tsne_visualization.png"),
    "version_tsne_png_path": str(BASE_DIR / "version_tsne_visualization.png"),
    "sentence_best_vgae_output_path": str(BASE_DIR / "sentence_best_vgae_output.pt"),
    "sentence_best_node_vectors_csv_path": str(
        BASE_DIR / "sentence_best_node_vectors.csv"
    ),
    "sentence_best_version_similarity_csv_path": str(
        BASE_DIR / "sentence_best_version_similarity_matrix.csv"
    ),
    "sentence_best_tsne_png_path": str(BASE_DIR / "sentence_best_tsne_visualization.png"),
    "sentence_best_chapter_tsne_png_path": str(
        BASE_DIR / "sentence_best_chapter_tsne_visualization.png"
    ),
    "sentence_best_version_tsne_png_path": str(
        BASE_DIR / "sentence_best_version_tsne_visualization.png"
    ),
    "sentence_best_config_path": str(BASE_DIR / "sentence_best_config.json"),
    "missing_edge_log_path": str(BASE_DIR / "missing_similarity_edges.json"),
    "input_dim": 256,
    "hidden_dim": 256,
    "latent_dim": 128,
    "epochs": 400,
    "learning_rate": 0.0015,
    "kl_weight": 0.001,
    "seed": 42,
    "device": "cuda_if_available",
    "hierarchy_weight": 0.05,
    "layer3_threshold": 0.15,
    "dropout": 0.15,
    "kl_anneal_epochs": 40,
    "kl_full_epochs": 160,
    "max_trials": 8,
    "trial_timeout_seconds": 1800,
    "sentence_silhouette_target": 0.15,
    "chapter_silhouette_target": 0.10,
    "embedding_std_target": 0.02,
    "l2_top_k": 2,
    "fallback_min_similarity_degree": 3,
    "fallback_weight": 0.01,
    "distance_tau": 0.05,
    "l0_similarity_weight": 2.0,
    "max_encoder_edge_weight": 2.0,
    "feature_noise_scale": 0.05,
    "structural_feature_scale": 0.7,
    "min_logvar": -6.0,
    "max_logvar": 2.0,
    "gradient_clip_norm": 1.0,
    "weight_decay": 1e-4,
    "use_mu_for_recon": True,
    "similarity_recon_weight": 1.0,
    "hierarchy_recon_weight": 0.25,
    "log_every": 20,
    "tsne_perplexity": 30,
    "tsne_random_state": 42,
    "missing_edge_sample_limit": 200,
    "training_log_path": str(BASE_DIR / "training_log.json"),
    "best_config_path": str(BASE_DIR / "best_config.json"),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def require_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        from torch_geometric.utils import negative_sampling, to_undirected
    except Exception as exc:
        raise RuntimeError(
            "Missing or incompatible dependencies. Please install compatible builds of "
            "torch, torch_geometric, scikit-learn, matplotlib, and numpy for Python 3.10. "
            f"Original import error: {exc}"
        ) from exc

    return {
        "plt": plt,
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "TSNE": TSNE,
        "silhouette_score": silhouette_score,
        "cosine_similarity": cosine_similarity,
        "Data": Data,
        "GCNConv": GCNConv,
        "negative_sampling": negative_sampling,
        "to_undirected": to_undirected,
    }


@dataclass(frozen=True)
class SimilarityCandidate:
    u: int
    v: int
    distance: float
    source_index: int


def resolve_existing_path(primary: str, fallbacks: list[str] | None = None) -> Path:
    paths = [Path(primary)]
    paths.extend(Path(p) for p in (fallbacks or []))
    for path in paths:
        if path.exists():
            return path
    tried = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"Input file not found. Tried:\n{tried}")


def set_seed(seed: int, torch_module: Any | None = None) -> None:
    random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)


def add_node(
    node_id: str,
    meta: dict[str, Any],
    node_id_map: dict[str, int],
    node_meta: list[dict[str, Any]],
) -> int:
    if node_id in node_id_map:
        return node_id_map[node_id]
    idx = len(node_meta)
    node_id_map[node_id] = idx
    full_meta = {"node_id": node_id, **meta}
    node_meta.append(full_meta)
    return idx


def load_tree_graph(tree_path: Path) -> tuple[
    dict[str, int],
    list[dict[str, Any]],
    list[tuple[int, int, float, str]],
    list[int],
    list[int],
    list[int],
]:
    log(f"Loading tree JSON: {tree_path.name}")
    with tree_path.open("r", encoding="utf-8") as f:
        root = json.load(f)

    node_id_map: dict[str, int] = {}
    node_meta: list[dict[str, Any]] = []
    hierarchy_edges: list[tuple[int, int, float, str]] = []
    version_indices: list[int] = []
    chapter_indices: list[int] = []
    sentence_indices: list[int] = []

    for version_order, version_node in enumerate(root.get("children", [])):
        if version_node.get("type") != "version":
            continue
        version = str(version_node.get("name") or version_node.get("version"))
        if not version:
            raise ValueError(f"Version node without a name at order {version_order}")

        version_id = f"version_{version}"
        version_idx = add_node(
            version_id,
            {
                "type": "version",
                "version": version,
                "chapter": "",
                "sentence": "",
                "text": "",
                "source_index": version_node.get("index", version_order),
            },
            node_id_map,
            node_meta,
        )
        version_indices.append(version_idx)

        for chapter_node in version_node.get("children", []):
            if chapter_node.get("type") != "chapter":
                continue
            chapter_number = int(chapter_node["chapter_number"])
            chapter_id = f"chapter_{version}_{chapter_number}"
            chapter_idx = add_node(
                chapter_id,
                {
                    "type": "chapter",
                    "version": version,
                    "chapter": chapter_number,
                    "sentence": "",
                    "text": "",
                    "sentence_count": chapter_node.get("sentence_count"),
                },
                node_id_map,
                node_meta,
            )
            chapter_indices.append(chapter_idx)
            hierarchy_edges.append(
                (
                    version_idx,
                    chapter_idx,
                    float(CONFIG["hierarchy_weight"]),
                    "version_chapter",
                )
            )

            for sentence_node in chapter_node.get("children", []):
                if sentence_node.get("type") != "sentence":
                    continue
                sentence_number = int(sentence_node["sentence_number"])
                sentence_id = f"sent_{version}_{chapter_number}_{sentence_number}"
                sentence_idx = add_node(
                    sentence_id,
                    {
                        "type": "sentence",
                        "version": version,
                        "chapter": chapter_number,
                        "sentence": sentence_number,
                        "text": sentence_node.get("text", ""),
                    },
                    node_id_map,
                    node_meta,
                )
                sentence_indices.append(sentence_idx)
                hierarchy_edges.append(
                    (
                        chapter_idx,
                        sentence_idx,
                        float(CONFIG["hierarchy_weight"]),
                        "chapter_sentence",
                    )
                )

    type_counts = Counter(meta["type"] for meta in node_meta)
    log(
        "Tree loaded: "
        f"{type_counts.get('version', 0)} versions, "
        f"{type_counts.get('chapter', 0)} chapters, "
        f"{type_counts.get('sentence', 0)} sentences, "
        f"{len(hierarchy_edges)} hierarchy edges"
    )
    return (
        node_id_map,
        node_meta,
        hierarchy_edges,
        version_indices,
        chapter_indices,
        sentence_indices,
    )


def similarity_weight(distance: float, config: dict[str, Any]) -> float:
    if abs(distance) <= 1e-12:
        return float(config.get("l0_similarity_weight", 2.0))
    return math.exp(-distance / float(config["distance_tau"]))


def pair_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def layer_name_with_config(distance: float, config: dict[str, Any]) -> str:
    if abs(distance) <= 1e-12:
        return "L0"
    if distance <= 0.1:
        return "L1"
    if distance <= float(config.get("layer3_threshold", 0.2)):
        return "L2"
    return "L3"


def apply_hierarchy_weight(
    hierarchy_edges: list[tuple[int, int, float, str]], config: dict[str, Any]
) -> list[tuple[int, int, float, str]]:
    weight = float(config["hierarchy_weight"])
    return [(u, v, weight, edge_type) for u, v, _, edge_type in hierarchy_edges]


def load_similarity_candidates(
    similarity_path: Path,
    node_id_map: dict[str, int],
    config: dict[str, Any],
) -> tuple[dict[tuple[int, int], SimilarityCandidate], dict[str, Any]]:
    log(f"Loading similarity JSON: {similarity_path.name}")
    with similarity_path.open("r", encoding="utf-8") as f:
        raw_edges = json.load(f)
    if not isinstance(raw_edges, list):
        raise ValueError("Similarity JSON must be a list of edge records")

    candidates: dict[tuple[int, int], SimilarityCandidate] = {}
    missing_samples: list[dict[str, Any]] = []
    missing_count = 0
    invalid_distance_count = 0
    duplicate_count = 0
    distance_layer_counts = Counter()

    for i, edge in enumerate(raw_edges):
        try:
            chapter = int(edge["chapter_number"])
            sentence = int(edge["sentence_number"])
            src_version = str(edge["original_text_version"])
            dst_version = str(edge["modified_text_version"])
            distance = float(edge["normalized_distance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid similarity edge at index {i}: {exc}") from exc

        if not math.isfinite(distance):
            invalid_distance_count += 1
            continue
        distance = min(max(distance, 0.0), 1.0)

        src_id = f"sent_{src_version}_{chapter}_{sentence}"
        dst_id = f"sent_{dst_version}_{chapter}_{sentence}"
        src_idx = node_id_map.get(src_id)
        dst_idx = node_id_map.get(dst_id)
        if src_idx is None or dst_idx is None:
            missing_count += 1
            if len(missing_samples) < int(config["missing_edge_sample_limit"]):
                missing_samples.append(
                    {
                        "edge_index": i,
                        "src_id": src_id,
                        "dst_id": dst_id,
                        "missing_src": src_idx is None,
                        "missing_dst": dst_idx is None,
                        "normalized_distance": distance,
                    }
                )
            continue

        key = pair_key(src_idx, dst_idx)
        existing = candidates.get(key)
        if existing is None or distance < existing.distance:
            if existing is not None:
                duplicate_count += 1
            candidates[key] = SimilarityCandidate(src_idx, dst_idx, distance, i)
        else:
            duplicate_count += 1

    for candidate in candidates.values():
        distance_layer_counts[layer_name_with_config(candidate.distance, config)] += 1

    report = {
        "raw_similarity_edges": len(raw_edges),
        "valid_unique_sentence_pairs": len(candidates),
        "missing_edge_count": missing_count,
        "invalid_distance_count": invalid_distance_count,
        "duplicate_or_weaker_pair_count": duplicate_count,
        "distance_layer_counts_before_pruning": dict(distance_layer_counts),
        "missing_samples": missing_samples,
    }
    with Path(config["missing_edge_log_path"]).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if missing_count:
        log(
            f"Skipped {missing_count} similarity edges whose sentence node was absent; "
            f"sample saved to {Path(config['missing_edge_log_path']).name}"
        )
    log(
        "Similarity candidates: "
        f"{len(raw_edges)} raw, {len(candidates)} valid unique pairs"
    )
    return candidates, report


def layer_name(distance: float) -> str:
    return layer_name_with_config(distance, CONFIG)


def prune_similarity_edges(
    candidates: dict[tuple[int, int], SimilarityCandidate],
    sentence_indices: list[int],
    config: dict[str, Any],
) -> tuple[list[tuple[int, int, float, str]], dict[str, Any]]:
    log("Pruning similarity edges with L0/L1/L2/L3 policy")
    kept: dict[tuple[int, int], tuple[float, str, float]] = {}
    l2_neighbors: dict[int, list[SimilarityCandidate]] = defaultdict(list)
    all_neighbors: dict[int, list[SimilarityCandidate]] = defaultdict(list)
    stats = Counter()

    for key, candidate in candidates.items():
        all_neighbors[candidate.u].append(candidate)
        all_neighbors[candidate.v].append(candidate)
        layer = layer_name_with_config(candidate.distance, config)
        stats[f"{layer}_input"] += 1

        if layer in {"L0", "L1"}:
            kept[key] = (
                similarity_weight(candidate.distance, config),
                layer,
                candidate.distance,
            )
            stats[f"{layer}_kept"] += 1
        elif layer == "L2":
            l2_neighbors[candidate.u].append(candidate)
            l2_neighbors[candidate.v].append(candidate)
        else:
            stats["L3_deleted"] += 1

    selected_l2: set[tuple[int, int]] = set()
    top_k = int(config["l2_top_k"])
    for node_idx, neighbors in l2_neighbors.items():
        # L2只保留距离最小的邻居，等价于保留权重最大的弱相似边。
        best = sorted(neighbors, key=lambda c: c.distance)[:top_k]
        for candidate in best:
            selected_l2.add(pair_key(candidate.u, candidate.v))

    for key in selected_l2:
        candidate = candidates[key]
        kept[key] = (
            similarity_weight(candidate.distance, config),
            f"L2_top{top_k}",
            candidate.distance,
        )
    stats[f"L2_kept_top{top_k}_union"] = len(selected_l2)

    similarity_degree = Counter()
    for u, v in kept:
        if u == v:
            similarity_degree[u] += 1
        else:
            similarity_degree[u] += 1
            similarity_degree[v] += 1

    fallback_target = int(config["fallback_min_similarity_degree"])
    fallback_weight = float(config["fallback_weight"])
    fallback_edges_added = 0
    fallback_self_loops = 0

    for node_idx in sentence_indices:
        if similarity_degree[node_idx] >= fallback_target:
            continue
        neighbor_candidates = sorted(
            all_neighbors.get(node_idx, []),
            key=lambda c: (c.distance, c.v if c.u == node_idx else c.u, c.source_index),
        )
        for candidate in neighbor_candidates:
            if similarity_degree[node_idx] >= fallback_target:
                break
            key = pair_key(candidate.u, candidate.v)
            if key in kept:
                continue
            kept[key] = (fallback_weight, "fallback_min_distance", candidate.distance)
            fallback_edges_added += 1
            if candidate.u == candidate.v:
                similarity_degree[candidate.u] += 1
            else:
                similarity_degree[candidate.u] += 1
                similarity_degree[candidate.v] += 1

        if similarity_degree[node_idx] < fallback_target:
            key = (node_idx, node_idx)
            if key not in kept:
                kept[key] = (fallback_weight, "fallback_self_loop", 0.0)
                fallback_edges_added += 1
                fallback_self_loops += 1
                similarity_degree[node_idx] += 1

    pruned_edges = [(u, v, weight, source) for (u, v), (weight, source, _) in kept.items()]
    degrees = [similarity_degree[idx] for idx in sentence_indices]
    stats.update(
        {
            "fallback_edges_added": fallback_edges_added,
            "fallback_self_loops": fallback_self_loops,
            "final_similarity_edges": len(pruned_edges),
            "final_similarity_degree_min": min(degrees) if degrees else 0,
            "final_similarity_degree_mean": sum(degrees) / len(degrees)
            if degrees
            else 0.0,
            "final_similarity_degree_lt_target": sum(
                1 for degree in degrees if degree < fallback_target
            ),
        }
    )
    log(
        "Similarity pruning complete: "
        f"{stats['final_similarity_edges']} kept, "
        f"{stats['fallback_edges_added']} fallback edges"
    )
    return pruned_edges, dict(stats)


def init_node_features(
    deps: dict[str, Any],
    node_meta: list[dict[str, Any]],
    hidden_dim: int,
    config: dict[str, Any],
):
    torch = deps["torch"]
    node_count = len(node_meta)
    x = torch.randn(node_count, hidden_dim, dtype=torch.float32) * float(
        config.get("feature_noise_scale", 0.05)
    )
    scale = float(config.get("structural_feature_scale", 0.7))

    version_orders = [
        float(meta.get("source_index", 0) or 0)
        for meta in node_meta
        if meta.get("type") == "version"
    ]
    max_version_order = max(version_orders) if version_orders else 1.0
    max_chapter = max(
        [int(meta.get("chapter", 0) or 0) for meta in node_meta if meta.get("chapter", "") != ""]
        or [1]
    )
    max_sentence = max(
        [int(meta.get("sentence", 0) or 0) for meta in node_meta if meta.get("sentence", "") != ""]
        or [1]
    )

    def put(idx: int, dim: int, value: float) -> None:
        if dim < hidden_dim:
            x[idx, dim] = float(value)

    for idx, meta in enumerate(node_meta):
        node_type = meta["type"]
        if node_type == "version":
            type_dim = 0
        elif node_type == "chapter":
            type_dim = 1
        else:
            type_dim = 2
        put(idx, type_dim, scale)

        if node_type == "version":
            version_index = float(meta.get("source_index", 0) or 0)
            version_pos = version_index / max(1.0, max_version_order)
            put(idx, 3, scale * version_pos)
            put(idx, 4, scale * math.sin(version_pos * math.tau))
            put(idx, 5, scale * math.cos(version_pos * math.tau))
        elif node_type == "chapter":
            chapter = meta.get("chapter", 0)
            chapter_number = int(chapter) if chapter != "" else 0
            chapter_pos = chapter_number / max(1, max_chapter)
            put(idx, 6, scale * chapter_pos)
            put(idx, 7, scale * math.sin(chapter_pos * math.tau))
            put(idx, 8, scale * math.cos(chapter_pos * math.tau))
        else:
            chapter = meta.get("chapter", 0)
            sentence = meta.get("sentence", 0)
            chapter_number = int(chapter) if chapter != "" else 0
            sentence_number = int(sentence) if sentence != "" else 0
            chapter_pos = chapter_number / max(1, max_chapter)
            sentence_pos = sentence_number / max(1, max_sentence)
            put(idx, 6, scale * chapter_pos)
            put(idx, 7, scale * math.sin(chapter_pos * math.tau))
            put(idx, 8, scale * math.cos(chapter_pos * math.tau))
            put(idx, 9, scale * sentence_pos)
            put(idx, 10, scale * math.sin(sentence_pos * math.tau))
            put(idx, 11, scale * math.cos(sentence_pos * math.tau))

    return x


def build_pyg_data(
    deps: dict[str, Any],
    node_count: int,
    hierarchy_edges: list[tuple[int, int, float, str]],
    similarity_edges: list[tuple[int, int, float, str]],
    config: dict[str, Any],
):
    torch = deps["torch"]
    Data = deps["Data"]
    to_undirected = deps["to_undirected"]

    all_edges = hierarchy_edges + similarity_edges
    if not all_edges:
        raise ValueError("Graph has no edges")

    edge_index = torch.tensor(
        [[u for u, _, _, _ in all_edges], [v for _, v, _, _ in all_edges]],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([w for _, _, w, _ in all_edges], dtype=torch.float32)

    edge_index, edge_weight = to_undirected(
        edge_index, edge_weight, num_nodes=node_count, reduce="mean"
    )
    edge_weight = edge_weight.clamp(
        min=0.0, max=float(config.get("max_encoder_edge_weight", 2.0))
    )
    x = torch.randn(node_count, int(config["input_dim"]), dtype=torch.float32) * 0.01
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, num_nodes=node_count)
    log(
        "PyG Data built: "
        f"{node_count} nodes, {edge_index.size(1)} directed edges after to_undirected"
    )
    return data


def build_pyg_data_with_meta(
    deps: dict[str, Any],
    node_meta: list[dict[str, Any]],
    hierarchy_edges: list[tuple[int, int, float, str]],
    similarity_edges: list[tuple[int, int, float, str]],
    config: dict[str, Any],
):
    torch = deps["torch"]
    Data = deps["Data"]
    to_undirected = deps["to_undirected"]

    all_edges = hierarchy_edges + similarity_edges
    if not all_edges:
        raise ValueError("Graph has no edges")

    def make_edge_tensors(edges: list[tuple[int, int, float, str]]):
        if not edges:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
            )
        return (
            torch.tensor(
                [[u for u, _, _, _ in edges], [v for _, v, _, _ in edges]],
                dtype=torch.long,
            ),
            torch.tensor([w for _, _, w, _ in edges], dtype=torch.float32),
        )

    edge_index, edge_weight = make_edge_tensors(all_edges)
    edge_index, edge_weight = to_undirected(
        edge_index, edge_weight, num_nodes=len(node_meta), reduce="mean"
    )
    edge_weight = edge_weight.clamp(
        min=0.0, max=float(config.get("max_encoder_edge_weight", 2.0))
    )
    x = init_node_features(deps, node_meta, int(config["input_dim"]), config)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=len(node_meta),
    )
    hierarchy_edge_index, _ = make_edge_tensors(hierarchy_edges)
    similarity_edge_index, _ = make_edge_tensors(similarity_edges)
    data.hierarchy_edge_index = to_undirected(
        hierarchy_edge_index, num_nodes=len(node_meta)
    )
    data.similarity_edge_index = to_undirected(
        similarity_edge_index, num_nodes=len(node_meta)
    )
    log(
        "PyG Data built: "
        f"{len(node_meta)} nodes, {edge_index.size(1)} directed edges after "
        f"to_undirected, hierarchy_weight={config['hierarchy_weight']}"
    )
    return data


def make_model_class(deps: dict[str, Any]):
    torch = deps["torch"]
    nn = deps["nn"]
    F = deps["F"]
    GCNConv = deps["GCNConv"]
    negative_sampling = deps["negative_sampling"]

    class GCNVariationalEncoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            model_config: dict[str, Any],
        ):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv_mu = GCNConv(hidden_dim, latent_dim)
            self.conv_logvar = GCNConv(hidden_dim, latent_dim)
            self.model_config = model_config

        def forward(self, x, edge_index, edge_weight=None):
            dropout = float(self.model_config.get("dropout", 0.15))
            h1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
            h1 = F.dropout(h1, p=dropout, training=self.training)

            h2 = F.relu(self.conv2(h1, edge_index, edge_weight=edge_weight))
            # 残差连接保留上一层特征，降低GCN过度平滑导致的向量坍缩风险。
            h2 = h2 + h1
            h2 = F.dropout(h2, p=dropout, training=self.training)

            mu = self.conv_mu(h2, edge_index, edge_weight=edge_weight)
            logvar = self.conv_logvar(h2, edge_index, edge_weight=edge_weight).clamp(
                min=float(self.model_config.get("min_logvar", -6.0)),
                max=float(self.model_config.get("max_logvar", 2.0)),
            )
            return (
                mu,
                logvar,
            )

    class TrainableFeatureVGAE(nn.Module):
        def __init__(
            self,
            num_nodes: int,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            initial_x,
            model_config: dict[str, Any],
        ):
            super().__init__()
            self.node_features = nn.Parameter(initial_x.clone())
            self.model_config = model_config
            self.encoder = GCNVariationalEncoder(
                input_dim, hidden_dim, latent_dim, model_config
            )
            self.mu = None
            self.logvar = None

        def encode(self, edge_index, edge_weight=None):
            self.mu, self.logvar = self.encoder(
                self.node_features, edge_index, edge_weight
            )
            if self.training and not bool(self.model_config.get("use_mu_for_recon", True)):
                std = torch.exp(0.5 * self.logvar)
                eps = torch.randn_like(std)
                return self.mu + eps * std
            return self.mu

        @staticmethod
        def decode_logits(z, edge_index):
            return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        def recon_loss(self, z, pos_edge_index, num_nodes: int, exclusion_edge_index=None):
            if pos_edge_index.numel() == 0:
                return z.new_tensor(0.0)
            neg_edge_index = negative_sampling(
                edge_index=exclusion_edge_index
                if exclusion_edge_index is not None
                else pos_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=pos_edge_index.size(1),
                method="sparse",
            )
            pos_logits = self.decode_logits(z, pos_edge_index)
            neg_logits = self.decode_logits(z, neg_edge_index)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )
            return pos_loss + neg_loss

        def kl_loss(self):
            if self.mu is None or self.logvar is None:
                raise RuntimeError("encode must be called before kl_loss")
            kl = 1.0 + self.logvar - self.mu.pow(2) - self.logvar.exp()
            return -0.5 * torch.mean(torch.sum(kl, dim=1))

    return TrainableFeatureVGAE


def get_kl_weight(epoch: int, total_epochs: int, config: dict[str, Any]) -> float:
    anneal_start = int(config.get("kl_anneal_epochs", 100))
    full_epoch = int(config.get("kl_full_epochs", 300))
    target = float(config.get("kl_weight", 1.0))
    if epoch < anneal_start:
        return 0.0
    if epoch < full_epoch:
        progress = (epoch - anneal_start) / max(1, full_epoch - anneal_start)
        return target * progress
    return target


def train_vgae(deps: dict[str, Any], data, config: dict[str, Any]):
    torch = deps["torch"]
    Model = make_model_class(deps)
    device = (
        torch.device("cuda")
        if config["device"] == "cuda_if_available" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    data = data.to(device)
    model = Model(
        num_nodes=data.num_nodes,
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        initial_x=data.x,
        model_config=config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    log(f"Training VGAE on {device} for {config['epochs']} epochs")
    history: list[dict[str, float]] = []
    start_time = time.monotonic()
    timeout_seconds = float(config.get("trial_timeout_seconds", 1800))
    for epoch in range(1, int(config["epochs"]) + 1):
        if time.monotonic() - start_time > timeout_seconds:
            raise TimeoutError(
                f"Trial exceeded {timeout_seconds:.0f}s before epoch {epoch}"
            )
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.edge_index, data.edge_weight)
        similarity_recon = model.recon_loss(
            z,
            data.similarity_edge_index,
            data.num_nodes,
            exclusion_edge_index=data.edge_index,
        )
        hierarchy_recon = model.recon_loss(
            z,
            data.hierarchy_edge_index,
            data.num_nodes,
            exclusion_edge_index=data.edge_index,
        )
        recon = (
            float(config.get("similarity_recon_weight", 1.0)) * similarity_recon
            + float(config.get("hierarchy_recon_weight", 0.25)) * hierarchy_recon
        )
        kl = model.kl_loss()
        current_kl_weight = get_kl_weight(epoch, int(config["epochs"]), config)
        loss = recon + current_kl_weight * kl
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}: "
                f"loss={float(loss.detach().cpu())}, "
                f"recon={float(recon.detach().cpu())}, "
                f"kl={float(kl.detach().cpu())}"
            )
        loss.backward()
        clip_norm = float(config.get("gradient_clip_norm", 0.0))
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        item = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "recon_loss": float(recon.detach().cpu()),
            "similarity_recon_loss": float(similarity_recon.detach().cpu()),
            "hierarchy_recon_loss": float(hierarchy_recon.detach().cpu()),
            "kl_loss": float(kl.detach().cpu()),
            "kl_weight": float(current_kl_weight),
        }
        history.append(item)
        if epoch == 1 or epoch % int(config["log_every"]) == 0 or epoch == config["epochs"]:
            log(
                f"Epoch {epoch:04d} | "
                f"loss={item['loss']:.4f} "
                f"recon={item['recon_loss']:.4f} "
                f"sim={item['similarity_recon_loss']:.4f} "
                f"hier={item['hierarchy_recon_loss']:.4f} "
                f"kl={item['kl_loss']:.4f} "
                f"kl_w={item['kl_weight']:.6f}"
            )

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.edge_index, data.edge_weight).detach().cpu()
    return model, embeddings, history, data.cpu()


def labels_and_vectors(
    embeddings,
    node_meta: list[dict[str, Any]],
    node_indices: list[int],
    label_field: str,
):
    vectors = embeddings[node_indices].numpy()
    labels = [node_meta[idx].get(label_field) for idx in node_indices]
    return vectors, labels


def safe_silhouette(deps: dict[str, Any], vectors, labels, title: str) -> float | None:
    np = deps["np"]
    silhouette_score = deps["silhouette_score"]
    if len(vectors) < 3:
        log(f"{title} silhouette skipped: not enough samples")
        return None
    labels_array = np.array(labels)
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        log(f"{title} silhouette skipped: less than two labels")
        return None
    counts = Counter(labels)
    valid_mask = np.array([counts[label] >= 2 for label in labels])
    if valid_mask.sum() < 3 or len(set(labels_array[valid_mask].tolist())) < 2:
        log(f"{title} silhouette skipped: clusters are too small")
        return None
    score = float(silhouette_score(vectors[valid_mask], labels_array[valid_mask]))
    log(f"{title} silhouette score: {score:.4f}")
    return score


def evaluate_and_save(
    deps: dict[str, Any],
    embeddings,
    node_meta: list[dict[str, Any]],
    version_indices: list[int],
    chapter_indices: list[int],
    sentence_indices: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    np = deps["np"]
    plt = deps["plt"]
    TSNE = deps["TSNE"]
    cosine_similarity = deps["cosine_similarity"]

    sentence_vectors, sentence_chapter_labels = labels_and_vectors(
        embeddings, node_meta, sentence_indices, "chapter"
    )
    chapter_vectors, chapter_number_labels = labels_and_vectors(
        embeddings, node_meta, chapter_indices, "chapter"
    )

    sentence_silhouette = safe_silhouette(
        deps, sentence_vectors, sentence_chapter_labels, "Sentence same-chapter"
    )
    chapter_silhouette = safe_silhouette(
        deps, chapter_vectors, chapter_number_labels, "Chapter same-name"
    )

    version_names = [str(node_meta[idx]["version"]) for idx in version_indices]
    version_vectors = embeddings[version_indices].numpy()
    version_similarity = cosine_similarity(version_vectors)
    save_version_similarity_csv(
        version_names, version_similarity, Path(config["version_similarity_csv_path"])
    )

    save_tsne_plot(
        deps,
        sentence_vectors,
        sentence_chapter_labels,
        Path(config["tsne_png_path"]),
        config,
    )
    save_node_type_tsne_plot(
        deps,
        chapter_vectors,
        chapter_number_labels,
        Path(config["chapter_tsne_png_path"]),
        config,
        title="Chapter Node t-SNE by Chapter Number",
        colorbar_label="Chapter number",
        point_size=18,
    )
    save_node_type_tsne_plot(
        deps,
        version_vectors,
        version_names,
        Path(config["version_tsne_png_path"]),
        config,
        title="Version Node t-SNE",
        colorbar_label="Version index",
        point_size=72,
        annotate=True,
    )

    return {
        "sentence_same_chapter_silhouette": sentence_silhouette,
        "chapter_same_name_silhouette": chapter_silhouette,
        "version_names": version_names,
        "version_similarity_shape": list(np.array(version_similarity).shape),
        "embedding_stats": embedding_stats(deps, embeddings),
    }


def evaluate_metrics_only(
    deps: dict[str, Any],
    embeddings,
    node_meta: list[dict[str, Any]],
    chapter_indices: list[int],
    sentence_indices: list[int],
) -> dict[str, Any]:
    sentence_vectors, sentence_chapter_labels = labels_and_vectors(
        embeddings, node_meta, sentence_indices, "chapter"
    )
    chapter_vectors, chapter_number_labels = labels_and_vectors(
        embeddings, node_meta, chapter_indices, "chapter"
    )
    sentence_silhouette = safe_silhouette(
        deps, sentence_vectors, sentence_chapter_labels, "Sentence same-chapter"
    )
    chapter_silhouette = safe_silhouette(
        deps, chapter_vectors, chapter_number_labels, "Chapter same-name"
    )
    stats = embedding_stats(deps, embeddings)
    return {
        "sentence_same_chapter_silhouette": sentence_silhouette,
        "chapter_same_name_silhouette": chapter_silhouette,
        "embedding_stats": stats,
    }


def embedding_stats(deps: dict[str, Any], embeddings) -> dict[str, float]:
    torch = deps["torch"]
    tensor = embeddings.detach().cpu() if hasattr(embeddings, "detach") else embeddings
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "l2_mean": float(torch.linalg.norm(tensor, dim=1).mean().item()),
    }


def save_version_similarity_csv(version_names: list[str], matrix, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version"] + version_names)
        for name, row in zip(version_names, matrix):
            writer.writerow([name] + [f"{float(value):.8f}" for value in row])
    log(f"Saved version similarity matrix: {path.name}")


def save_tsne_plot(
    deps: dict[str, Any],
    vectors,
    labels: list[Any],
    path: Path,
    config: dict[str, Any],
) -> None:
    np = deps["np"]
    plt = deps["plt"]
    TSNE = deps["TSNE"]

    sample_count = len(vectors)
    if sample_count < 3:
        log("t-SNE skipped: not enough sentence nodes")
        return
    perplexity = min(int(config["tsne_perplexity"]), max(2, (sample_count - 1) // 3))
    log(f"Running t-SNE for {sample_count} sentence nodes (perplexity={perplexity})")
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(config["tsne_random_state"]),
    ).fit_transform(vectors)

    label_values = np.array([int(label) if label != "" else -1 for label in labels])
    fig, ax = plt.subplots(figsize=(11, 9), dpi=220)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=label_values,
        cmap="tab20",
        s=7,
        alpha=0.78,
        linewidths=0,
    )
    ax.set_title("Sentence Node t-SNE by Chapter")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Chapter number")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved t-SNE visualization: {path.name}")


def save_node_type_tsne_plot(
    deps: dict[str, Any],
    vectors,
    labels: list[Any],
    path: Path,
    config: dict[str, Any],
    title: str,
    colorbar_label: str,
    point_size: int,
    annotate: bool = False,
) -> None:
    np = deps["np"]
    plt = deps["plt"]
    TSNE = deps["TSNE"]

    sample_count = len(vectors)
    if sample_count < 3:
        log(f"{title} skipped: not enough nodes")
        return
    perplexity = min(int(config["tsne_perplexity"]), max(2, (sample_count - 1) // 3))
    log(f"Running {title} for {sample_count} nodes (perplexity={perplexity})")
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(config["tsne_random_state"]),
    ).fit_transform(vectors)

    unique_labels = {label: i for i, label in enumerate(sorted(set(labels), key=str))}
    label_values = np.array([unique_labels[label] for label in labels])
    fig, ax = plt.subplots(figsize=(10, 8), dpi=220)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=label_values,
        cmap="tab20",
        s=point_size,
        alpha=0.82,
        linewidths=0.25,
        edgecolors="white" if annotate else "none",
    )
    if annotate:
        for (x_coord, y_coord), label in zip(coords, labels):
            ax.annotate(
                str(label),
                (x_coord, y_coord),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"Saved {title}: {path.name}")


def save_node_vectors_csv(
    embeddings,
    node_meta: list[dict[str, Any]],
    path: Path,
) -> None:
    latent_dim = embeddings.shape[1]
    header = [
        "node_id",
        "type",
        "version",
        "chapter",
        "sentence",
    ] + [f"dim_{i}" for i in range(latent_dim)]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        array = embeddings.numpy()
        for idx, meta in enumerate(node_meta):
            writer.writerow(
                [
                    meta.get("node_id", ""),
                    meta.get("type", ""),
                    meta.get("version", ""),
                    meta.get("chapter", ""),
                    meta.get("sentence", ""),
                ]
                + [f"{float(value):.8f}" for value in array[idx]]
            )
    log(f"Saved node vectors CSV: {path.name}")


def save_torch_output(
    deps: dict[str, Any],
    model,
    embeddings,
    node_id_map: dict[str, int],
    node_meta: list[dict[str, Any]],
    config: dict[str, Any],
    graph_stats: dict[str, Any],
    pruning_stats: dict[str, Any],
    eval_stats: dict[str, Any],
    training_history: list[dict[str, float]],
) -> None:
    torch = deps["torch"]
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    payload = {
        "model_state_dict": model.state_dict(),
        "node_embeddings": embeddings,
        "node_id_map": node_id_map,
        "node_meta": node_meta,
        "config": serializable_config,
        "graph_stats": graph_stats,
        "pruning_stats": pruning_stats,
        "eval_stats": eval_stats,
        "training_history": training_history,
    }
    torch.save(payload, config["vgae_output_path"])
    log(f"Saved VGAE output bundle: {Path(config['vgae_output_path']).name}")


def build_graph_stats(
    node_meta: list[dict[str, Any]],
    hierarchy_edges: list[tuple[int, int, float, str]],
    similarity_edges: list[tuple[int, int, float, str]],
    data,
    similarity_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "node_counts": dict(Counter(meta["type"] for meta in node_meta)),
        "hierarchy_edge_count_directed_before_undirected": len(hierarchy_edges),
        "similarity_edge_count_before_undirected": len(similarity_edges),
        "edge_count_after_to_undirected": int(data.edge_index.size(1)),
        "similarity_load_report": similarity_report,
    }


def fmt_score(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }


def with_artifact_paths(config: dict[str, Any], prefix: str) -> dict[str, Any]:
    artifact_config = config.copy()
    if prefix == "sentence_best":
        artifact_config["vgae_output_path"] = config["sentence_best_vgae_output_path"]
        artifact_config["node_vectors_csv_path"] = config[
            "sentence_best_node_vectors_csv_path"
        ]
        artifact_config["version_similarity_csv_path"] = config[
            "sentence_best_version_similarity_csv_path"
        ]
        artifact_config["tsne_png_path"] = config["sentence_best_tsne_png_path"]
        artifact_config["chapter_tsne_png_path"] = config[
            "sentence_best_chapter_tsne_png_path"
        ]
        artifact_config["version_tsne_png_path"] = config[
            "sentence_best_version_tsne_png_path"
        ]
        artifact_config["best_config_path"] = config["sentence_best_config_path"]
    return artifact_config


def make_trial_config(base_config: dict[str, Any], trial: int) -> dict[str, Any]:
    config = base_config.copy()
    trial_grid = [
        {"hierarchy_weight": 0.05, "learning_rate": 0.0015, "kl_weight": 0.0010, "hierarchy_recon_weight": 0.25},
        {"hierarchy_weight": 0.03, "learning_rate": 0.0010, "kl_weight": 0.0010, "hierarchy_recon_weight": 0.20},
        {"hierarchy_weight": 0.08, "learning_rate": 0.0010, "kl_weight": 0.0007, "hierarchy_recon_weight": 0.20},
        {"hierarchy_weight": 0.05, "learning_rate": 0.0007, "kl_weight": 0.0007, "hierarchy_recon_weight": 0.15},
        {"hierarchy_weight": 0.02, "learning_rate": 0.0007, "kl_weight": 0.0005, "hierarchy_recon_weight": 0.15},
        {"hierarchy_weight": 0.10, "learning_rate": 0.0007, "kl_weight": 0.0005, "hierarchy_recon_weight": 0.10},
        {"hierarchy_weight": 0.05, "learning_rate": 0.0005, "kl_weight": 0.0003, "hierarchy_recon_weight": 0.10},
        {"hierarchy_weight": 0.03, "learning_rate": 0.0005, "kl_weight": 0.0002, "hierarchy_recon_weight": 0.10},
    ]
    grid_item = trial_grid[trial % len(trial_grid)]
    config["trial_index"] = trial
    config.update(grid_item)
    config["seed"] = int(base_config["seed"]) + trial
    return config


def train_single_trial(
    deps: dict[str, Any],
    config: dict[str, Any],
    node_meta: list[dict[str, Any]],
    hierarchy_edges: list[tuple[int, int, float, str]],
    candidates: dict[tuple[int, int], SimilarityCandidate],
    version_indices: list[int],
    chapter_indices: list[int],
    sentence_indices: list[int],
    similarity_report: dict[str, Any],
) -> dict[str, Any]:
    # 每个trial使用自己的随机种子、层级权重和剪枝配置，避免复用上一次的图结构。
    set_seed(int(config["seed"]), deps["torch"])
    weighted_hierarchy_edges = apply_hierarchy_weight(hierarchy_edges, config)
    similarity_edges, pruning_stats = prune_similarity_edges(
        candidates, sentence_indices, config
    )
    data = build_pyg_data_with_meta(
        deps, node_meta, weighted_hierarchy_edges, similarity_edges, config
    )
    model, embeddings, training_history, trained_data = train_vgae(deps, data, config)
    metrics = evaluate_metrics_only(
        deps, embeddings, node_meta, chapter_indices, sentence_indices
    )
    graph_stats = build_graph_stats(
        node_meta,
        weighted_hierarchy_edges,
        similarity_edges,
        trained_data,
        similarity_report,
    )
    graph_stats["trial_index"] = config.get("trial_index")
    graph_stats["hierarchy_weight"] = config["hierarchy_weight"]
    return {
        "model": model,
        "embeddings": embeddings,
        "training_history": training_history,
        "data": trained_data,
        "similarity_edges": similarity_edges,
        "pruning_stats": pruning_stats,
        "metrics": metrics,
        "graph_stats": graph_stats,
    }


def trial_combined_score(metrics: dict[str, Any]) -> float:
    sent = metrics.get("sentence_same_chapter_silhouette")
    chap = metrics.get("chapter_same_name_silhouette")
    std = float(metrics.get("embedding_stats", {}).get("std", 0.0) or 0.0)
    l2_mean = float(metrics.get("embedding_stats", {}).get("l2_mean", 0.0) or 0.0)
    std_bonus = min(std, 0.2)
    collapse_penalty = 0.5 if std < 0.01 or l2_mean < 0.1 else 0.0
    return (
        (sent if sent is not None else -1.0)
        + (chap if chap is not None else -1.0)
        + std_bonus
        - collapse_penalty
    )


def trial_sentence_priority_score(metrics: dict[str, Any]) -> float:
    sent = metrics.get("sentence_same_chapter_silhouette")
    chap = metrics.get("chapter_same_name_silhouette")
    std = float(metrics.get("embedding_stats", {}).get("std", 0.0) or 0.0)
    l2_mean = float(metrics.get("embedding_stats", {}).get("l2_mean", 0.0) or 0.0)
    collapse_penalty = 0.5 if std < 0.01 or l2_mean < 0.1 else 0.0
    return (
        (sent if sent is not None else -1.0)
        + 0.25 * (chap if chap is not None else -1.0)
        + 0.25 * min(std, 0.2)
        - collapse_penalty
    )


def reached_targets(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    sent = metrics.get("sentence_same_chapter_silhouette")
    chap = metrics.get("chapter_same_name_silhouette")
    std = metrics.get("embedding_stats", {}).get("std", 0.0)
    return (
        sent is not None
        and chap is not None
        and sent > float(config["sentence_silhouette_target"])
        and chap > float(config["chapter_silhouette_target"])
        and std > float(config["embedding_std_target"])
    )


def save_best_artifacts(
    deps: dict[str, Any],
    result: dict[str, Any],
    node_id_map: dict[str, int],
    node_meta: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    result["model"].to("cpu")
    save_node_vectors_csv(
        result["embeddings"], node_meta, Path(config["node_vectors_csv_path"])
    )
    save_torch_output(
        deps,
        result["model"],
        result["embeddings"],
        node_id_map,
        node_meta,
        config,
        result["graph_stats"],
        result["pruning_stats"],
        result["metrics"],
        result["training_history"],
    )
    save_json(config["best_config_path"], serializable_config(config))


def auto_train_loop(
    deps: dict[str, Any],
    base_config: dict[str, Any],
    node_id_map: dict[str, int],
    node_meta: list[dict[str, Any]],
    hierarchy_edges: list[tuple[int, int, float, str]],
    candidates: dict[tuple[int, int], SimilarityCandidate],
    version_indices: list[int],
    chapter_indices: list[int],
    sentence_indices: list[int],
    similarity_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    max_trials = int(base_config.get("max_trials", 10))
    best_score = -999.0
    best_sentence_score = -999.0
    best_result: dict[str, Any] | None = None
    best_config: dict[str, Any] | None = None
    best_sentence_result: dict[str, Any] | None = None
    best_sentence_config: dict[str, Any] | None = None
    training_log: list[dict[str, Any]] = []
    torch = deps["torch"]

    for trial in range(max_trials):
        config = make_trial_config(base_config, trial)
        log(
            f"Trial {trial + 1}/{max_trials} start | "
            f"hierarchy_weight={config['hierarchy_weight']:.4f}, "
            f"lr={config['learning_rate']:.5f}, "
            f"L2_top_k={config['l2_top_k']}, "
            f"L3_threshold={config['layer3_threshold']}"
        )
        started = time.monotonic()
        result: dict[str, Any] | None = None
        try:
            result = train_single_trial(
                deps,
                config,
                node_meta,
                hierarchy_edges,
                candidates,
                version_indices,
                chapter_indices,
                sentence_indices,
                similarity_report,
            )
            metrics = result["metrics"]
            combined = trial_combined_score(metrics)
            sentence_priority = trial_sentence_priority_score(metrics)
            elapsed = time.monotonic() - started
            sent = metrics.get("sentence_same_chapter_silhouette")
            chap = metrics.get("chapter_same_name_silhouette")
            std = metrics.get("embedding_stats", {}).get("std", 0.0)
            log(
                f"Trial {trial + 1}: sent={fmt_score(sent)}, "
                f"chap={fmt_score(chap)}, std={std:.4f}, "
                f"combined={combined:.4f}, "
                f"sentence_priority={sentence_priority:.4f}, "
                f"elapsed={elapsed:.1f}s"
            )
            trial_log = {
                "trial": trial + 1,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "config": serializable_config(config),
                "metrics": metrics,
                "pruning_stats": result["pruning_stats"],
                "training_history": result["training_history"],
                "combined_score": combined,
                "sentence_priority_score": sentence_priority,
            }
            training_log.append(trial_log)

            if combined > best_score:
                best_score = combined
                old_best = best_result
                best_result = result
                best_config = config
                save_best_artifacts(deps, result, node_id_map, node_meta, config)
                log(f"New best trial saved: {trial + 1}, combined={combined:.4f}")
                if old_best is not None:
                    del old_best

            if sentence_priority > best_sentence_score:
                best_sentence_score = sentence_priority
                old_sentence_best = best_sentence_result
                best_sentence_result = result
                best_sentence_config = config
                sentence_artifact_config = with_artifact_paths(config, "sentence_best")
                save_best_artifacts(
                    deps,
                    result,
                    node_id_map,
                    node_meta,
                    sentence_artifact_config,
                )
                log(
                    f"New sentence-best trial saved: {trial + 1}, "
                    f"sentence_priority={sentence_priority:.4f}"
                )
                if old_sentence_best is not None and old_sentence_best is not best_result:
                    del old_sentence_best

            save_json(base_config["training_log_path"], training_log)
            if reached_targets(metrics, config):
                log("Convergence targets reached; stopping auto loop.")
                break
        except Exception as exc:
            elapsed = time.monotonic() - started
            log(f"Trial {trial + 1} failed after {elapsed:.1f}s: {exc}")
            training_log.append(
                {
                    "trial": trial + 1,
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "config": serializable_config(config),
                    "error": repr(exc),
                }
            )
            save_json(base_config["training_log_path"], training_log)
        finally:
            # 夜间运行时尽量主动释放本轮临时图和CUDA缓存。
            if (
                result is not None
                and result is not best_result
                and result is not best_sentence_result
            ):
                del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_result is None or best_config is None:
        raise RuntimeError("All trials failed; no best model was produced")
    if best_sentence_result is None or best_sentence_config is None:
        raise RuntimeError("All trials failed; no sentence-best model was produced")

    save_json(base_config["training_log_path"], training_log)
    return best_result, best_config, best_sentence_result, best_sentence_config


def main() -> int:
    try:
        deps = require_dependencies()
        set_seed(int(CONFIG["seed"]), deps["torch"])

        tree_path = resolve_existing_path(CONFIG["tree_json_path"])
        similarity_path = resolve_existing_path(
            CONFIG["similarity_json_path"], CONFIG.get("similarity_json_fallbacks", [])
        )

        (
            node_id_map,
            node_meta,
            hierarchy_edges,
            version_indices,
            chapter_indices,
            sentence_indices,
        ) = load_tree_graph(tree_path)

        candidates, similarity_report = load_similarity_candidates(
            similarity_path, node_id_map, CONFIG
        )
        (
            best_result,
            best_config,
            best_sentence_result,
            best_sentence_config,
        ) = auto_train_loop(
            deps,
            CONFIG,
            node_id_map,
            node_meta,
            hierarchy_edges,
            candidates,
            version_indices,
            chapter_indices,
            sentence_indices,
            similarity_report,
        )

        eval_stats = evaluate_and_save(
            deps,
            best_result["embeddings"],
            node_meta,
            version_indices,
            chapter_indices,
            sentence_indices,
            best_config,
        )
        save_node_vectors_csv(
            best_result["embeddings"], node_meta, Path(best_config["node_vectors_csv_path"])
        )
        save_torch_output(
            deps,
            best_result["model"],
            best_result["embeddings"],
            node_id_map,
            node_meta,
            best_config,
            best_result["graph_stats"],
            best_result["pruning_stats"],
            eval_stats,
            best_result["training_history"],
        )
        save_json(best_config["best_config_path"], serializable_config(best_config))
        sentence_artifact_config = with_artifact_paths(
            best_sentence_config, "sentence_best"
        )
        sentence_eval_stats = evaluate_and_save(
            deps,
            best_sentence_result["embeddings"],
            node_meta,
            version_indices,
            chapter_indices,
            sentence_indices,
            sentence_artifact_config,
        )
        save_node_vectors_csv(
            best_sentence_result["embeddings"],
            node_meta,
            Path(sentence_artifact_config["node_vectors_csv_path"]),
        )
        save_torch_output(
            deps,
            best_sentence_result["model"],
            best_sentence_result["embeddings"],
            node_id_map,
            node_meta,
            sentence_artifact_config,
            best_sentence_result["graph_stats"],
            best_sentence_result["pruning_stats"],
            sentence_eval_stats,
            best_sentence_result["training_history"],
        )
        save_json(
            sentence_artifact_config["best_config_path"],
            serializable_config(sentence_artifact_config),
        )
        log("Done.")
        return 0
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
