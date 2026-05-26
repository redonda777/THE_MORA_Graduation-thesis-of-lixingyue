from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations, islice
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def log(message: str) -> None:
    print(message, flush=True)


def read_version_similarity_matrix(path: Path) -> tuple[list[str], dict[tuple[str, str], float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        versions = header[1:]
        pair_scores: dict[tuple[str, str], float] = {}
        for row in reader:
            version_a = row[0]
            for version_b, raw_value in zip(versions, row[1:]):
                if version_a == version_b:
                    continue
                key = tuple(sorted((version_a, version_b)))
                pair_scores[key] = max(pair_scores.get(key, -1.0), float(raw_value))
    return versions, pair_scores


def write_version_pair_ranking(
    pair_scores: dict[tuple[str, str], float], output_path: Path
) -> list[dict[str, Any]]:
    rows = []
    for (version_a, version_b), similarity in pair_scores.items():
        rows.append(
            {
                "version_a": version_a,
                "version_b": version_b,
                "similarity": similarity,
                "distance_1_minus_similarity": 1.0 - similarity,
            }
        )
    rows.sort(key=lambda row: row["similarity"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "rank",
            "version_a",
            "version_b",
            "similarity",
            "distance_1_minus_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def require_networkx():
    try:
        import networkx as nx
        from networkx.algorithms import community
    except Exception as exc:
        raise RuntimeError(
            "NetworkX is required for Girvan-Newman community detection. "
            "Install it in the mora environment with: pip install networkx"
        ) from exc
    return nx, community


def add_similarity_edges_to_graph(
    nx,
    graph,
    pair_scores: dict[tuple[str, str], float],
    threshold: float,
    top_k: int,
) -> None:
    keep_edges: set[tuple[str, str]] = set()
    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (version_a, version_b), similarity in pair_scores.items():
        if similarity >= threshold:
            keep_edges.add(tuple(sorted((version_a, version_b))))
        neighbors[version_a].append((version_b, similarity))
        neighbors[version_b].append((version_a, similarity))

    for version, items in neighbors.items():
        for neighbor, _ in sorted(items, key=lambda item: item[1], reverse=True)[:top_k]:
            keep_edges.add(tuple(sorted((version, neighbor))))

    for version_a, version_b in sorted(keep_edges):
        similarity = pair_scores[(version_a, version_b)]
        distance = max(1.0 - similarity, 1e-6)
        graph.add_edge(
            version_a,
            version_b,
            similarity=similarity,
            weight=similarity,
            distance=distance,
        )


def weighted_girvan_newman_best_partition(
    nx,
    community,
    graph,
    max_communities: int,
) -> dict[str, Any]:
    if graph.number_of_edges() == 0:
        communities = [sorted(component) for component in nx.connected_components(graph)]
        return {
            "communities": communities,
            "community_count": len(communities),
            "modularity": 0.0,
            "evaluated_partitions": [],
        }

    def most_valuable_edge(current_graph):
        betweenness = nx.edge_betweenness_centrality(
            current_graph, weight="distance"
        )
        return max(betweenness, key=betweenness.get)

    best: dict[str, Any] | None = None
    evaluated = []
    generator = community.girvan_newman(graph, most_valuable_edge=most_valuable_edge)
    for partition in islice(generator, max(1, max_communities - 1)):
        communities = [sorted(group) for group in partition]
        community_count = len(communities)
        modularity = community.modularity(graph, communities, weight="weight")
        item = {
            "community_count": community_count,
            "modularity": modularity,
            "communities": communities,
        }
        evaluated.append(item)
        if best is None or modularity > best["modularity"]:
            best = item
        if community_count >= max_communities:
            break

    if best is None:
        communities = [sorted(component) for component in nx.connected_components(graph)]
        best = {
            "community_count": len(communities),
            "modularity": 0.0,
            "communities": communities,
        }
    return {
        **best,
        "evaluated_partitions": evaluated,
    }


def write_membership_csv(communities: list[list[str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "community_id"])
        for community_id, nodes in enumerate(communities, start=1):
            for node in nodes:
                writer.writerow([node, community_id])


def load_tree_versions_and_chapters(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    with path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    versions = []
    chapters_by_version: dict[str, list[int]] = {}
    for version_node in root.get("children", []):
        if version_node.get("type") != "version":
            continue
        version = str(version_node.get("name") or version_node.get("version"))
        versions.append(version)
        chapters = []
        for chapter_node in version_node.get("children", []):
            if chapter_node.get("type") == "chapter":
                chapters.append(int(chapter_node["chapter_number"]))
        chapters_by_version[version] = sorted(chapters)
    return versions, chapters_by_version


def similarity_from_distance(distance: float, tau: float, transform: str) -> float:
    distance = min(max(distance, 0.0), 1.0)
    if transform == "linear":
        return 1.0 - distance
    return math.exp(-distance / tau)


def build_chapter_pair_scores(
    similarity_json_path: Path,
    tau: float,
    transform: str,
) -> tuple[
    dict[int, dict[tuple[str, str], list[float]]],
    dict[tuple[str, str], list[float]],
]:
    with similarity_json_path.open("r", encoding="utf-8") as f:
        raw_edges = json.load(f)

    chapter_scores: dict[int, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    book_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for edge in raw_edges:
        try:
            chapter = int(edge["chapter_number"])
            version_a = str(edge["original_text_version"])
            version_b = str(edge["modified_text_version"])
            distance = float(edge["normalized_distance"])
        except (KeyError, TypeError, ValueError):
            continue
        if version_a == version_b or not math.isfinite(distance):
            continue
        pair = tuple(sorted((version_a, version_b)))
        similarity = similarity_from_distance(distance, tau, transform)
        chapter_scores[chapter][pair].append(similarity)
        book_scores[pair].append(similarity)
    return chapter_scores, book_scores


def aggregate_scores(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    if mode == "max":
        return ordered[0]
    if mode == "top3_mean":
        return sum(ordered[:3]) / min(3, len(ordered))
    return sum(values) / len(values)


def aggregate_pair_scores(
    raw_scores: dict[tuple[str, str], list[float]], mode: str
) -> dict[tuple[str, str], float]:
    return {pair: aggregate_scores(values, mode) for pair, values in raw_scores.items()}


def write_chapter_affinity_edges(
    chapter_scores: dict[int, dict[tuple[str, str], list[float]]],
    output_path: Path,
    aggregation: str,
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "chapter",
                "version_a",
                "version_b",
                "similarity",
                "supporting_sentence_pairs",
            ]
        )
        for chapter in sorted(chapter_scores):
            for (version_a, version_b), values in sorted(chapter_scores[chapter].items()):
                writer.writerow(
                    [
                        chapter,
                        version_a,
                        version_b,
                        f"{aggregate_scores(values, aggregation):.8f}",
                        len(values),
                    ]
                )


def write_book_affinity_matrix(
    versions: list[str],
    book_pair_scores: dict[tuple[str, str], float],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version"] + versions)
        for version_a in versions:
            row = [version_a]
            for version_b in versions:
                if version_a == version_b:
                    row.append("1.00000000")
                else:
                    pair = tuple(sorted((version_a, version_b)))
                    if pair in book_pair_scores:
                        row.append(f"{book_pair_scores[pair]:.8f}")
                    else:
                        row.append("0.00000000")
            writer.writerow(row)


def write_book_affinity_sources(
    versions: list[str],
    book_pair_scores: dict[tuple[str, str], float],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version"] + versions)
        for version_a in versions:
            row = [version_a]
            for version_b in versions:
                if version_a == version_b:
                    row.append("self")
                elif tuple(sorted((version_a, version_b))) in book_pair_scores:
                    row.append("sentence_aggregation")
                else:
                    row.append("no_shared_chapter")
            writer.writerow(row)


def find_missing_pair_scores(
    versions: list[str],
    pair_scores: dict[tuple[str, str], float],
) -> list[dict[str, str]]:
    missing = []
    for version_a, version_b in combinations(versions, 2):
        if tuple(sorted((version_a, version_b))) not in pair_scores:
            missing.append({"version_a": version_a, "version_b": version_b})
    return missing


def detect_chapter_communities(
    nx,
    community,
    versions: list[str],
    chapter_scores: dict[int, dict[tuple[str, str], list[float]]],
    aggregation: str,
    threshold: float,
    top_k: int,
    max_communities: int,
) -> list[dict[str, Any]]:
    results = []
    for chapter in sorted(chapter_scores):
        graph = nx.Graph()
        graph.add_nodes_from(versions)
        pair_scores = aggregate_pair_scores(chapter_scores[chapter], aggregation)
        add_similarity_edges_to_graph(nx, graph, pair_scores, threshold, top_k)
        best = weighted_girvan_newman_best_partition(
            nx, community, graph, max_communities
        )
        results.append(
            {
                "chapter": chapter,
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "community_count": best["community_count"],
                "modularity": best["modularity"],
                "communities": best["communities"],
            }
        )
    return results


def write_chapter_membership_csv(
    chapter_results: list[dict[str, Any]], output_path: Path
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chapter", "version", "community_id"])
        for item in chapter_results:
            for community_id, versions in enumerate(item["communities"], start=1):
                for version in versions:
                    writer.writerow([item["chapter"], version, community_id])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5 Girvan-Newman community detection and affinity ranking."
    )
    parser.add_argument(
        "--version-matrix",
        type=Path,
        default=BASE_DIR / "version_similarity_matrix.csv",
        help="Version similarity matrix from the VGAE stage.",
    )
    parser.add_argument(
        "--tree-json",
        type=Path,
        default=BASE_DIR / "mora_v4.1_0406.json",
        help="Tree JSON used to discover versions and chapters.",
    )
    parser.add_argument(
        "--similarity-json",
        type=Path,
        default=BASE_DIR / "total_formal_all_sentence_adjusted_distance_aggressive_llm.json",
        help="Sentence similarity JSON used to aggregate chapter affinities.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version-threshold", type=float, default=0.6)
    parser.add_argument("--version-top-k", type=int, default=3)
    parser.add_argument("--version-max-communities", type=int, default=8)
    parser.add_argument("--chapter-threshold", type=float, default=0.35)
    parser.add_argument("--chapter-top-k", type=int, default=4)
    parser.add_argument("--chapter-max-communities", type=int, default=6)
    parser.add_argument(
        "--chapter-aggregation",
        choices=["mean", "max", "top3_mean"],
        default="mean",
    )
    parser.add_argument(
        "--sentence-similarity-transform",
        choices=["linear", "exp"],
        default="linear",
        help="Convert normalized distance to similarity before chapter aggregation.",
    )
    parser.add_argument("--distance-tau", type=float, default=0.05)
    parser.add_argument(
        "--skip-chapter",
        action="store_true",
        help="Only run version pair ranking and version community detection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nx, community = require_networkx()

    log("Loading version similarity matrix")
    versions, version_pair_scores = read_version_similarity_matrix(args.version_matrix)
    ranking = write_version_pair_ranking(
        version_pair_scores, args.output_dir / "version_pair_similarity_ranking.csv"
    )
    log(f"Saved version pair ranking ({len(ranking)} pairs)")

    version_graph = nx.Graph()
    version_graph.add_nodes_from(versions)
    add_similarity_edges_to_graph(
        nx,
        version_graph,
        version_pair_scores,
        threshold=args.version_threshold,
        top_k=args.version_top_k,
    )
    version_best = weighted_girvan_newman_best_partition(
        nx, community, version_graph, args.version_max_communities
    )
    version_payload = {
        "source_matrix": str(args.version_matrix),
        "threshold": args.version_threshold,
        "top_k": args.version_top_k,
        "node_count": version_graph.number_of_nodes(),
        "edge_count": version_graph.number_of_edges(),
        **version_best,
    }
    (args.output_dir / "version_communities.json").write_text(
        json.dumps(version_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_membership_csv(
        version_best["communities"],
        args.output_dir / "version_community_membership.csv",
    )
    log(
        "Saved version communities: "
        f"{version_best['community_count']} communities, "
        f"modularity={version_best['modularity']:.4f}"
    )

    summary: dict[str, Any] = {
        "version_pair_count": len(ranking),
        "version_top_pairs": ranking[:20],
        "version_community_count": version_best["community_count"],
        "version_modularity": version_best["modularity"],
    }

    if not args.skip_chapter:
        log("Loading tree and sentence similarities for chapter-level affinities")
        tree_versions, _ = load_tree_versions_and_chapters(args.tree_json)
        chapter_scores, book_scores_raw = build_chapter_pair_scores(
            args.similarity_json,
            args.distance_tau,
            args.sentence_similarity_transform,
        )
        write_chapter_affinity_edges(
            chapter_scores,
            args.output_dir / "chapter_affinity_edges.csv",
            args.chapter_aggregation,
        )
        book_scores = aggregate_pair_scores(book_scores_raw, args.chapter_aggregation)
        missing_book_pairs = find_missing_pair_scores(tree_versions, book_scores)
        write_book_affinity_matrix(
            tree_versions,
            book_scores,
            args.output_dir / "book_affinity_matrix.csv",
        )
        write_book_affinity_sources(
            tree_versions,
            book_scores,
            args.output_dir / "book_affinity_sources.csv",
        )
        chapter_results = detect_chapter_communities(
            nx,
            community,
            tree_versions,
            chapter_scores,
            aggregation=args.chapter_aggregation,
            threshold=args.chapter_threshold,
            top_k=args.chapter_top_k,
            max_communities=args.chapter_max_communities,
        )
        (args.output_dir / "chapter_communities.json").write_text(
            json.dumps(chapter_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_chapter_membership_csv(
            chapter_results, args.output_dir / "chapter_community_membership.csv"
        )
        summary.update(
            {
                "chapter_count": len(chapter_results),
                "chapter_aggregation": args.chapter_aggregation,
                "sentence_similarity_transform": args.sentence_similarity_transform,
                "chapter_threshold": args.chapter_threshold,
                "chapter_top_k": args.chapter_top_k,
                "book_affinity_missing_pair_count": len(missing_book_pairs),
                "book_affinity_missing_pairs": missing_book_pairs,
                "book_affinity_missing_pair_policy": "Written as 0.0 because these version pairs have no shared chapters and therefore no direct sentence-level evidence.",
            }
        )
        log(f"Saved chapter communities for {len(chapter_results)} chapters")

    (args.output_dir / "community_detection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Done. Outputs are in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
