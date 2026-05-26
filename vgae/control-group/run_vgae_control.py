from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CONTROL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import vgae_training  # noqa: E402


def control_path(name: str) -> str:
    return str(CONTROL_DIR / name)


vgae_training.CONFIG.update(
    {
        "tree_json_path": str(BASE_DIR / "mora_v4.1_0406.json"),
        "similarity_json_path": control_path("standard_edit_count_sentence_edges.json"),
        "similarity_json_fallbacks": [],
        "output_dir": str(CONTROL_DIR),
        "vgae_output_path": control_path("vgae_output.pt"),
        "node_vectors_csv_path": control_path("node_vectors.csv"),
        "version_similarity_csv_path": control_path("version_similarity_matrix.csv"),
        "tsne_png_path": control_path("tsne_visualization.png"),
        "chapter_tsne_png_path": control_path("chapter_tsne_visualization.png"),
        "version_tsne_png_path": control_path("version_tsne_visualization.png"),
        "sentence_best_vgae_output_path": control_path("sentence_best_vgae_output.pt"),
        "sentence_best_node_vectors_csv_path": control_path("sentence_best_node_vectors.csv"),
        "sentence_best_version_similarity_csv_path": control_path(
            "sentence_best_version_similarity_matrix.csv"
        ),
        "sentence_best_tsne_png_path": control_path("sentence_best_tsne_visualization.png"),
        "sentence_best_chapter_tsne_png_path": control_path(
            "sentence_best_chapter_tsne_visualization.png"
        ),
        "sentence_best_version_tsne_png_path": control_path(
            "sentence_best_version_tsne_visualization.png"
        ),
        "sentence_best_config_path": control_path("sentence_best_config.json"),
        "missing_edge_log_path": control_path("missing_similarity_edges.json"),
        "training_log_path": control_path("training_log.json"),
        "best_config_path": control_path("best_config.json"),
    }
)


if __name__ == "__main__":
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    raise SystemExit(vgae_training.main())
