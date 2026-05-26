# Standard edit_count Control Group

Run date: 2026-05-10

## Input

The control input is `standard_edit_count_sentence_edges.json`.

It was generated from `../total_formal_all_sentence_adjusted_distance_aggressive_llm.json` by replacing each edge's `normalized_distance` with:

```text
relation_summary.edit_count / max(len(original_text), len(modified_text), 1)
```

Input summary:

- Raw edges: 28,550
- Written edges: 28,550
- Missing `edit_count`: 0
- Invalid records: 0
- Distance layers before VGAE pruning: L0 = 7,603, L1 = 37, L2 = 616, L3 = 20,294

## VGAE Result

Best and sentence-best selected the same trial:

- Trial index: 2
- Seed: 44
- Hierarchy weight: 0.08
- Learning rate: 0.001
- KL weight: 0.0007
- Hierarchy reconstruction weight: 0.2
- Sentence same-chapter silhouette: 0.0280154496
- Chapter same-name silhouette: 0.2522784770
- Combined score: 0.4437035322
- Sentence priority score: 0.1319374703

Generated VGAE artifacts:

- `vgae_output.pt`
- `node_vectors.csv`
- `version_similarity_matrix.csv`
- `tsne_visualization.png`
- `chapter_tsne_visualization.png`
- `version_tsne_visualization.png`
- `sentence_best_vgae_output.pt`
- `sentence_best_node_vectors.csv`
- `sentence_best_version_similarity_matrix.csv`
- `sentence_best_tsne_visualization.png`
- `sentence_best_chapter_tsne_visualization.png`
- `sentence_best_version_tsne_visualization.png`

## GVNM Result

GVNM output directory: `gvnm_output/`

- Version pair count: 66
- Version community count: 2
- Version modularity: -0.0007768735
- Chapter communities generated: 77
- Book affinity missing version pairs: 3

Top version pairs:

1. `fy` - `hs`: 0.99971116
2. `hs` - `wb`: 0.99965185
3. `fy` - `wb`: 0.99958026

## Quick Comparison With Adjusted-Distance Experiment

The original adjusted-distance run's root `training_log.json` records:

- Comprehensive best: sentence silhouette 0.0247064643, chapter silhouette 0.4513472915, combined score 0.6361582391
- Sentence best: sentence silhouette 0.0760090947, chapter silhouette 0.3492985964, sentence priority score 0.2048311159

Against that baseline, this standard edit_count control group keeps a comparable but slightly higher comprehensive sentence silhouette, while the chapter-level clustering and sentence-priority result are weaker.
