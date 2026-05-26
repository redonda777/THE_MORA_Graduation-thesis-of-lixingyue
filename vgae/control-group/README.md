# edit_count control group

This directory contains the control-group run that replaces the adjusted/LLM
sentence distance with standard edit distance from `relation_summary.edit_count`.

Pipeline:

1. `generate_edit_count_control_input.py`
   - Reads `../total_formal_all_sentence_adjusted_distance_aggressive_llm.json`.
   - Writes `standard_edit_count_sentence_edges.json`.
   - Replaces `normalized_distance` with `edit_count / max(len(original_text), len(modified_text), 1)`.

2. `run_vgae_control.py`
   - Reuses `../vgae_training.py`.
   - Writes VGAE artifacts into this directory.

3. `run_gvnm_control.py`
   - Reuses `../gvnm/community_detection.py`.
   - Writes GVNM artifacts into `gvnm_output/`.
