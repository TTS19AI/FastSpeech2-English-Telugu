# Historical notebook source and coverage

`colab_history.py` is a guarded source-history export of the supplied Colab notebook, not a standalone production runner. It preserves the original body and adds a leading unconditional SystemExit guard. Direct execution and import stop before any original statement runs. The original upload remains unchanged.

The export contains earlier failed attempts, retries, code-generation cells, manual listening decisions, recovery steps, training commands, package installation and later successful verification. Do not remove the guard and run everything. Some cells assume variables or artifacts from earlier interactive steps; some deliberately stop for review, overwrite scratch outputs or start training. Including this source does not make full raw-to-training replay independently verified.

## Where to look

Use `stage_index.csv` for exact original/export line mappings of STEP headings. `guarded_line = original_line + leading_guard_lines_added`, as recorded in `source_review.json`.

| Area | Search anchor in the source |
|---|---|
| Source retrieval and dataset audits | `ai4bharat/Svarah`, `SPRINGLab/IndicTTS_Telugu`, revision constants |
| English grouped split | `def select_groups`, `recording_group_key`, `train_candidate` |
| English conditioning policy | `demographic_proxy`, `Proxy groups are not verified speaker identities` |
| Telugu constrained split | `scipy.optimize import milp`, `joint constrained selection` |
| English WAV/LAB export and selection top-up | `CORPUS_ROOT`, `alignment_selected.jsonl`, `top_up` |
| English alignment audit and feature adapter | STEP 4.17 through 4.30 |
| English train-only normalization fit | STEP 4.31; `StandardScaler.partial_fit` loop over training only |
| Telugu pronunciation inventory and export | STEP 5.1 through 5.8 and the later export repair |
| Telugu train-only acoustic aligner | STEP 5.11 through 5.13 |
| Telugu split exclusions and audits | STEP 5.15 through 5.20 |
| Telugu symbols, extraction and normalization | STEP 5.21 through 5.27; training-only fit at STEP 5.25 |
| Vocoder provenance and reconstruction | STEP 6.1 through 6.4 and repair cells |
| Checkpoint and integration smoke tests | STEP 6.6 through 6.9 and tracking repair |
| Production training and recovery | STEP 7 and later ORIGINAL ENGLISH / TELUGU training sections |
| Assignment frontends and final results | `test_sentences_tokenized.json`, saved `inference.py` generation cells |
| Anonymous model downloads | `use_cookies=False`, `clean_model_downloads.json` |
| Fresh-environment inference | `system_site_packages`, `clean_inference_verification.json`, `MPLBACKEND=Agg` |

The two statistics fits explicitly use positive-duration training positions and population standard deviations. Validation manifests are checked for overlap and conditioning coverage but their pitch/energy observations are not fitted. Existing successful-feature manifests and manual decisions are required to reproduce the original final subsets.

## Entry points for normal use

Use `<language>/inference.py`, `<language>/train.py`, `<language>/preprocess.py`, `<language>/normalize.py` and `tools/plot_variance.py`, with the resources and working directories described in the project README. Frozen-input inference was successfully tested in a fresh Python environment using packaged source and anonymous, SHA-256-verified downloads. Set `MPLBACKEND=Agg` for subprocesses; the Colab notebook backend is not installed in that environment.

The historical export is additional audit evidence. It does not supersede selected checkpoint metadata, normalization records, final test tokens, or the original/repaired experiment distinction. The earlier English data-repair experiment is not the selected English model.

## Review boundary

The export contains 21077 source lines and passes Python AST parsing. No obvious hard-coded Hugging Face/GitHub token, literal secret/password assignment or private-key block was found by the recorded pattern scan. This scan is not an exhaustive credential audit. The historical workflow was not executed during review. No dataset audio, checkpoints or notebook cell outputs are embedded in this add-on.
