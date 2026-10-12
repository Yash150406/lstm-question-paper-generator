# Progress

Boxes get ticked only after the phase check passes.

## Phase 0: Setup
- [x] Create the repo layout
- [x] Write `pyproject.toml`, `requirements.txt`, `.gitignore` and `Makefile`
- [x] Write `config/default.yaml` and `config/subjects_topics.yaml`
- [x] Implement `config.py` and `seed.py`
- [x] Run `pip install -e .` and `pytest`

Check: `python -c "import qpgen"` runs without errors.

## Phase 1: Dataset
- [ ] Seed files for all four subjects, at least 250 questions each, every topic and difficulty
- [ ] `concepts.yaml` and the question templates
- [ ] `labels.py` with the verb rules and tests
- [ ] `build_dataset.py`: merge, clean, exact and near duplicate removal, length filter
- [ ] Optional MMLU import, tagged `mmlu`
- [ ] Stratified 80/10/10 split saved as three CSVs
- [ ] Counts by subject, topic and difficulty. Fail if a subject has fewer than 500

Check: `questions.csv` matches the schema, has no duplicates, split sizes close to 80/10/10.

## Phase 2: Tokenizer and dataset class
- [ ] `tokenize`, `detokenize` and `Vocab` with special and control tokens, save and load
- [ ] PyTorch `Dataset` with input/target pairs and the ignore mask
- [ ] `collate_fn` with padding
- [ ] Round-trip tests for the tokenizer

Check: three printed sequences start with the control tokens. Unknown-word rate on validation printed, looked into if above 5%.

## Phase 3: Model
- [ ] `QuestionLSTM`
- [ ] Tests for output shapes and weight tying, parameter count printed
- [ ] Test that initial loss is close to ln(vocab size)
- [ ] Test that the model overfits 32 questions to near zero loss

Check: `pytest tests/test_model.py` passes.

## Phase 4: Training
- [ ] `train.py` with Adam, clipping, LR scheduler and early stopping
- [ ] Training log CSV and loss curve
- [ ] Best checkpoint with weights, model config and vocabulary
- [ ] Full training run, time and best validation perplexity recorded

Check: validation loss falls then flattens. `model_best.pt` loads and runs.

## Phase 5: Generation
- [ ] Sampling with temperature, top-k, blocked tokens and repeated trigram block
- [ ] Batch generation
- [ ] Same seed gives same output (test)
- [ ] 5 samples printed per subject

Check: no output holds a control token, `<pad>`, `<bos>` or `<unk>`.

## Phase 6: Filters and classifiers
- [ ] TF-IDF topic centroids and difficulty logistic regression, saved to `classifiers.pkl`
- [ ] Classifier accuracy on real test questions
- [ ] Per-topic relevance threshold from the validation percentile
- [ ] Seven filters, each returning a pass flag and a reason
- [ ] Fallback rounds and the `from_bank` fill
- [ ] One passing and one failing test per filter

Check: `generate_questions` returns the requested number for every cell.

## Phase 7: Baseline
- [ ] Trigram model with stupid backoff per topic and difficulty, with the back-off rule
- [ ] Same interface as the LSTM generator
- [ ] Validation perplexity

Check: the trigram model generates for every cell.

## Phase 8: Paper builder and export
- [ ] `PaperSpec`, the three presets and `build_paper`
- [ ] Marks, uniqueness and topic coverage assertions
- [ ] DOCX, PDF and JSON export
- [ ] Tests for totals, uniqueness, determinism and that the files open

Check: a balanced paper totals 70 marks. DOCX and PDF layout checked by eye.

## Phase 9: CLI and app
- [ ] `cli.py` with `generate` and `paper`
- [ ] Streamlit app with the three tabs
- [ ] Missing checkpoint message
- [ ] Regenerate, Edit and Remove per question

Check: app runs, a paper generates for each subject, all three file types download.

## Phase 10: Evaluation
- [ ] Every metric for the LSTM, the trigram and the real questions
- [ ] Three generation seeds, mean and standard deviation
- [ ] Ablations
- [ ] Plots, `metrics.md` and `samples.md`
- [ ] Blind human rating sheet and a script that averages the scores

Check: `reports/` holds every expected file.

## Phase 11: Report and polish
- [ ] Finish `README.md`
- [ ] Fill in the report outline
- [ ] Clean run: new venv, then `make data`, `make train`, `make eval`, `make test`
- [ ] Screenshots of the app and a sample paper

Check: every acceptance test and final checklist item holds.

## Decisions

**2026-10-12. `data/processed/` is not committed.**
`make data` rebuilds it from `data/raw/` in seconds, so committing it would just duplicate the seed files. Alternative was committing the CSVs for convenience.

**2026-10-12. Requirements pin direct dependencies only.**
A full `pip freeze` pulls in Windows-only and CUDA-specific packages that break installs on other machines. Pinning the 13 direct packages to the versions from the first clean install keeps runs repeatable without that problem.

**2026-10-12. Short subject keys.**
Subjects are keyed `dbms`, `os`, `cn`, `dsa` in config, file names (`seed_dbms.csv`) and control tokens (`<subj_dbms>`). The full name is kept for display.

**2026-10-12. Make is optional on Windows.**
Windows has no `make` by default. The Makefile is there for Linux, macOS and anyone who installs make. Each target is a single command that can be run by hand, and the README lists them.

**2026-10-12. Training runs on the laptop GPU.**
Training uses the RTX 4050 through the CUDA 13.0 build of torch 2.14.1. `requirements.txt` still pins the plain `torch==2.14.1`, so a CPU-only machine installs fine. The code picks CUDA when it is there and falls back to CPU. The README gives the extra install line for the GPU build.
