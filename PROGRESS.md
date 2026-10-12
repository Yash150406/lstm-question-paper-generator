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
- [x] Seed files for all four subjects, at least 250 questions each, every topic and difficulty
- [x] `concepts.yaml` and the question templates
- [x] `labels.py` with the verb rules and tests
- [x] `build_dataset.py`: merge, clean, exact and near duplicate removal, length filter
- [x] Optional MMLU import, tagged `mmlu`
- [x] Stratified 80/10/10 split saved as three CSVs
- [x] Counts by subject, topic and difficulty. Fail if a subject has fewer than 500

Result: 3,445 questions (DBMS 873, OS 863, CN 867, DSA 842), 39% easy, 41% medium, 20% hard. Split 2,756 / 344 / 345.

Check: `questions.csv` matches the schema, has no duplicates, split sizes close to 80/10/10.

## Phase 2: Tokenizer and dataset class
- [x] `tokenize`, `detokenize` and `Vocab` with special and control tokens, save and load
- [x] PyTorch `Dataset` with input/target pairs and the ignore mask
- [x] `collate_fn` with padding
- [x] Round-trip tests for the tokenizer

Result: vocab of 920 tokens (4 special, 39 control, 877 words seen at least twice in train). All 2,756 train sequences fit in 50 tokens. Unknown word rate on validation is 2.68%.

Check: three printed sequences start with the control tokens. Unknown-word rate on validation printed, looked into if above 5%.

## Phase 3: Model
- [x] `QuestionLSTM`
- [x] Tests for output shapes and weight tying, parameter count printed
- [x] Test that initial loss is close to ln(vocab size)
- [x] Test that the model overfits 32 questions to near zero loss

Result: 1,289,112 parameters with the 920 token vocab. Initial loss on a real training batch is 6.8240 against ln(920) = 6.8244. The 32 question batch overfits to under 0.05 loss in 300 Adam steps.

Check: `pytest tests/test_model.py` passes.

## Phase 4: Training
- [ ] `train.py` with Adam, clipping, LR scheduler and early stopping
- [ ] Training log CSV and loss curve
- [ ] Best checkpoint with weights, model config and vocabulary
- [ ] Full training run, time and best validation perplexity recorded
- [ ] One CPU training run timed for the report, next to the GPU time
- [ ] Two GPU runs with seed 42 give the same validation loss

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
- [ ] `upload.py`: PDF, text and typed-name input
- [ ] Subject and topic detection, with a clear message for unsupported material
- [ ] Upload similarity in ranking and `from_notes` template fill
- [ ] CLI `--from-file` and the app uploader
- [ ] Tests for each input type and the unsupported case

Check: app runs, a paper generates for each subject, all three file types download. `paper --from-file` on DBMS notes gives a 70-mark DBMS paper on the matching topics.

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

**2026-10-12. Papers can be built from uploaded notes.**
A user can upload a PDF or notes, paste text, or type a chapter name. The text is matched to the fixed subject and topic list with the same TF-IDF centroids the filters use, and the LSTM generates for the matched topics. Questions closer to the notes rank higher, and key terms from the notes fill the templates (tagged `from_notes`). Adds `pypdf` for reading PDFs. Alternative was generating straight from the notes with a pretrained language model, which is out of scope and would sideline the LSTM.

**2026-10-12. Deterministic mode for GPU runs.**
cuDNN's LSTM kernels can give slightly different results between runs even with the same seed. `set_seed` turns on PyTorch's deterministic mode and sets `CUBLAS_WORKSPACE_CONFIG` so a repeat run on the same machine matches. It costs a little speed. One CPU run is also timed so the report can state CPU training time, since the project is meant to train on a CPU.

**2026-10-12. Seed questions are at least 5 words.**
The quality rule drops anything under 5 words, which removed about 190 short seed questions like "What is a stack?". Instead of loosening the rule, those questions were extended with the subject ("in data structures") or "Give one example.", and short "Explain X." ones became "Explain X with an example.".

**2026-10-12. Template expansion settings.**
Each topic has 15 concepts and 4 concept pairs. Every concept fills 2 easy, 2 medium and 1 hard template picked at random with seed 42, and every pair fills one compare template. With the seed bank this gives 3,445 questions at roughly 40/40/20, clearing the 3,000 floor with some margin. With 12 concepts per topic the bank came out at 2,991, just short.

**2026-10-12. MMLU import is off by default.**
The import is in place (`data.use_mmlu`), but only 8 stems from the two computer science subsets match a topic and are not built around the answer options, and most of those are long numeric word problems over the 40 word limit. It adds nothing useful, so the default build runs offline from seed files and templates.

**2026-10-12. The subject column holds the full subject name.**
`questions.csv` stores "DBMS", "Operating Systems" and so on, since that is what shows up in the app and on papers. The short key is looked up from config where needed.

**2026-10-12. Control-token targets are masked by swapping them for `<pad>`.**
The dataset replaces control-token targets with the pad id, so a single `ignore_index` in the loss covers both pad and control positions. Alternative was a separate mask tensor, which is more code for the same result.

**2026-10-12. Model is about 1.3M parameters, not 2.5M.**
The parameter estimate assumed a vocabulary near 8,000 words. The training split only has 877 words seen twice or more, so the embedding (tied with the decoder) is much smaller. The LSTM layers themselves are the planned 2 x 256.

**2026-10-12. Overfit check uses one question per topic.**
If two questions share the same subject, topic and difficulty and start the same way, no model can tell which one comes next, so the loss can never reach zero. The overfit test picks one medium question from each of the 32 topics so every control prefix is unique, and turns dropout off for that test.
