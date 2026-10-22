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
- [x] `train.py` with Adam, clipping, LR scheduler and early stopping
- [x] Training log CSV and loss curve
- [x] Best checkpoint with weights, model config and vocabulary
- [x] Full training run, time and best validation perplexity recorded
- [x] One CPU training run timed for the report, next to the GPU time
- [x] Two GPU runs with seed 42 give the same validation loss

Result (from `reports/`):
- GPU (RTX 4050): best epoch 72 of 77, val loss 1.337, val perplexity 3.81, 36.4 s.
- Repeat GPU run with seed 42: identical log and identical weights.
- CPU: best epoch 68 of 73, val loss 1.309, val perplexity 3.70, 251 s (about 4 minutes). CPU and GPU kernels round differently, so the two runs drift apart slightly. The GPU run is the main model.

Check: validation loss falls then flattens. `model_best.pt` loads and runs.

## Phase 5: Generation
- [x] Sampling with temperature, top-k, blocked tokens and repeated trigram block
- [x] Batch generation
- [x] Same seed gives same output (test)
- [x] 5 samples printed per subject

Result: 4,800 raw samples (50 for each of the 96 subject, topic and difficulty cells) in 3.4 s on the GPU. None hold a banned token and all end with `<eos>` inside 40 tokens. Samples stay on topic and lean on template wording, for example "Explain the working of TCP connection termination." Some slip, such as "integrity integrity constraints", which the filters are there to catch.

Check: no output holds a control token, `<pad>`, `<bos>` or `<unk>`.

## Phase 6: Filters and classifiers
- [x] TF-IDF topic centroids and difficulty logistic regression, saved to `classifiers.pkl`
- [x] Classifier accuracy on real test questions
- [x] Per-topic relevance threshold from the validation percentile
- [x] Seven filters, each returning a pass flag and a reason
- [x] Fallback rounds and the `from_bank` fill
- [x] One passing and one failing test per filter

Result:
- On the 345 real test questions the nearest-centroid topic classifier scores 84.6% and the difficulty classifier 99.1% (`reports/classifier_accuracy.json`). Difficulty is easy to predict because labels and templates both follow the opening verb, so this number says more about the data than the classifier.
- Topic thresholds (10th percentile on validation) range from 0.09 to 0.21.
- `generate_questions` returned 10 questions for all 96 cells in 9.8 s on the GPU, all generated, none from the bank.
- One round of 50 raw candidates per cell: 19.6% pass all filters. Of the rejects, 3,010 of 4,800 candidates are exact copies of training questions, 528 are off topic and 300 are too close to an accepted question. The copy rate is the main weakness so far and is looked at again in Phase 10 (dropout 0.4 and hidden size 128 are the planned fixes).

Check: `generate_questions` returns the requested number for every cell.

## Phase 7: Baseline
- [x] Trigram model with stupid backoff per topic and difficulty, with the back-off rule
- [x] Same interface as the LSTM generator
- [x] Validation perplexity

Result: validation perplexity 10.17 for the trigram model against 3.81 for the LSTM (`reports/baseline.json`, `reports/training_log.csv`). It samples for all 96 cells, and 31 hard cells with under 20 training questions use their topic model. Its samples are less coherent ("Differentiate between shared memory."), yet 24.9% pass the filters against 19.6% for the LSTM, because it copies whole training questions less often.

Check: the trigram model generates for every cell.

## Phase 8: Paper builder and export
- [x] `PaperSpec`, the three presets and `build_paper`
- [x] Marks, uniqueness and topic coverage assertions
- [x] DOCX, PDF and JSON export
- [x] Tests for totals, uniqueness, determinism and that the files open

Result: a balanced 70 mark paper builds for every subject in 1.0 to 1.6 s on the GPU, all 18 questions generated. The same seed and spec give byte-identical JSON for all three presets. DOCX (checked in Word) and PDF both show the header, section headings with mark totals, numbered questions and marks on the right. Wording is often template-like and a few slips get through, for example "Implement relational algebra for relational algebra and 3NF."

Check: a balanced paper totals 70 marks. DOCX and PDF layout checked by eye.

## Phase 9: CLI and app
- [x] `cli.py` with `generate` and `paper`
- [x] Streamlit app with the three tabs
- [x] Missing checkpoint message
- [x] Regenerate, Edit and Remove per question
- [x] `upload.py`: PDF, text and typed-name input
- [x] Subject and topic detection, with a clear message for unsupported material
- [x] Upload similarity in ranking and `from_notes` template fill
- [x] CLI `--from-file` and the app uploader
- [x] Tests for each input type and the unsupported case

Result:
- A4: `generate` prints 5 questions ending in . with no < or >. A5: two `paper` runs with seed 42 give identical JSON.
- A8: with no checkpoint the app shows "Run `make train` first" and no traceback (`tests/test_app.py`).
- A9: `paper --from-file` on a DBMS notes PDF detects DBMS with Normalization (57%) and Concurrency control (29%) and builds a 70 mark paper split by those weights. Unrelated text (history, a recipe) is refused with the list of supported subjects.
- In the browser: a paper generates, Regenerate swaps one question, Remove drops the total and Fill brings it back, DOCX, PDF and JSON buttons appear once marks add up, the Try the model tab shows passed and rejected samples with reasons, and pasted notes are detected.
- Weak spot: with notes, the novel questions the model writes often mix terms wrongly ("Armstrong normal form", "archival dependency"). This is the same copy problem seen in Phase 6, and is looked at in Phase 10.

Check: app runs, a paper generates for each subject, all three file types download. `paper --from-file` on DBMS notes gives a 70-mark DBMS paper on the matching topics.

## Phase 10: Evaluation
- [x] Every metric for the LSTM, the trigram and the real questions
- [x] Three generation seeds, mean and standard deviation
- [x] Ablations
- [x] Plots, `metrics.md` and `samples.md`
- [x] Blind human rating sheet and a script that averages the scores

Result (all from `reports/metrics.md`):
- Perplexity: LSTM 3.81 validation and 4.02 test, trigram 10.17 and 9.34. The LSTM fits real questions far better.
- The trigram wins on raw novelty (0.61 against 0.37), filter pass rate (28% against 22%) and topic accuracy (0.93 against 0.80). The LSTM wins on raw difficulty accuracy (0.999 against 0.73) and on diversity after filtering (distinct-2 0.41 against 0.36, self-BLEU 0.53 against 0.56).
- No control tokens: topic accuracy falls from 0.78 to 0.03 and difficulty accuracy to 0.33 (chance), and every sample is the same question. Conditioning clearly works, so the condition vector variant was not built.
- Temperature trades topic accuracy for novelty: 0.6 gives 0.86 and 0.23, 1.0 gives 0.71 and 0.49.
- The copy problem is not fixed by the two planned remedies. Dropout 0.4 leaves novelty at 0.36. Hidden size 128 raises novelty to 0.73 only because it stops following the topic (topic accuracy 0.19) and repeats itself (self-BLEU 0.94). Hidden size 512 is about the same as 256.
- Phrase check (filter 8, added after the first run): the LSTM pass rate drops from 22% to 11% and its filtered output improves, with topic accuracy 0.86 to 0.90 and relevance 0.267 to 0.290. Building 10 questions for every cell still works, with 51 of 960 coming from the bank. The trigram pass rate only moves from 28% to 25%, since a trigram model rarely produces a word pair it has not seen.
- Human rating sheet (60 questions, 20 per source, shuffled) and key are in `reports/`. Ratings still need the student and two classmates; `python -m qpgen.evaluate --ratings a.csv b.csv c.csv` averages them.

Check: `reports/` holds every expected file.

## Phase 11: Report and polish
- [x] Finish `README.md`
- [x] Fill in the report outline
- [x] Clean run: new venv, then `make data`, `make train`, `make eval`, `make test`
- [x] Screenshots of the app and a sample paper

Result: a fresh clone from GitHub, set up with only the README steps in a new virtual environment, ran `make data`, `make train`, `make eval` and `make test` with exit code 0 each. Training gave the same best epoch 75 and val perplexity 3.84, all 118 tests passed, and the regenerated `reports/metrics.md` was identical to the committed one. Still open outside the code: the human ratings and a check of the report's references.

Check: every acceptance test and final checklist item holds.

## Phase 12: Paper pattern, Bloom levels and topics
- [x] Custom pattern: marks, questions printed, questions to attempt and difficulty per section, in the app and as `--pattern`
- [x] "Answer any N of M" sections, totals counted from the attempted questions
- [x] Bloom level per question from the opening verb, CO per topic from config and editable per paper
- [x] CO and BL columns in DOCX and PDF, CO-wise marks in the app and JSON
- [x] Seed and temperature shown as Paper variant # and Creativity
- [x] App split into Make a paper, Add a topic and Model insights pages
- [x] Add a topic page with checks, save and an in-app retrain

Result: 149 tests pass. A 1:6, 5:5:3, 8:2:1 DBMS paper totals 29 marks and prints CO and BL columns with "Answer any 3 of 5". In a temporary copy of the config, adding a 90-question Caching topic to OS and retraining took 72 s on the GPU (best val perplexity 3.95), and the model then wrote Caching questions at all three levels ("Define cache miss in operating systems.", "Analyze the cost of cache miss in detail."). The committed model and reports were not touched.

Check: a custom pattern paper builds and exports with the right total, and a new topic can be added and used after a retrain.

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

**2026-10-12. Max epochs raised from 40 to 100.**
With the cap at 40, validation loss was still dropping at the last epoch (perplexity 4.96), so training never reached the point where it flattens and early stopping never fired. With a cap of 100, early stopping ends the run at epoch 77 and the best checkpoint is from epoch 72. Patience, learning rate and the scheduler are unchanged.

**2026-10-12. Extra training runs use a tag.**
`python -m qpgen.train --tag <name>` writes `model_best_<name>.pt` and `training_log_<name>.csv` so repeat runs, the CPU timing run and later ablations do not overwrite the main model.

**2026-10-12. Topic classifier is the nearest TF-IDF centroid.**
The same centroids give the relevance score for filter 6 and the topic prediction for evaluation, so there is one model to explain instead of two. A logistic regression for topic was the alternative.

**2026-10-12. Fallback re-filters the whole candidate pool.**
Each extra round adds 50 candidates to a pool and the filters run again over the pool, so a question rejected only as a near duplicate in round one can still be picked later. After 5 rounds the pool is filtered once more with thresholds relaxed by 20%, then the bank fills any gap.

**2026-10-12. Trigram probabilities are normalised and unigrams use add-one.**
Plain stupid backoff gives scores that do not sum to one, which makes perplexity meaningless. The backoff scores for each context are divided by their sum, and add-one smoothing on unigrams keeps every word above zero. Sampling uses the same temperature, top-k, banned tokens and trigram blocking as the LSTM.

**2026-10-12. Generators share one call signature.**
`LSTMSampler` and `TrigramSampler` both take (subject, topic, difficulty, n, temperature, top_k, seed). `generate_questions` uses whichever sampler its context holds, so the baseline goes through exactly the same filters and fallback.

**2026-10-12. Topic allocation uses the largest remainder method.**
Each topic's fair share is count x weight / total weight. Every topic gets the floor of its share and the leftover questions go to the largest remainders, so no topic gets more than one above its share. When a section has at least as many questions as topics, any topic left at zero takes one from the topic furthest above its share.

**2026-10-12. Repeats across the paper use the same overlap limit as the filters.**
The builder asks for 3 extra candidates per topic and skips any whose word overlap with a question already on the paper is 0.7 or more. If a topic still falls short it asks again with a new seed, up to 5 times.

**2026-10-12. Upload topics are found sentence by sentence.**
Scoring a whole set of notes against each topic at once let the main topic hide the others. Each sentence now goes to its closest topic if it clears that topic's threshold, and a topic is picked when it holds at least 15% of those sentences (up to 4 topics, weighted by share). English stop words are dropped before scoring, since TF-IDF over common words made any English text, such as a history paragraph, look about 0.19 similar to every subject. The subject cut-off is 0.05; real notes scored around 0.14 and unrelated text 0.00.

**2026-10-12. Removing a question blocks downloads until the section is filled.**
Exported papers must always add up to the declared marks, so after a Remove the app shows the shortfall and a Fill button instead of the download buttons.

**2026-10-12. Hidden size ablations change the embedding size too.**
Weight tying needs the embedding and hidden sizes to match, so the 128 and 512 runs use 128 and 512 for both.

**2026-10-12. Self-BLEU is computed on a sample.**
Each set scores 200 seeded hypotheses against all the other questions in the set instead of every question, which keeps `make eval` to a few minutes. The setting is `eval.self_bleu_sample`.

**2026-10-12. Ablation models are trained once and reused.**
`make eval` trains a tagged checkpoint for each ablation only if it is missing, then reuses it on later runs.

**2026-10-12. Eighth filter: phrase check.**
After the copy filter, the questions that survive are novel but often glue two topics into a term that does not exist, like "relational cover" or "archival dependency". The phrase check rejects a generated question if any pair of adjacent words never appears in a training question. Neither planned fix for copying helped (dropout 0.4 left novelty at 0.36, hidden size 128 lost the topic), and allowing training copies would have filled papers with questions the model had already seen. The check is skipped for bank and notes questions, and can be turned off with `filters.check_phrases`.

**2026-10-12. The report is kept out of the repo.**
The report follows the outline (introduction, background, dataset, method, experiments, results, limitations, conclusion, references, sample paper) and is submitted separately, so it is not committed. Its numbers are read from `reports/`. Human ratings and a check of each reference are still to do.

**2026-10-13. LSTM layers are stacked by hand.**
A fresh clone showed `make train` failing on the GPU: training finished, then the process crashed on exit (0xC0000409, issue #3). The cause is cuDNN's inter-layer dropout in `nn.LSTM(num_layers=2, dropout=0.3)` under torch 2.14.1+cu130 on Windows; five lines of plain PyTorch reproduce it, and `os._exit` does not avoid it because the crash is in native teardown. The model now stacks one single-layer `nn.LSTM` per layer with its own dropout between them, which is the same maths and never creates that dropout state. Turning cuDNN off also worked but was twice as slow. The change draws dropout masks differently, so the model and every report were regenerated. Phase results above are as measured at the time. The regenerated numbers (all in `reports/`): best epoch 75, validation perplexity 3.84 and test 4.05, 44 s on the GPU and 261 s on the CPU, repeat GPU runs still identical. LSTM raw novelty 0.38, pass rate 12%, filtered topic accuracy 0.90; the trigram numbers are unchanged. Ablations tell the same story as before.

**2026-10-22. Paper total comes from the pattern.**
Total marks are the sum of questions to attempt times marks for each section, so a custom pattern can total anything. The presets still add up to 70, and `paper.total_marks` is kept only to check that. In an "any N of M" section all M questions are generated and printed.

**2026-10-22. Difficulty follows marks unless set.**
The model only knows easy, medium and hard, so each mark value is mapped with `paper.difficulty_by_marks` (up to 3 marks easy, up to 7 medium, above that hard). The pattern table can override it per row.

**2026-10-22. Bloom level comes from the opening verb.**
The same idea as the difficulty labels: design is BL6, evaluate or justify BL5, compare or differentiate BL4, solve, write a query or implement BL3, define or list BL1, and anything else BL2. It is a rule, not a model, and is recomputed when a question is edited.

**2026-10-22. CO map lives in subjects_topics.yaml.**
Each subject maps every topic to a course outcome, two topics per CO by default. The app shows the map as an editable table, and the edited map is stored on the paper spec.

**2026-10-22. New topics need a retrain.**
A topic is a control token, so the LSTM cannot write for a topic it has not been trained on. The Add a topic page saves at least 30 questions to `student_questions.csv`, adds the topic to `subjects_topics.yaml` as text so the comments stay, and runs the dataset build, training and classifiers. Until then the topic is listed as needing a retrain and cannot be picked. A small topic can miss the validation split, so its relevance threshold falls back to its training questions.
