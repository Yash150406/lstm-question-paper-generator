# Handoff

Last updated 2026-10-22.

## Status

All 12 build phases are done and checked off in [PROGRESS.md](../PROGRESS.md). The pipeline runs end to end: `make data`, `make train`, `make eval`, `make test`, the CLI and the Streamlit app. Every number in the README comes from `reports/`.

## Done

- Question bank of 3,445 questions (1,045 seed, 2,400 template) over 4 subjects, 32 topics and 3 difficulty levels.
- 2-layer LSTM with control tokens, trained on GPU (44 s) and CPU (261 s). Repeat GPU runs with seed 42 give identical weights.
- Eight filters (the seven planned plus a phrase check), fallback rounds and bank fill.
- Trigram baseline with the same interface and filters.
- Paper builder with three presets, DOCX, PDF and JSON export, regenerate, fill and edit.
- Papers from uploaded notes, pasted text or a typed chapter name.
- Evaluation over three seeds with ablations, plots, samples and a blind rating sheet.
- Phase 12: custom paper patterns with any N of M, Bloom levels and CO mapping on every question, an Add a topic page with in-app retrain, and the app split into three pages.
- Bugs #1, #2 and #3 found, fixed and closed on GitHub. #3 was a GPU crash on exit after training, fixed by stacking the LSTM layers by hand.

## Pending

- [ ] Human ratings: Yash and two classmates fill in `reports/human_rating_sheet.csv` (1 to 5 for fluency, relevance and difficulty fit), then `python -m qpgen.evaluate --ratings a.csv b.csv c.csv` writes `reports/human_ratings.md`.
- [ ] Check the references in the report against the original papers before submitting.

## Known weak spots

- The LSTM repeats training questions a lot (raw novelty 0.38). Dropout 0.4 and hidden size 128 did not fix it. The copy filter and phrase check keep papers clean, at the cost of a 12% pass rate and about 5% of paper questions coming from the bank.
- The trigram baseline beats the LSTM on novelty, pass rate and topic accuracy. This is reported as is.
- Difficulty accuracy is near 1 because labels follow the opening verb.
- Bloom levels come from a verb rule, not a model, so a question that opens with an unlisted verb gets BL2.
- A topic added in the app is only as good as the questions pasted in, and retraining overwrites the training reports in `reports/`.

## Next step

Collect the human ratings, add the table to the report and the README results section, and rerun `make eval` only if the config changes.

## Ideas not built

From similar projects on GitHub (amfoss/qujini, priyam-hub/TestPaperPro, Niraj-Kamdar/question-paper-generator): a frontend redesign with a print-style paper preview, several paper sets in one go, sub-parts such as Q1 (a) and (b), an answer key, and a question bank browser for editing saved questions.
