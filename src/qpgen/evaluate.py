"""Every number and plot in the report: LSTM against the trigram baseline and real questions, plus ablations."""
import argparse
import json
import math
import random
from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader

from qpgen.baseline import TrigramSampler
from qpgen.config import DEFAULT_CONFIG, Config, load_config
from qpgen.data.build_dataset import words
from qpgen.data.dataset import QuestionDataset, build_vocab, collate_fn
from qpgen.filters import Context, filter_candidates, load_context, score
from qpgen.generate import LSTMSampler
from qpgen.train import evaluate_loss, fit, load_model, pick_device, tagged

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"LSTM": "#2a78d6", "Trigram": "#eb6834", "Real test": "#1baf7a"}   # validated categorical slots 1-3
INK, MUTED = "#2b2b2b", "#8a8a85"
METRICS = ["novelty", "distinct_1", "distinct_2", "self_bleu", "relevance", "topic_accuracy",
           "difficulty_accuracy", "pass_rate"]


def distinct(texts: list[str], n: int) -> float:
    """Unique n-grams divided by all n-grams over the set."""
    grams = [tuple(w[i:i + n]) for w in map(words, texts) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def self_bleu(texts: list[str], sample: int, seed: int) -> float:
    """Mean BLEU of sampled questions against all the others in the set. Lower means more varied."""
    # ponytail: scores a seeded sample of hypotheses, not all of them; raise self_bleu_sample for the full set
    toks = [words(t) for t in texts]
    idx = random.Random(seed).sample(range(len(toks)), min(sample, len(toks)))
    smooth = SmoothingFunction().method1
    return float(np.mean([sentence_bleu(toks[:i] + toks[i + 1:], toks[i], smoothing_function=smooth) for i in idx]))


def text_metrics(texts: list[str], topics: list[str], levels: list[str], ctx: Context, train_set: set[str]) -> dict:
    """All per-set metrics that only need the question texts and the requested cell."""
    s = ctx.scorer
    vecs = s.vectors(texts)
    rows = [s.topics.index(t) for t in topics]
    pred_level = s.difficulty.predict(s.vectorizer.transform(texts))
    return {"novelty": float(np.mean([t.lower().strip() not in train_set for t in texts])),
            "distinct_1": distinct(texts, 1), "distinct_2": distinct(texts, 2),
            "self_bleu": self_bleu(texts, ctx.cfg.eval["self_bleu_sample"], ctx.cfg.seed),
            "relevance": float(np.mean(np.sum(vecs * s.centroids[rows], axis=1))),
            "topic_accuracy": float(np.mean((vecs @ s.centroids.T).argmax(axis=1) == np.array(rows))),
            "difficulty_accuracy": float(np.mean(pred_level == np.array(levels)))}


def cells(cfg: Config) -> list[tuple[str, str, str]]:
    """Every subject, topic and difficulty combination."""
    return [(info["name"], t, lv) for info in cfg.subjects.values() for t in info["topics"] for lv in cfg.marks]


def run(sampler, ctx: Context, seed: int, n: int, temperature: float | None = None) -> tuple[list, list, list]:
    """n raw candidates per cell, scored and filtered. Returns raw, accepted, rejected."""
    raw, acc, rej = [], [], []
    for subject, topic, level in cells(ctx.cfg):
        out = sampler(subject, topic, level, n, temperature, None, seed)
        scored = [{**c, "subject": subject, "topic": topic, "level": level} for c in score(out, ctx, topic, level)]
        a, r = filter_candidates(scored, ctx, topic)
        raw, acc, rej = raw + scored, acc + a, rej + r
    return raw, acc, rej


def set_metrics(items: list[dict], ctx: Context, train_set: set[str], pass_rate: float | None) -> dict:
    """Metrics for a list of generated candidates."""
    m = text_metrics([c["text"] for c in items], [c["topic"] for c in items], [c["level"] for c in items], ctx, train_set)
    return {**m, "pass_rate": pass_rate, "count": len(items)}


def compare(ctx: Context, tri: TrigramSampler, train_set: set[str]) -> tuple[dict, dict]:
    """Raw and filtered metrics for both models over every seed. Also returns the seed-42 runs."""
    cfg, out, first = ctx.cfg, {}, {}
    for name, sampler in (("LSTM", ctx.sampler), ("Trigram", tri)):
        for seed in cfg.eval["seeds"]:
            raw, acc, rej = run(sampler, ctx, seed, cfg.eval["candidates_per_cell"])
            rate = len(acc) / len(raw)
            out.setdefault(f"{name} raw", []).append(set_metrics(raw, ctx, train_set, rate))
            out.setdefault(f"{name} filtered", []).append(set_metrics(acc, ctx, train_set, rate))
            first.setdefault(name, (raw, acc, rej))
            print(f"{name} seed {seed}: {len(raw)} raw, pass rate {rate:.1%}")
    return out, first


def mean_std(runs: list[dict]) -> dict:
    """Mean and standard deviation of each metric across seeds."""
    keys = [k for k in runs[0] if runs[0][k] is not None]
    return {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                "std": round(float(np.std([r[k] for r in runs])), 4)} for k in keys}


def perplexities(ctx: Context, tri: TrigramSampler, data: dict) -> dict:
    """Validation and test perplexity for the LSTM and the trigram model."""
    model, vocab = load_model(ctx.cfg.path("checkpoint"), pick_device(ctx.cfg.train["device"]))
    out = {"LSTM": {}, "Trigram": {}}
    for split in ("val", "test"):
        loader = DataLoader(QuestionDataset(data[split], vocab, ctx.cfg), 64, collate_fn=collate_fn)
        out["LSTM"][split] = round(math.exp(evaluate_loss(model, loader, next(model.parameters()).device)), 2)
        out["Trigram"][split] = round(tri.perplexity(data[split]), 2)
    return out


def ablation_model(cfg: Config, tag: str, model_cfg: dict, use_controls: bool = True) -> tuple:
    """Load an ablation checkpoint, training it first if it is not there."""
    device = pick_device(cfg.train["device"])
    path = tagged(cfg.path("checkpoint"), tag)
    if not path.exists():
        print(f"Training ablation model '{tag}'")
        fit(cfg, device, tag, model_cfg, use_controls)
    best = pd.read_csv(tagged(cfg.path("training_log"), tag))["val_ppl"].min()
    model, vocab = load_model(path, device)
    return LSTMSampler(model, vocab, cfg, use_controls), float(best)


def ablations(ctx: Context, train_set: set[str]) -> list[dict]:
    """Raw-output metrics for each ablation, seed 42."""
    cfg, base = ctx.cfg, dict(ctx.cfg.model)
    main_ppl = float(pd.read_csv(cfg.path("training_log"))["val_ppl"].min())
    variants = [(f"temperature {t}", ctx.sampler, main_ppl, t) for t in cfg.eval["temperatures"]]
    for h in cfg.eval["hidden_sizes"]:
        if h != base["hidden_dim"]:
            sampler, ppl = ablation_model(cfg, f"hidden{h}", {**base, "embed_dim": h, "hidden_dim": h})
            variants.append((f"hidden size {h}", sampler, ppl, None))
    drop = cfg.eval["ablation_dropout"]
    sampler, ppl = ablation_model(cfg, f"dropout{int(drop * 10)}", {**base, "dropout": drop})
    variants.append((f"dropout {drop}", sampler, ppl, None))
    sampler, ppl = ablation_model(cfg, "nocontrol", base, use_controls=False)
    variants.append(("no control tokens", sampler, ppl, None))

    rows = []
    for name, sampler, ppl, temp in variants:
        vctx = replace(ctx, sampler=sampler)
        raw, acc, _ = run(vctx.sampler, vctx, cfg.seed, cfg.eval["candidates_per_cell"], temp)
        rows.append({"variant": name, "val_ppl": round(ppl, 2),
                     **{k: round(v, 4) for k, v in set_metrics(raw, ctx, train_set, len(acc) / len(raw)).items()
                        if k in METRICS}})
        print(f"ablation {name}: {rows[-1]}")
    return rows


def style(ax) -> None:
    """Recessive axes: no top or right spine, muted ticks, light grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x" if ax.get_yaxis().get_visible() else "y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_perplexity(ppl: dict, path) -> None:
    """Validation and test perplexity, LSTM next to trigram. Lower is better."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(2)
    for i, name in enumerate(("LSTM", "Trigram")):
        vals = [ppl[name]["val"], ppl[name]["test"]]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=COLORS[name], label=name, edgecolor="white", linewidth=2)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, color=INK, padding=2)
    ax.set_xticks(x, ["validation", "test"])
    ax.set_ylabel("perplexity (lower is better)", color=INK)
    ax.legend(frameon=False)
    style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(raw: list[dict], ctx: Context, path) -> None:
    """Requested difficulty against what the difficulty classifier predicts, LSTM raw output."""
    levels = list(ctx.cfg.marks)
    pred = ctx.scorer.difficulty.predict(ctx.scorer.vectorizer.transform([c["text"] for c in raw]))
    m = np.zeros((3, 3), dtype=int)
    for want, got in zip([c["level"] for c in raw], pred):
        m[levels.index(want), levels.index(got)] += 1
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            share = m[i, j] / m[i].sum()
            ax.text(j, i, f"{m[i, j]}\n{share:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if share > 0.5 else INK)
    ax.set_xticks(range(3), levels)
    ax.set_yticks(range(3), levels)
    ax.set_xlabel("predicted difficulty", color=INK)
    ax.set_ylabel("requested difficulty", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_relevance(first: dict, test: pd.DataFrame, ctx: Context, path) -> None:
    """Mean topic relevance per topic for LSTM, trigram and real test questions."""
    topics = ctx.scorer.topics
    series = {name: pd.DataFrame(first[name][0]).groupby("topic")["relevance"].mean() for name in ("LSTM", "Trigram")}
    series["Real test"] = pd.Series({t: float(np.mean(ctx.scorer.relevance(
        test.loc[test["topic"] == t, "question"].tolist(), t))) for t in topics})
    fig, ax = plt.subplots(figsize=(7.5, 10))
    y = np.arange(len(topics))
    for i, (name, s) in enumerate(series.items()):
        ax.barh(y + (i - 1) * 0.27, s.reindex(topics).to_numpy(), 0.25, color=COLORS[name], label=name)
    ax.set_yticks(y, topics)
    ax.invert_yaxis()
    ax.set_xlabel("mean TF-IDF cosine to topic centroid", color=INK)
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_samples(first: dict, cfg: Config, path) -> None:
    """Generated questions per subject, with a few rejected ones and why."""
    raw, acc, rej = first["LSTM"]
    rng = random.Random(cfg.seed)
    keep, drop = cfg.eval["samples_per_subject"] - cfg.eval["rejected_per_subject"], cfg.eval["rejected_per_subject"]
    lines = ["# Sample questions", "", f"LSTM output, seed {cfg.seed}. Accepted questions passed all seven filters and the phrase check.", ""]
    for info in cfg.subjects.values():
        lines += [f"## {info['name']}", "", "| Topic | Difficulty | Question | Result |", "| --- | --- | --- | --- |"]
        for c in rng.sample([c for c in acc if c["subject"] == info["name"]], keep):
            lines.append(f"| {c['topic']} | {c['level']} | {c['text']} | accepted |")
        for c in rng.sample([c for c in rej if c["subject"] == info["name"]], drop):
            lines.append(f"| {c['topic']} | {c['level']} | {c['text']} | rejected: {c['reason']} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rating_sheet(first: dict, test: pd.DataFrame, cfg: Config) -> None:
    """A shuffled blind sheet of LSTM, trigram and real questions, plus a separate answer key."""
    rng = random.Random(cfg.seed)
    k = cfg.eval["human_per_source"]
    pool = [(c["text"], c["topic"], c["level"], "LSTM") for c in rng.sample(first["LSTM"][1], k)]
    pool += [(c["text"], c["topic"], c["level"], "Trigram") for c in rng.sample(first["Trigram"][1], k)]
    pool += [(r.question, r.topic, r.difficulty, "Real") for r in test.sample(k, random_state=cfg.seed).itertuples()]
    rng.shuffle(pool)
    sheet = pd.DataFrame([{"id": i + 1, "topic": t, "difficulty": lv, "question": q,
                           "fluency": "", "relevance": "", "difficulty_fit": ""} for i, (q, t, lv, _) in enumerate(pool)])
    sheet.to_csv(cfg.path("rating_sheet"), index=False)
    pd.DataFrame({"id": sheet["id"], "source": [p[3] for p in pool]}).to_csv(cfg.path("rating_key"), index=False)


def average_ratings(cfg: Config, files: list[str]) -> pd.DataFrame:
    """Mean 1 to 5 score per source over every filled-in sheet."""
    key = pd.read_csv(cfg.path("rating_key"))
    sheets = pd.concat([pd.read_csv(f).assign(rater=f) for f in files]).merge(key, on="id")
    table = sheets.groupby("source")[["fluency", "relevance", "difficulty_fit"]].mean().round(2)
    rows = [f"| {src} | {r.fluency} | {r.relevance} | {r.difficulty_fit} |" for src, r in table.iterrows()]
    md = ["# Human ratings", "", f"Mean 1 to 5 score from {len(files)} raters.", "",
          "| Source | Fluency | Relevance | Difficulty fit |", "| --- | --- | --- | --- |", *rows]
    cfg.path("human_ratings_md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return table


def fmt(stat: dict | None) -> str:
    """mean ± std, or a dash when there is nothing to show."""
    return "-" if not stat else f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def write_metrics_md(result: dict, cfg: Config) -> None:
    """The comparison tables the report quotes."""
    s, ppl, cls = result["summary"], result["perplexity"], result["classifier_ceiling"]
    cols = ["LSTM raw", "LSTM filtered", "Trigram raw", "Trigram filtered"]
    lines = ["# Evaluation", "",
             f"{cfg.eval['candidates_per_cell']} raw candidates per cell for each of the {len(cells(cfg))} cells, "
             f"seeds {', '.join(map(str, cfg.eval['seeds']))}. Values are mean ± standard deviation over seeds. "
             "Filtered means the candidates that pass all seven filters and the phrase check.", "",
             "## Perplexity", "", "| Model | Validation | Test |", "| --- | --- | --- |"]
    lines += [f"| {m} | {ppl[m]['val']} | {ppl[m]['test']} |" for m in ("LSTM", "Trigram")]
    lines += ["", "## Generated questions", "", "| Metric | " + " | ".join(cols) + " | Real test |",
              "| --- " * (len(cols) + 2) + "|"]
    for k in METRICS:
        real = result["real_test"].get(k)
        lines.append(f"| {k} | " + " | ".join(fmt(s[c].get(k)) for c in cols) +
                     f" | {'-' if real is None else f'{real:.3f}'} |")
    lines += ["", f"Classifier ceiling on real test questions: topic {cls['topic_accuracy']:.3f}, "
              f"difficulty {cls['difficulty_accuracy']:.3f}.", "", "## Ablations (seed 42, raw output)", "",
              "| Variant | Val perplexity | " + " | ".join(METRICS) + " |", "| --- " * (len(METRICS) + 2) + "|"]
    for r in result["ablations"]:
        lines.append(f"| {r['variant']} | {r['val_ppl']} | " + " | ".join(f"{r[k]:.3f}" for k in METRICS) + " |")
    cfg.path("metrics_md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Run the whole evaluation, or average filled-in rating sheets with --ratings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--ratings", nargs="+", help="filled-in copies of human_rating_sheet.csv")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.ratings:
        print(average_ratings(cfg, args.ratings))
        return
    ctx = load_context(str(args.config))
    data = {s: pd.read_csv(cfg.path("processed_dir") / f"{s}.csv") for s in ("train", "val", "test")}
    train_set = set(data["train"]["question"].str.lower().str.strip())
    tri = TrigramSampler(cfg, build_vocab(cfg, data["train"]), data["train"])

    runs, first = compare(ctx, tri, train_set)
    test = data["test"]
    result = {"perplexity": perplexities(ctx, tri, data),
              "summary": {k: mean_std(v) for k, v in runs.items()},
              "real_test": text_metrics(test["question"].tolist(), test["topic"].tolist(),
                                        test["difficulty"].tolist(), ctx, train_set),
              "classifier_ceiling": json.loads(cfg.path("classifier_report").read_text(encoding="utf-8"))["test"],
              "ablations": ablations(ctx, train_set)}
    cfg.path("metrics_json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_metrics_md(result, cfg)
    plot_perplexity(result["perplexity"], cfg.path("perplexity_bar"))
    plot_confusion(first["LSTM"][0], ctx, cfg.path("difficulty_confusion"))
    plot_relevance(first, test, ctx, cfg.path("relevance_by_topic"))
    write_samples(first, cfg, cfg.path("samples_md"))
    write_rating_sheet(first, test, cfg)
    print(cfg.path("metrics_md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
