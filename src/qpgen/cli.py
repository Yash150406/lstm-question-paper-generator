"""Command line: generate questions for one cell, or build and export a paper."""
import argparse
import sys
from functools import partial

from qpgen.config import DEFAULT_CONFIG
from qpgen.export import export
from qpgen.filters import generate_questions, load_context
from qpgen.paper import build_paper, spec_from_preset
from qpgen.upload import analyze_upload, read_upload


def cmd_generate(args: argparse.Namespace) -> int:
    """Print n filtered questions for a subject, topic and difficulty."""
    ctx = load_context(str(args.config))
    subject = ctx.cfg.subjects[ctx.cfg.subject_key(args.subject)]["name"]
    for q in generate_questions(subject, args.topic, args.difficulty, args.n, args.temperature,
                                args.top_k, args.seed, ctx=ctx):
        print(q["text"])
    return 0


def resolve_topics(args: argparse.Namespace, ctx) -> tuple[str, list[str], dict, dict | None]:
    """Subject, topics, weights and upload info, either from flags or from --from-file."""
    cfg = ctx.cfg
    if not args.from_file:
        info = cfg.subjects[cfg.subject_key(args.subject)]
        topics = [t.strip() for t in args.topics.split(",")] if args.topics else info["topics"]
        return info["name"], topics, {}, None
    text = read_upload(args.from_file)
    found = analyze_upload(text, cfg, ctx.scorer)
    if not found["supported"]:
        raise SystemExit(found["message"])
    print(f"Detected {found['subject']}: {', '.join(found['topics'])}")
    return found["subject"], list(found["topics"]), found["topics"], {"text": text, "key_terms": found["key_terms"]}


def cmd_paper(args: argparse.Namespace) -> int:
    """Build a paper and write it in the requested formats."""
    ctx = load_context(str(args.config))
    cfg = ctx.cfg
    subject, topics, weights, upload = resolve_topics(args, ctx)
    overrides = {k: v for k, v in (("institute", args.institute), ("exam_title", args.exam_title),
                                   ("duration_minutes", args.duration)) if v}
    spec = spec_from_preset(cfg, subject, topics, args.preset, weights, **overrides)
    paper = build_paper(spec, args.seed, cfg, partial(generate_questions, ctx=ctx, upload=upload))
    formats = ["docx", "pdf", "json"] if args.format == "all" else [args.format, "json"]
    for path in export(paper, cfg, cfg.subject_key(subject), list(dict.fromkeys(formats))):
        print(f"Wrote {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    """Argument parser with the two subcommands."""
    p = argparse.ArgumentParser(prog="python -m qpgen.cli", description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="print questions for one subject, topic and difficulty")
    g.add_argument("--subject", required=True)
    g.add_argument("--topic", required=True)
    g.add_argument("--difficulty", required=True, choices=["easy", "medium", "hard"])
    g.add_argument("-n", type=int, default=5)
    g.add_argument("--seed", type=int)
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-k", type=int)
    g.set_defaults(func=cmd_generate)

    pp = sub.add_parser("paper", help="build a paper and export it")
    pp.add_argument("--subject", help="needed unless --from-file is given")
    pp.add_argument("--topics", help="comma separated, default is every topic of the subject")
    pp.add_argument("--from-file", help="PDF, .txt or .md notes, or a typed chapter name")
    pp.add_argument("--preset", default="balanced", choices=["balanced", "easy_heavy", "hard_heavy"])
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--format", default="all", choices=["docx", "pdf", "json", "all"])
    pp.add_argument("--institute")
    pp.add_argument("--exam-title")
    pp.add_argument("--duration", type=int)
    pp.set_defaults(func=cmd_paper)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parser().parse_args(argv)
    if args.command == "paper" and not (args.subject or args.from_file):
        parser().error("paper needs --subject or --from-file")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
