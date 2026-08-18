#!/usr/bin/env python3
"""Export 44-ФЗ documents from ЕИС (zakupki.gov.ru) into a pipeline corpus.

Procurement documentation is the densest near-duplicate source available in
Russian: every notification is generated from the same regulated template, so
documents differ only in customer, object, dates and amounts. That is the exact
distractor structure SID injects synthetically — here it comes for free, at
scale, from a real corpus.

Two halves, deliberately separate:

    fetch   talks to https://int44.zakupki.gov.ru/eis-integration/services/
            and saves raw archives. Needs a token and a Russian IP.
    build   turns already-downloaded archives into
            {"file_name","index","raw_text","document_id","title"} chunks.
            No network, no credentials — runs anywhere, including CI.

So the machine that has ЕИС access does not have to be the machine that builds
the dataset: rsync the raw archives and run ``build`` wherever you like.

Access, as of 2025 (this trips up every older guide):

  * the FTP dump ``ftp://fz223free@ftp.zakupki.gov.ru`` was **closed on
    2025-01-01** — recipes based on it no longer work at all;
  * individuals get a token at https://zakupki.gov.ru/pmd/auth/welcome
    (Госуслуги → «Регистрация нового потребителя машиночитаемых данных» →
    «Физическое лицо»); it goes in ``EIS_TOKEN``;
  * legal entities sign requests with a qualified ЭЦП against ``getDocsLE2``;
    the ЭЦП path is not implemented here;
  * zakupki.gov.ru refuses connections from non-Russian address space, so
    ``fetch`` must run from a Russian IP regardless of credentials.

Examples:
    # 0. keep the live schema next to your data — element names do drift
    python scripts/build_zakupki_corpus.py xsd --out data/zakupki/getDocsIP.xsd

    # 1. pull a month of electronic-auction notifications for two regions
    export EIS_TOKEN=5d035886-...
    python scripts/build_zakupki_corpus.py fetch \
        --regions 72,77 --date-from 2025-06-01 --date-to 2025-06-30 \
        --raw-dir data/zakupki/raw

    # 2. build the corpus (offline, no token)
    python scripts/build_zakupki_corpus.py build \
        --raw-dir data/zakupki/raw --out-dir data/zakupki

    # 3. how duplicated is this slice, really?
    python scripts/build_zakupki_corpus.py stats --corpus data/zakupki/zakupki_index.jsonl

Then point SID at it:
    python run_sid.py all --config config_sid.yaml --corpus data/zakupki/zakupki_index.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import replace
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arqg.utils import append_jsonl, ensure_parent, log, read_jsonl, setup_logging
from arqg.zakupki.client import (BASE_URL, DEFAULT_DOCUMENT_TYPES, SERVICES, SUBSYSTEMS,
                                 EisClient, EisConfig, EisError)
from arqg.zakupki.corpus import ChunkOptions, build_corpus, duplicate_report, write_report
from arqg.zakupki.merge import SourceSpec, build_merged, write_manifest
from arqg.zakupki.parse import parse_path
from arqg.zakupki.tabular import (PROFILES, detect_profile, iter_docs,
                                  parse_column_overrides)

DEFAULT_RAW_DIR = "data/zakupki/raw"
DEFAULT_OUT_DIR = "data/zakupki"
MANIFEST = "manifest.jsonl"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def daterange(start: str, end: str) -> list[str]:
    a = dt.date.fromisoformat(start)
    b = dt.date.fromisoformat(end)
    if b < a:
        raise SystemExit(f"--date-to {end} is before --date-from {start}")
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def archive_name(url: str, provenance: dict[str, str]) -> str:
    """Stable local filename for an archive URL (so re-runs resume, not re-download)."""
    tail = re.sub(r"[^\w.-]+", "_", url.rsplit("/", 1)[-1].split("?")[0])[:80]
    stem = "_".join(filter(None, [provenance.get("date", ""),
                                  provenance.get("org_region", ""),
                                  provenance.get("document_type", "")]))
    if not tail or tail in (".", "_"):
        tail = f"{abs(hash(url)) % 10**12:012d}.zip"
    if not tail.lower().endswith(".zip"):
        tail += ".zip"
    return f"{stem}__{tail}" if stem else tail


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = EisConfig(token=args.token or os.environ.get("EIS_TOKEN", ""),
                    service=args.service, base_url=args.base_url,
                    ws_namespace=args.ws_namespace, mode=args.mode,
                    timeout=args.timeout, retries=args.retries, pause=args.pause)
    try:
        client = EisClient(cfg)
    except EisError as e:
        log.error("%s", e)
        return 2

    os.makedirs(args.raw_dir, exist_ok=True)
    manifest_path = os.path.join(args.raw_dir, MANIFEST)
    already = {rec.get("url") for rec in read_jsonl(manifest_path)} \
        if os.path.exists(manifest_path) else set()

    if args.reestr_numbers:
        pairs: Iterable[tuple[str, dict[str, str]]] = (
            (url, {"reestr_number": num, "subsystem": args.subsystem})
            for num in args.reestr_numbers
            for url in _safe(client.by_reestr_number, num, subsystem=args.subsystem)
        )
    else:
        dates = ([args.date] if args.date
                 else daterange(args.date_from, args.date_to))
        log.info("zakupki: %d day(s) × %d region(s) × %d type(s) = %d requests",
                 len(dates), len(args.regions), len(args.doc_types),
                 len(dates) * len(args.regions) * len(args.doc_types))
        pairs = client.iter_org_region_archives(args.regions, args.doc_types, dates,
                                                subsystem=args.subsystem)

    n_new = n_skipped = n_failed = 0
    for url, provenance in pairs:
        if url in already:
            n_skipped += 1
            continue
        dest = os.path.join(args.raw_dir, archive_name(url, provenance))
        if args.dry_run:
            log.info("zakupki: would download %s -> %s", url, dest)
            n_new += 1
        else:
            try:
                client.download(url, dest)
            except EisError as e:
                log.warning("zakupki: %s", e)
                n_failed += 1
                continue
            append_jsonl(manifest_path, {"url": url, "path": dest, **provenance})
            already.add(url)
            n_new += 1
        if args.limit and n_new >= args.limit:
            log.info("zakupki: --limit %d reached", args.limit)
            break

    log.info("zakupki: fetch done — %d new, %d already present, %d failed",
             n_new, n_skipped, n_failed)
    return 0 if n_new or n_skipped else 1


def _safe(fn, *a, **kw) -> list[str]:
    try:
        return fn(*a, **kw)
    except EisError as e:
        log.warning("zakupki: %s", e)
        return []


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def cmd_build(args: argparse.Namespace) -> int:
    source = args.input or args.raw_dir
    if not os.path.exists(source):
        log.error("no such path: %s (run `fetch` first, or pass --input)", source)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    corpus_path = args.corpus or os.path.join(args.out_dir, "zakupki_index.jsonl")
    docs_path = "" if args.no_docs else os.path.join(args.out_dir, "zakupki_documents.jsonl")
    report_path = os.path.join(args.out_dir, "zakupki_report.json")

    opts = ChunkOptions(max_chars=args.max_chars, merge_below=args.merge_below,
                        min_chars=args.min_chars,
                        max_chunks_per_doc=args.max_chunks_per_doc)

    docs = parse_path(source)
    if args.limit:
        docs = _take(docs, args.limit)
    stats = build_corpus(docs, corpus_path, opts=opts, docs_path=docs_path)
    if not stats["n_chunks"]:
        log.error("no chunks produced from %s — is it really an EIS archive?", source)
        return 1

    records = list(read_jsonl(corpus_path))
    report = {"source": source, "corpus": corpus_path, **stats,
              **duplicate_report(records, examples=args.examples)}
    write_report(report, report_path)
    _print_report(report)
    return 0


def _take(it, n: int):
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x


# --------------------------------------------------------------------------- #
# from-table
# --------------------------------------------------------------------------- #
def cmd_from_table(args: argparse.Namespace) -> int:
    """Build the same corpus from a third-party dump instead of the live service."""
    if not os.path.exists(args.input):
        log.error("no such file: %s", args.input)
        return 2

    if args.profile == "auto":
        profile = detect_profile(args.input)
        if profile is None:
            log.error("could not recognise %s — pass --profile with --column overrides "
                      "(known profiles: %s)", args.input, ", ".join(sorted(PROFILES)))
            return 2
    else:
        profile = PROFILES[args.profile]
    if args.column:
        profile = replace(profile, columns={**profile.columns,
                                            **parse_column_overrides(args.column)})
    if args.delimiter:
        profile = replace(profile, delimiter=args.delimiter)
    if not profile.columns:
        log.error("profile %r has no columns; supply them with --column canonical=source",
                  profile.name)
        return 2
    log.info("zakupki: %s — %s (licence: %s)", profile.name, profile.source or args.input,
             profile.licence)

    os.makedirs(args.out_dir, exist_ok=True)
    corpus_path = args.corpus or os.path.join(args.out_dir, f"zakupki_{profile.name}.jsonl")
    report_path = os.path.join(args.out_dir, f"zakupki_{profile.name}_report.json")
    docs_path = "" if args.no_docs else os.path.join(
        args.out_dir, f"zakupki_{profile.name}_documents.jsonl")

    # Record cards are short: merging sections would collapse a row into a single
    # chunk, and a one-chunk document can never form a neighbour window.
    opts = ChunkOptions(max_chars=args.max_chars, merge_below=args.merge_below,
                        min_chars=args.min_chars,
                        max_chunks_per_doc=args.max_chunks_per_doc)

    stats = build_corpus(iter_docs(args.input, profile, limit=args.limit),
                         corpus_path, opts=opts, docs_path=docs_path)
    if not stats["n_chunks"]:
        log.error("no chunks produced from %s", args.input)
        return 1

    records = list(read_jsonl(corpus_path))
    report = {"source": profile.source or args.input, "profile": profile.name,
              "licence": profile.licence, "corpus": corpus_path, **stats,
              **duplicate_report(records, examples=args.examples)}
    write_report(report, report_path)
    _print_report(report)
    _warn_thin_documents(records)
    return 0


# --------------------------------------------------------------------------- #
# dumps
# --------------------------------------------------------------------------- #
def materialise_download(archive: str, target: str, member: str,
                         out_dir: str) -> None:
    """Turn a fetched blob into ``target``.

    Kaggle serves a zip whose member is the dump; Hugging Face serves the
    ``.xlsx`` itself. Office Open XML is a zip too, so ``is_zipfile`` alone is
    not enough — only extract when the named member is actually inside.
    """
    import zipfile

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if member in names:
                zf.extract(member, out_dir)
                os.remove(archive)
                return
            if member.endswith((".xlsx", ".xlsm")):
                os.replace(archive, target)
                return
        raise FileNotFoundError(
            f"{member} not found in archive (has {', '.join(names[:6])})")
    os.replace(archive, target)


def cmd_dumps(args: argparse.Namespace) -> int:
    """Download the published dumps, so `merge` is reproducible from nothing.

    All three are served without credentials — Kaggle over its public API, the
    HF file over the resolve endpoint. Existing files are left alone, so a failed
    run resumes rather than re-fetching 150 MB.
    """
    import httpx

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = args.profiles or [n for n, p in PROFILES.items() if p.download_url]
    ready: list[tuple[str, str]] = []
    failed: list[str] = []

    for name in wanted:
        profile = PROFILES.get(name)
        if profile is None or not profile.download_url:
            log.error("no download URL for profile %r", name)
            failed.append(name)
            continue
        target = os.path.join(args.out_dir, profile.member)
        if os.path.exists(target) and not args.force:
            log.info("zakupki: %s already present (%s)", profile.member, name)
            ready.append((name, target))
            continue

        archive = os.path.join(args.out_dir, f"{name}.download")
        log.info("zakupki: fetching %s (%s, licence: %s)",
                 name, profile.source, profile.licence)
        try:
            with httpx.Client(timeout=600.0, follow_redirects=True) as http:
                with http.stream("GET", profile.download_url) as r:
                    r.raise_for_status()
                    with open(archive, "wb") as f:
                        for block in r.iter_bytes(1 << 20):
                            f.write(block)
        except httpx.HTTPError as e:
            log.error("zakupki: %s failed: %s", name, e)
            failed.append(name)
            continue

        try:
            materialise_download(archive, target, profile.member, args.out_dir)
        except FileNotFoundError as e:
            log.error("zakupki: %s: %s", name, e)
            failed.append(name)
            continue
        log.info("zakupki: %s -> %s (%.1f MB)", name, target,
                 os.path.getsize(target) / 1e6)
        ready.append((name, target))

    if not ready:
        return 1
    print("\n— дампы готовы —")
    for name, path in ready:
        print(f"  {name:16} {path}  [{PROFILES[name].licence}]")
    # No `:profile` suffix: every downloaded dump is recognised from its header.
    inputs = " \\\n    ".join(f"--input {path}" for _, path in ready)
    print("\nСобрать один корпус:\n")
    print(f"  python scripts/build_zakupki_corpus.py merge \\\n    {inputs} \\\n"
          f"    --name zakupki_all --out-dir {DEFAULT_OUT_DIR}\n")
    if failed:
        log.warning("zakupki: %d dump(s) could not be fetched: %s",
                    len(failed), ", ".join(failed))
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def _resolve_source(spec: str, overrides: list[str], delimiter: str) -> SourceSpec:
    """``path[:profile[:limit]]`` -> a SourceSpec, detecting the profile if omitted."""
    parts = spec.split(":")
    path = parts[0]
    # a Windows drive letter or a URL-ish path would break the naive split
    if len(parts) > 1 and not os.path.exists(path) and os.path.exists(spec):
        path, parts = spec, [spec]
    if not os.path.exists(path):
        raise SystemExit(f"no such file: {path}")
    name = parts[1] if len(parts) > 1 and parts[1] else "auto"
    limit = int(parts[2]) if len(parts) > 2 and parts[2] else 0

    profile = detect_profile(path) if name == "auto" else PROFILES.get(name)
    if profile is None:
        raise SystemExit(
            f"could not recognise {path}; name a profile as <path>:<profile> "
            f"(known: {', '.join(sorted(PROFILES))})")
    if overrides:
        profile = replace(profile, columns={**profile.columns,
                                            **parse_column_overrides(overrides)})
    if delimiter:
        profile = replace(profile, delimiter=delimiter)
    return SourceSpec(path=path, profile=profile, limit=limit)


def cmd_merge(args: argparse.Namespace) -> int:
    """Fold every available dump into one corpus with a metadata sidecar."""
    specs = [_resolve_source(s, args.column, args.delimiter) for s in args.input]
    for spec in specs:
        log.info("zakupki: source %s — %s (licence: %s%s)", spec.profile.name,
                 spec.path, spec.profile.licence,
                 f", limit {spec.limit}" if spec.limit else "")

    os.makedirs(args.out_dir, exist_ok=True)
    corpus_path = args.corpus or os.path.join(args.out_dir, f"{args.name}.jsonl")
    meta_path = "" if args.no_meta else os.path.join(args.out_dir, f"{args.name}_meta.jsonl")
    docs_path = "" if args.no_meta else os.path.join(args.out_dir, f"{args.name}_documents.jsonl")
    manifest_path = os.path.join(args.out_dir, f"{args.name}_manifest.json")
    report_path = os.path.join(args.out_dir, f"{args.name}_report.json")

    opts = ChunkOptions(max_chars=args.max_chars, merge_below=args.merge_below,
                        min_chars=args.min_chars,
                        max_chunks_per_doc=args.max_chunks_per_doc)
    stats = build_merged(specs, corpus_path, meta_path=meta_path,
                         docs_path=docs_path, opts=opts)
    if not stats["n_chunks"]:
        log.error("no chunks produced")
        return 1

    write_manifest(manifest_path, specs, stats, corpus_path, meta_path)
    report = {"corpus": corpus_path, "metadata": meta_path, **stats,
              **duplicate_report(read_jsonl(corpus_path), examples=args.examples)}
    write_report(report, report_path)
    _print_report(report)
    if meta_path:
        print(f"метаданные (чанки):    {meta_path}")
        print(f"метаданные (документы):{docs_path}")
    return 0


def _warn_thin_documents(records: list[dict]) -> None:
    """Windows need ≥2 chunks in a file; say so loudly if most rows give one."""
    per_file: dict[str, int] = {}
    for rec in records:
        per_file[rec["file_name"]] = per_file.get(rec["file_name"], 0) + 1
    singles = sum(1 for n in per_file.values() if n < 2)
    if singles:
        log.warning("zakupki: %d/%d documents have a single chunk — those cannot form "
                    "neighbour windows; lower --merge-below or --min-chars",
                    singles, len(per_file))


def cmd_stats(args: argparse.Namespace) -> int:
    if not os.path.exists(args.corpus):
        log.error("no such corpus: %s", args.corpus)
        return 2
    report = {"corpus": args.corpus,
              **duplicate_report(read_jsonl(args.corpus), examples=args.examples)}
    if args.report:
        write_report(report, args.report)
    _print_report(report)
    return 0


def _print_report(report: dict) -> None:
    ex = report.get("exact_duplicates", {})
    st = report.get("structural_duplicates", {})
    parsed = report.get("n_documents_parsed")
    files = report.get("n_files", 0)
    print("\n— выгрузка ЕИС —")
    print(f"документов:            {files}"
          + (f" (разобрано {parsed}, дублей отброшено {parsed - files})"
             if parsed and parsed != files else ""))
    print(f"чанков:                {report.get('n_chunks', 0)}")
    if "mean_chunk_chars" in report:
        print(f"средний размер чанка:  {report['mean_chunk_chars']} симв.")
    if report.get("document_types"):
        print(f"типы документов:       {json.dumps(report['document_types'], ensure_ascii=False)}")
    print(f"точные дубликаты:      {ex.get('n_groups', 0)} групп, "
          f"{ex.get('share_of_chunks', 0):.1%} чанков")
    print(f"шаблонные дубликаты:   {st.get('n_groups', 0)} групп, "
          f"{st.get('share_of_chunks', 0):.1%} чанков, "
          f"средний Jaccard {st.get('mean_jaccard_within_group', 0):.2f}")
    print()


# --------------------------------------------------------------------------- #
# xsd
# --------------------------------------------------------------------------- #
def cmd_xsd(args: argparse.Namespace) -> int:
    """Save the live integration schema — the only authority on element order."""
    import httpx
    url = f"{args.base_url.rstrip('/')}/{args.service}?xsd={args.service}-ws-api.xsd"
    log.info("zakupki: GET %s", url)
    try:
        r = httpx.get(url, timeout=60.0, follow_redirects=True,
                      headers={"User-Agent": "arqg-zakupki/1.0"})
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.error("cannot reach %s: %s", url, e)
        log.error("zakupki.gov.ru refuses connections from non-Russian address space; "
                  "run this from a Russian IP.")
        return 1
    ensure_parent(args.out)
    with open(args.out, "wb") as f:
        f.write(r.content)
    log.info("zakupki: schema saved -> %s (%d bytes)", args.out, len(r.content))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    # accepted on either side of the subcommand — `--log-level DEBUG build …`
    # and `build --log-level DEBUG` both work
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO")

    # ---- fetch ---- #
    f = sub.add_parser("fetch", parents=[common],
                       help="download raw archives from the ЕИС data services")
    f.add_argument("--token", default="", help="ЕИС token (default: $EIS_TOKEN)")
    f.add_argument("--service", default="getDocsIP", choices=sorted(SERVICES),
                   help="getDocsIP = token (физлицо); getDocsLE2 = ЭЦП (юрлицо)")
    f.add_argument("--base-url", default=BASE_URL)
    f.add_argument("--ws-namespace", default="",
                   help="override the request namespace if the XSD says otherwise")
    f.add_argument("--mode", default="PROD", choices=("PROD", "TEST"))
    f.add_argument("--subsystem", default="PRIZ", choices=sorted(SUBSYSTEMS),
                   help="; ".join(f"{k} — {v}" for k, v in SUBSYSTEMS.items()))
    f.add_argument("--regions", type=csv_list, default=["77"],
                   help="comma-separated region codes, e.g. 72,77")
    f.add_argument("--doc-types", type=csv_list, default=list(DEFAULT_DOCUMENT_TYPES),
                   help="comma-separated documentType44 values")
    f.add_argument("--date", default="", help="single day, YYYY-MM-DD")
    f.add_argument("--date-from", default="", help="range start, YYYY-MM-DD")
    f.add_argument("--date-to", default="", help="range end, YYYY-MM-DD")
    f.add_argument("--reestr-numbers", type=csv_list, default=[],
                   help="fetch by registry number instead of region/date")
    f.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    f.add_argument("--limit", type=int, default=0, help="stop after N new archives")
    f.add_argument("--timeout", type=float, default=180.0)
    f.add_argument("--retries", type=int, default=4)
    f.add_argument("--pause", type=float, default=2.0,
                   help="seconds between calls; the services are quota-limited")
    f.add_argument("--dry-run", action="store_true",
                   help="resolve archive URLs but do not download them")

    # ---- build ---- #
    b = sub.add_parser("build", parents=[common], help="turn raw archives into a corpus (offline)")
    b.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    b.add_argument("--input", default="", help="explicit .zip/.xml/dir instead of --raw-dir")
    b.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    b.add_argument("--corpus", default="", help="corpus path (default: <out-dir>/zakupki_index.jsonl)")
    b.add_argument("--no-docs", action="store_true", help="skip the document-level dump")
    b.add_argument("--limit", type=int, default=0, help="stop after N documents")
    b.add_argument("--max-chars", type=int, default=1400)
    b.add_argument("--merge-below", type=int, default=600)
    b.add_argument("--min-chars", type=int, default=120)
    b.add_argument("--max-chunks-per-doc", type=int, default=40)
    b.add_argument("--examples", type=int, default=5)

    # ---- from-table ---- #
    t = sub.add_parser("from-table", parents=[common],
                       help="build the corpus from a third-party CSV/XLSX dump "
                            "(no token, no Russian IP)")
    t.add_argument("--input", required=True, help="path to the .csv / .xlsx dump")
    t.add_argument("--profile", default="auto",
                   choices=["auto", *sorted(PROFILES)],
                   help="; ".join(f"{k} — {v.source or v.notes}" for k, v in PROFILES.items()))
    t.add_argument("--column", action="append", default=[], metavar="CANON=COLUMN",
                   help="map a canonical field onto a source column (repeatable)")
    t.add_argument("--delimiter", default="", help="override the CSV delimiter")
    t.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    t.add_argument("--corpus", default="")
    t.add_argument("--no-docs", action="store_true")
    t.add_argument("--limit", type=int, default=0, help="stop after N documents")
    t.add_argument("--max-chars", type=int, default=1400)
    t.add_argument("--merge-below", type=int, default=250,
                   help="fold sections shorter than this into a neighbour. 0 keeps "
                        "every section separate, which leaves one-line sections that "
                        "thousands of documents share verbatim")
    t.add_argument("--min-chars", type=int, default=40)
    t.add_argument("--max-chunks-per-doc", type=int, default=40)
    t.add_argument("--examples", type=int, default=5)

    # ---- dumps ---- #
    d = sub.add_parser("dumps", parents=[common],
                       help="download the published dumps (no credentials needed)")
    d.add_argument("--out-dir", default=os.path.join(DEFAULT_OUT_DIR, "dumps"))
    d.add_argument("--profiles", type=csv_list, default=[],
                   help="comma-separated subset (default: every downloadable profile)")
    d.add_argument("--force", action="store_true", help="re-download files already present")

    # ---- merge ---- #
    m = sub.add_parser("merge", parents=[common],
                       help="fold several dumps into ONE corpus + metadata sidecar")
    m.add_argument("--input", action="append", required=True, metavar="PATH[:PROFILE[:LIMIT]]",
                   help="a dump to include; repeat for each. PROFILE defaults to "
                        "auto-detection, LIMIT caps the rows taken from that file")
    m.add_argument("--name", default="zakupki", help="base name for the output files")
    m.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    m.add_argument("--corpus", default="")
    m.add_argument("--no-meta", action="store_true",
                   help="skip both metadata sidecars")
    m.add_argument("--column", action="append", default=[], metavar="CANON=COLUMN",
                   help="column overrides applied to every source")
    m.add_argument("--delimiter", default="")
    m.add_argument("--max-chars", type=int, default=1400)
    m.add_argument("--merge-below", type=int, default=250,
                   help="fold sections shorter than this into a neighbour "
                        "(default 250; see the README for the trade-off)")
    m.add_argument("--min-chars", type=int, default=40)
    m.add_argument("--max-chunks-per-doc", type=int, default=40)
    m.add_argument("--examples", type=int, default=5)

    # ---- stats ---- #
    s = sub.add_parser("stats", parents=[common], help="near-duplicate report over an existing corpus")
    s.add_argument("--corpus", default=os.path.join(DEFAULT_OUT_DIR, "zakupki_index.jsonl"))
    s.add_argument("--report", default="", help="also write the report as JSON here")
    s.add_argument("--examples", type=int, default=5)

    # ---- xsd ---- #
    x = sub.add_parser("xsd", parents=[common], help="download the integration XSD")
    x.add_argument("--service", default="getDocsIP", choices=sorted(SERVICES))
    x.add_argument("--base-url", default=BASE_URL)
    x.add_argument("--out", default=os.path.join(DEFAULT_OUT_DIR, "getDocsIP-ws-api.xsd"))

    args = p.parse_args(argv)
    if args.command == "fetch" and not args.reestr_numbers:
        if not args.date and not (args.date_from and args.date_to):
            p.error("fetch needs --date, or --date-from with --date-to, "
                    "or --reestr-numbers")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    return {"fetch": cmd_fetch, "build": cmd_build, "from-table": cmd_from_table,
            "dumps": cmd_dumps, "merge": cmd_merge,
            "stats": cmd_stats, "xsd": cmd_xsd}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
