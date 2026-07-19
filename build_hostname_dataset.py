"""
Collapse the URL-level splits into hostname-level datasets.

Going hostname-only means 'en.wikipedia.org' would otherwise appear 14,311
times as an identical row. Deduplicating stops the model from spending
capacity memorizing a handful of over-represented hosts, and stops those
hosts from dominating the metrics.

Safe to run after prepare_data.py: splits are grouped by registered domain,
and every hostname sits under exactly one registered domain, so no hostname
can straddle two splits. This script asserts that rather than assuming it.

Usage:
    python build_hostname_dataset.py --indir data/splits --outdir data/hosts
"""

import argparse
import os

import pandas as pd

from url_normalizer import normalize

SPLITS = ["train", "val", "test"]


def section(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def load_split(indir, name):
    path = os.path.join(indir, f"{name}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run prepare_data.py first")
    return pd.read_csv(path)


def to_hostnames(df, on_conflict):
    """Collapse URL rows to unique hostnames, resolving label conflicts."""
    parsed = df["url"].map(normalize)
    df = df.assign(
        host=parsed.map(lambda d: d["host"]),
        is_ip=parsed.map(lambda d: d["is_ip"]),
        is_punycode=parsed.map(lambda d: d["is_punycode"]),
        malformed=parsed.map(lambda d: d["malformed"]),
    )

    n_malformed = int(df["malformed"].sum())
    n_empty = int((df["host"].str.len() == 0).sum())
    df = df[df["host"].str.len() > 0]

    # A hostname carrying both labels is a genuine ambiguity, not noise:
    # usually a legitimate host that also served a phishing page.
    grp = df.groupby("host")
    stats = grp.agg(
        label_mean=("label", "mean"),
        n_urls=("label", "size"),
        domain=("domain", "first"),
        is_ip=("is_ip", "first"),
        is_punycode=("is_punycode", "first"),
    ).reset_index()

    conflicted = stats[(stats["label_mean"] > 0) & (stats["label_mean"] < 1)]
    n_conflict = len(conflicted)

    if on_conflict == "drop":
        stats = stats[(stats["label_mean"] == 0) | (stats["label_mean"] == 1)]
        stats["label"] = stats["label_mean"].astype(int)
    elif on_conflict == "phish":
        # Any phishing URL on the host taints the host. Conservative from a
        # security standpoint; inflates the phishing class.
        stats["label"] = (stats["label_mean"] > 0).astype(int)
    elif on_conflict == "majority":
        stats["label"] = (stats["label_mean"] >= 0.5).astype(int)
    else:
        raise SystemExit(f"unknown --on-conflict value: {on_conflict}")

    stats = stats.drop(columns=["label_mean"])
    return stats, {
        "malformed": n_malformed,
        "empty_host": n_empty,
        "conflicted": n_conflict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/splits")
    ap.add_argument("--outdir", default="data/hosts")
    ap.add_argument(
        "--on-conflict", default="drop", choices=["drop", "phish", "majority"],
        help="what to do with a hostname that carries both labels",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = {}

    for name in SPLITS:
        section(f"{name.upper()}")
        df = load_split(args.indir, name)
        print(f"  {len(df):,} URL rows in")

        hosts, diag = to_hostnames(df, args.on_conflict)

        print(f"  malformed URLs:      {diag['malformed']:,}")
        print(f"  empty hostnames:     {diag['empty_host']:,} (dropped)")
        print(f"  conflicted hosts:    {diag['conflicted']:,} "
              f"(--on-conflict={args.on_conflict})")
        print(f"  -> {len(hosts):,} unique hostnames "
              f"({len(hosts)/max(len(df),1):.1%} of rows)")

        rate = hosts["label"].mean()
        print(f"  phishing rate: {rate:.1%}  "
              f"(was {df['label'].mean():.1%} at URL level)")

        # how much redundancy did we remove?
        print(f"  median URLs collapsed per host: "
              f"{hosts['n_urls'].median():.0f}, "
              f"max {hosts['n_urls'].max():,}")

        out[name] = hosts

    # ---- the invariant that makes this safe --------------------------
    section("LEAKAGE CHECK")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(out[a]["host"]) & set(out[b]["host"])
        assert not overlap, f"{len(overlap)} hostnames leaked {a}<->{b}"
        print(f"  {a} <-> {b}: no shared hostnames")

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(out[a]["domain"]) & set(out[b]["domain"])
        assert not overlap, f"{len(overlap)} domains leaked {a}<->{b}"
    print("  registered-domain grouping still intact")

    section("WRITING")
    cols = ["host", "domain", "label", "n_urls", "is_ip", "is_punycode"]
    for name, hosts in out.items():
        path = os.path.join(args.outdir, f"{name}.csv")
        hosts[cols].to_csv(path, index=False)
        print(f"  {path}  ({len(hosts):,} rows)")

    total = sum(len(h) for h in out.values())
    print(f"\n  {total:,} hostnames total across all splits")
    print("\n  Note: n_urls is kept for analysis only -- do NOT train on it.")
    print("  It encodes how often a host appeared in the source crawl,")
    print("  which is collection metadata, not a property of the hostname.")


if __name__ == "__main__":
    main()