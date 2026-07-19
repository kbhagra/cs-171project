"""
Phishing URL detector -- dataset preparation.

Loads the raw Kaggle CSV, runs sanity checks, deduplicates, and writes a
train/val/test split grouped by registered domain (eTLD+1) so that no domain
appears in more than one split.

Usage:
    python prepare_data.py --csv data/raw.csv --outdir data/splits
"""

import argparse
import os

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

from url_normalizer import normalize

# tldextract normally fetches the public-suffix list over the network on first
# use. suffix_list_urls=() forces the bundled snapshot instead: no network call,
# and identical results on every machine and every rerun.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

SEED = 42


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def resolve_csv(path):
    """Accept a CSV path, or a directory containing exactly one CSV.

    kagglehub returns a directory, so passing it straight through is an easy
    mistake -- handle it instead of failing deep inside pandas.
    """
    path = os.path.expanduser(path)

    if os.path.isdir(path):
        csvs = sorted(f for f in os.listdir(path) if f.lower().endswith(".csv"))
        if len(csvs) == 1:
            resolved = os.path.join(path, csvs[0])
            print(f"note: '{path}' is a directory; using {csvs[0]}")
            return resolved
        if not csvs:
            raise SystemExit(
                f"'{path}' is a directory and contains no CSV files.\n"
                f"Contents: {os.listdir(path)}"
            )
        raise SystemExit(
            f"'{path}' is a directory containing several CSVs.\n"
            f"Pass one explicitly: {csvs}"
        )

    if not os.path.exists(path):
        raise SystemExit(f"no such file: {path}")

    return path


def load_raw(csv_path):
    df = pd.read_csv(resolve_csv(csv_path))
    print(f"loaded {len(df):,} rows, columns: {df.columns.tolist()}")

    # Column names vary between mirrors of this dataset.
    url_col = _find_col(df, ["url", "URL", "urls"])
    label_col = _find_col(df, ["status", "label", "Label", "type", "result"])
    df = df[[url_col, label_col]].rename(
        columns={url_col: "url", label_col: "raw_label"}
    )

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 0]
    return df


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"could not find a column among {candidates}; got {df.columns.tolist()}"
    )


def normalize_labels(df, phishing_values):
    """Map the raw label column to 1 = phishing, 0 = benign.

    The dataset docs claim 0 = phishing, 1 = legitimate, but mirrors disagree.
    Always eyeball the printout below before trusting it.
    """
    vals = df["raw_label"].astype(str).str.strip().str.lower()
    phishing_values = {str(v).strip().lower() for v in phishing_values}

    unknown = set(vals.unique()) - phishing_values
    print(f"\nraw label values: {sorted(set(vals.unique()))}")
    print(f"treating as phishing: {sorted(phishing_values)}")
    print(f"treating as benign:   {sorted(unknown)}")

    df["label"] = vals.isin(phishing_values).astype(int)
    return df.drop(columns=["raw_label"])


# --------------------------------------------------------------------------
# sanity checks
# --------------------------------------------------------------------------

def sanity_checks(df):
    print("\n" + "=" * 62)
    print("SANITY CHECKS -- read these before going further")
    print("=" * 62)

    n = len(df)
    counts = df["label"].value_counts()
    print(f"\nclass balance:")
    for lab in sorted(counts.index):
        name = "phishing" if lab == 1 else "benign  "
        print(f"  {name} (label={lab}): {counts[lab]:>7,}  ({counts[lab]/n:.1%})")

    print("\nsample URLs by class (verify the labels look right!):")
    for lab in sorted(counts.index):
        name = "phishing" if lab == 1 else "benign"
        print(f"\n  --- {name} ---")
        for u in df[df["label"] == lab]["url"].head(5):
            print(f"    {u[:100]}")

    # The path-length confound: if benign entries are bare domains and phishing
    # entries are full URLs, a model learns "has a path", not "is phishing".
    df["_len"] = df["url"].str.len()
    df["_has_path"] = df["url"].str.replace(
        r"^https?://", "", regex=True
    ).str.contains("/")

    print("\nURL length by class:")
    print(df.groupby("label")["_len"].describe()[
        ["mean", "50%", "75%", "max"]
    ].round(1).to_string())

    path_rate = df.groupby("label")["_has_path"].mean()
    print("\nfraction with a URL path:")
    for lab in sorted(path_rate.index):
        name = "phishing" if lab == 1 else "benign  "
        print(f"  {name}: {path_rate[lab]:.1%}")

    gap = abs(path_rate.get(1, 0) - path_rate.get(0, 0))
    if gap > 0.30:
        print(
            f"\n  *** WARNING: {gap:.0%} gap in path presence between classes."
            f"\n  *** Your model may just be learning 'has a path'."
            f"\n  *** Check feature importances carefully, and consider"
            f"\n  *** reporting results on the path-only subset as well."
        )

    return df.drop(columns=["_len", "_has_path"])


# --------------------------------------------------------------------------
# dedup + grouping
# --------------------------------------------------------------------------

def add_domain(df):
    """Registered domain (eTLD+1) -- the grouping key for the split.

    Derived from the NORMALIZED hostname, not the raw URL. This matters:
    tldextract finds no suffix for raw IP addresses, and an earlier version
    fell back to the whole URL string, which gave two URLs on the same IP
    two different group keys. They could then land in different splits, and
    the leak only became visible after collapsing to hostnames.

    Falling back to the hostname itself guarantees that identical hostnames
    always share a group key.
    """
    def reg_domain(host):
        if not host:
            return ""
        ext = _EXTRACT(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        # IP addresses, single-label hosts, malformed entries
        return host

    df["host"] = df["url"].map(lambda u: normalize(u)["host"])
    before = len(df)
    df = df[df["host"].str.len() > 0]
    if before != len(df):
        print(f"dropped {before - len(df):,} rows with an unparseable hostname")

    df["domain"] = df["host"].map(reg_domain)
    return df


def deduplicate(df):
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"\ndropped {before - len(df):,} exact duplicate URLs "
          f"({len(df):,} remain)")

    # A domain labelled both phishing and benign is a labelling conflict.
    # Grouping would force it entirely into one split, poisoning that split.
    per_domain = df.groupby("domain")["label"].nunique()
    conflicted = set(per_domain[per_domain > 1].index)
    if conflicted:
        n_rows = df["domain"].isin(conflicted).sum()
        print(f"{len(conflicted):,} domains have BOTH labels "
              f"({n_rows:,} rows) -- dropping them as ambiguous")
        df = df[~df["domain"].isin(conflicted)]

    return df


def report_concentration(df):
    sizes = df.groupby("domain").size().sort_values(ascending=False)
    print(f"\n{len(sizes):,} unique domains across {len(df):,} URLs")
    print(f"top domains by URL count (this is why we group-split):")
    for dom, cnt in sizes.head(8).items():
        print(f"    {cnt:>6,}  {dom}")


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def grouped_split(df, test_frac=0.15, val_frac=0.15):
    """Split so that every registered domain lands in exactly one split."""
    groups = df["domain"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=SEED)
    trainval_idx, test_idx = next(gss.split(df, groups=groups))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    # val_frac is a fraction of the whole, so rescale it for the second split.
    inner = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=inner, random_state=SEED)
    tr_idx, va_idx = next(
        gss2.split(trainval, groups=trainval["domain"].values)
    )
    train, val = trainval.iloc[tr_idx], trainval.iloc[va_idx]

    return train, val, test


def verify_split(train, val, test):
    print("\n" + "=" * 62)
    print("SPLIT SUMMARY")
    print("=" * 62)
    total = len(train) + len(val) + len(test)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rate = part["label"].mean()
        print(f"  {name:<6} {len(part):>7,} rows  ({len(part)/total:.0%})  "
              f"{part['domain'].nunique():>6,} domains  "
              f"phish rate {rate:.1%}")

    # The whole point of the exercise -- assert it actually held.
    for a_name, a, b_name, b in [
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ]:
        overlap = set(a["domain"]) & set(b["domain"])
        assert not overlap, f"{len(overlap)} domains leaked {a_name}<->{b_name}"
        hoverlap = set(a["host"]) & set(b["host"])
        assert not hoverlap, (
            f"{len(hoverlap)} hostnames leaked {a_name}<->{b_name} -- the "
            f"grouping key does not fully determine the hostname")
    print("\n  verified: zero domain AND hostname overlap between splits")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to the raw CSV")
    ap.add_argument("--outdir", default="data/splits")
    ap.add_argument(
        "--phishing-values", nargs="+", default=["0"],
        help="raw label value(s) meaning PHISHING. Dataset docs say 0, but "
             "CHECK the sample URLs printed above before trusting this.",
    )
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = load_raw(args.csv)
    df = normalize_labels(df, args.phishing_values)
    df = sanity_checks(df)
    df = add_domain(df)
    df = deduplicate(df)
    report_concentration(df)

    train, val, test = grouped_split(df, args.test_frac, args.val_frac)
    verify_split(train, val, test)

    os.makedirs(args.outdir, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(args.outdir, f"{name}.csv")
        part[["url", "host", "domain", "label"]].to_csv(out, index=False)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()