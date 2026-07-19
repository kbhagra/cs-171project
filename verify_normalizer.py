"""
Verify normalization against real data.

Unit tests prove normalize() handles the cases we thought of. This checks
what it does to the 700k+ cases we didn't, and re-measures the path
confound now that trailing-slash formatting has been removed.

Usage:
    python verify_normalization.py --csv data/splits/train.csv
"""

import argparse

import pandas as pd

from url_normalizer import normalize


def section(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits/train.csv")
    ap.add_argument("--sample", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"{len(df):,} rows from {args.csv}")

    # ---- 1. eyeball it -------------------------------------------------
    section("1. SPOT CHECK -- read every line, look for mangling")
    for u in df["url"].sample(args.sample, random_state=1):
        n = normalize(u)["normalized"]
        flag = "  <-- EMPTY!" if not n else ""
        print(f"  {str(u)[:52]:<54} -> {n[:48]}{flag}")

    # ---- 2. apply to everything ---------------------------------------
    section("2. APPLYING TO FULL SPLIT")
    parsed = df["url"].map(normalize)
    df["norm"] = parsed.map(lambda d: d["normalized"])
    df["host"] = parsed.map(lambda d: d["host"])
    df["npath"] = parsed.map(lambda d: d["path"])
    df["had_scheme"] = parsed.map(lambda d: d["had_scheme"])
    df["is_ip"] = parsed.map(lambda d: d["is_ip"])
    df["is_puny"] = parsed.map(lambda d: d["is_punycode"])

    empties = (df["norm"].str.len() == 0).sum()
    print(f"  empty results: {empties:,}  (should be 0 or near it)")
    nohost = (df["host"].str.len() == 0).sum()
    print(f"  empty hostnames: {nohost:,}")

    # ---- 3. THE number -------------------------------------------------
    section("3. PATH CONFOUND, RE-MEASURED (was 33.8% before)")

    has_path = df["npath"].str.len() > 0
    rates = has_path.groupby(df["label"]).mean()
    for lab in sorted(rates.index):
        name = "phishing" if lab == 1 else "benign  "
        print(f"  {name}: {rates[lab]:.1%} have a path")

    gap = abs(rates.max() - rates.min())
    print(f"\n  NEW GAP: {gap:.1%}")
    if gap > 0.30:
        print("    Still severe -> go hostname-only for the main model.")
    elif gap > 0.15:
        print("    Moderate -> keep path features, but include has_path")
        print("    explicitly and watch its feature importance.")
    else:
        print("    Resolved. Most of the old gap was trailing slashes.")
        print("    Path features are safe to use.")

    # ---- 4. other artifacts -------------------------------------------
    section("4. REMAINING ARTIFACTS")

    sch = df.groupby("label")["had_scheme"].mean()
    print("  had_scheme by class (the merge artifact -- do NOT use as a feature):")
    for lab in sorted(sch.index):
        name = "phishing" if lab == 1 else "benign  "
        print(f"    {name}: {sch[lab]:.1%}")

    print("\n  structural flags by class (these ARE legitimate features):")
    for col in ["is_ip", "is_puny"]:
        r = df.groupby("label")[col].mean()
        vals = "  ".join(f"label={l}: {v:.3%}" for l, v in r.items())
        print(f"    {col:<9} {vals}")

    # host duplication tells you how much n-gram signal is repeated
    section("5. HOSTNAME REDUNDANCY")
    print(f"  {df['host'].nunique():,} unique hostnames "
          f"for {len(df):,} rows")
    print(f"  top repeated hostnames:")
    for h, c in df["host"].value_counts().head(8).items():
        print(f"    {c:>6,}  {h}")


if __name__ == "__main__":
    main()