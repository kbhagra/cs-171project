"""
Inspect the raw phishing URL CSV before designing features.

Answers, in order:
  1. Is the label direction what we think it is?
  2. Do these entries have schemes and paths, or are they bare hostnames?
  3. Is the path-length confound present?
  4. Which hosting providers / TLDs dominate, and will grouping break?

Usage:
    python inspect_data.py --csv data/new_data_urls.csv
"""

import argparse
import os
from collections import Counter

import pandas as pd
import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

PHISH, BENIGN = 0, 1   # per the dataset docs; check 1 confirms or refutes this


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


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


def load(csv_path):
    csv_path = resolve_csv(csv_path)
    df = pd.read_csv(csv_path)
    print(f"{len(df):,} rows | columns: {df.columns.tolist()}")

    url_col = next((c for c in df.columns if c.lower() in
                    ("url", "urls", "domain")), df.columns[0])
    lab_col = next((c for c in df.columns if c.lower() in
                    ("status", "label", "type", "result")), df.columns[-1])
    print(f"using url column '{url_col}', label column '{lab_col}'")

    df = df[[url_col, lab_col]].rename(columns={url_col: "url",
                                                lab_col: "label"})
    df["url"] = df["url"].astype(str).str.strip()
    return df[df["url"].str.len() > 0]


# -- 1. label direction ----------------------------------------------------

def check_labels(df):
    section("1. LABEL DIRECTION -- read the URLs, don't trust the docs")

    print("\nclass counts:")
    for lab, cnt in df["label"].value_counts().sort_index().items():
        print(f"  label={lab}: {cnt:>7,}  ({cnt/len(df):.1%})")

    # Random sample, NOT head() -- the file is sorted, so head() shows you
    # the alphabetical top and nothing else.
    for lab in sorted(df["label"].unique()):
        print(f"\n  --- 12 RANDOM urls with label={lab} ---")
        for u in df[df["label"] == lab]["url"].sample(
            12, random_state=0
        ):
            print(f"    {u[:95]}")

    print("\n  >> Which block looks like real websites? That block is benign.")


# -- 2. structure ----------------------------------------------------------

def check_structure(df):
    section("2. URL STRUCTURE -- decides which features are even possible")

    has_scheme = df["url"].str.contains("://", regex=False)
    stripped = df["url"].str.replace(r"^[a-zA-Z]+://", "", regex=True)
    has_path = stripped.str.contains("/", regex=False)
    has_query = df["url"].str.contains(r"\?", regex=True)
    has_port = stripped.str.contains(r":\d", regex=True)

    print(f"\n  overall: scheme {has_scheme.mean():>6.1%} | "
          f"path {has_path.mean():>6.1%} | "
          f"query {has_query.mean():>6.1%} | port {has_port.mean():>6.1%}")

    print("\n  by class:")
    for lab in sorted(df["label"].unique()):
        m = df["label"] == lab
        print(f"    label={lab}:  scheme {has_scheme[m].mean():>6.1%} | "
              f"path {has_path[m].mean():>6.1%} | "
              f"query {has_query[m].mean():>6.1%}")

    print("\n  DECISION RULE:")
    if has_path.mean() < 0.15:
        print("    Paths are nearly absent -> build hostname-only features.")
        print("    Drop slash counts, path keywords, query-param features.")
    elif has_path.mean() > 0.85:
        print("    Paths nearly always present -> full URL features are safe.")
    else:
        print("    MIXED. Write path features to return 0 when absent, and")
        print("    add an explicit has_path flag so the model can condition")
        print("    on it instead of confounding it with the label.")

    return has_path


# -- 3. the confound -------------------------------------------------------

def check_confound(df, has_path):
    section("3. PATH-LENGTH CONFOUND -- the project-killer")

    df = df.assign(_len=df["url"].str.len(), _path=has_path)

    print("\n  URL length by class:")
    print(df.groupby("label")["_len"].describe()
          [["mean", "50%", "75%", "max"]].round(1).to_string())

    rates = df.groupby("label")["_path"].mean()
    print("\n  fraction with a path, by class:")
    for lab, r in rates.items():
        print(f"    label={lab}: {r:.1%}")

    gap = abs(rates.max() - rates.min())
    print(f"\n  gap: {gap:.1%}")
    if gap > 0.30:
        print("    *** SEVERE. A model can hit high accuracy using nothing")
        print("    *** but 'does this have a slash'. Mitigations:")
        print("    ***   a) strip to hostname for ALL rows, everywhere")
        print("    ***   b) or report a second score on the path-only subset")
    elif gap > 0.15:
        print("    Moderate. Keep has_path as an explicit feature and check")
        print("    it isn't dominating your feature importances.")
    else:
        print("    Fine. No special handling needed.")


# -- 4. grouping viability -------------------------------------------------

def check_grouping(df):
    section("4. GROUPING -- will eTLD+1 splitting hold up?")

    def reg(u):
        e = _EXTRACT(u)
        return f"{e.domain}.{e.suffix}".lower() if e.suffix else (e.domain or u).lower()

    df = df.assign(domain=df["url"].map(reg))
    sizes = df.groupby("domain").size().sort_values(ascending=False)

    print(f"\n  {len(sizes):,} unique domains for {len(df):,} URLs "
          f"(avg {len(df)/max(len(sizes),1):.1f} URLs/domain)")
    print(f"  largest domain holds {sizes.iloc[0]/len(df):.2%} of all rows")
    print(f"  top 10 domains hold  {sizes.head(10).sum()/len(df):.2%}")

    print("\n  top 15 domains:")
    for dom, cnt in sizes.head(15).items():
        sub = df[df["domain"] == dom]
        pr = (sub["label"] == PHISH).mean()
        print(f"    {cnt:>7,}  {dom:<34} phish-ish {pr:>5.0%}")

    top_share = sizes.head(10).sum() / len(df)
    print("\n  DECISION RULE:")
    if top_share > 0.20:
        print("    Top domains are too concentrated. eTLD+1 grouping would")
        print("    dump a huge block into one split. Use the FULL hostname")
        print("    as the group key for shared-hosting providers instead.")
    else:
        print("    Concentration is fine. eTLD+1 grouping works as written.")

    # TLD distribution differs sharply between classes in most phishing data
    print("\n  top TLDs by class:")
    for lab in sorted(df["label"].unique()):
        tlds = Counter(df[df["label"] == lab]["domain"]
                       .str.rsplit(".", n=1).str[-1])
        top = ", ".join(f"{t}({c:,})" for t, c in tlds.most_common(8))
        print(f"    label={lab}: {top}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = load(args.csv)
    check_labels(df)
    has_path = check_structure(df)
    check_confound(df, has_path)
    check_grouping(df)

    print("\n" + "=" * 66)
    print("Paste sections 2, 3 and 4 back to decide the feature set.")
    print("=" * 66)


if __name__ == "__main__":
    main()