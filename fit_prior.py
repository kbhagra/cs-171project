"""
Fit feature priors from the TRAINING SPLIT ONLY.

Replaces hardcoded word lists with lists learned from data. This matters
because hardcoded lists chosen after looking at whole-dataset statistics
are a subtle form of leakage: no code touched the test set, but knowledge
about it reached the feature definition through the analyst.

Everything here reads data/hosts/train.csv and nothing else. The output
priors.json is a fitted artifact, exactly like a TF-IDF vocabulary --
refit it whenever the training split changes, and never refit on val/test.

Statistical care:
  - every list has a minimum support threshold, so rare tokens can't
    qualify on one or two lucky examples
  - selection uses the Wilson lower confidence bound on the phishing rate,
    not the raw rate, so a token seen 30 times must be much more skewed
    than one seen 3000 times to make the cut

Usage:
    python fit_priors.py --train data/hosts/train.csv --out priors.json
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict

import pandas as pd
import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

# Support is counted in DISTINCT REGISTERED DOMAINS, not hostnames.
# One campaign with 543 subdomains is one observation, not 543. Counting
# hostnames let single campaigns ('kylelierman', 'zuzdn') qualify as if
# they were general phishing vocabulary.
MIN_TLD_DOMAINS = 30
MIN_TOKEN_DOMAINS = 20
MIN_FANOUT = 20          # distinct hostnames before a domain is multi-tenant
N_BRANDS = 400
MIN_TOKEN_LEN = 4

# A category with no benign examples cannot be evaluated: we can't tell
# "genuinely malicious" from "postdates our benign crawl". This dataset's
# benign corpus is roughly 2009-2012 (posterous, wetpaint, freebase,
# myspace), so TLDs delegated after that (.app 2018, .dev 2019, .icu,
# .cfd, .cyou) are 100% phishing by construction. Requiring benign support
# excludes them instead of learning the collection date as a feature.
MIN_BENIGN_DOMAINS = 10

# TLDs need MUCH stronger benign representation than tokens do. With only
# a floor of 10, nearly every non-anglophone ccTLD qualified -- .jp at
# 94.9%, .pl at 93.4%, .pt at 88.5%. Japan and Poland are not phishing
# havens; the benign crawl is a US/English directory, so foreign ccTLDs
# have thin benign coverage and plenty of recent phishing. A model trained
# on that learns "non-English country -> phishing", which is a geography
# detector and a fairness problem, not a phishing detector.
MIN_TLD_BENIGN_DOMAINS = 100
MIN_TLD_BENIGN_FRACTION = 0.20

Z = 1.96                 # 95% confidence


def wilson_lower(k, n, z=Z):
    """Lower bound of the Wilson score interval for a proportion.

    Penalises small samples: 3/3 gives a much lower bound than 3000/3000.
    """
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / d


def section(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def parse_hosts(hosts):
    """Pre-split every hostname once: (subdomain, sld, tld, regdomain)."""
    out = []
    for h in hosts:
        e = _EXTRACT(h)
        sld, tld = e.domain or "", e.suffix or ""
        reg = f"{sld}.{tld}" if sld and tld else h
        out.append((e.subdomain or "", sld, tld, reg))
    return out


# --------------------------------------------------------------------------

def fit_risky_tlds(parsed, labels, base_rate):
    """TLDs confidently phishing-skewed, counted over distinct domains.

    Returns (qualified, unverifiable). 'unverifiable' holds TLDs with
    enough data but too few benign domains to evaluate -- almost always
    TLDs that postdate the benign crawl.
    """
    dom_label = {}
    for (_, _, tld, reg), y in zip(parsed, labels):
        if tld:
            dom_label[(tld, reg)] = y

    tot, phish = Counter(), Counter()
    for (tld, _), y in dom_label.items():
        tot[tld] += 1
        phish[tld] += y

    good, no_benign, thin_benign = [], [], []
    for tld, n in tot.items():
        if n < MIN_TLD_DOMAINS:
            continue
        benign = n - phish[tld]
        rate = phish[tld] / n

        # postdates the benign crawl: cannot be evaluated at all
        if benign < MIN_BENIGN_DOMAINS:
            no_benign.append((tld, n, rate, benign))
            continue

        # benign coverage too thin to distinguish "risky TLD" from
        # "our benign crawl did not cover this country"
        if (benign < MIN_TLD_BENIGN_DOMAINS or
                benign / n < MIN_TLD_BENIGN_FRACTION):
            thin_benign.append((tld, n, rate, benign))
            continue

        lb = wilson_lower(phish[tld], n)
        if lb > base_rate:
            good.append((tld, n, rate, lb))

    good.sort(key=lambda r: -r[3])
    no_benign.sort(key=lambda r: -r[1])
    thin_benign.sort(key=lambda r: -r[1])
    return good, no_benign, thin_benign


def fit_multitenant(parsed, labels, base_rate):
    """Registered domains hosting many distinct subdomains.

    Two lists, so you can ablate them separately:
      high_fanout   -- label-free: purely structural, any domain with many
                       distinct hostnames (includes wikipedia.org)
      abused_host   -- fanout AND confidently phishing-skewed: the free
                       hosting providers actually being used for phishing
    """
    hosts_per = defaultdict(set)
    phish_per, tot_per = Counter(), Counter()
    for (sub, sld, tld, reg), y in zip(parsed, labels):
        key = reg
        hosts_per[key].add(sub)
        tot_per[key] += 1
        phish_per[key] += y

    high_fanout, abused = [], []
    for reg, subs in hosts_per.items():
        fan = len(subs)
        if fan < MIN_FANOUT:
            continue
        n = tot_per[reg]
        rate = phish_per[reg] / n
        lb = wilson_lower(phish_per[reg], n)
        high_fanout.append((reg, fan, rate, lb))
        if lb > base_rate:
            abused.append((reg, fan, rate, lb))

    high_fanout.sort(key=lambda r: -r[1])
    abused.sort(key=lambda r: -r[1])
    return high_fanout, abused


# EXTERNAL prior -- not learned from this dataset.
#
# Learning brands from the benign corpus does not work here: that corpus is
# a ~2010 web directory, so the learned list came out as universities and
# defunct sites (berkeley, posterous, wetpaint) while the brands actually
# impersonated -- paypal, apple, microsoft -- were absent entirely. 'paypal'
# appeared 2,906 times at 100% phishing and was then filtered out for having
# no benign examples, killing the strongest signal in the data.
#
# This list is public knowledge about which brands attackers impersonate
# (the kind of thing published in APWG reports), not anything mined from
# your files. Using outside knowledge is not leakage; mining your own test
# set is. Provenance is recorded in priors.json so the writeup can say so.
EXTERNAL_BRANDS = [
    "paypal", "apple", "icloud", "appleid", "amazon", "microsoft", "office",
    "office365", "outlook", "hotmail", "onedrive", "sharepoint", "netflix",
    "facebook", "instagram", "whatsapp", "google", "gmail", "yahoo",
    "linkedin", "dropbox", "adobe", "docusign", "steam", "spotify", "ebay",
    "alibaba", "aliexpress", "wechat", "telegram", "twitter", "tiktok",
    "chase", "wellsfargo", "bankofamerica", "citibank", "capitalone",
    "hsbc", "barclays", "lloyds", "natwest", "santander", "halifax",
    "rakuten", "smbc", "mufg", "aeon", "jcb", "auone", "docomo", "softbank",
    "dhl", "fedex", "usps", "royalmail", "correos", "poste",
    "coinbase", "binance", "metamask", "blockchain", "kraken", "ledger",
    "netflix", "disney", "roblox", "runescape", "battlenet", "epicgames",
    "irs", "hmrc", "gov", "nhs", "medicare", "socialsecurity",
    "americanexpress", "mastercard", "visa", "discover", "westernunion",
    "zelle", "venmo", "cashapp", "revolut", "monzo", "wise",
]


def fit_brands(parsed, labels):
    """Return the external brand list, plus observed counts for reporting.

    Nothing is fitted here -- the list is fixed. The counts are computed on
    train purely so the printout shows which brands actually occur in your
    data and how skewed they are.
    """
    seen = Counter()
    for (sub, sld, _, _), y in zip(parsed, labels):
        blob = (sub + sld).lower()
        for b in EXTERNAL_BRANDS:
            if b in blob:
                seen[b] += 1
    return [(b, seen.get(b, 0)) for b in EXTERNAL_BRANDS]


def fit_suspicious_tokens(parsed, labels, base_rate, brand_set):
    """Tokens that are confidently over-represented in phishing hostnames.

    Chunks come from splitting the subdomain and SLD on separators. This
    misses concatenated tokens ('servicehelpdesk'), which is fine -- those
    are what the character n-grams in the model are for.
    """
    # token -> set of distinct registered domains containing it
    tok_domains = defaultdict(set)
    dom_label = {}
    for (sub, sld, _, reg), y in zip(parsed, labels):
        dom_label[reg] = y
        for part in (sub, sld):
            for c in re.split(r"[^a-z]+", part.lower()):
                if len(c) >= MIN_TOKEN_LEN:
                    tok_domains[c].add(reg)

    rows = []
    for tok, domains in tok_domains.items():
        n = len(domains)
        if n < MIN_TOKEN_DOMAINS or tok in brand_set:
            continue
        k = sum(dom_label[d] for d in domains)
        benign = n - k
        if benign < MIN_BENIGN_DOMAINS:
            continue
        lb = wilson_lower(k, n)
        if lb > base_rate:
            rows.append((tok, n, k / n, lb))

    rows.sort(key=lambda r: -r[3])
    return rows


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/hosts/train.csv")
    ap.add_argument("--out", default="priors.json")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    hosts = df["host"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    base_rate = sum(labels) / len(labels)

    print(f"fitting on {len(df):,} training hostnames")
    print(f"base phishing rate: {base_rate:.1%}")
    print("(a list only qualifies if its Wilson lower bound beats this)")

    parsed = parse_hosts(hosts)

    # ---- TLDs ----------------------------------------------------------
    section("RISKY TLDs")
    tlds, no_benign, thin_benign = fit_risky_tlds(parsed, labels, base_rate)
    print(f"{len(tlds)} TLDs qualified "
          f"(min {MIN_TLD_DOMAINS} domains, {MIN_BENIGN_DOMAINS} benign)")
    print(f"\n{'tld':<14}{'domains':>9}{'phish%':>9}{'lower':>9}")
    for tld, n, rate, lb in tlds[:args.show]:
        print(f"{tld:<14}{n:>9,}{rate:>8.1%}{lb:>9.1%}")

    if no_benign:
        print(f"\n  EXCLUDED -- {len(no_benign)} TLDs with almost no benign "
              f"examples (<{MIN_BENIGN_DOMAINS} domains).")
        print("  These postdate the benign crawl (.icu 2018, .app 2018,")
        print("  .dev 2019, .cfd, .cyou). They are 100% phishing by")
        print("  construction; keeping them teaches the collection date.")
        print(f"\n  {'tld':<12}{'domains':>9}{'phish%':>9}{'benign':>8}")
        for tld, n, rate, benign in no_benign[:10]:
            print(f"  {tld:<12}{n:>9,}{rate:>8.1%}{benign:>8}")

    if thin_benign:
        print(f"\n  EXCLUDED -- {len(thin_benign)} TLDs with benign coverage "
              f"too thin to judge")
        print(f"  (need >={MIN_TLD_BENIGN_DOMAINS} benign domains AND "
              f">={MIN_TLD_BENIGN_FRACTION:.0%} benign).")
        print("  Mostly non-anglophone ccTLDs. The benign crawl is a")
        print("  US/English directory, so these look malicious only because")
        print("  legitimate sites from those countries were never collected.")
        print("  Keeping them yields a geography detector, not a phishing one.")
        print(f"\n  {'tld':<12}{'domains':>9}{'phish%':>9}{'benign':>8}")
        for tld, n, rate, benign in thin_benign[:12]:
            print(f"  {tld:<12}{n:>9,}{rate:>8.1%}{benign:>8}")

    # ---- multi-tenant hosts --------------------------------------------
    section("MULTI-TENANT / ABUSED HOSTING")
    fanout, abused = fit_multitenant(parsed, labels, base_rate)
    print(f"{len(fanout)} high-fanout domains (>= {MIN_FANOUT} subdomains)")
    print(f"{len(abused)} of those are phishing-skewed")
    print(f"\n{'domain':<34}{'subs':>7}{'phish%':>9}{'lower':>9}")
    for reg, fan, rate, lb in abused[:args.show]:
        print(f"{reg:<34}{fan:>7,}{rate:>8.1%}{lb:>9.1%}")

    print(f"\nhigh-fanout but NOT phishing-skewed (sanity check):")
    abused_set = {r[0] for r in abused}
    clean = [r for r in fanout if r[0] not in abused_set][:10]
    for reg, fan, rate, lb in clean:
        print(f"  {reg:<32}{fan:>7,}{rate:>8.1%}")

    # ---- brands --------------------------------------------------------
    section("BRANDS (EXTERNAL list -- not learned from this dataset)")
    brands = fit_brands(parsed, labels)
    brand_set = {b for b, _ in brands}
    present = [(b, c) for b, c in brands if c > 0]
    print(f"{len(brands)} brands in the external list, "
          f"{len(present)} appear in train")
    print("\n  most frequent in your training hostnames:")
    for b, c in sorted(present, key=lambda r: -r[1])[:20]:
        print(f"    {b:<20}{c:>8,}")
    absent = [b for b, c in brands if c == 0]
    if absent:
        print(f"\n  not present in train ({len(absent)}): "
              + ", ".join(absent[:12]))

    # ---- suspicious tokens ---------------------------------------------
    section("SUSPICIOUS TOKENS")
    toks = fit_suspicious_tokens(parsed, labels, base_rate, brand_set)
    print(f"{len(toks)} tokens qualified "
          f"(min {MIN_TOKEN_DOMAINS} distinct domains, "
          f"{MIN_BENIGN_DOMAINS} benign)")
    print(f"\n{'token':<20}{'domains':>9}{'phish%':>9}{'lower':>9}")
    for tok, n, rate, lb in toks[:args.show]:
        print(f"{tok:<20}{n:>9,}{rate:>8.1%}{lb:>9.1%}")

    # ---- write ---------------------------------------------------------
    priors = {
        "_meta": {
            "fitted_on": args.train,
            "n_train_hosts": len(df),
            "base_phishing_rate": round(base_rate, 4),
            "min_tld_domains": MIN_TLD_DOMAINS,
            "min_token_domains": MIN_TOKEN_DOMAINS,
            "min_benign_domains": MIN_BENIGN_DOMAINS,
            "min_fanout": MIN_FANOUT,
            "min_tld_benign_domains": MIN_TLD_BENIGN_DOMAINS,
            "min_tld_benign_fraction": MIN_TLD_BENIGN_FRACTION,
            "selection": "Wilson 95% lower bound > base rate",
        },
        "_provenance": {
            "risky_tlds": "LEARNED from train split only",
            "high_fanout": "LEARNED from train split only",
            "abused_hosting": "LEARNED from train split only",
            "suspicious_tokens": "LEARNED from train split only",
            "brands": ("EXTERNAL -- public knowledge of commonly-phished "
                       "brands. Not derived from this dataset. The benign "
                       "corpus predates most of these brands' prominence, "
                       "so learning them from it was not possible."),
            "excluded_no_benign_tlds": ("TLDs postdating the benign crawl; "
                                        "excluded to avoid learning "
                                        "collection date"),
            "excluded_thin_benign_tlds": ("TLDs with insufficient benign "
                                          "coverage, mostly non-anglophone "
                                          "ccTLDs; excluded to avoid "
                                          "geographic bias"),
        },
        "risky_tlds": [t for t, *_ in tlds],
        "high_fanout": [d for d, *_ in fanout],
        "abused_hosting": [d for d, *_ in abused],
        "brands": [b for b, _ in brands],
        "suspicious_tokens": [t for t, *_ in toks],
        "excluded_no_benign_tlds": [t for t, *_ in no_benign],
        "excluded_thin_benign_tlds": [t for t, *_ in thin_benign],
    }

    with open(args.out, "w") as fh:
        json.dump(priors, fh, indent=2)

    section("WRITTEN")
    print(f"  {args.out}")
    for k, v in priors.items():
        if not k.startswith("_"):
            tag = "external" if k == "brands" else "learned"
            if k.startswith("excluded"):
                tag = "excluded"
            print(f"    {k:<30} {len(v):>6,}  [{tag}]")
    print("\n  Refit this whenever the training split changes.")
    print("  Never refit on val or test.")


if __name__ == "__main__":
    main()