"""
Hostname feature extraction for the phishing detector.

Every feature here is computed from the hostname string alone -- nothing
from the path, nothing from the crawl, nothing fitted on the dataset. That
means the same function works unchanged on a URL typed into your demo app.

Deliberately NOT included:
  had_scheme  - merge artifact (8.6% benign vs 42.5% phishing in this file,
                which reflects how sources were stitched together, not
                phishing behaviour)
  n_urls      - collection metadata; unavailable at inference time
  path/query  - the main model is hostname-only by design

Word lists and TLD lists are LEARNED, not hardcoded. Run fit_priors.py to
produce priors.json from the training split, then call load_priors() before
extracting. The small hardcoded lists below are a fallback only, used when
no priors.json is available -- for example in the demo app, where there is
no training split to fit on.

Why this matters: hardcoded lists chosen after looking at whole-dataset
statistics leak test information into training through the analyst, even
though no code touched the test set.
"""

import json
import math
import re
from collections import Counter

import pandas as pd
import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

# --------------------------------------------------------------------------
# FALLBACK priors -- used only when no priors.json has been loaded.
# These are deliberately small and generic. The real lists come from
# fit_priors.py, fitted on train only.
# --------------------------------------------------------------------------

BRANDS = {
    "paypal", "apple", "icloud", "amazon", "microsoft", "office365", "outlook",
    "netflix", "facebook", "instagram", "whatsapp", "google", "gmail", "yahoo",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "barclays",
    "santander", "dhl", "fedex", "ups", "usps", "linkedin", "dropbox",
    "adobe", "steam", "coinbase", "binance", "metamask", "blockchain",
    "netflix", "spotify", "ebay", "alibaba", "wechat", "docusign", "irs",
    "hmrc", "amex", "visa", "mastercard", "orange", "telstra",
}

ACTION_WORDS = {
    "login", "signin", "sign-in", "logon", "verify", "verification", "secure",
    "security", "account", "accounts", "update", "confirm", "confirmation",
    "password", "passwd", "credential", "auth", "authenticate", "recovery",
    "recover", "unlock", "suspend", "suspended", "alert", "warning", "billing",
    "invoice", "payment", "pay", "refund", "webform", "helpdesk", "support",
    "service", "customer", "client", "portal", "access", "owa", "webmail",
    "mail", "wallet", "validate", "activity", "limited", "restricted",
}

# TLDs repeatedly flagged in abuse reporting. A prior, not a measurement.
RISKY_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "icu", "online", "site",
    "club", "work", "link", "click", "loan", "download", "stream", "bid",
    "win", "review", "country", "kim", "party", "science", "date", "faith",
    "racing", "cricket", "accountant", "men", "trade", "webcam", "buzz",
    "rest", "cyou", "quest", "sbs", "cc", "pw", "su", "info", "biz",
}

# Providers offering free subdomains -- heavily abused for phishing.
FREE_HOSTING = {
    "000webhostapp.com", "godaddysites.com", "duckdns.org", "workers.dev",
    "blogspot.com", "wordpress.com", "weebly.com", "wixsite.com", "web.app",
    "firebaseapp.com", "netlify.app", "vercel.app", "github.io", "gitlab.io",
    "glitch.me", "repl.co", "herokuapp.com", "pages.dev", "r2.dev",
    "azurewebsites.net", "sharepoint.com", "myshopify.com", "square.site",
    "usite.pro", "wcomhost.com", "sites.google.com", "form.jotform.com",
    "typeform.com", "surveyheart.com", "ngrok.io", "serveo.net",
}

# is_shortener was removed as a feature: at hostname level a shortener
# collapses to a single row ('bit.ly'), so the flag carries almost no
# information in this representation. It belongs to the full-URL variant.

HIGH_FANOUT = set()
ABUSED_HOSTING = set()
_PRIORS_LOADED = False


def load_priors(path="priors.json"):
    """Load learned priors, replacing the fallback lists.

    Call this once before extracting features. Raises if the file is
    missing rather than silently falling back, because silently using
    generic lists would make your results hard to interpret.
    """
    global BRANDS, ACTION_WORDS, RISKY_TLDS, FREE_HOSTING
    global HIGH_FANOUT, ABUSED_HOSTING, _PRIORS_LOADED

    with open(path) as fh:
        pri = json.load(fh)

    BRANDS = set(pri["brands"])
    ACTION_WORDS = set(pri["suspicious_tokens"])
    RISKY_TLDS = set(pri["risky_tlds"])
    HIGH_FANOUT = set(pri["high_fanout"])
    ABUSED_HOSTING = set(pri["abused_hosting"])
    FREE_HOSTING = ABUSED_HOSTING
    _PRIORS_LOADED = True

    meta = pri.get("_meta", {})
    print(f"loaded priors from {path} "
          f"(fitted on {meta.get('n_train_hosts', '?')} train hosts): "
          f"{len(BRANDS)} brands, {len(ACTION_WORDS)} tokens, "
          f"{len(RISKY_TLDS)} risky TLDs, {len(ABUSED_HOSTING)} abused hosts")
    return pri


def priors_loaded():
    return _PRIORS_LOADED


VOWELS = set("aeiou")
_HEXISH = re.compile(r"[0-9a-f]{8,}")
_DIGIT_RUN = re.compile(r"\d+")
_CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxyz]+")
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _entropy(s):
    """Shannon entropy in bits per character. Random-looking hosts score high."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _longest_run(pattern, s):
    runs = pattern.findall(s)
    return max((len(r) for r in runs), default=0)


def _split_words(s):
    """Split a host label into alpha chunks, for token matching."""
    return [w for w in re.split(r"[^a-z]+", s.lower()) if w]


# Attackers concatenate without separators ('0000111servicehelpdesk'), so
# exact token matching misses almost everything. Substring matching fixes
# that, but short keywords then produce nonsense hits -- 'ups' inside
# 'backups', 'bt' inside almost anything. So: substring-match keywords of
# 5+ characters, exact-token-match the short ones.
_MIN_SUBSTR = 6


def _keyword_hits(host_alpha, word_set, token_set):
    """Count keyword matches, substring for long words, tokens for short."""
    hits = {w for w in word_set if len(w) >= _MIN_SUBSTR and w in host_alpha}
    hits |= {w for w in word_set if len(w) < _MIN_SUBSTR and w in token_set}
    return hits


# --------------------------------------------------------------------------
# the extractor
# --------------------------------------------------------------------------

def extract(host):
    """Return a flat dict of features for one hostname."""
    host = (host or "").strip().lower()
    f = {}

    ext = _EXTRACT(host)
    subdomain = ext.subdomain or ""
    sld = ext.domain or ""
    tld = ext.suffix or ""
    reg_domain = f"{sld}.{tld}" if sld and tld else host

    labels = [l for l in host.split(".") if l]
    alpha = re.sub(r"[^a-z]", "", host)
    digits = re.sub(r"\D", "", host)

    # ---- family 1: shape ------------------------------------------------
    f["host_len"] = len(host)
    f["n_labels"] = len(labels)
    f["n_dots"] = host.count(".")
    f["n_hyphens"] = host.count("-")
    f["n_digits"] = len(digits)
    f["longest_label_len"] = max((len(l) for l in labels), default=0)
    f["leftmost_label_len"] = len(labels[0]) if labels else 0
    f["sld_len"] = len(sld)
    f["subdomain_len"] = len(subdomain)
    f["subdomain_depth"] = len([l for l in subdomain.split(".") if l])
    f["tld_len"] = len(tld)
    f["has_subdomain"] = int(bool(subdomain))

    # ---- family 2: character composition --------------------------------
    f["digit_ratio"] = len(digits) / len(host) if host else 0.0
    f["longest_digit_run"] = _longest_run(_DIGIT_RUN, host)
    f["entropy"] = _entropy(host)
    f["sld_entropy"] = _entropy(sld)
    f["vowel_ratio"] = (sum(1 for c in alpha if c in VOWELS) / len(alpha)
                        if alpha else 0.0)
    f["longest_consonant_run"] = _longest_run(_CONSONANT_RUN, alpha)
    f["n_nonalnum"] = sum(1 for c in host if not c.isalnum() and c != ".")
    f["has_nonascii"] = int(any(ord(c) > 127 for c in host))
    f["has_hexish"] = int(bool(_HEXISH.search(host)))
    f["starts_with_digit"] = int(bool(host) and host[0].isdigit())

    # ---- family 3: token signals ----------------------------------------
    words = set(_split_words(host))
    sub_words = set(_split_words(subdomain))
    sld_words = set(_split_words(sld))

    host_alpha = "".join(_split_words(host))
    sub_alpha = "".join(_split_words(subdomain))
    sld_alpha = "".join(_split_words(sld))

    brand_hits = _keyword_hits(host_alpha, BRANDS, words)
    f["n_brand_hits"] = len(brand_hits)
    f["has_brand"] = int(bool(brand_hits))

    # The strongest single hand-crafted feature: a brand name appears
    # somewhere in the host, but is NOT the registered domain.
    # 'paypal.com' -> 0.  'paypal.secure-verify.tk' -> 1.
    sld_brands = _keyword_hits(sld_alpha, BRANDS, sld_words)
    f["brand_not_in_sld"] = int(bool(brand_hits) and not sld_brands)

    action_hits = _keyword_hits(host_alpha, ACTION_WORDS, words)
    f["n_action_words"] = len(action_hits)
    f["has_action_word"] = int(bool(action_hits))
    f["action_in_subdomain"] = int(
        bool(_keyword_hits(sub_alpha, ACTION_WORDS, sub_words)))
    f["action_in_sld"] = int(
        bool(_keyword_hits(sld_alpha, ACTION_WORDS, sld_words)))

    # crude "is this pronounceable / word-like" proxy
    f["n_word_chunks"] = len(_split_words(host))
    f["mean_chunk_len"] = (sum(len(w) for w in _split_words(host)) /
                           max(len(_split_words(host)), 1))

    # ---- family 4: TLD and hosting --------------------------------------
    f["is_risky_tld"] = int(tld.split(".")[-1] in RISKY_TLDS if tld else 0)
    f["is_cctld"] = int(len(tld.split(".")[-1]) == 2 if tld else 0)
    # WARNING: these two are domain-membership lookups. Because the splits
    # are grouped by registered domain, no val/test domain appears in train,
    # so both are ALWAYS 0 during evaluation. Drop them from the training
    # matrix (see DOMAIN_LOOKUP_FEATURES) -- they are kept here because they
    # ARE useful in the demo app, where a real URL may well sit on a known
    # abused host.
    f["is_abused_hosting"] = int(reg_domain in FREE_HOSTING)
    f["is_high_fanout"] = int(reg_domain in HIGH_FANOUT)
    f["is_ip"] = int(bool(_IPV4.match(host)))
    f["is_punycode"] = int("xn--" in host)
    f["no_tld"] = int(not tld)

    return f


# Features that are lookups against domains seen in training. Under a
# domain-grouped split these are structurally zero on val/test, so they
# must be excluded from model training or they will silently do nothing.
DOMAIN_LOOKUP_FEATURES = ["is_abused_hosting", "is_high_fanout"]

FEATURE_NAMES = sorted(extract("example.com").keys())
TRAIN_FEATURE_NAMES = [f for f in FEATURE_NAMES
                       if f not in DOMAIN_LOOKUP_FEATURES]


def extract_frame(hosts):
    """Vectorized-ish extraction over an iterable of hostnames -> DataFrame."""
    return pd.DataFrame([extract(h) for h in hosts],
                        columns=FEATURE_NAMES).astype("float32")


def tld_of(host):
    """Raw TLD string, for one-hot encoding fitted on train only."""
    return _EXTRACT((host or "").lower()).suffix or ""


# --------------------------------------------------------------------------

if __name__ == "__main__":
    demo = [
        "0000111servicehelpdesk.godaddysites.com",
        "srnbc-card.com.52kfu.top",
        "paypal.com",
        "paypal.secure-verify.tk",
        "en.wikipedia.org",
        "6pygzc.fartit.com",
        "192.0.2.5",
        "xn--pypal-4ve.com",
    ]
    df = extract_frame(demo)
    df.insert(0, "host", demo)
    cols = ["host", "host_len", "entropy", "digit_ratio", "longest_digit_run",
            "vowel_ratio", "brand_not_in_sld", "n_action_words",
            "is_risky_tld", "is_abused_hosting", "is_ip", "is_punycode"]
    pd.set_option("display.width", 200)
    print(df[cols].to_string(index=False))
    print(f"\n{len(FEATURE_NAMES)} features: {FEATURE_NAMES}")