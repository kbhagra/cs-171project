"""
URL normalization for the phishing detector.

Pure function, no state, no fitting -- so it behaves identically on train,
val and test and cannot leak information between splits.

The flags exist so you can ablate each normalization choice later: train
once with a flag on, once with it off, and compare val PR-AUC. That turns
each judgement call into a measured decision.
"""

import re
from urllib.parse import urlsplit

# Module level: compiled once, not on each of 722k calls.
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _manual_split(body):
    """Fallback parser for URLs urlsplit refuses to handle.

    urlsplit raises ValueError on things like an unmatched '[' in the host
    (it reads that as IPv6 bracket notation). Real scraped data contains
    plenty of malformed URLs, and one bad row must not kill a 700k-row run.
    """
    rest = _SCHEME.sub("", body)

    frag = rest.find("#")
    if frag != -1:
        rest = rest[:frag]

    q = rest.find("?")
    query = rest[q + 1:] if q != -1 else ""
    if q != -1:
        rest = rest[:q]

    slash = rest.find("/")
    if slash != -1:
        netloc, path = rest[:slash], rest[slash:]
    else:
        netloc, path = rest, ""

    return netloc, path, query


def normalize(url, *, strip_scheme=True, lower_path=True,
              strip_www=True, strip_trailing_slash=True):
    """Normalize a URL and return its parts plus metadata flags.

    Never raises: malformed input falls back to a manual split and is
    reported via the 'malformed' flag.

    Returns a dict with:
        normalized    reassembled URL string (feed this to your n-grams)
        host          hostname, always lowercased (DNS is case-insensitive)
        path          path component, '' if absent
        query         query string without the '?'
        port          port as a string, '' if absent
        had_scheme    was a scheme present in the input?
        has_userinfo  was there a 'user@' before the host?
        is_ip         is the host a raw IPv4 address?
        is_punycode   does the host contain an 'xn--' label?
        malformed     did urlsplit reject this and force the fallback?
    """
    # --- 1. clean the raw string first; everything below assumes this ran
    raw = str(url).strip().strip("\"'")
    raw = raw.replace("\\", "/")

    # --- 2. record scheme presence BEFORE stripping it
    had_scheme = bool(_SCHEME.match(raw))

    # --- 3. strip the scheme, then re-add a placeholder so urlsplit can
    # find the host. Without this, a bare 'google.com/search' parses with
    # an empty netloc and the whole string lands in .path -- silently.
    body = _SCHEME.sub("", raw) if (strip_scheme and had_scheme) else raw
    if not _SCHEME.match(body):
        body = "http://" + body

    malformed = False
    try:
        parts = urlsplit(body)
        netloc, rawpath, query = parts.netloc, parts.path, parts.query
    except ValueError:
        malformed = True
        netloc, rawpath, query = _manual_split(body)

    # --- 4. netloc surgery must happen BEFORE deriving host from it

    # userinfo: split on the LAST '@' so 'user:pa@ss@host' works. This
    # only looks at netloc, so an '@' in the path is not a false positive.
    has_userinfo = "@" in netloc
    if has_userinfo:
        netloc = netloc.rsplit("@", 1)[1]

    # port: rpartition from the right, after userinfo is gone. Skip when the
    # host is bracketed IPv6 ('[::1]') -- those colons are part of the address.
    port = ""
    if netloc.endswith("]"):
        pass
    elif ":" in netloc:
        netloc, _, port = netloc.rpartition(":")

    # --- 5. now build host
    host = netloc.lower()

    # anchored + includes the dot, so 'wwwtest.com' is left alone
    if strip_www and host.startswith("www."):
        host = host[4:]

    # --- 6. path: case-sensitive by spec, so lowering is optional
    path = rawpath.lower() if lower_path else rawpath

    # strip ONLY a bare trailing slash; '/includes/aol/' is a real path
    if strip_trailing_slash and path == "/":
        path = ""

    # --- 7. flags worth keeping as features
    is_ip = bool(_IPV4.match(host))
    is_punycode = "xn--" in host   # homograph signal: preserved, never stripped

    # --- 8. reassemble
    normalized = host
    if port:
        normalized += ":" + port
    normalized += path
    if query:
        normalized += "?" + query

    return {
        "normalized": normalized,
        "host": host,
        "path": path,
        "query": query,
        "port": port,
        "had_scheme": had_scheme,
        "has_userinfo": has_userinfo,
        "is_ip": is_ip,
        "is_punycode": is_punycode,
        "malformed": malformed,
    }


if __name__ == "__main__":
    # quick smoke test: python url_normalizer.py
    for u in [
        "0000111servicehelpdesk.godaddysites.com",
        "www.cs.arizona.edu/patterns/weaving/books/tws_mach_2.pdf",
        "http://user@evil.com:8080/Login/",
        "nlcblog.org/",
    ]:
        print(f"{u[:55]:<57} -> {normalize(u)['normalized']}")