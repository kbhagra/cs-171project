"""
Tests for normalize().

Write normalize() in url_normalizer.py until every one of these passes.
Run with:  python test_normalizer.py

Each test names the trap it's guarding against. If one fails, read the
failure message before changing code -- it usually tells you which
ordering assumption you broke.
"""

from url_normalizer import normalize


CASES = []


def case(name, url, expect, **kwargs):
    CASES.append((name, url, expect, kwargs))


# -- the big one: no scheme on 74% of your rows ---------------------------
case("bare hostname parses as host, not path",
     "google.com/search",
     {"host": "google.com", "path": "/search"})

case("bare hostname, no path",
     "bmspakistan.com",
     {"host": "bmspakistan.com", "path": ""})

case("scheme present still parses",
     "https://google.com/search",
     {"host": "google.com", "path": "/search"})

case("had_scheme flag records the truth (for your ablation)",
     "http://example.com", {"had_scheme": True})

case("had_scheme false when absent",
     "example.com", {"had_scheme": False})


# -- host vs path case asymmetry -----------------------------------------
case("host lowercased unconditionally (DNS is case-insensitive)",
     "GOOGLE.COM/Path", {"host": "google.com"}, lower_path=False)

case("path case preserved when lower_path=False",
     "google.com/Login/Verify", {"path": "/Login/Verify"}, lower_path=False)

case("path lowercased when lower_path=True",
     "google.com/Login/Verify", {"path": "/login/verify"}, lower_path=True)


# -- trailing slash: only the bare one ------------------------------------
case("bare trailing slash stripped",
     "nlcblog.org/", {"path": "", "normalized": "nlcblog.org"})

case("meaningful trailing slash NOT stripped",
     "notransportes.com.br/includes/aol/",
     {"path": "/includes/aol/"})


# -- www ------------------------------------------------------------------
case("www stripped when flagged",
     "www.syedgakbar.com/products/", {"host": "syedgakbar.com"},
     strip_www=True)

case("www kept when not flagged",
     "www.syedgakbar.com/products/", {"host": "www.syedgakbar.com"},
     strip_www=False)

case("www NOT stripped mid-hostname",
     "wwwtest.com/", {"host": "wwwtest.com"}, strip_www=True)


# -- phishing-relevant structure ------------------------------------------
case("userinfo @ split off, host is what follows",
     "http://user@evil.com/login",
     {"host": "evil.com", "has_userinfo": True})

case("no false positive on @ in path",
     "example.com/contact@us", {"has_userinfo": False})

case("raw IP host detected",
     "http://192.168.1.1/login", {"host": "192.168.1.1", "is_ip": True})

case("normal host is not an IP",
     "example.com", {"is_ip": False})

case("port split off host",
     "example.com:8080/admin", {"host": "example.com", "port": "8080"})

case("punycode PRESERVED -- homograph signal, do not strip",
     "xn--pypal-4ve.com", {"host": "xn--pypal-4ve.com", "is_punycode": True})

case("non-ascii host preserved",
     "paypaI.com", {"host": "paypai.com"})


# -- messy real-world input -----------------------------------------------
case("whitespace stripped",
     "  example.com/path  ", {"host": "example.com"})

case("backslashes converted",
     "example.com\\admin\\login", {"path": "/admin/login"})

case("query separated from path",
     "youtube.com/watch?v=6KKxCoGLRQs",
     {"path": "/watch", "query": "v=6KKxCoGLRQs"})

case("fragment separated",
     "example.com/page#section", {"path": "/page"})

case("very long url survives",
     "example.com/" + "a" * 3900, {"host": "example.com"})

case("real sample from your phishing class",
     "0000111servicehelpdesk.godaddysites.com",
     {"host": "0000111servicehelpdesk.godaddysites.com", "path": ""})

case("real sample from your benign class",
     "www.cs.arizona.edu/patterns/weaving/books/tws_mach_2.pdf",
     {"host": "cs.arizona.edu", "path": "/patterns/weaving/books/tws_mach_2.pdf"})


# -- purity: no hidden state ----------------------------------------------
def test_deterministic():
    u = "www.Example.com/Path/"
    a = normalize(u)
    b = normalize(u)
    assert a == b, "normalize() is not deterministic -- do you have state?"


def run():
    passed = failed = 0
    for name, url, expect, kwargs in CASES:
        try:
            got = normalize(url, **kwargs)
        except Exception as e:
            print(f"  ERROR  {name}\n         {type(e).__name__}: {e}")
            failed += 1
            continue

        bad = {k: (v, got.get(k, "<missing>"))
               for k, v in expect.items() if got.get(k, "<missing>") != v}
        if bad:
            print(f"  FAIL   {name}")
            print(f"         input: {url[:70]!r}")
            for k, (want, actual) in bad.items():
                print(f"         {k}: expected {want!r}, got {actual!r}")
            failed += 1
        else:
            passed += 1

    try:
        test_deterministic()
        passed += 1
    except AssertionError as e:
        print(f"  FAIL   determinism: {e}")
        failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)