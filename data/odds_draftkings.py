"""
DraftKings odds scraper.

DraftKings exposes a public "eventgroup" JSON used by their own web
sportsbook. MLB is eventGroupId 84240. The response groups offers by
offerCategories -> offerSubcategories; we need the "Game Lines"
category which contains three subcategories: Moneyline, Run Line, Total.

Endpoint:
    GET /sites/US-SB/api/v5/eventgroups/84240?format=json

This has been stable for years but structure may drift. All fetch
failures return []; the client falls back to Pinnacle (or vice versa).
"""
from __future__ import annotations

import http.cookiejar as _cookiejar
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import error, request

from .odds_models import OddsBook, OddsSnapshot, make_event_id, pick_main_run_line
from .team_names import try_normalize_team

log = logging.getLogger(__name__)

_MLB_EVENT_GROUP_ID = 84240
_URL = (
    "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/"
    f"{_MLB_EVENT_GROUP_ID}?format=json"
)
_LANDING_URL = "https://sportsbook.draftkings.com/leagues/baseball/mlb"

# Chrome 122 on Windows — matches the client hints a real browser sends,
# which is what DK's anti-bot layer (Datadome) checks against. Bare urllib
# requests with just a Mozilla UA and Referer get 403'd even from clean
# residential IPs because no sec-ch-ua / sec-fetch-* headers = not a browser.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_SEC_CH_UA = '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"'

_LANDING_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": _UA,
    "sec-ch-ua": _SEC_CH_UA,
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_API_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": _UA,
    "Referer": _LANDING_URL,
    "Origin": "https://sportsbook.draftkings.com",
    "sec-ch-ua": _SEC_CH_UA,
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

# Module-level cookie jar + opener so cookies set by the warmup GET are
# attached to the subsequent API GET. Reused across polls.
_COOKIE_JAR: Optional[_cookiejar.CookieJar] = None
_OPENER = None

# Monotonic time after which we should re-run the warmup. Datadome tokens
# typically live ~1 hour; we refresh at 45 min to be safe.
_COOKIES_FRESH_UNTIL: Optional[float] = None
_COOKIE_TTL_SEC = 45 * 60

# Stealth init script — injected into every page in the browser context
# before any site JS runs. Hides the most common "this is an automated
# browser" signals that Datadome checks (webdriver flag, missing plugins,
# missing chrome runtime, suspicious permissions behaviour).
_STEALTH_JS = r"""
(() => {
  // Hide webdriver flag
  Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

  // Fake a plausible plugins array (real Chrome reports >0 plugins)
  Object.defineProperty(navigator, 'plugins', {
    get: () => [
      { name: 'Chrome PDF Plugin' },
      { name: 'Chrome PDF Viewer' },
      { name: 'Native Client' },
    ],
  });

  // Fake language list
  Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en'],
  });

  // Ensure window.chrome exists (headless Chromium sometimes lacks it)
  if (!window.chrome) {
    window.chrome = { runtime: {} };
  } else if (!window.chrome.runtime) {
    window.chrome.runtime = {};
  }

  // Normalise permissions API — headless returns 'denied' for notifications
  // but real Chrome returns 'default' until user decides.
  try {
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
      parameters && parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery.call(window.navigator.permissions, parameters)
    );
  } catch (_e) { /* ignore */ }

  // WebGL vendor/renderer spoof — headless reports 'Google SwiftShader'
  try {
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
      if (p === 37445) return 'Intel Inc.';           // UNMASKED_VENDOR_WEBGL
      if (p === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
      return getParameter.call(this, p);
    };
  } catch (_e) { /* ignore */ }
})();
"""


def _get_opener():
    global _COOKIE_JAR, _OPENER
    if _OPENER is None:
        _COOKIE_JAR = _cookiejar.CookieJar()
        _OPENER = request.build_opener(request.HTTPCookieProcessor(_COOKIE_JAR))
    return _OPENER


_SCHEMA_SAMPLE_WRITTEN = False


def _dump_schema_sample(body: str, url: str) -> None:
    """One-shot per process: save a DK response to disk and log its
    top-level shape. Used to learn DK's current JSON schema so we can
    write a parser. Writes to <project_root>/dk_sample.json.
    """
    global _SCHEMA_SAMPLE_WRITTEN
    if _SCHEMA_SAMPLE_WRITTEN:
        return
    try:
        import pathlib
        # data/ -> project root
        root = pathlib.Path(__file__).resolve().parent.parent
        sample_path = root / "dk_sample.json"
        sample_path.write_text(body[:400_000], encoding="utf-8")
        log.warning("DK sample saved to %s (%d bytes from %s)",
                    sample_path, len(body), url[:120])
    except Exception as e:  # noqa: BLE001
        log.debug("DK sample save failed: %s", e)
    try:
        top = json.loads(body)
    except Exception as e:  # noqa: BLE001
        log.warning("DK sample JSON parse failed: %s", e)
        _SCHEMA_SAMPLE_WRITTEN = True
        return
    if isinstance(top, dict):
        keys = list(top.keys())
        log.warning("DK response top-level keys: %s", keys[:30])
        for k in keys[:12]:
            v = top[k]
            if isinstance(v, list):
                log.warning("  .%s: list[%d]", k, len(v))
                if v and isinstance(v[0], dict):
                    log.warning("    [0] keys: %s", list(v[0].keys())[:20])
            elif isinstance(v, dict):
                log.warning("  .%s: dict keys=%s", k, list(v.keys())[:20])
    elif isinstance(top, list):
        log.warning("DK response is a list[%d]", len(top))
        if top and isinstance(top[0], dict):
            log.warning("  [0] keys: %s", list(top[0].keys())[:20])
    _SCHEMA_SAMPLE_WRITTEN = True


def _urllib_warmup(timeout: float = 10.0) -> None:
    """Fallback warmup: plain urllib GET to the landing page.

    Not sufficient to defeat Datadome's JS challenge on its own — DK will
    still return 403 to the subsequent API call. Kept so the module
    degrades gracefully if Playwright isn't installed.
    """
    opener = _get_opener()
    req = request.Request(_LANDING_URL, headers=_LANDING_HEADERS)
    try:
        with opener.open(req, timeout=timeout) as resp:
            resp.read(4096)
    except (error.URLError, error.HTTPError, TimeoutError) as e:
        log.debug("DK urllib warmup failed: %s", e)


def _fetch_via_playwright(url: str = _URL, timeout_sec: float = 30.0) -> Any:
    """Fetch the DK API through a warmed Chromium context.

    DK is fronted by Akamai Bot Manager, which inspects both cookies AND
    TLS fingerprint (JA3/JA4). Even with correct cookies, urllib's TLS
    fingerprint gives it away and Akamai returns 403. The reliable fix is
    to make the actual API request through Chromium's own network stack
    (context.request.get) so cookies, TLS signature, client hints, and
    headers are all the same as when the page was loaded.

    Returns the parsed JSON. Raises RuntimeError (wrapped with status) on
    failure so the outer fetch_draftkings_snapshots can log and return [].
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "playwright not installed — run `pip install playwright` "
            "and `python -m playwright install chromium`"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            context = browser.new_context(
                user_agent=_UA,
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            context.add_init_script(_STEALTH_JS)
            page = context.new_page()

            # Passive capture strategy: DK's own frontend JS fetches the
            # odds data when it renders the MLB landing page. Akamai
            # won't 403 DK's own calls. We record every JSON API response
            # during the load and pick the biggest one whose body looks
            # like an eventgroup/event list.
            api_candidates: list = []

            def _on_response(resp):
                try:
                    if resp.request.method != "GET":
                        return
                    u = resp.url
                    # DK-hosted JSON APIs only
                    if "draftkings.com" not in u:
                        return
                    if "/api/" not in u and "/sites/" not in u:
                        return
                    ct = (resp.headers or {}).get("content-type", "").lower()
                    if "json" not in ct:
                        return
                    api_candidates.append(resp)
                except Exception:  # noqa: BLE001
                    pass

            page.on("response", _on_response)

            status = 0
            body = ""
            try:
                page.goto(
                    _LANDING_URL,
                    wait_until="domcontentloaded",
                    timeout=int(timeout_sec * 1000),
                )
            except Exception as goto_err:  # noqa: BLE001
                log.debug("DK goto raised (continuing): %s", goto_err)
            # Human-ish activity so Akamai's sensor doesn't flag us, plus
            # give DK's frontend time to fetch its odds data.
            try:
                page.mouse.move(200, 200)
                page.wait_for_timeout(300)
                page.mouse.move(650, 420, steps=12)
                page.mouse.wheel(0, 500)
                page.wait_for_timeout(300)
                page.mouse.wheel(0, -150)
            except Exception as mv_err:  # noqa: BLE001
                log.debug("DK mouse sim raised (non-fatal): %s", mv_err)
            # Wait for frontend XHRs to finish.
            page.wait_for_timeout(7000)

            # Diagnostic: log what DK called (first pass — can remove later).
            log.warning(
                "DK captured %d JSON API responses during load",
                len(api_candidates),
            )
            for r in api_candidates[:25]:
                try:
                    log.warning("  DK api: %d  %s", r.status, r.url[:180])
                except Exception:  # noqa: BLE001
                    pass

            # Pick the response most likely to be the eventgroup payload:
            # prefer anything containing "eventgroup" in URL, otherwise
            # fall back to the biggest JSON body that looks like it has
            # event/offer data.
            def _score(r) -> int:
                u = r.url.lower()
                s = 0
                if "eventgroup" in u:
                    s += 1000
                if str(_MLB_EVENT_GROUP_ID) in u:
                    s += 500
                if "mlb" in u or "baseball" in u:
                    s += 100
                if "categor" in u:
                    s += 50
                if r.status == 200:
                    s += 10
                return s

            scored = sorted(
                (r for r in api_candidates if r.status == 200),
                key=_score,
                reverse=True,
            )
            for r in scored:
                try:
                    b = r.text()
                except Exception:  # noqa: BLE001
                    continue
                if not b:
                    continue
                # Does it look like an odds-data payload? Accept either
                # the old eventGroup shape or the new sportscontent shape
                # (markets / selections / offers / events arrays).
                head = b[:6000].lower()
                looks_like = (
                    '"eventgroup"' in head
                    or '"events"' in head
                    or '"offercategories"' in head
                    or '"offersubcategorydescriptors"' in head
                    or '"markets"' in head
                    or '"selections"' in head
                    or '"offers"' in head
                )
                if looks_like:
                    status = r.status
                    body = b
                    log.warning(
                        "DK passive capture OK — %s (%d bytes)",
                        r.url[:180],
                        len(b),
                    )
                    _dump_schema_sample(b, r.url)
                    break
            if status != 200:
                log.warning(
                    "DK passive capture: no candidate matched eventgroup "
                    "shape; sampled %d responses",
                    len(api_candidates),
                )
        finally:
            browser.close()

    if status != 200:
        # Truncate body so the log line stays readable.
        snippet = (body or "")[:160].replace("\n", " ")
        raise RuntimeError(f"DK API HTTP {status}: {snippet}")
    return json.loads(body)


def _playwright_warmup(timeout_sec: float = 30.0) -> bool:
    """Legacy warmup that only populated the urllib cookie jar.

    Kept for diagnostic purposes but no longer used by the hot path —
    _fetch_via_playwright does the whole thing end-to-end.

    Returns True on success, False if Playwright is unavailable or the
    browser session fails.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        log.warning(
            "DK warmup: playwright not installed. "
            "Run `pip install playwright` and `playwright install chromium`."
        )
        return False

    try:
        with sync_playwright() as p:
            # --disable-blink-features=AutomationControlled removes the
            # Chrome/Chromium banner that says the browser is being
            # controlled by automated test software and also strips the
            # easiest-to-detect webdriver signal at the CDP layer.
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                context = browser.new_context(
                    user_agent=_UA,
                    viewport={"width": 1280, "height": 900},
                    locale="en-US",
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                # Run stealth overrides before any page JS executes.
                context.add_init_script(_STEALTH_JS)
                page = context.new_page()
                # Use domcontentloaded rather than networkidle — DK has
                # long-poll streams / live price feeds that keep the network
                # perpetually busy, so networkidle never fires. DOM-ready is
                # enough to kick off Datadome's challenge script.
                try:
                    page.goto(
                        _LANDING_URL,
                        wait_until="domcontentloaded",
                        timeout=int(timeout_sec * 1000),
                    )
                except Exception as goto_err:  # noqa: BLE001
                    # Page may have partially loaded and set cookies already.
                    # Keep going — we'll check cookies below.
                    log.debug("DK playwright goto raised (continuing): %s", goto_err)
                # Poll for the Datadome cookie (named 'datadome') — it appears
                # only after the challenge JS computes and posts its answer.
                # Give it up to 12 seconds.
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    if any(c.get("name") == "datadome" for c in context.cookies()):
                        break
                    page.wait_for_timeout(500)
                cookies = context.cookies()
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001 — playwright raises many types
        log.warning("DK playwright warmup failed: %s", e)
        return False

    if not cookies:
        log.warning("DK playwright warmup: browser returned no cookies")
        return False

    has_datadome = any(c.get("name") == "datadome" for c in cookies)
    if not has_datadome:
        log.warning(
            "DK playwright warmup: no 'datadome' cookie in %d captured "
            "cookies — Datadome challenge did not complete. Cookie names: %s",
            len(cookies),
            [c.get("name") for c in cookies][:20],
        )

    # Inject the browser's cookies into our urllib cookie jar.
    _get_opener()  # ensure _COOKIE_JAR is initialised
    assert _COOKIE_JAR is not None
    # Clear stale cookies from any prior warmup.
    _COOKIE_JAR.clear()
    for c in cookies:
        domain = c.get("domain", "") or ""
        expires = c.get("expires")
        # Playwright uses -1 for session cookies; cookiejar wants None.
        expires_val = int(expires) if isinstance(expires, (int, float)) and expires > 0 else None
        ck = _cookiejar.Cookie(
            version=0,
            name=c.get("name", ""),
            value=c.get("value", ""),
            port=None,
            port_specified=False,
            domain=domain,
            domain_specified=bool(domain),
            domain_initial_dot=domain.startswith("."),
            path=c.get("path", "/") or "/",
            path_specified=True,
            secure=bool(c.get("secure", False)),
            expires=expires_val,
            discard=expires_val is None,
            comment=None,
            comment_url=None,
            rest={},
            rfc2109=False,
        )
        _COOKIE_JAR.set_cookie(ck)

    global _COOKIES_FRESH_UNTIL
    _COOKIES_FRESH_UNTIL = time.monotonic() + _COOKIE_TTL_SEC
    log.warning(
        "DK playwright warmup OK — captured %d cookies (datadome=%s), "
        "valid until +%dm",
        len(cookies),
        "yes" if has_datadome else "NO",
        _COOKIE_TTL_SEC // 60,
    )
    return True


def _ensure_warm(force: bool = False, timeout_sec: float = 30.0) -> None:
    """Warm the cookie jar if empty, stale, or force=True."""
    _get_opener()
    need = (
        force
        or (_COOKIE_JAR is not None and len(_COOKIE_JAR) == 0)
        or (_COOKIES_FRESH_UNTIL is not None and time.monotonic() > _COOKIES_FRESH_UNTIL)
    )
    if not need:
        return
    if not _playwright_warmup(timeout_sec=timeout_sec):
        # Fall back so existing behaviour is preserved on envs without
        # Playwright — DK will still 403 but we won't crash.
        _urllib_warmup()


def _http_get_json(url: str = _URL, timeout: float = 10.0) -> Any:
    """Fetch the DK eventgroup JSON.

    Delegates to _fetch_via_playwright so cookies + TLS fingerprint all
    match a real Chrome session. The `timeout` param is respected (passed
    through as the Playwright navigation/request timeout); urllib is no
    longer in the picture for DK because Akamai's TLS check rejects it.
    """
    return _fetch_via_playwright(url=url, timeout_sec=max(timeout, 30.0))


def _parse_utc(s: str) -> datetime:
    # DraftKings returns "2026-04-19T23:05:00.0000000Z" or similar
    s = s.rstrip("Z")
    # Trim fractional-second precision to 6 digits for fromisoformat
    if "." in s:
        head, frac = s.split(".")
        frac = frac[:6]
        s = f"{head}.{frac}"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _american_from_display(do: Optional[dict]) -> Optional[int]:
    """Parse DK's displayOdds dict from the new sportscontent schema.

    DK encodes the American odds string with Unicode characters — minus is
    U+2212 (−, not ASCII ``-``) and plus is U+002B (``+`` but delivered
    escaped as ``\\u002b``). ASCII int() chokes on the unicode minus, so we
    substitute before parsing. Falls back to decimal conversion.
    """
    if not do:
        return None
    am = do.get("american")
    if am:
        s = str(am).replace("\u2212", "-").replace("\u002b", "+").replace("+", "")
        try:
            return int(s)
        except ValueError:
            pass
    dec = do.get("decimal")
    if dec is not None:
        try:
            d = float(dec)
            if d <= 1.0:
                return 0
            if d >= 2.0:
                return int(round((d - 1.0) * 100))
            return int(round(-100.0 / (d - 1.0)))
        except (TypeError, ValueError):
            pass
    return None


def _parse_sportscontent_snapshots(payload: dict) -> list[OddsSnapshot]:
    """Parse DK's new ``sportscontent`` relational schema.

    Shape (fields we use):
      events:     [{id, name, startEventDate,
                    participants:[{name, venueRole: "Home"|"Away"}]}]
      markets:    [{id, eventId, marketType:{name}}]       # Moneyline, Run Line, Total
      selections: [{id, marketId, label, displayOdds:{american,decimal},
                    points?, outcomeType: "Home"|"Away"|"Over"|"Under"}]

    Joins markets→selections via marketId and events→markets via eventId.
    Returns one OddsSnapshot per event that we could fully identify.
    """
    events = payload.get("events") or []
    markets = payload.get("markets") or []
    selections = payload.get("selections") or []
    if not events:
        return []

    # Index markets by eventId -> {market_name: market}
    markets_by_event: dict[str, dict[str, dict]] = {}
    for m in markets:
        ev_id = str(m.get("eventId") or "")
        if not ev_id:
            continue
        mtype_name = ((m.get("marketType") or {}).get("name")
                      or m.get("name") or "").strip()
        if not mtype_name:
            continue
        markets_by_event.setdefault(ev_id, {})[mtype_name] = m

    # Index selections by marketId
    sels_by_market: dict[str, list[dict]] = {}
    for s in selections:
        mid = str(s.get("marketId") or "")
        if not mid:
            continue
        sels_by_market.setdefault(mid, []).append(s)

    polled_at = datetime.now(timezone.utc)
    out: list[OddsSnapshot] = []

    for ev in events:
        ev_id = str(ev.get("id") or "")
        if not ev_id:
            continue

        # Participants carry venueRole = "Home"/"Away".
        home_raw = away_raw = ""
        for pt in ev.get("participants") or []:
            role = (pt.get("venueRole") or "").strip().lower()
            name = pt.get("name") or ""
            if role == "home":
                home_raw = name
            elif role == "away":
                away_raw = name
        # Fallback: split event name "AWAY @ HOME".
        if not home_raw or not away_raw:
            ev_name = ev.get("name") or ""
            if " @ " in ev_name:
                a, h = ev_name.split(" @ ", 1)
                away_raw = away_raw or a
                home_raw = home_raw or h

        home = try_normalize_team(home_raw)
        away = try_normalize_team(away_raw)
        if not home or not away:
            log.debug("DK: unknown teams %r / %r (event=%r)",
                      away_raw, home_raw, ev.get("name"))
            continue

        try:
            start_utc = _parse_utc(ev.get("startEventDate") or "")
        except Exception:  # noqa: BLE001
            start_utc = datetime.now(timezone.utc)

        event_id = make_event_id(start_utc, away, home)
        ev_markets = markets_by_event.get(ev_id, {})

        # --- Moneyline ---
        home_ml = away_ml = None
        ml_market = ev_markets.get("Moneyline")
        if ml_market:
            mid = str(ml_market.get("id") or "")
            for sel in sels_by_market.get(mid, []):
                side = (sel.get("outcomeType") or "").strip().lower()
                am = _american_from_display(sel.get("displayOdds"))
                if am is None:
                    continue
                if side == "home":
                    home_ml = am
                elif side == "away":
                    away_ml = am

        # --- Run Line (standard ±1.5) ---
        # DK's "Run Line" market can contain BOTH the main line (fav -1.5 /
        # dog +1.5) AND the reverse (fav +1.5 / dog -1.5) as separate
        # selections. Previously we overwrote home_rl_line with whichever
        # home-side selection came last, which flipped the sign on games
        # where DK listed the reverse pair after the main. Collect every
        # ±1.5 candidate per side, then pick the main via ML direction.
        home_rl_cands: list[tuple[float, int]] = []
        away_rl_cands: list[tuple[float, int]] = []
        rl_market = ev_markets.get("Run Line")
        if rl_market:
            mid = str(rl_market.get("id") or "")
            for sel in sels_by_market.get(mid, []):
                side = (sel.get("outcomeType") or "").strip().lower()
                try:
                    pts_f = float(sel.get("points")) if sel.get("points") is not None else None
                except (TypeError, ValueError):
                    pts_f = None
                if pts_f is None or abs(abs(pts_f) - 1.5) > 0.01:
                    continue
                am = _american_from_display(sel.get("displayOdds"))
                if am is None:
                    continue
                if side == "home":
                    home_rl_cands.append((pts_f, am))
                elif side == "away":
                    away_rl_cands.append((pts_f, am))
        home_rl_line, home_rl_odds, away_rl_odds = pick_main_run_line(
            home_rl_cands, away_rl_cands, home_ml, away_ml,
        )

        # --- Total ---
        total_line = over_odds = under_odds = None
        tot_market = ev_markets.get("Total")
        if tot_market:
            mid = str(tot_market.get("id") or "")
            by_line: dict[float, dict[str, int]] = {}
            for sel in sels_by_market.get(mid, []):
                side = (sel.get("outcomeType") or "").strip().lower()
                if side not in ("over", "under"):
                    continue
                try:
                    pts_f = float(sel.get("points")) if sel.get("points") is not None else None
                except (TypeError, ValueError):
                    pts_f = None
                if pts_f is None:
                    continue
                am = _american_from_display(sel.get("displayOdds"))
                if am is None:
                    continue
                by_line.setdefault(pts_f, {})[side] = am
            # Pick the line whose juice is closest to -110/-110 — that's the
            # "main" number DK is posting.
            best = None
            best_score = 1e9
            for ln, sides in by_line.items():
                if "over" not in sides or "under" not in sides:
                    continue
                score = abs(sides["over"] + 110) + abs(sides["under"] + 110)
                if score < best_score:
                    best_score = score
                    best = ln
            if best is not None:
                total_line = best
                over_odds = by_line[best]["over"]
                under_odds = by_line[best]["under"]

        out.append(OddsSnapshot(
            book=OddsBook.DRAFTKINGS,
            event_id=event_id,
            home_team=home,
            away_team=away,
            game_time_utc=start_utc,
            home_ml=home_ml,
            away_ml=away_ml,
            home_rl_line=home_rl_line,
            home_rl_odds=home_rl_odds,
            away_rl_odds=away_rl_odds,
            total_line=total_line,
            over_odds=over_odds,
            under_odds=under_odds,
            polled_at_utc=polled_at,
            native_event_id=ev_id,
        ))

    return out


def fetch_draftkings_snapshots() -> list[OddsSnapshot]:
    """Fetch today's MLB slate from DraftKings and normalize."""
    try:
        payload = _http_get_json()
    except error.HTTPError as e:
        log.warning("DraftKings HTTP %s: %s", e.code, e.reason)
        return []
    except (error.URLError, TimeoutError, json.JSONDecodeError) as e:
        log.warning("DraftKings fetch failed: %s", e)
        return []
    except RuntimeError as e:
        # Raised by _fetch_via_playwright on non-200 or missing Playwright.
        log.warning("DraftKings fetch failed: %s", e)
        return []

    if not isinstance(payload, dict):
        log.warning("DraftKings: unexpected payload type %s",
                    type(payload).__name__)
        return []

    # New relational sportscontent schema (events + markets + selections).
    if "selections" in payload and "markets" in payload and "events" in payload:
        snaps = _parse_sportscontent_snapshots(payload)
        log.warning("DraftKings parsed %d event snapshots (sportscontent)",
                    len(snaps))
        return snaps

    log.warning("DraftKings: unknown payload shape (top keys=%s)",
                list(payload.keys())[:10])
    return []
