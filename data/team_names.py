"""
Team name normalization across sources.
"""
from __future__ import annotations

_CANONICAL = {
    "Arizona Diamondbacks":  ("arizona diamondbacks", "arizona", "diamondbacks", "d-backs", "az"),
    "Atlanta Braves":        ("atlanta braves", "atlanta", "braves", "atl"),
    "Baltimore Orioles":     ("baltimore orioles", "baltimore", "orioles", "bal"),
    "Boston Red Sox":        ("boston red sox", "boston", "red sox", "bos"),
    "Chicago Cubs":          ("chicago cubs", "cubs", "chc"),
    "Chicago White Sox":     ("chicago white sox", "white sox", "cws", "chw"),
    "Cincinnati Reds":       ("cincinnati reds", "cincinnati", "reds", "cin"),
    "Cleveland Guardians":   ("cleveland guardians", "cleveland", "guardians", "cle", "cleveland indians", "indians"),
    "Colorado Rockies":      ("colorado rockies", "colorado", "rockies", "col"),
    "Detroit Tigers":        ("detroit tigers", "detroit", "tigers", "det"),
    "Houston Astros":        ("houston astros", "houston", "astros", "hou"),
    "Kansas City Royals":    ("kansas city royals", "kansas city", "royals", "kc", "kcr"),
    "Los Angeles Angels":    ("los angeles angels", "la angels", "angels", "laa", "anaheim angels"),
    "Los Angeles Dodgers":   ("los angeles dodgers", "la dodgers", "dodgers", "lad"),
    "Miami Marlins":         ("miami marlins", "miami", "marlins", "mia"),
    "Milwaukee Brewers":     ("milwaukee brewers", "milwaukee", "brewers", "mil"),
    "Minnesota Twins":       ("minnesota twins", "minnesota", "twins", "min"),
    "New York Mets":         ("new york mets", "ny mets", "mets", "nym"),
    "New York Yankees":      ("new york yankees", "ny yankees", "yankees", "nyy"),
    "Oakland Athletics":     ("oakland athletics", "oakland", "athletics", "as", "oak",
                              "sacramento athletics"),
    "Philadelphia Phillies": ("philadelphia phillies", "philadelphia", "phillies", "phi"),
    "Pittsburgh Pirates":    ("pittsburgh pirates", "pittsburgh", "pirates", "pit"),
    "San Diego Padres":      ("san diego padres", "san diego", "padres", "sd", "sdp"),
    "San Francisco Giants":  ("san francisco giants", "san francisco", "giants", "sf", "sfg"),
    "Seattle Mariners":      ("seattle mariners", "seattle", "mariners", "sea"),
    "St. Louis Cardinals":   ("st. louis cardinals", "st louis cardinals", "st. louis",
                              "st louis", "cardinals", "stl"),
    "Tampa Bay Rays":        ("tampa bay rays", "tampa bay", "rays", "tb", "tbr"),
    "Texas Rangers":         ("texas rangers", "texas", "rangers", "tex"),
    "Toronto Blue Jays":     ("toronto blue jays", "toronto", "blue jays", "tor"),
    "Washington Nationals":  ("washington nationals", "washington", "nationals", "wsh", "was"),
}

_ALIAS_TO_CANONICAL = {}
for canonical, aliases in _CANONICAL.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for a in aliases:
        _ALIAS_TO_CANONICAL[a.lower()] = canonical


def normalize_team(raw_name):
    if not raw_name:
        raise KeyError("empty team name")
    key = raw_name.strip().lower()
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    key = key.replace("*", "").strip()
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    # Handle DraftKings' "<ABBR> <Nickname>" format like "DET Tigers",
    # "BOS Red Sox", "TB Rays". Strip the 2-3 letter prefix if it's a
    # known abbreviation and retry with the tail.
    parts = key.split()
    if len(parts) >= 2 and 2 <= len(parts[0]) <= 3 and parts[0].isalpha():
        # Prefer matching by the abbreviation itself first — it's more
        # reliable when two teams share a nickname (e.g. White Sox / Red Sox
        # both end with "sox" but DK writes "CWS White Sox" vs "BOS Red Sox").
        abbr_upper = parts[0].upper()
        if abbr_upper in FG_ABBR_TO_CANONICAL:
            return FG_ABBR_TO_CANONICAL[abbr_upper]
        tail = " ".join(parts[1:])
        if tail in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[tail]
    raise KeyError(f"unknown team name: {raw_name!r}")


def try_normalize_team(raw_name):
    try:
        return normalize_team(raw_name)
    except KeyError:
        return None


FG_ABBR_TO_CANONICAL = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CHW": "Chicago White Sox", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KCR": "Kansas City Royals", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees",
    "OAK": "Oakland Athletics", "ATH": "Oakland Athletics", "SAC": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres", "SD": "San Diego Padres",
    "SFG": "San Francisco Giants", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals", "WAS": "Washington Nationals", "WSH": "Washington Nationals",
}


def normalize_fg_abbr(abbr):
    if not abbr:
        return None
    return FG_ABBR_TO_CANONICAL.get(abbr.strip().upper())
