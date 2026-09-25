import re, unicodedata
from anyascii import anyascii

def to_ascii(s):
    """Convert any script (Hindi, Kannada, French accents) to plain lowercase English letters."""
    return anyascii(unicodedata.normalize("NFKC", str(s))).lower()

# Short forms in business names -> one standard form
NAME_MAP = {
    "pvt": "private", "pvtltd": "private limited", "ltd": "limited", "ltda": "limited",
    "corp": "corporation", "inc": "incorporated", "co": "company", "cos": "company",
    "intl": "international", "mfg": "manufacturing", "svc": "services", "svcs": "services",
    "bros": "brothers", "assoc": "associates", "mgmt": "management", "tech": "technologies",
    "dept": "department", "ctr": "center", "centre": "center", "natl": "national",
}
# Legal / generic words removed for the core name
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "llc", "llp", "pllc",
    "plc", "lp", "the", "and", "of", "opc", "sarl", "sas", "sa", "eurl", "sci", "gmbh",
}

def clean_name(s):
    """Full cleaned business name."""
    s = to_ascii(s)
    s = re.sub(r"\[.*?\]\(.*?\)", " ", s)                      # markdown links
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)                # urls
    s = s.split("|")[0]                                         # text after | is junk
    s = re.sub(r"\.(com|in|net|org|co|fr|us|biz)\b", " ", s)    # 'abc.com' -> 'abc'
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)                           # remove punctuation
    words = [NAME_MAP.get(w, w) for w in s.split()]
    out = []
    for w in words:                                             # drop repeated words
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

def core_name(clean):
    """Cleaned name without legal/generic words."""
    return " ".join(w for w in clean.split() if w not in LEGAL)

# Address short forms (US, India, France)
ADDR_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "ln": "lane", "ct": "court", "blvd": "boulevard", "bd": "boulevard",
    "hwy": "highway", "pkwy": "parkway", "pl": "place", "cir": "circle", "trl": "trail",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building",
    "n": "north", "s": "south", "e": "east", "w": "west", "hno": "house", "h": "house",
    "opp": "opposite", "nr": "near", "mg": "mahatma gandhi", "chem": "chemin",
}

US_STATES = {
    "alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co",
    "connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga","hawaii":"hi","idaho":"id",
    "illinois":"il","indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la",
    "maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi","minnesota":"mn",
    "mississippi":"ms","missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv",
    "new hampshire":"nh","new jersey":"nj","new mexico":"nm","new york":"ny",
    "north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok","oregon":"or",
    "pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd",
    "tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa",
    "west virginia":"wv","wisconsin":"wi","wyoming":"wy","district of columbia":"dc",
}
STATE_ABBR = set(US_STATES.values())
STATE_RE = re.compile(r"\b(" + "|".join(sorted(US_STATES, key=len, reverse=True)) + r")\b")

def clean_addr(s, country=""):
    """Cleaned address."""
    s = to_ascii(s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    if country == "US":
        s = STATE_RE.sub(lambda m: US_STATES[m.group(1)], s)
        s = re.sub(r"\bcity of\b|\btown of\b", " ", s)
    words = [w if (country == "US" and w in STATE_ABBR) else ADDR_MAP.get(w, w) for w in s.split()]
    return " ".join(words)

def addr_numbers(clean):
    """All numbers in the address (house no, PIN, unit) as a sorted string."""
    return " ".join(sorted(set(re.findall(r"\d+", clean))))