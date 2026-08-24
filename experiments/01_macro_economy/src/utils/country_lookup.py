"""Country name to ISO3 mapping and prompt entity extractor."""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 2-letter codes that are also common English words / prepositions.
# These must NOT participate in regex \b word-boundary matching because
# they cause false positives (e.g. "qatar" ending with "ar" → ARG).
# They are ONLY matched when the user types them as a standalone token
# (handled specially in extract_entities_from_prompt).
# ---------------------------------------------------------------------------
_AMBIGUOUS_SHORT_CODES: dict[str, str] = {
    "in": "IND", "it": "ITA", "no": "NOR", "us": "USA", "be": "BEL",
    "me": "MNE", "to": "TON", "my": "MYS", "do": "DOM", "am": "ARM",
    "an": "AND",  # Andorra
    "ar": "ARG",
    "at": "AUT",
    "de": "DEU",
    "id": "IDN",
}

# Comprehensive mapping of country names, 3-letter codes, demonyms,
# and historical aliases to ISO3.
# NOTE: Ambiguous 2-letter codes listed in _AMBIGUOUS_SHORT_CODES are
# intentionally excluded here to prevent false substring matching.
COUNTRY_MAP = {
    # ── A ──────────────────────────────────────────────────────────────
    "afghanistan": "AFG", "afganistan": "AFG", "afg": "AFG", "afghan": "AFG", "afgan": "AFG",
    "albania": "ALB", "alb": "ALB", "albanian": "ALB",
    "algeria": "DZA", "dza": "DZA", "algerian": "DZA",
    "andorra": "AND", "and": "AND",
    "angola": "AGO", "ago": "AGO", "angolan": "AGO",
    "antigua and barbuda": "ATG", "atg": "ATG",
    "argentina": "ARG", "arg": "ARG", "argentinian": "ARG", "argentine": "ARG",
    "armenia": "ARM", "arm": "ARM", "armenian": "ARM",
    "australia": "AUS", "aus": "AUS", "oz": "AUS", "au": "AUS", "australian": "AUS",
    "austria": "AUT", "aut": "AUT", "austrian": "AUT",
    "azerbaijan": "AZE", "aze": "AZE", "azerbaijani": "AZE",

    # ── B ──────────────────────────────────────────────────────────────
    "bahamas": "BHS", "bhs": "BHS", "bahamian": "BHS",
    "bahrain": "BHR", "bhr": "BHR", "bahraini": "BHR",
    "bangladesh": "BGD", "bgd": "BGD", "bd": "BGD", "bangladeshi": "BGD",
    "barbados": "BRB", "brb": "BRB", "barbadian": "BRB",
    "belarus": "BLR", "blr": "BLR", "belarusian": "BLR",
    "belgium": "BEL", "bel": "BEL", "belgian": "BEL",
    "belize": "BLZ", "blz": "BLZ",
    "benin": "BEN", "ben": "BEN", "beninese": "BEN",
    "bhutan": "BTN", "btn": "BTN", "bhutanese": "BTN",
    "bolivia": "BOL", "bol": "BOL", "bolivian": "BOL",
    "bosnia and herzegovina": "BIH", "bih": "BIH", "bosnia": "BIH", "bosnian": "BIH",
    "botswana": "BWA", "bwa": "BWA",
    "brazil": "BRA", "bra": "BRA", "br": "BRA", "brazilian": "BRA",
    "brunei": "BRN", "brn": "BRN",
    "bulgaria": "BGR", "bgr": "BGR", "bulgarian": "BGR",
    "burkina faso": "BFA", "bfa": "BFA", "burkinabe": "BFA",
    "burundi": "BDI", "bdi": "BDI",

    # ── C ──────────────────────────────────────────────────────────────
    "cabo verde": "CPV", "cape verde": "CPV", "cpv": "CPV",
    "cambodia": "KHM", "khm": "KHM", "cambodian": "KHM",
    "cameroon": "CMR", "cmr": "CMR", "cameroonian": "CMR",
    "canada": "CAN", "can": "CAN", "ca": "CAN", "canadian": "CAN",
    "central african republic": "CAF", "caf": "CAF",
    "chad": "TCD", "tcd": "TCD", "chadian": "TCD",
    "chile": "CHL", "chl": "CHL", "cl": "CHL", "chilean": "CHL",
    "china": "CHN", "chn": "CHN", "prc": "CHN", "cn": "CHN", "chinese": "CHN",
    "colombia": "COL", "col": "COL", "co": "COL", "colombian": "COL",
    "comoros": "COM", "com": "COM",
    "congo": "COG", "cog": "COG", "republic of the congo": "COG",
    "dr congo": "COD", "cod": "COD", "drc": "COD", "democratic republic of the congo": "COD",
    "costa rica": "CRI", "cri": "CRI", "costa rican": "CRI",
    "cote d'ivoire": "CIV", "ivory coast": "CIV", "civ": "CIV",
    "croatia": "HRV", "hrv": "HRV", "croatian": "HRV",
    "cuba": "CUB", "cub": "CUB", "cuban": "CUB",
    "cyprus": "CYP", "cyp": "CYP", "cypriot": "CYP",
    "czech republic": "CZE", "cze": "CZE", "czechia": "CZE", "cz": "CZE", "czech": "CZE",

    # ── D ──────────────────────────────────────────────────────────────
    "denmark": "DNK", "dnk": "DNK", "dk": "DNK", "danish": "DNK",
    "djibouti": "DJI", "dji": "DJI",
    "dominica": "DMA", "dma": "DMA",
    "dominican republic": "DOM", "dom": "DOM", "dominican": "DOM",

    # ── E ──────────────────────────────────────────────────────────────
    "ecuador": "ECU", "ecu": "ECU", "ecuadorian": "ECU",
    "egypt": "EGY", "egy": "EGY", "eg": "EGY", "egyptian": "EGY",
    "el salvador": "SLV", "slv": "SLV", "salvadoran": "SLV",
    "equatorial guinea": "GNQ", "gnq": "GNQ",
    "eritrea": "ERI", "eri": "ERI", "eritrean": "ERI",
    "estonia": "EST", "est": "EST", "estonian": "EST",
    "eswatini": "SWZ", "swaziland": "SWZ", "swz": "SWZ",
    "ethiopia": "ETH", "eth": "ETH", "ethiopian": "ETH",

    # ── F ──────────────────────────────────────────────────────────────
    "fiji": "FJI", "fji": "FJI", "fijian": "FJI",
    "finland": "FIN", "fin": "FIN", "fi": "FIN", "finnish": "FIN",
    "france": "FRA", "fra": "FRA", "fr": "FRA", "french": "FRA",

    # ── G ──────────────────────────────────────────────────────────────
    "gabon": "GAB", "gab": "GAB", "gabonese": "GAB",
    "gambia": "GMB", "gmb": "GMB", "gambian": "GMB",
    "georgia": "GEO", "geo": "GEO", "georgian": "GEO",
    "germany": "DEU", "deu": "DEU", "deutschland": "DEU", "german": "DEU",
    "ghana": "GHA", "gha": "GHA", "ghanaian": "GHA",
    "greece": "GRC", "grc": "GRC", "gr": "GRC", "greek": "GRC",
    "grenada": "GRD", "grd": "GRD",
    "guatemala": "GTM", "gtm": "GTM", "guatemalan": "GTM",
    "guinea": "GIN", "gin": "GIN", "guinean": "GIN",
    "guinea-bissau": "GNB", "gnb": "GNB",
    "guyana": "GUY", "guy": "GUY", "guyanese": "GUY",

    # ── H ──────────────────────────────────────────────────────────────
    "haiti": "HTI", "hti": "HTI", "haitian": "HTI",
    "honduras": "HND", "hnd": "HND", "honduran": "HND",
    "hungary": "HUN", "hun": "HUN", "hu": "HUN", "hungarian": "HUN",

    # ── I ──────────────────────────────────────────────────────────────
    "iceland": "ISL", "isl": "ISL", "icelandic": "ISL",
    "india": "IND", "ind": "IND", "indian": "IND",
    "indonesia": "IDN", "idn": "IDN", "indonesian": "IDN",
    "iran": "IRN", "irn": "IRN", "iranian": "IRN", "persia": "IRN",
    "iraq": "IRQ", "irq": "IRQ", "iraqi": "IRQ",
    "ireland": "IRL", "irl": "IRL", "ie": "IRL", "irish": "IRL",
    "israel": "ISR", "isr": "ISR", "il": "ISR", "israeli": "ISR",
    "italy": "ITA", "ita": "ITA", "italian": "ITA",

    # ── J ──────────────────────────────────────────────────────────────
    "jamaica": "JAM", "jam": "JAM", "jamaican": "JAM",
    "japan": "JPN", "jpn": "JPN", "jp": "JPN", "japanese": "JPN",
    "jordan": "JOR", "jor": "JOR", "jordanian": "JOR",

    # ── K ──────────────────────────────────────────────────────────────
    "kazakhstan": "KAZ", "kaz": "KAZ", "kazakh": "KAZ",
    "kenya": "KEN", "ken": "KEN", "kenyan": "KEN",
    "kiribati": "KIR", "kir": "KIR",
    "north korea": "PRK", "prk": "PRK", "dprk": "PRK",
    "south korea": "KOR", "korea": "KOR", "kor": "KOR", "kr": "KOR", "korean": "KOR",
    "kuwait": "KWT", "kwt": "KWT", "kuwaiti": "KWT",
    "kyrgyzstan": "KGZ", "kgz": "KGZ", "kyrgyz": "KGZ",

    # ── L ──────────────────────────────────────────────────────────────
    "laos": "LAO", "lao": "LAO", "lao pdr": "LAO",
    "latvia": "LVA", "lva": "LVA", "latvian": "LVA",
    "lebanon": "LBN", "lbn": "LBN", "lebanese": "LBN",
    "lesotho": "LSO", "lso": "LSO",
    "liberia": "LBR", "lbr": "LBR", "liberian": "LBR",
    "libya": "LBY", "lby": "LBY", "libyan": "LBY",
    "liechtenstein": "LIE", "lie": "LIE",
    "lithuania": "LTU", "ltu": "LTU", "lithuanian": "LTU",
    "luxembourg": "LUX", "lux": "LUX",

    # ── M ──────────────────────────────────────────────────────────────
    "madagascar": "MDG", "mdg": "MDG", "malagasy": "MDG",
    "malawi": "MWI", "mwi": "MWI", "malawian": "MWI",
    "malaysia": "MYS", "mys": "MYS", "malaysian": "MYS",
    "maldives": "MDV", "mdv": "MDV", "maldivian": "MDV",
    "mali": "MLI", "mli": "MLI", "malian": "MLI",
    "malta": "MLT", "mlt": "MLT", "maltese": "MLT",
    "marshall islands": "MHL", "mhl": "MHL",
    "mauritania": "MRT", "mrt": "MRT", "mauritanian": "MRT",
    "mauritius": "MUS", "mus": "MUS", "mauritian": "MUS",
    "mexico": "MEX", "mex": "MEX", "mx": "MEX", "mexican": "MEX",
    "micronesia": "FSM", "fsm": "FSM",
    "moldova": "MDA", "mda": "MDA", "moldovan": "MDA",
    "monaco": "MCO", "mco": "MCO",
    "mongolia": "MNG", "mng": "MNG", "mongolian": "MNG",
    "montenegro": "MNE", "mne": "MNE", "montenegrin": "MNE",
    "morocco": "MAR", "mar": "MAR", "moroccan": "MAR",
    "mozambique": "MOZ", "moz": "MOZ", "mozambican": "MOZ",
    "myanmar": "MMR", "mmr": "MMR", "burma": "MMR", "burmese": "MMR",

    # ── N ──────────────────────────────────────────────────────────────
    "namibia": "NAM", "nam": "NAM", "namibian": "NAM",
    "nauru": "NRU", "nru": "NRU",
    "nepal": "NPL", "npl": "NPL", "nepalese": "NPL", "nepali": "NPL",
    "netherlands": "NLD", "nld": "NLD", "holland": "NLD", "nl": "NLD", "dutch": "NLD",
    "new zealand": "NZL", "nzl": "NZL", "nz": "NZL", "kiwi": "NZL",
    "nicaragua": "NIC", "nic": "NIC", "nicaraguan": "NIC",
    "niger": "NER", "ner": "NER", "nigerien": "NER",
    "nigeria": "NGA", "nga": "NGA", "ng": "NGA", "nigerian": "NGA",
    "north macedonia": "MKD", "mkd": "MKD", "macedonia": "MKD", "macedonian": "MKD",
    "norway": "NOR", "nor": "NOR", "norwegian": "NOR",

    # ── O ──────────────────────────────────────────────────────────────
    "oman": "OMN", "omn": "OMN", "omani": "OMN",

    # ── P ──────────────────────────────────────────────────────────────
    "pakistan": "PAK", "pak": "PAK", "pk": "PAK", "pakistani": "PAK",
    "palau": "PLW", "plw": "PLW",
    "panama": "PAN", "pan": "PAN", "panamanian": "PAN",
    "papua new guinea": "PNG", "png": "PNG",
    "paraguay": "PRY", "pry": "PRY", "paraguayan": "PRY",
    "peru": "PER", "per": "PER", "pe": "PER", "peruvian": "PER",
    "philippines": "PHL", "phl": "PHL", "ph": "PHL", "filipino": "PHL", "philippine": "PHL",
    "poland": "POL", "pol": "POL", "pl": "POL", "polish": "POL",
    "portugal": "PRT", "prt": "PRT", "pt": "PRT", "portuguese": "PRT",

    # ── Q ──────────────────────────────────────────────────────────────
    "qatar": "QAT", "qat": "QAT", "qa": "QAT", "qatari": "QAT",

    # ── R ──────────────────────────────────────────────────────────────
    "romania": "ROU", "rou": "ROU", "romanian": "ROU",
    "russia": "RUS", "rus": "RUS", "russian federation": "RUS", "ru": "RUS", "russian": "RUS", "soviet": "RUS",
    "rwanda": "RWA", "rwa": "RWA", "rwandan": "RWA",

    # ── S ──────────────────────────────────────────────────────────────
    "saint kitts and nevis": "KNA", "kna": "KNA",
    "saint lucia": "LCA", "lca": "LCA",
    "saint vincent and the grenadines": "VCT", "vct": "VCT",
    "samoa": "WSM", "wsm": "WSM", "samoan": "WSM",
    "san marino": "SMR", "smr": "SMR",
    "sao tome and principe": "STP", "stp": "STP",
    "saudi arabia": "SAU", "sau": "SAU", "saudi": "SAU", "sa": "SAU",
    "senegal": "SEN", "sen": "SEN", "senegalese": "SEN",
    "serbia": "SRB", "srb": "SRB", "serbian": "SRB",
    "seychelles": "SYC", "syc": "SYC",
    "sierra leone": "SLE", "sle": "SLE",
    "singapore": "SGP", "sgp": "SGP", "sg": "SGP", "singaporean": "SGP",
    "slovakia": "SVK", "svk": "SVK", "slovak": "SVK",
    "slovenia": "SVN", "svn": "SVN", "slovenian": "SVN",
    "solomon islands": "SLB", "slb": "SLB",
    "somalia": "SOM", "som": "SOM", "somali": "SOM",
    "south africa": "ZAF", "zaf": "ZAF", "za": "ZAF", "south african": "ZAF",
    "south sudan": "SSD", "ssd": "SSD",
    "spain": "ESP", "esp": "ESP", "es": "ESP", "spanish": "ESP",
    "sri lanka": "LKA", "lka": "LKA", "sri lankan": "LKA", "ceylon": "LKA",
    "sudan": "SDN", "sdn": "SDN", "sudanese": "SDN",
    "suriname": "SUR", "sur": "SUR", "surinamese": "SUR",
    "sweden": "SWE", "swe": "SWE", "se": "SWE", "swedish": "SWE",
    "switzerland": "CHE", "che": "CHE", "ch": "CHE", "swiss": "CHE",
    "syria": "SYR", "syr": "SYR", "syrian": "SYR",

    # ── T ──────────────────────────────────────────────────────────────
    "taiwan": "TWN", "twn": "TWN", "taiwanese": "TWN",
    "tajikistan": "TJK", "tjk": "TJK", "tajik": "TJK",
    "tanzania": "TZA", "tza": "TZA", "tanzanian": "TZA",
    "thailand": "THA", "tha": "THA", "th": "THA", "thai": "THA",
    "timor-leste": "TLS", "east timor": "TLS", "tls": "TLS",
    "togo": "TGO", "tgo": "TGO", "togolese": "TGO",
    "tonga": "TON", "ton": "TON", "tongan": "TON",
    "trinidad and tobago": "TTO", "tto": "TTO", "trinidadian": "TTO",
    "tunisia": "TUN", "tun": "TUN", "tunisian": "TUN",
    "turkey": "TUR", "tur": "TUR", "turkiye": "TUR", "tr": "TUR", "turkish": "TUR",
    "turkmenistan": "TKM", "tkm": "TKM", "turkmen": "TKM",
    "tuvalu": "TUV", "tuv": "TUV",

    # ── U ──────────────────────────────────────────────────────────────
    "uganda": "UGA", "uga": "UGA", "ugandan": "UGA",
    "ukraine": "UKR", "ukr": "UKR", "ua": "UKR", "ukrainian": "UKR",
    "united arab emirates": "ARE", "uae": "ARE", "ae": "ARE", "emirati": "ARE",
    "united kingdom": "GBR", "uk": "GBR", "gbr": "GBR", "britain": "GBR",
    "great britain": "GBR", "england": "GBR", "english": "GBR",
    "scotland": "GBR", "scottish": "GBR", "wales": "GBR", "welsh": "GBR",
    "northern ireland": "GBR", "british": "GBR",
    "united states": "USA", "united states of america": "USA", "usa": "USA",
    "america": "USA", "american": "USA",
    "uruguay": "URY", "ury": "URY", "uruguayan": "URY",
    "uzbekistan": "UZB", "uzb": "UZB", "uzbek": "UZB",

    # ── V ──────────────────────────────────────────────────────────────
    "vanuatu": "VUT", "vut": "VUT",
    "vatican": "VAT", "vat": "VAT", "holy see": "VAT",
    "venezuela": "VEN", "ven": "VEN", "venezuelan": "VEN",
    "vietnam": "VNM", "vnm": "VNM", "vn": "VNM", "vietnamese": "VNM",

    # ── Y ──────────────────────────────────────────────────────────────
    "yemen": "YEM", "yem": "YEM", "yemeni": "YEM",

    # ── Z ──────────────────────────────────────────────────────────────
    "zambia": "ZMB", "zmb": "ZMB", "zambian": "ZMB",
    "zimbabwe": "ZWE", "zwe": "ZWE", "zimbabwean": "ZWE",
}


# Comprehensive list of macro/economic terms for domain intent checking
ECONOMIC_KEYWORDS = {
    "gdp", "growth", "economy", "economic", "macro", "inflation", "recession", "crisis",
    "debt", "trade", "forecast", "situation", "market", "interest", "rate", "unemployment",
    "financial", "development", "poverty", "investment", "export", "import", "fiscal",
    "monetary", "central bank", "currency", "dollar", "exchange", "policy", "outlook",
    "per capita", "cpi", "deficit", "surplus", "gdp_pc", "income", "wage", "yield",
    "twin", "analogs", "scenario", "history", "trajectory", "per-capita", "percapita",
}


def is_relevant_economic_prompt(prompt: str, available_iso3: set[str] | None = None) -> bool:
    """Check if a prompt is relevant to macroeconomic analysis or country scenario forecasting.
    
    Returns True if the prompt contains economic keywords, a country name/ISO3, or a year.
    Returns False for off-topic queries like 'tell me a joke', 'recipe for cake', etc.
    """
    clean_text = prompt.lower()
    
    # 1. Check for country names or ISO3 codes
    iso3, year, _ = extract_entities_from_prompt(prompt, available_iso3)
    if iso3 is not None:
        return True
        
    # 2. Check for explicit 4-digit years
    if year is not None:
        return True

    # 3. Check for economic domain keywords
    words = set(re.findall(r"\b[a-z_]{3,}\b", clean_text))
    if words.intersection(ECONOMIC_KEYWORDS):
        return True

    return False


def extract_entities_from_prompt(prompt: str, available_iso3: set[str] | None = None) -> tuple[str | None, int | None, int | None]:
    """Parse a natural language prompt to extract (iso3, year, horizon).
    
    Examples:
        "can you tell about bangladesh and 2005 economic situation" -> ("BGD", 2005, None)
        "bd 2027 10 year forecast" -> ("BGD", 2027, 10)
        "oman 2009" -> ("OMN", 2009, None)
        "qatar 2018" -> ("QAT", 2018, None)
    """
    clean_text = prompt.lower()
    
    # 1. Extract year (4-digit number between 1900 and 2039)
    years = re.findall(r"\b(19\d\d|20[0-3]\d)\b", prompt)
    year = int(years[0]) if years else None
    
    # 2. Extract horizon (1, 3, 5, 10 years)
    horizon = None
    if re.search(r"\b(10[\s-]*years?|10y)\b", clean_text):
        horizon = 10
    elif re.search(r"\b(3[\s-]*years?|3y)\b", clean_text):
        horizon = 3
    elif re.search(r"\b(1[\s-]*years?|1y)\b", clean_text):
        horizon = 1
    elif re.search(r"\b(5[\s-]*years?|5y)\b", clean_text):
        horizon = 5
        
    # 3. Extract country/ISO3
    # Strategy: try longer names first (sorted by length desc) so that
    # "south korea" matches before "korea", "united kingdom" before "uk", etc.
    # Only keys with len >= 3 are used in regex matching; shorter keys are
    # handled separately to avoid false substring matches.
    iso3 = None

    # 3a. Try all keys with length >= 3 (safe for \b matching)
    safe_entries = [(name, code) for name, code in COUNTRY_MAP.items() if len(name) >= 3]
    for name, code in sorted(safe_entries, key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, clean_text):
            iso3 = code
            break

    # 3b. If not found, try ambiguous 2-letter codes but ONLY as standalone
    #     whitespace-delimited tokens (not as part of another word).
    if not iso3:
        tokens = set(clean_text.split())
        for short_code, code in _AMBIGUOUS_SHORT_CODES.items():
            if short_code in tokens:
                iso3 = code
                break

    # 3c. Fallback: match raw uppercase 3-letter ISO3 codes from the panel
    if not iso3 and available_iso3:
        words = re.findall(r"\b[A-Z]{3}\b", prompt)
        for w in words:
            if w in available_iso3:
                iso3 = w
                break

    # 3d. Fallback: fuzzy matching for misspelled country names (e.g. "afganistan" -> AFG, "columbia" -> COL)
    if not iso3:
        import difflib
        words = re.findall(r"\b[a-z]{4,}\b", clean_text)
        skip_words = ECONOMIC_KEYWORDS.union({"year", "years", "horizon", "tell", "about", "what", "how", "with", "from", "into", "over", "under", "than", "more", "show", "give", "prediction"})
        candidate_tokens = [w for w in words if w not in skip_words]
        country_names = [k for k in COUNTRY_MAP.keys() if len(k) >= 4]

        for token in candidate_tokens:
            matches = difflib.get_close_matches(token, country_names, n=1, cutoff=0.75)
            if matches:
                matched_code = COUNTRY_MAP[matches[0]]
                if available_iso3 is None or matched_code in available_iso3:
                    iso3 = matched_code
                    break

    return iso3, year, horizon


