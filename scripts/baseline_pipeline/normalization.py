import re
import unicodedata
import text_unidecode

LEGAL_SUFFIXES = {
    # English / International
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company',
    'ltd', 'limited', 'pvt', 'private', 'llc', 'llp', 'pllc',
    'enterprises', 'enterprise', 'industries', 'industry',
    'technologies', 'technology', 'services', 'service', 'solutions', 'solution',
    'group', 'holdings', 'ventures', 'consulting', 'associates', 'trading',
    'systems', 'system', 'international', 'holdings', 'holding',
    # French
    'sarl', 'sas', 'sasu', 'sci', 'sa', 'snc', 'eurl', 'ste', 'gmbh', 'bv',
    'gie', 'ei', 'scp', 'selarl', 'earl', 'scic', 'sem',
    'participations', 'distribution', 'france', 'etablissement', 'ets',
    'cie', 'compagnie', 'groupe'
}

# Leading articles and prepositions to skip when comparing brand roots
ARTICLES_AND_PREP = {
    'the', 'a', 'an',
    'le', 'la', 'les', 'l', 'un', 'une', 'du', 'des', 'de', 'd',
    'el', 'los', 'las'
}

ADDR_ABBR = {
    # English
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ct': 'court', 'ln': 'lane', 'pl': 'place',
    'sq': 'square', 'terr': 'terrace', 'pkwy': 'parkway',
    'hwy': 'highway', 'fl': 'floor', 'ste': 'suite', 'apt': 'apartment',
    'bldg': 'building', 'no': 'number', 'nr': 'near', 'opp': 'opposite',
    # French
    'r': 'rue', 'av': 'avenue', 'bd': 'boulevard', 'ch': 'chemin',
    'imp': 'impasse', 'all': 'allee', 'rte': 'route', 'crs': 'cours',
    'qu': 'quai', 'pass': 'passage', 'bat': 'batiment', 'res': 'residence',
    'st': 'saint', 'ste': 'sainte', 'zi': 'zone industrielle', 'za': 'zone activite'
}

STATE_MAP = {
    # US states (2-letter to full)
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire',
    'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee',
    'tx': 'texas', 'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington',
    'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
    # India states
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar',
    'cg': 'chhattisgarh', 'dl': 'delhi', 'ga': 'goa', 'gj': 'gujarat', 'hr': 'haryana',
    'hp': 'himachal pradesh', 'jh': 'jharkhand', 'ka': 'karnataka', 'kl': 'kerala',
    'mp': 'madhya pradesh', 'mh': 'maharashtra', 'mn': 'manipur', 'ml': 'meghalaya',
    'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha', 'pb': 'punjab', 'rj': 'rajasthan',
    'sk': 'sikkim', 'tn': 'tamil nadu', 'tg': 'telangana', 'ts': 'telangana',
    'tr': 'tripura', 'up': 'uttar pradesh', 'uk': 'uttarakhand', 'wb': 'west bengal'
}

SPLIT_DIGIT_LETTER = re.compile(r'(\d+)\s*([a-zA-Z]+)')
NUMBER_INDICATORS = re.compile(r'\b(?:no|n°|nº|#|num|numero)\b\.?', re.IGNORECASE)
PUNCT_RE = re.compile(r'[^a-zA-Z0-9\s]')
DOMAIN_RE = re.compile(r'\.(com|org|net|in|co|io|fr)\b')
URL_RE = re.compile(r'https?://(?:www\.)?')
MULTI_SPACE_RE = re.compile(r'\s+')
NUMBER_RE = re.compile(r'\b\d+\b')


def clean_string(s):
    if not s or s == 'null':
        return ''
    s = text_unidecode.unidecode(s)
    s = s.lower()
    s = URL_RE.sub('', s)
    s = DOMAIN_RE.sub(' ', s)
    s = NUMBER_INDICATORS.sub(' ', s)
    # Split digits attached to letters (e.g. '5b' -> '5 b', '14c' -> '14 c', '9bis' -> '9 bis')
    s = SPLIT_DIGIT_LETTER.sub(r'\1 \2', s)
    s = PUNCT_RE.sub(' ', s)
    s = MULTI_SPACE_RE.sub(' ', s).strip()
    return s


def normalize_name(name):
    cleaned = clean_string(name)
    if not cleaned:
        return '', '', ''
    tokens = cleaned.split()
    # Strip legal suffixes
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = tokens
    # Strip leading articles for core comparison
    if len(core_tokens) > 1 and core_tokens[0] in ARTICLES_AND_PREP:
        core_tokens = core_tokens[1:]
    core_name = ' '.join(core_tokens)
    token_sorted = ' '.join(sorted(tokens))
    return cleaned, core_name, token_sorted


def extract_numbers(cleaned):
    """
    Extracts all numeric tokens as clean canonical strings without leading zeros
    (so '0034' and '34' match as '34'), and identifies the primary street number.
    """
    raw_nums = NUMBER_RE.findall(cleaned)
    int_nums = set()
    ordered_nums = []
    for r in raw_nums:
        val = str(int(r))
        int_nums.add(val)
        ordered_nums.append(val)
    primary_num = ordered_nums[0] if ordered_nums else None
    return int_nums, primary_num


def normalize_address(addr):
    cleaned = clean_string(addr)
    if not cleaned:
        return '', set(), None, ''
    tokens = cleaned.split()
    expanded_tokens = []
    for t in tokens:
        if t in ADDR_ABBR:
            expanded_tokens.append(ADDR_ABBR[t])
        elif t in STATE_MAP:
            expanded_tokens.append(STATE_MAP[t])
        else:
            expanded_tokens.append(t)
    expanded_addr = ' '.join(expanded_tokens)
    int_nums, primary_num = extract_numbers(cleaned)
    token_sorted = ' '.join(sorted(expanded_tokens))
    return expanded_addr, int_nums, primary_num, token_sorted
