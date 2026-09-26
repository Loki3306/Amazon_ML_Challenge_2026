import os


def load_tsv_records(path, country_filter=None):
    """
    Reads a TSV file with header: entity_id, business_name, business_address, country.
    Optionally filters by country string.
    Returns: dict of entity_id -> (business_name, business_address, country)
    """
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''

            if country_filter is None or country == country_filter:
                records[eid] = (name, addr, country)
    return records


def load_ground_truth(path):
    """
    Reads ground truth TSV: source1_entity_id, matched_entity_ids.
    Returns: dict of source1_entity_id -> set of matched_entity_ids
    """
    gt = {}
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0]
            mids = parts[1].split(',') if len(parts) > 1 and parts[1] else []
            gt[s1_id] = set(mids)
    return gt


def stream_s1_records(path, country_filter=None):
    """
    Generator yielding (entity_id, business_name, business_address, country).
    """
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''

            if country_filter is None or country == country_filter:
                yield eid, name, addr, country


def get_distinct_countries(path):
    """
    Returns the ordered list of distinct countries present in the file.
    """
    countries = []
    seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            c = parts[3] if len(parts) > 3 else ''
            if c and c not in seen:
                seen.add(c)
                countries.append(c)
    return countries
