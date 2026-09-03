import geoip2.database

_city_reader = None
_asn_reader = None

def _get_city_reader():
    global _city_reader
    if _city_reader is None:
        _city_reader = geoip2.database.Reader("data/geoip/GeoLite2-City.mmdb")
    return _city_reader

def _get_asn_reader():
    global _asn_reader
    if _asn_reader is None:
        _asn_reader = geoip2.database.Reader("data/geoip/GeoLite2-ASN.mmdb")
    return _asn_reader

def resolve_country(ip):
    try:
        response = _get_city_reader().city(ip)
        return response.country.iso_code or "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def resolve_asn(ip):
    try:
        response = _get_asn_reader().asn(ip)
        return response.autonomous_system_organization or "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def add_geo_columns(df):
    """Adds geo_country and asn columns resolved from src_ip to a transactions DataFrame."""
    df = df.copy()
    df["geo_country"] = df["src_ip"].apply(resolve_country)
    df["asn"] = df["src_ip"].apply(resolve_asn)
    return df


if __name__ == "__main__":
    from parsers import load_transactions

    df = load_transactions("data/raw/synthetic_bitcoin_metadata.csv")
    df = add_geo_columns(df.head(20))
    print(df[["src_ip", "geo_country", "asn"]])