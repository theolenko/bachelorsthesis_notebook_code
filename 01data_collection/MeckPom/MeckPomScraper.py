# MeckPomScraper.py — Extrahiert nur Meta­daten 20.05.2025

import os
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter, Retry

# Konfiguration 
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
META_JSON         = os.path.join(SCRIPT_DIR, 'MeckPom_Meta.json')
EXCEL_PATH        = os.path.join(SCRIPT_DIR, 'MeckPom_FilterWords.xlsx')
LOG_FILE          = os.path.join(SCRIPT_DIR, 'scraper.log')

SCRAPING_MODE     = "fixed"   # oder "rolling"
FIXED_START_DATE  = "16.05.2023"
RETENTION_DAYS    = 7 #nur bei rolling relevant

MV_SEARCH_URL     = "https://www.dokumentation.landtag-mv.de/parldok/Fulltext/Search"
MV_RESULT_URL     = "https://www.dokumentation.landtag-mv.de/parldok/Fulltext/Resultpage"
MV_BASE_PDF_URL   = "https://www.dokumentation.landtag-mv.de/parldok"

# Excel lädt Spalte "Filter"
df = pd.read_excel(EXCEL_PATH, sheet_name='FilterWords')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Session mit Retry
session = requests.Session()
retries = Retry(
    total=3, backoff_factor=0.5,
    status_forcelist=[429,500,502,503,504],
    allowed_methods=["GET","POST"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
}


def get_date_from() -> str:
    if SCRAPING_MODE.lower() == "fixed":
        return FIXED_START_DATE
    return (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime("%d.%m.%Y")


def init_meta_file():
    """Legt die Meta-Datei an, falls sie noch nicht existiert."""
    if not os.path.exists(META_JSON):
        with open(META_JSON, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)


def load_existing_meta() -> list:
    """Lädt vorhandene Metadaten, fängt I/O- und JSON-Fehler ab."""
    try:
        with open(META_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Lade Meta-Datei fehlgeschlagen: {e}")
        return []


def save_meta(data: list):
    """Speichert Metadaten, protokolliert I/O-Fehler."""
    try:
        with open(META_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logging.error(f"Speichern der Meta-Datei fehlgeschlagen: {e}")


def build_payload(term: str) -> str:
    """Payload als JSON-String für MV-Search."""
    date_from = get_date_from()
    payload = {
        "devicekey": None,
        "max": 1000,
        "withfilter": False,
        "sort": 0,
        "limit": {"Length": 100},
        "tags": [
            {"type": "0", "field": "Alle", "ored": True, "fulltext": term},
            {"type": "9", "id": date_from, "field": "datefrom", "ored": True}
        ]
    }
    return json.dumps(payload)


def fetch_docs_and_queryid(term: str) -> tuple[int, list, str | None]:
    """
    Sendet die POST-Anfrage an MV_SEARCH_URL für `term` und
    gibt count, docs sowie ggf. die queryid für Pagination zurück.
    """
    payload = build_payload(term)
    resp = session.post(
        MV_SEARCH_URL,
        data={'data': payload},
        headers=HEADERS,
        timeout=60
    )
    if resp.status_code != 200:
        logging.error(f"HTTP {resp.status_code} bei Suche '{term}'")
        return 0, [], None

    try:
        data = resp.json()
    except ValueError as e:
        logging.error(f"JSON-Parse-Error bei Suche '{term}': {e}")
        return 0, [], None

    data_str = data.get('data')
    if not data_str:
        logging.info(f"Keine Ergebnisse für '{term}'")
        return 0, [], None

    try:
        data_obj = json.loads(data_str)
    except ValueError as e:
        logging.error(f"JSON-Decode(data) Fehler für '{term}': {e}")
        return 0, [], None

    count   = data_obj.get('count', 0)
    docs    = data_obj.get('docs', [])
    queryid = data_obj.get('queryid')
    logging.info(f"Found {count} docs for '{term}' (queryid={queryid})")
    return count, docs, queryid


def process_mv_docs(docs: list, term: str) -> list[dict]:
    """
    Wandelt rohe MV-JSON-`docs` in unsere Meta-Struktur um.
    """
    results = []
    for doc in docs:
        link = doc.get("link")
        #elemente ohne link bzw. fehlerhaftem link werden übersprungen
        if not link:
            continue
        url      = MV_BASE_PDF_URL + link
        besch    = doc.get("title", "").strip()
        date_txt = doc.get("date", "")
        results.append({
            "ElementID":         url,
            "Beschreibungstext": besch,
            "Datum":             date_txt,
            "Landtag":           "MV",
            "FilterDetails":     [term]
        })
    return results


def send_request(term: str) -> list[dict]:
    """
    Zentrale Aufrufstelle: holt count/docs/queryid, verarbeitet erste Seite
    und paginiert bei Bedarf.
    """
    count, docs, queryid = fetch_docs_and_queryid(term)
    if count == 0:
        return []

    results = process_mv_docs(docs, term)

    if count > len(docs) and queryid:
        start = len(docs)
        logging.info(f"{count} Treffer für '{term}', paginiere ab {start} …")
        while True:
            pag_payload = {
                "devicekey": None,
                "queryid":   queryid,
                "limit":     {"Start": start, "Length": 100}
            }
            r2 = session.post(
                MV_RESULT_URL,
                data={'data': json.dumps(pag_payload)},
                headers=HEADERS,
                timeout=60
            )
            if r2.status_code != 200:
                logging.error(f"Paginate HTTP {r2.status_code} bei '{term}'")
                break

            try:
                page_data = r2.json()
            except ValueError as e:
                logging.error(f"Paginierung JSON-Parse-Error für '{term}': {e}")
                break

            pd_str    = page_data.get("data", "")
            pd_obj    = json.loads(pd_str or "{}")
            more_docs = pd_obj.get("docs", [])
            if not more_docs:
                break

            results.extend(process_mv_docs(more_docs, term))
            start += len(more_docs)

    return results


def main():
    init_meta_file()
    all_meta     = load_existing_meta()
    existing_ids = {e['ElementID'] for e in all_meta}

    for term in df['Filter'].dropna():
        logging.info(f"Verarbeite Filterwort '{term}'")
        try:
            metas = send_request(term)
            for m in metas:
                eid = m['ElementID']
                if eid in existing_ids:
                    for ent in all_meta:
                        if ent['ElementID'] == eid and term not in ent['FilterDetails']:
                            ent['FilterDetails'].append(term)
                            logging.info(f"Filter '{term}' zu {eid} hinzugefügt")
                            break
                else:
                    all_meta.append(m)
                    existing_ids.add(eid)
        except Exception as e:
            logging.error(f"Fehler bei Verarbeitung von '{term}': {e}")
        time.sleep(2)

    save_meta(all_meta)
    logging.info("Metadaten-Scraping abgeschlossen.")


if __name__ == '__main__':
    main()
