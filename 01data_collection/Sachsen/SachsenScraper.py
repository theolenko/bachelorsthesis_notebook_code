import os
import json
import logging
import re
from datetime import datetime
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# Configuration 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META_JSON = os.path.join(SCRIPT_DIR, 'Sachsen_Meta.json')
EXCEL_PATH = os.path.join(SCRIPT_DIR, 'Sachsen_FilterWords.xlsx')
LOG_FILE = os.path.join(SCRIPT_DIR, 'scraper.log')
BASE_URL = 'https://edas.landtag.sachsen.de/redas/query'
FIXED_START_DATE = '16.05.2023'

# Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# HTTP session with retries 
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=['GET']
)
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Helper functions 

def init_meta_file():
    if not os.path.exists(META_JSON):
        with open(META_JSON, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)


def load_existing_meta():
    try:
        with open(META_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        logging.warning('Bestehende Meta-Datei ungültig oder nicht lesbar, starte neu.')
        return []


def save_meta(data):
    with open(META_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info('Meta-Daten in %s gespeichert.', META_JSON)


def extract_date_from_text(text: str) -> str:
    # sucht dd.mm.yyyy
    m = re.search(r"\b(\d{2}\.\d{2}\.\d{4})\b", text)
    if m:
        return m.group(1)
    return ""


def fetch_metadata(filter_word: str) -> list[dict]:
    """
    Sendet Anfrage an EDAS API und liefert Rohdaten zurück.
    """
    # convert FIXED_START_DATE to YYYY-MM-DD
    dt = datetime.strptime(FIXED_START_DATE, '%d.%m.%Y')
    start_iso = dt.strftime('%Y-%m-%d')

    params = [
            ("text", filter_word),
            ("pageNumber", 0),
            ("pageSize", 1000000),
            ("sortId", 4),
            ("wahlperiode", 7),
            ("wahlperiode", 8),
            ("dokArt", "APr"),
            ("dokArt", "Drs"),
            ("dokArt", "GVBl"),
            ("dokArt", "PlPr"),
            ("anfangsDatum", start_iso),
            ("nurErstinitiative", "false"),
            ("nurBasisdokument", "true"),
            ("durchsucheVolltext", "true"),
            ("durchsucheTitel", "true"),
            ("durchsucheAbstract", "true"),
        ]
    try:
        r = session.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error("Fehler bei EDAS-Anfrage für '%s': %s", filter_word, e)
        return []


def process_response(data: list, filter_word: str) -> list[dict]:
    """
    Wandelt API-Antwort in unsere Meta-Struktur um.
    Gruppiert alle Dateien eines Items in ein Element.
    """
    results = []
    base_pdf = 'https://edas.landtag.sachsen.de/redas/'
    for item in data:
        title       = item.get('titel', '').strip()
        author_ref  = item.get('fundstelleAutor', '').strip()
        date_txt    = extract_date_from_text(author_ref)
        besch       = f"{title}, {author_ref}" if author_ref else title

        # 1) Sammle alle Dateien für dieses Item
        parts = []
        for file in item.get('dateien', []):
            url_rel = file.get('url', '')
            if not url_rel:
                continue
            # Label nehmen wir aus dem 'name', oder fallback auf Dateiname
            label = file.get('name', file.get('filename', '')).strip()
            url   = base_pdf + url_rel.replace('download?', 'download/file?')
            parts.append(f"{label}:{url}")

        if not parts:
            # kein PDF → überspringen
            continue

        # 2) ElementID ist nun die Gruppe aller PDFs
        element_id = " | ".join(parts)

        results.append({
            'ElementID':        element_id,
            'Beschreibungstext': besch,
            'Datum':             date_txt,
            'Landtag':           'SN',
            'FilterDetails':     [filter_word]
        })

    return results



def send_request(filter_word: str) -> list[dict]:
    raw = fetch_metadata(filter_word)
    if not raw:
        return []
    return process_response(raw, filter_word)

# Main 

def main():
    init_meta_file()
    existing = load_existing_meta()
    # dict by ElementID
    all_meta = {e['ElementID']: e for e in existing}

    df = pd.read_excel(EXCEL_PATH, sheet_name='FilterWords')
    for term in df['Filter'].dropna():
        logging.info("Verarbeite Filterwort '%s'", term)
        entries = send_request(term)
        for e in entries:
            eid = e['ElementID']
            if eid in all_meta:
                if term not in all_meta[eid]['FilterDetails']:
                    all_meta[eid]['FilterDetails'].append(term)
                    logging.info("Filter '%s' hinzugefügt zu %s", term, eid)
            else:
                all_meta[eid] = e
        # kurze Pause
        import time; time.sleep(1)

    save_meta(list(all_meta.values()))
    logging.info('Scraping für Sachsen abgeschlossen.')

if __name__ == '__main__':
    main()
