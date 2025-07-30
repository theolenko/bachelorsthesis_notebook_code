# Scraper 17.05.2025 – Option 1: In-Memory Grouping
import os
import re
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Settings
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
META_JSON         = os.path.join(SCRIPT_DIR, 'SachsenAnhalt_Meta.json')
EXCEL_PATH        = os.path.join(SCRIPT_DIR, 'SachsenAnhalt_FilterWords.xlsx')
SCRAPING_MODE     = "fixed"
FIXED_START_DATE  = "16.05.2023"  # Format: dd.mm.yyyy

# HTTP Session with Retries
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Helpers
def adjust_date_in_payload(payload_str: str) -> str:
    base = datetime.strptime(FIXED_START_DATE, "%d.%m.%Y")
    return (
        payload_str
        .replace("17.01.2025", base.strftime("%d.%m.%Y"))
        .replace("2025 01 17", base.strftime("%Y %m %d"))
    )

DATE_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{4}")
def extract_date(text: str) -> str:
    m = DATE_PATTERN.search(text)
    return m.group() if m else "Kein Datum gefunden"

# Core Functions
def get_payload_from_excel(term: str) -> dict | None:
    df = pd.read_excel(EXCEL_PATH, sheet_name='FilterWords')
    row = df[df['Filter'] == term]
    if row.empty:
        logging.warning(f"Kein Payload für '{term}'")
        return None
    raw = row.iloc[0]['Payload']
    try:
        return json.loads(adjust_date_in_payload(raw))
    except Exception as e:
        logging.error(f"Payload-Fehler für '{term}': {e}")
        return None


def construct_report_url(report_id: str) -> str:
    return (
        f"https://padoka.landtag.sachsen-anhalt.de/portal/report.tt.html"
        f"?report_id={report_id}&start=0&chunksize=1000000"
    )


def extract_data_from_page(html: str, term: str) -> list[dict]:
    soup = BeautifulSoup(html, 'html.parser')
    container = soup.find('div', id='results-container')
    if not container:
        return []
    records = container.find_all(
        'div', class_=lambda v: v and 'efxRecordRepeater' in v
    )
    cutoff = datetime.strptime(FIXED_START_DATE, '%d.%m.%Y')
    results = []
    for el in records:
        title = el.find('h3', class_=lambda v: v and 'h5' in v)
        span = el.find('span', class_='h6')
        besch = f"{title.get_text(strip=True) if title else ''}, {span.get_text(strip=True) if span else ''}".strip(', ')
        date_str = extract_date(besch)
        try:
            if datetime.strptime(date_str, '%d.%m.%Y') < cutoff:
                continue
        except:
            continue
        pdf_div = el.find('div', class_=lambda v: v and 'mt-4' in v)
        if not pdf_div:
            continue
        links = pdf_div.find_all('a', href=True, class_=lambda v: v and 'btn-white' in v)
        if not links:
            continue
        parts = []
        for a in links:
            label = a.find('span', class_='icon-text')
            lbl = label.get_text(strip=True) if label else 'Dokument'
            url = a['href'].strip()
            parts.append(f"{lbl}:{url}")
        eid = ' | '.join(parts)
        entry = {
            'ElementID': eid,
            'Beschreibungstext': besch,
            'Datum': date_str,
            'Landtag': 'SA',
            'FilterDetails': [term]
        }
        results.append(entry)
    return results


def send_request(term: str) -> list[dict]:
    payload = get_payload_from_excel(term)
    if not payload:
        return []
    base_url = "https://padoka.landtag.sachsen-anhalt.de/portal/browse.tt.json"
    r = session.post(base_url, json=payload, timeout=60)
    if r.status_code != 200:
        logging.error(f"HTTP {r.status_code} für '{term}'")
        return []
    data = r.json()
    report_id = data.get('report_id')
    if report_id:
        logging.info(f"Fetching report_id {report_id} for '{term}'")
        page = session.get(construct_report_url(report_id), timeout=60).text
        return extract_data_from_page(page, term)
    search_id = data.get('search_id')
    if search_id:
        logging.info(f"No report_id for '{term}', retrying with search_id {search_id}")
        time.sleep(10)
        fallback_payload = {
            "action": "SearchAndDisplay",
            "sources": ["lsa.lissh"],
            "report": {"rhl":"main","rhlmode":"add","format":"generic2-short","mime":"html","sort":"WEBSO1/D WEBSO2"},
            "dataSet": "2","id": search_id
        }
        r2 = session.post(base_url, json=fallback_payload, timeout=60)
        if r2.status_code == 200:
            data2 = r2.json()
            rid2 = data2.get('report_id')
            if rid2:
                logging.info(f"Fetching fallback report_id {rid2} for '{term}'")
                page = session.get(construct_report_url(rid2), timeout=60).text
                return extract_data_from_page(page, term)
        logging.error(f"Fallback also returned no report_id for '{term}'")
        return []
    logging.warning(f"No report_id and no search_id for '{term}'")
    return []


def main():
    # 1) In-Memory laden
    all_entries: dict[str, dict] = {}
    if os.path.exists(META_JSON):
        try:
            with open(META_JSON, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for e in existing:
                all_entries[e['ElementID']] = e
        except Exception:
            logging.warning("Konnte META_JSON nicht lesen, starte leer.")
    # 2) Scraping und Gruppierung
    filters = pd.read_excel(EXCEL_PATH, sheet_name='FilterWords')['Filter'].dropna()
    for term in filters:
        logging.info(f"Processing {term}")
        hits = send_request(term)
        for doc in hits:
            eid = doc['ElementID']
            if eid not in all_entries:
                all_entries[eid] = doc
            else:
                entry = all_entries[eid]
                if term not in entry['FilterDetails']:
                    entry['FilterDetails'].append(term)
        time.sleep(20)
    # 3) Schreibung
    final_list = list(all_entries.values())
    try:
        with open(META_JSON, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Wrote {len(final_list)} records to {META_JSON}")
    except Exception as e:
        logging.error(f"Fehler beim Schreiben von {META_JSON}: {e}")

if __name__ == '__main__':
    main()
