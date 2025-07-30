# extrahiert nur die MetaDaten 16.05.2025
import pandas as pd
import json
import requests
from datetime import datetime
from requests.adapters import HTTPAdapter, Retry
import logging
import time
import os
import re
from bs4 import BeautifulSoup

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("scraper.log", mode='a', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

# Settings
target_json = os.path.join(os.path.dirname(__file__), 'Brandenburg_Meta.json')
excel_path = os.path.join(os.path.dirname(__file__), 'Brandenburg_FilterWords.xlsx')
SCRAPING_MODE = "fixed"
FIXED_START_DATE = "16.05.2023"

# Session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5,
                status_forcelist=[429,500,502,503,504],
                allowed_methods=["GET","POST"])
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.parlamentsdokumentation.brandenburg.de/"
}

# Pre-load existing meta and track, handle empty or invalid JSON
existing_meta = []
existing_meta = []
if os.path.exists(target_json):
    try:
        if os.path.getsize(target_json) > 0:
            with open(target_json, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Konnte bestehende Meta-Datei nicht lesen ({e}), starte leer.")

# load filters
df = pd.read_excel(excel_path, sheet_name='FilterWords')

# helper 
def adjust_date_in_payload(payload_str):
    base = datetime.strptime(FIXED_START_DATE, "%d.%m.%Y")
    return payload_str.replace("16.02.2025", base.strftime("%d.%m.%Y")).replace("2025 02 16", base.strftime("%Y %m %d"))

#helper functions for send_request

def get_initial_report_data(term: str) -> dict | None:
    """
    Führt die erste POST-Anfrage aus und gibt das JSON zurück,
    oder None, wenn kein report_id da ist.
    """
    base_url = "https://www.parlamentsdokumentation.brandenburg.de/portal/browse.tt.json"
    row = df[df['Filter'] == term]
    if row.empty:
        logging.warning(f"No payload for {term}")
        return None

    payload_str = adjust_date_in_payload(row.iloc[0]['Payload'])
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError as e:
        logging.error(f"Payload-Decode-Error für '{term}': {e}")
        return None
    r = session.post(base_url, json=payload, headers=headers, timeout=60)
    if r.status_code != 200:
        logging.error(f"HTTP {r.status_code} for '{term}'")
        return None

    data = r.json()
    if "report_id" not in data:
        logging.info(f"No report_id for '{term}'")
        return None

    return data

#zweite anfrage stellen um optimierte HTML zu erhalten (bessere gruppierung der dokumente)
def refine_report_via_search_id(data: dict, term: str) -> str:
    """
    Wenn in data eine search_id steckt, sendet eine DisplayChange-Anfrage
    und liefert ggf. eine neue report_id zurück.
    """
    search_id = data.get("search_id")
    if not search_id:
        return data["report_id"]

    logging.info(f"Fallback-View für '{term}' via search_id {search_id}")
    time.sleep(1)
    fallback_payload = {
        "action":"DisplayChange",
        "report":{  "rhl":"main",
                    "rhlmode":"add",
                    "format":"generic2-short",
                    "mime":"html",
                    "sort":"WEBSO1/D WEBSO2/D WEBSO3"
                },
            "id":search_id,
            "start":0,
            "dataSet":"1"
    }
    try:
        r2 = session.post(
            "https://www.parlamentsdokumentation.brandenburg.de/portal/browse.tt.json",
            json=fallback_payload, headers=headers, timeout=60
        )
        r2.raise_for_status()
        data2 = r2.json()
        return data2.get("report_id", data["report_id"])
    except Exception as e:
        logging.warning(f"Fallback fehlgeschlagen für '{term}': {e}")
        return data["report_id"]

#holt den finalen HTML Baum
def fetch_report_page(report_id: str) -> str | None:
    """
    Holt das HTML zu einer gegebenen report_id.
    """
    url = (
        "https://www.parlamentsdokumentation.brandenburg.de/portal/report.tt.html"
        f"?report_id={report_id}&start=0&chunksize=1000000"
    )
    r = session.get(url, headers=headers, timeout=60)
    if r.status_code != 200:
        logging.error(f"Report-Page HTTP {r.status_code}")
        return None
    return r.text

#parst den HTML Baum 
def parse_meta_elements(html: str, term: str) -> list[dict]:
    """
    Parst alle <div class="record mb-3 | efxRecordRepeater"> aus dem HTML,
    extrahiert Datum, Links, Beschreibungstext und gibt eine Liste von Metadicts zurück.
    """
    soup = BeautifulSoup(html, "html.parser")
    elems = soup.find_all("div", class_="record mb-3 | efxRecordRepeater")
    cutoff = datetime.strptime(FIXED_START_DATE, "%d.%m.%Y")
    results = []

    for el in elems:
        date_tag = el.find("span", class_="h6")
        besch = date_tag.text.strip() if date_tag else ""
        dt_txt = besch
        m = re.search(r"\d{2}\.\d{2}\.\d{4}", dt_txt)
        if not m: 
            continue
        doc_date = datetime.strptime(m.group(), "%d.%m.%Y")
        if doc_date < cutoff: 
            continue

        # 1 Element → alle Links gruppieren
        parts = []
        for a in el.find_all("a", class_="dropdown-item", href=True):
            label = a.find("span", class_="icon-text")
            lbl = label.get_text(strip=True) if label else "Dokument"
            parts.append(f"{lbl}:{a['href']}")
        element_id = " | ".join(parts)

        results.append({
            "ElementID":    element_id,
            "Beschreibung": besch,
            "Datum":         m.group(),
            "Landtag": "BB",
            "FilterDetails":[term]
        })



    return results

# extract meta only
def send_request(term: str) -> list[dict]:
    initial = get_initial_report_data(term)
    if not initial:
        return []

    # opt. bessere View holen
    report_id = refine_report_via_search_id(initial, term)

    # Seite holen und parsen
    html = fetch_report_page(report_id)
    if not html:
        return []

    return parse_meta_elements(html, term)



def main():
    
    # 1) Vorhandene Meta-Daten laden
    all_results = {e['ElementID']: e for e in existing_meta}

    # 2) Scrapen und in unser Dict einfügen/aktualisieren
    for term in df['Filter']:
        logging.info(f"Processing {term}")
        for m in send_request(term):
            eid = m['ElementID']
            if eid in all_results:
                entry = all_results[eid]
                if term not in entry['FilterDetails']:
                    entry['FilterDetails'].append(term)
                    logging.info(f"Appended filter '{term}' to existing document {eid}")
            else:
                all_results[eid] = m
        time.sleep(20)

    # 3) Alles auf einmal in die JSON schreiben
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(list(all_results.values()), f, ensure_ascii=False, indent=4)
    logging.info("Scraping done.")


if __name__ == '__main__':
    main()

