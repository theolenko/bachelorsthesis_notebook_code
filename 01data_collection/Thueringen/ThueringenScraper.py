# ThueringenScraper.py — Extrahiert nur die MetaDaten 23.05.2025

import pandas as pd
import json
import requests
import logging
import time
import os
import re
from datetime import datetime
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

#  Settings 
script_dir        = os.path.dirname(os.path.abspath(__file__))
target_json       = os.path.join(script_dir, 'Thueringen_Meta.json')
excel_path        = os.path.join(script_dir, 'Thueringen_FilterWords.xlsx')
SCRAPING_MODE     = "fixed" #optional rolling
FIXED_START_DATE  = "16.05.2023"

BASE_URL   = "https://parldok.thueringer-landtag.de"
SEARCH_URL = f"{BASE_URL}/ParlDok/freiesuche"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html",
    "Referer": BASE_URL
}

#  Session mit Retry 
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

#  Bestehende Meta laden 
existing_meta = []
if os.path.exists(target_json):
    try:
        if os.path.getsize(target_json) > 0:
            with open(target_json, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Konnte bestehende Meta-Datei nicht lesen ({e}), starte mit leerer Liste.")

#  Filter-Wörter laden für POST Anfragen
try:
    df = pd.read_excel(excel_path, sheet_name='FilterWords')
except Exception as e:
    logging.error(f"Fehler beim Laden der Excel-Datei: {e}")
    df = pd.DataFrame(columns=['Filter'])

#  Hilfsfunktionen 
def get_date_from() -> str:
    """Gibt das Startdatum zurück (fixed oder rolling)."""
    if SCRAPING_MODE.lower() == "fixed":
        return FIXED_START_DATE
    # implement rolling if needed
    return FIXED_START_DATE

def fetch_initial_content(term: str) -> str | None:
    """Erste POST-Anfrage, liefert HTML oder None."""
    try:
        date_from = datetime.strptime(get_date_from(), "%d.%m.%Y")
        post_data = {
            "SearchType": "1",  # 1 für Suche im Dokument, 0 für Suche im Titel/Summary
            "LegislaturPeriodenNummer": "", #wenn, keine ausgewählt, suche über alle wie in diesem Fall
            "Datum": "",
            "DatumVon": date_from,
            "DatumBis": "",
            "SearchWords": term,
            "Operation": "1", #mehrere Wörter zB kommunale haushalte werden mit UND verbunden bei 1 und ODER bei 0
            "UrheberSonstigeId": "",
            "DokumententypId": "",
            "BeratungsstandId": ""
        }
        
        r = session.post(SEARCH_URL, data=post_data, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logging.error(f"Fehler beim Abrufen initialer Seite für '{term}': {e}")
        return None

def fetch_next_page(page: int) -> str | None:
    """GET der paginierten Ergebnisse."""
    try:
        url = f"{SEARCH_URL}/{page}"
        r = session.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logging.error(f"Fehler beim Abrufen Seite {page}: {e}")
        return None

def no_results_found(html: str) -> bool:
    """Prüft, ob 'Keine Dokumente gefunden' angezeigt wird."""
    soup = BeautifulSoup(html, "html.parser")
    msg = soup.find("p", class_="message")
    return bool(msg and "keine dokumente gefunden" in msg.get_text(strip=True).lower())

def get_current_page_number(html: str) -> int:
    """Extrahiert die aktuelle Seitennummer aus dem screenreader-Header."""
    soup = BeautifulSoup(html, "html.parser")
    hdr = soup.find("h2", class_="screenreader")
    if hdr:
        txt = " ".join(hdr.stripped_strings)
        m = re.search(r"Seite\s*(\d+)", txt, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 1

def extract_date(text: str) -> str:
    """Findet das erste Datum im Format DD.MM.YYYY."""
    m = re.search(r'\d{2}\.\d{2}\.\d{4}', text)
    return m.group() if m else ""

def extract_author(text: str) -> str:
    return text.strip() if text else "Unbekannt"

def parse_meta_elements(html: str, term: str) -> list[dict]:
    """
    Parst alle <li class="row tlt_search_result">,
    extrahiert Link, Titel, Datum, Autor.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for li in soup.find_all("li", class_="row tlt_search_result"):
        col11 = li.find("div", class_="col-11")
        if not col11:
            continue

        # Titel & Link
        title_div = col11.find("div", class_="row title")
        a = title_div.find("a", href=True) if title_div else None
        title = a.get_text(strip=True) if a else "Unbekannt"
        href  = a['href'] if a and a['href'] != "#" else ""
        link  = f"{BASE_URL}{href}" if href else ""
        if not link:
            continue

        # Datum & Autor
        info_rows = col11.find("div", class_="row resultinfo")
        rows      = info_rows.find_all("div", class_="row") if info_rows else []
        date_txt  = rows[0].get_text(strip=True) if len(rows) > 0 else "" #wenn wirklich 1 Eintrag da ist ...
        author_txt= rows[1].get_text(strip=True) if len(rows) > 1 else "" #wenn wirklich 2 Einträge da sind ...

        results.append({
            "ElementID":      link,
            "Beschreibung":   f"{title}, {extract_author(author_txt)}",
            "Datum":          extract_date(date_txt),
            "Landtag":        "TH",
            "FilterDetails":  [term]
        })

    return results

def send_request(term: str) -> list[dict]:
    """
    Führt initiale Suche und Pagination durch,
    bis keine neuen Seiten oder 'Seite 1' erneut erscheint.
    """
    html = fetch_initial_content(term)
    if not html:
        return []
    if no_results_found(html):
        logging.info(f"Für '{term}' keine Ergebnisse.")
        return []

    first_page = get_current_page_number(html)
    page        = first_page
    all_meta    = []

    while True:
        logging.info(f"{term}: verarbeite Seite {page}")
        metas = parse_meta_elements(html, term)
        if not metas:
            break
        all_meta.extend(metas)

        page += 1
        html = fetch_next_page(page)
        if not html or no_results_found(html):
            break
        if get_current_page_number(html) == first_page:
            break

        time.sleep(2)

    return all_meta

#  Main ——
def main():
    # 1) Index existing
    all_results = {e['ElementID']: e for e in existing_meta}

    # 2) Scrape per Filter
    for term in df['Filter'].dropna():
        logging.info(f"Starte Suche für Filterwort '{term}'")
        for m in send_request(term):
            eid = m['ElementID']
            if eid in all_results:
                entry = all_results[eid]
                if term not in entry['FilterDetails']:
                    entry['FilterDetails'].append(term)
                    logging.info(f"Filter '{term}' ergänzt bei {eid}")
            else:
                all_results[eid] = m
        time.sleep(2)

    # 3) Speichern
    with open(target_json, 'w', encoding='utf-8') as f:
        json.dump(list(all_results.values()), f, ensure_ascii=False, indent=4)
    logging.info("Thüringen-Scraping abgeschlossen.")

if __name__ == '__main__':
    main()
