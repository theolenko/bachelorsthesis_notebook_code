# BrandenburgPDFExtractor.py  — Extractor nach dem Scraper (Stand: 27.05.2025)

import json
import os
import tempfile
import logging
import re
import requests
import fitz
from datetime import datetime
from requests.adapters import HTTPAdapter, Retry

#wichtig. Datei im Verzeichnis aus https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/text-extraction/multi_column.py
from multi_column import column_boxes

# priorisierte Filterliste für große Dateien
IMPORTANT_FILTERS = {
    "vng", "pck", "verbundnetz gas", "ontras", "heimüller",
    "kaltefleiter", "lng", "wasserstoff", "enbw",
    "leag", "biomethan", "biogas", "erdgas", "arcelor mittal", "kernnetz"
}

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extractor.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

#  Dateien 
META_FILE   = os.path.join(os.path.dirname(__file__), 'Brandenburg_Meta.json')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'Brandenburg_Data.json')

#  HTTP-Session mit Retry 
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

#  Chunked-Flush Konfiguration --
BUFFER_SIZE = 50
buffer = []


def flush_buffer():
    """Anhängen des Puffers an OUTPUT_FILE und Leeren des Buffers."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            logging.warning("Alte OUTPUT_FILE ungültig, überschreibe neu.")
            data = []
    else:
        data = []

    data.extend(buffer)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Flushed {len(buffer)} Einträge nach {OUTPUT_FILE}")
    buffer.clear()

# Hilfsfunktionen

def format_text_with_json_friendly_structure(raw_text: str) -> str:
    """Ersetzt Pipes und newlines und bricht Sätze auf neue Zeilen um."""
    cleaned = raw_text.replace("|", " ").replace("\n", " ")
    formatted = re.sub(r'([\.\?\!]) ', r'\1\n', cleaned)
    return re.sub(r'\s+', ' ', formatted).strip()


def choose_filters_to_use(filters: list[str], total_pages: int) -> list[str]:
    """
    Wenn PDF >300 Seiten: nur IMPORTANT_FILTERS, falls welche
    existieren, sonst alle. Sonst alle.
    """
    if total_pages > 300:
        subset = [w for w in filters if w in IMPORTANT_FILTERS]
        if subset:
            logging.info(f"Große PDF ({total_pages} S.): Nur wichtige Filter genutzt {subset}")
            return subset
    return filters


def select_relevant_pages(doc: fitz.Document, filters_to_use: list[str]) -> set[int]:
    """
    Markiert alle Seiten ≤5, sonst nur solche mit Filtertreffern ±1 Nachbar.
    """
    total = doc.page_count
    if total <= 5:
        return set(range(total))
    pages = set()
    for i in range(total):
        text = doc.load_page(i).get_text("text") or ""
        lower = text.lower()
        if any(re.search(rf"{re.escape(w)}", lower) for w in filters_to_use):
            pages.update({i, i-1, i+1})
    return {p for p in pages if 0 <= p < total}


def parse_links(element_id: str) -> list[tuple[str, str]]:
    """
    Teilt "Label:URL | Label2:URL2" in [(Label, URL), ...].
    """
    parts = [x.strip() for x in element_id.split("|")]
    out = []
    for p in parts:
        if ":" in p:
            lbl, u = p.split(":", 1)
            out.append((lbl.strip(), u.strip()))
    return out


def extract_text(url: str, filters: list[str]) -> str:
    """
    Lädt das PDF, überspringt >5000 Seiten, wählt Filter,
    markiert relevante Seiten und extrahiert Text—
    mit Multicolumn bei is_plenar.
    """
    try:
        r = session.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Download-Fehler {e} für {url}")
        return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        pdf_path = tmp.name

    parts: list[str] = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count

        if total_pages > 5000:
            logging.info(f"Überspringe PDF ({total_pages} Seiten) für {url}")
            return ""
        
        #relevante Filter auswählen basierend auf Textlänge
        filters_to_use = choose_filters_to_use(filters, total_pages)
        #basierend auf Filtrern rellevante Seiten auswählen
        relevant = select_relevant_pages(doc, filters_to_use)

        for i in sorted(relevant):
            page = doc.load_page(i)
            # Spalten erkennen
            boxes = column_boxes(page)
            if boxes:
                # text aus den boxen zusammenführen
                col_texts = [page.get_text("text", clip=rect, sort=True) for rect in boxes]
                parts.append("\n".join(col_texts))
            else:
                parts.append(page.get_text("text") or "")
    except Exception as e:
        logging.error(f"PDF-Parsing-Fehler {e} für {url}")
    finally:
        doc.close()
        os.remove(pdf_path)

    full = "\n".join(parts)
    return format_text_with_json_friendly_structure(full)


# ende hilfsfunktionen


#  Main-Funktion 
def main():
    # 1) Metadaten laden
    if not os.path.exists(META_FILE):
        logging.error(f"Meta-Datei nicht gefunden: {META_FILE}")
        return

    try:
        with open(META_FILE, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except Exception as e:
        logging.error(f"Konnte Meta-Datei nicht laden: {e}")
        return

    # 2) Bereits extrahierte ElementIDs sammeln
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for rec in json.load(f):
                    processed.add(rec.get('ElementID'))
        except Exception:
            pass

    # 3) Durch alle Elemente iterieren
    for ent in entries:
        eid = ent.get('ElementID', '')
        if eid in processed:
            logging.info(f"Bereits extrahiert, überspringe {eid}")
            continue

        filters = ent.get('FilterDetails', [])
        logging.info(f"Verarbeite {eid} mit Filtern {filters}")

        parts = []
        for lbl, url in parse_links(eid):
            txt = extract_text(url, filters)
            parts.append(f"{lbl}: {txt}")

        ent['ExtrahierterText'] = "\n\n".join(parts)
        buffer.append(ent)

        if len(buffer) >= BUFFER_SIZE:
            flush_buffer()

    # 4) Letzte Puffer-Reste flushen
    if buffer:
        flush_buffer()

    logging.info("PDF-Extraktion abgeschlossen.")


if __name__ == "__main__":
    main()
