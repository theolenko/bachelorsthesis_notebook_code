import os
import json
import tempfile
import logging
import re
import requests
import fitz
from datetime import datetime
from requests.adapters import HTTPAdapter, Retry

#wichtig. Datei im Verzeichnis aus https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/text-extraction/multi_column.py
from multi_column import column_boxes

# Priorisierte Filterliste für große Dateien
IMPORTANT_FILTERS = {

}

# Dateien
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META_FILE = os.path.join(SCRIPT_DIR, 'MeckPom_Meta.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'MeckPom_Data.json')

# Logging
LOG_FILE = os.path.join(SCRIPT_DIR, 'extractor.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# HTTP-Session mit Retry
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

# Puffer-Konfiguration
BUFFER_SIZE = 50
buffer = []


def flush_buffer():
    """Anhängen des Puffers an OUTPUT_FILE und Leeren des Buffers."""
    try:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Alte OUTPUT_FILE ungültig, überschreibe neu: {e}")
        data = []

    data.extend(buffer)
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Flushed {len(buffer)} Einträge nach {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Fehler beim Schreiben nach {OUTPUT_FILE}: {e}")
    buffer.clear()


def format_text_with_json_friendly_structure(raw_text: str) -> str:
    """Ersetzt Pipes und newlines und bricht Sätze auf neue Zeilen um."""
    cleaned = raw_text.replace("|", " ").replace("\n", " ")
    formatted = re.sub(r'([\.\?\!]) ', r'\1\n', cleaned)
    return re.sub(r'\s+', ' ', formatted).strip()


def choose_filters_to_use(filters: list[str], total_pages: int) -> list[str]:
    """
    Wenn PDF >300 Seiten: nur IMPORTANT_FILTERS, falls vorhanden.
    Sonst alle.
    """
    if total_pages > 300:
        subset = [w for w in filters if w in IMPORTANT_FILTERS]
        if subset:
            logging.info(f"Große PDF ({total_pages} S.): Nur wichtige Filter genutzt {subset}")
            return subset
    return filters


def select_relevant_pages(doc: fitz.Document, filters_to_use: list[str]) -> set[int]:
    """Markiert alle Seiten ≤5, sonst nur solche mit Treffern ±1 Nachbar."""
    total = doc.page_count
    if total <= 5:
        return set(range(total))
    pages = set()
    for i in range(total):
        txt = (doc.load_page(i).get_text("text") or "").lower()
        if any(re.search(rf"{re.escape(w)}", txt) for w in filters_to_use):
            pages.update({i-1, i, i+1})
    return {p for p in pages if 0 <= p < total}


def extract_text(url: str, filters: list[str]) -> str:
    """
    Lädt das PDF, überspringt >5000 Seiten, wählt Filter,
    markiert relevante Seiten und extrahiert Text.
    """
    try:
        r = session.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Download-Fehler {e} für {url}")
        return ''

    tmp_pdf = None
    parts: list[str] = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(r.content)
            tmp_pdf = tmp.name

        doc = fitz.open(tmp_pdf)
        total = doc.page_count
        if total > 5000:
            logging.info(f"Überspringe PDF ({total} S.) für {url}")
            return ''
        
        #relevante Filter auswählen
        filters_to_use = choose_filters_to_use(filters, total)
        #relevante Seiten markieren
        relevant = select_relevant_pages(doc, filters_to_use)

        for i in sorted(relevant):
            page = doc.load_page(i)
            boxes = column_boxes(page)
            if boxes:
                texts = [page.get_text("text", clip=b, sort=True) for b in boxes]
                parts.append("\n".join(texts))
            else:
                parts.append(page.get_text("text") or '')

        
    except Exception as e:
        logging.error(f"PDF-Parsing-Fehler {e} für {url}")
    finally:
        doc.close()
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)

    full = "\n".join(parts)
    return format_text_with_json_friendly_structure(full)


def main():
    if not os.path.exists(META_FILE):
        logging.error(f"Meta-Datei nicht gefunden: {META_FILE}")
        return
    try:
        with open(META_FILE, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except Exception as e:
        logging.error(f"Konnte Meta-Datei nicht laden: {e}")
        return

    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for rec in json.load(f):
                    processed.add(rec.get('ElementID'))
        except Exception:
            pass

    for ent in entries:
        eid       = ent.get('ElementID', '')
        if eid in processed:
            logging.info(f"Bereits extrahiert, überspringe {eid}")
            continue
        filters   = ent.get('FilterDetails', [])
        logging.info(f"Verarbeite {eid} mit Filtern {filters}")

        #Text extrahieren
        txt = extract_text(eid, filters)
        ent['ExtrahierterText'] = txt
        buffer.append(ent)

        if len(buffer) >= BUFFER_SIZE:
            flush_buffer()

    if buffer:
        flush_buffer()
    logging.info("PDF-Extraktion abgeschlossen.")

if __name__ == '__main__':
    main()
