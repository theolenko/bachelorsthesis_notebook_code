# Extractor 17.05.2025 
import os
import re
import json
import tempfile
import logging
import requests
import fitz
from datetime import datetime
from requests.adapters import HTTPAdapter, Retry

#wichtig. Datei im Verzeichnis aus https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/text-extraction/multi_column.py
from multi_column import column_boxes

#priorisierte Filterliste für große Dateien
IMPORTANT_FILTERS = {
    "vng", "skw", "verbundnetz gas", "ontras", "heimüller",
    "kaltefleiter", "lng", "wasserstoff", "mibrag",
    "energiepark", "biomethan", "biogas", "erdgas"
}

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extractor.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Dateipfade
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
META_FILE    = os.path.join(SCRIPT_DIR, 'SachsenAnhalt_Meta.json')
OUTPUT_FILE  = os.path.join(SCRIPT_DIR, 'SachsenAnhalt_Data.json')

# HTTP-Session mit Retry
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

# Puffer-Größe
BUFFER_SIZE = 50
buffer = []

# Hilfsfunktion: Formatierung

def format_text_with_json_friendly_structure(raw_text: str) -> str:
    cleaned = raw_text.replace("|", " ").replace("\n", " ")
    # jede Satzende auf neue Zeile
    formatted = re.sub(r'([\.\?\!]) ', r'\1\n', cleaned)
    # wiederholte Leerzeichen entfernen
    return re.sub(r'\s+', ' ', formatted).strip()
    
#Funktion entscheided welche FIlter genutzt werden sollen - alle oder nur wichtigere   
def choose_filters_to_use(filters: list[str], total_pages: int) -> list[str]:
    if total_pages > 300:
        subset = [w for w in filters if w in IMPORTANT_FILTERS]
        if subset:
            logging.info(f"Große PDF ({total_pages} S.): Nur wichtige Filter genutzt {subset}")
            return subset
    return filters

#markiert basierend auf Filtern die relevanten Seiten für weitere Extraktion
def select_relevant_pages(doc: fitz.Document, filters_to_use: list[str]) -> set[int]:
    total = doc.page_count
    if total <= 5:
        return set(range(total))
    pages = set()
    for i in range(total):
        txt = (doc.load_page(i).get_text("text") or "").lower()
        if any(re.search(rf"{re.escape(w)}", txt) for w in filters_to_use):
            pages.update({i-1, i, i+1})
    return {p for p in pages if 0 <= p < total}


# Extraktion von PDF-Text
def extract_text(url: str, filters: list[str]) -> str:
    try:
        r = session.get(url, stream=True, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Download-Fehler {e} für {url}")
        return ''
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(r.content)
        path = tmp.name
    text = ''
    try:
        doc = fitz.open(path)
        total = doc.page_count
        if total > 5000:
            logging.info(f"Überspringe PDF ({total} Seiten) für {url}")
            return ""

        filters_to_use = choose_filters_to_use(filters, total)
        relevant      = select_relevant_pages(doc, filters_to_use)

        parts: list[str] = []
        for i in sorted(relevant):
            page = doc.load_page(i)
            boxes = column_boxes(page)
            if boxes:
                texts = [page.get_text("text", clip=b, sort=True) for b in boxes]
                parts.append("\n".join(texts))
            else:
                parts.append(page.get_text("text") or "")
                
        text = "\n".join(parts)


    except Exception as e:
        logging.error(f"PDF-Parsing-Fehler {e} für {url}")
    finally:
        doc.close()
        os.remove(path)

    return format_text_with_json_friendly_structure(text)
        

# Link-Parser
def parse_links(element_id: str) -> list[tuple[str, str]]:
    parts = [x.strip() for x in element_id.split('|')]
    out = []
    for p in parts:
        if ':' in p:
            lbl, u = p.split(':', 1)
            out.append((lbl.strip(), u.strip()))
    return out

# Hauptlogik
def main():
    #kein Meta File --> Abbruch
    if not os.path.exists(META_FILE):
        logging.error(f"Meta-Datei nicht gefunden: {META_FILE}")
        return
    try:
        entries = json.load(open(META_FILE, 'r', encoding='utf-8'))
    except Exception as e:
        logging.error(f"Konnte Meta laden: {e}")
        return
    
    #bereits gescrapte Elemente überspringen
    processed = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for rec in json.load(f):
                    processed.add(rec.get('ElementID'))
        except Exception:
            pass


    for ent in entries:
        eid = ent.get('ElementID', '')
        
        #wenn ID bereits vorhanden wird das Element übersprungen. 
        if eid in processed:
            logging.info(f"Bereits extrahiert, überspringe {eid}")
            continue

        filters = ent.get('FilterDetails', [])
        logging.info(f"Verarbeite {eid} mit Filtern {filters}")
        parts = []
        # alle Links parsen
        for lbl, url in parse_links(eid):
            # in extract_text sind jetzt:
            # 1) >300-Seiten-Logik
            # 2) Filter-Seiten-Logik
            # 3) Multi-Column-Logik, falls is_plenar=True
            txt = extract_text(url, filters)
            parts.append(f"{lbl}: {txt}")


        ent['ExtrahierterText'] = '\n\n'.join(parts)
        buffer.append(ent)
        if len(buffer) >= BUFFER_SIZE:
            # flush wie gehabt
            data = []
            if os.path.exists(OUTPUT_FILE):
                try:
                    data = json.load(open(OUTPUT_FILE, 'r', encoding='utf-8'))
                except Exception:
                    data = []
            data.extend(buffer)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Flushed {len(buffer)} records to {OUTPUT_FILE}")
            buffer.clear()

    # letzte Reste flushen
    if buffer:
        data = []
        if os.path.exists(OUTPUT_FILE):
            try:
                data = json.load(open(OUTPUT_FILE, 'r', encoding='utf-8'))
            except Exception:
                data = []
        data.extend(buffer)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Flushed {len(buffer)} records to {OUTPUT_FILE}")
        buffer.clear()

    logging.info("PDF-Extraktion abgeschlossen.")

if __name__ == '__main__':
    main()
