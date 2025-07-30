import json
import pandas as pd

# JSON-Datei einlesen
meta_data = 'SachsenAnhalt_Meta.json'
all_Data = 'SachsenAnhalt_Data.json'
with open(all_Data, "r", encoding="utf-8") as file:
    data = json.load(file)  # JSON in eine Python-Liste laden

all_elements = len(data)
print(f"Gesamtanzahl {all_elements}")

# Excel-Datei einlesen (angenommen, die Filterwörter stehen in der Spalte "Filter" im Sheet "FilterWords")
excel_path = "SachsenAnhalt_FilterWords.xlsx"
filter_words_df = pd.read_excel(excel_path, sheet_name="FilterWords")
filter_words = filter_words_df["Filter"].dropna().tolist()

# Erstelle ein Dictionary, das für jedes Filterwort die Anzahl der zugehörigen Ergebnisse zählt
results_per_filter = {word: 0 for word in filter_words}

# Durchlaufe alle Einträge in der JSON-Liste
for entry in data:
    # Wir gehen davon aus, dass in jedem Eintrag unter "FilterDetails" eine Liste mit den verwendeten Filterwörtern gespeichert ist
    filter_details = entry.get("FilterDetails", [])
    for word in filter_details:
        if word in results_per_filter:
            results_per_filter[word] += 1

# Ausgabe: Für jedes Filterwort die Anzahl der Ergebnisse
for word, count in results_per_filter.items():
    print(f"Filterwort '{word}': {count} Ergebnisse")