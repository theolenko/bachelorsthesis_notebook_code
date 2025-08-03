# 28.07.25
# preprocessing.py
# collection of preprocessing functions and helpers for cleaning text data in JSON or CSV files using the clean_text function and Segmenting.
# Supports both file output and in-memory return.

import json
import pandas as pd
import os
from src.cleaning import clean_text

def process_file(
    filepath: str,
    mode: str = "basic",
    file_format: str = "json",
    text_key: str = "ExtrahierterText",
    output_mode: str = "memory"
):
    """
    Clean the text in a JSON or CSV file using clean_text and either save the result or return it.

    Args:
        filepath (str): Path to the file to process.
        mode (str): Cleaning mode, "basic" or "extended".
        file_format (str): File type, "json" or "csv".
        text_key (str): Key/column with text to clean.
        output_mode (str): "file" to write output, "memory" to return cleaned data.
    Returns:
        Cleaned data (if output_mode is "memory"). (pd.DataFrame for json and csv)
    Raises:
        ValueError: If file_format is not supported.
    """
    # Only allow supported file formats and output_modes
    if file_format not in ("json", "csv"):
        raise ValueError("file_format must be 'json' or 'csv'")
    if output_mode not in ("file", "memory"):
        raise ValueError("output_mode must be 'file' or 'memory'")

    # JSON: clean each entry's text_key and write or return
    if file_format == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if output_mode == "file":
            # In-place im JSON-Objekt bereinigen und zurückschreiben
            for entry in data:
                if text_key in entry:
                    entry[text_key] = clean_text(entry[text_key], mode=mode)
            base, ext = filepath.rsplit('.', 1)
            new_path = f"{base}_cleaned.{ext}"
            with open(new_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else: #"memory"
            # DataFrame zurückliefern und EINMAL bereinigen
            df = pd.json_normalize(data)
            df[text_key] = df[text_key].astype(str).map(lambda txt: clean_text(txt, mode=mode))
            return df

    # CSV: clean the text_key column and write or return
    else:
        df = pd.read_csv(filepath)
        if text_key in df.columns:
            df[text_key] = df[text_key].astype(str).apply(lambda x: clean_text(x, mode=mode))
        if output_mode == "file":
            base, ext = filepath.rsplit('.', 1)
            new_path = f"{base}_cleaned.{ext}"
            df.to_csv(new_path, index=False)
        else:
            return df
        

# segment_documents function
# This function splits a list of JSON-like documents into segments based on token length 28.07.25
def segment_documents(
    documents: list[dict],
    text_key: str = "ExtrahierterText",
    min_tokens: int = 10,
    max_tokens: int = 15000,
    segment_size: int = 1000,
    min_segment_tokens: int = 30,
    verbose: bool = True
) -> list[dict]:
    """
    Split a list of JSON-like documents into filtered and size-limited segments for further processing.

    This function is intended for use with lists of dictionaries (e.g., loaded from JSON files),
    not for DataFrames or CSV data. Each document is filtered by minimum and maximum token length,
    then split into segments of up to segment_size tokens. Segments with too few tokens are removed.
    Functionality based on findings from exploratory_analysis.ipynb Notebook 28.07.25

    Args:
        documents (list[dict]): List of documents (dicts) to Segment and pd.DataFrame (will be converted to list of dicts internally).
        text_key (str): Key in each dict containing the text to split and filter.
        min_tokens (int): Minimum number of tokens required for a document to be kept.
        max_tokens (int): Maximum number of tokens allowed for a document to be kept.
        segment_size (int): Maximum number of tokens per Segment.
        min_segment_tokens (int): Minimum number of tokens required for a Segment to be kept.
        verbose (bool): If True, prints summary statistics after processing.

    Returns:
        pf.DataFrame: DataFrame containing the filtered and segmented documents.

    Note:
        This function cannot be used with CSVs or pandas DataFrames. It is only for lists of dicts (JSON-like data).
        Usage example:
        from src.io_utils import process_file, segment_documents
        cleaned = process_file("raw.json", mode="basic", output_mode="memory")
        segments = segment_documents(cleaned)

    """
    #also support for df input, transform it to list of dicts
    if isinstance(documents, pd.DataFrame):
        documents = documents.to_dict(orient="records")

    # Initialize counters and Segment list
    removed_empty = 0
    removed_too_short = 0
    removed_too_long = 0
    kept_docs = 0
    segments = []

    # Iterate through all documents and apply filtering and segmenting
    for entry in documents:
        # Get and clean the text for the current document
        text = entry.get(text_key, "").strip()
        if not text:
            # Skip and count documents with empty text
            removed_empty += 1
            continue
        tokens = text.split()
        token_count = len(tokens)
        if token_count <= min_tokens:
            # Skip and count documents that are too short
            removed_too_short += 1
            continue
        elif token_count > max_tokens:
            # Skip and count documents that are too long
            removed_too_long += 1
            continue
        # Document passes all filters and will be kept
        kept_docs += 1
        # Calculate how many segments are needed for this document
        num_segments = (token_count + segment_size - 1) // segment_size
        for i in range(num_segments):
            # Determine the start and end indices for the current Segment
            start = i * segment_size
            end = min(start + segment_size, token_count)
            segment_tokens = tokens[start:end]
            # Create a Segment dictionary with relevant metadata and Segment text
            segments.append({
                "ID": entry["ID"],
                "SegmentID": f"{entry['ID']}_{i+1}",
                "Landtag": entry["Landtag"],
                "Datum": entry["Datum"],
                "Beschreibungstext": entry.get("Beschreibungstext", ""),
                "FilterDetails": entry.get("FilterDetails", []),
                "Links": entry.get("Links", []),
                "SegmentText": " ".join(segment_tokens),
                "SegmentTokenLength": len(segment_tokens)
            })

    # Final small-Segment filter: remove segments with too few tokens
    filtered_segments = [
        s for s in segments
        if len(s["SegmentText"].split()) >= min_segment_tokens
    ]

    # Print summary statistics
    if verbose:
        #calculate variables for summary
        total_docs = len(documents)
        total_segments = len(segments)
        average_segments = total_segments / kept_docs if kept_docs else 0
        removed_small = total_segments - len(filtered_segments)

        #output summary
        print("===== Summary =====")
        print(f"Total original documents: {total_docs}")
        print(f"Removed (empty): {removed_empty}")
        print(f"Removed (≤ {min_tokens} tokens): {removed_too_short}")
        print(f"Removed (> {max_tokens} tokens): {removed_too_long}")
        print(f"Remaining valid documents: {kept_docs}")
        print(f"Total segments generated: {total_segments}")
        print(f"Average segments per document: {average_segments:.2f}")
        print(f"Removed (TokenLength < {min_segment_tokens}): {removed_small}")
        print(f"Remaining segments: {len(filtered_segments)}")

    return pd.DataFrame(filtered_segments)


#30.07.25 for checking if in-memory data matches file, maybe delete later on
def compare_data(a, b, fmt=None):
    """
    Compare two data sources of the same type: JSON file path, CSV file path,
    in‐memory list[dict], or pandas.DataFrame. Prints first diff or “identical.”
    Returns True if identical, False otherwise.

    Args:
        a, b: Either file paths (str) ending in .json or .csv, or in‐memory data:
              list[dict] for JSON-like, or pd.DataFrame for CSV-like.
        fmt (str, optional): Force format "json" or "csv". If None, inferred.

    Example usage:
        compare_data("clean1.json", "clean2.json")
        compare_data(df1, df2)
        compare_data(cleaned_list, "cleaned.json")
    """
    # Helper to load or serialize to text
    def to_text(x, fmt):
        if isinstance(x, str) and os.path.isfile(x):
            # path
            with open(x, "r", encoding="utf-8") as f:
                return f.read()
        elif fmt == "json":
            # list or dataframe -> JSON text
            if isinstance(x, pd.DataFrame):
                obj = json.loads(x.to_json(orient="records", force_ascii=False))
            else:
                obj = x
            return json.dumps(obj, ensure_ascii=False, indent=2)
        elif fmt == "csv":
            # dataframe or list -> CSV text
            if isinstance(x, str):
                # path fallback
                with open(x, "r", encoding="utf-8") as f:
                    return f.read()
            elif isinstance(x, pd.DataFrame):
                return x.to_csv(index=False)
            else:
                # list of dicts -> DataFrame
                df = pd.DataFrame(x)
                return df.to_csv(index=False)
        else:
            raise ValueError(f"Cannot serialize object of type {type(x)} as {fmt}")

    # Infer format if not provided
    def infer_format(x):
        if isinstance(x, str) and os.path.isfile(x):
            _, ext = os.path.splitext(x)
            return ext.lower().lstrip(".")
        elif isinstance(x, pd.DataFrame):
            return "csv"
        elif isinstance(x, list):
            return "json"
        else:
            raise ValueError(f"Cannot infer format for object of type {type(x)}")

    fmt_a = fmt or infer_format(a)
    fmt_b = fmt or infer_format(b)
    if fmt_a != fmt_b:
        raise ValueError(f"Formats differ: {fmt_a} vs {fmt_b}")
    fmt = fmt_a

    s1 = to_text(a, fmt)
    s2 = to_text(b, fmt)

    if s1 == s2:
        print("Data sources are identical")
        return True

    # find first difference
    min_len = min(len(s1), len(s2))
    for idx in range(min_len):
        if s1[idx] != s2[idx]:
            line = s1.count('\n', 0, idx) + 1
            col = idx - s1.rfind('\n', 0, idx)
            print(f"Difference at idx={idx} (line {line}, col {col}):")
            print(f"  first  has: {s1[idx]!r}")
            print(f"  second has: {s2[idx]!r}")
            break
    else:
        print(f"Match for first {min_len} chars, lengths differ:")
        print(f"  first  length = {len(s1)}")
        print(f"  second length = {len(s2)}")
    return False




