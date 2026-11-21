# Embeddings demo (Zoho content)

Two small Python demos for the Nexlevr AI Knowledge Systems beginner track.

## 1. `embeddings_demo.py` (Task 2)
Embeds 12 Zoho blog titles, prints the most similar pairs, runs a meaning-based search and draws a 2D map (`embeddings_map.png`).

## 2. `search_compare.py` (Task 3)
Compares keyword search (BM25) with embedding search on 50 Zoho documents (blog titles and help-centre FAQ questions). It prints the top 3 results for each method and saves them to `results.md`.

## How to run
```
pip install sentence-transformers scikit-learn matplotlib
python embeddings_demo.py
python search_compare.py
```
The first run downloads the `all-MiniLM-L6-v2` model (about 90 MB).

## Notes
- Documents are short titles and questions, so this is a small demo, not a benchmark.
- In the 2D map, 384 numbers are squashed to 2 dimensions, so distances are approximate.
