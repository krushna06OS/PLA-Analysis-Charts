Setup:
1. pip install -r requirements.txt
2. streamlit run app.py

Usage:
1. Run queries from queries.txt on BQ and save output in a Google spreadsheet.
2. Export the sheet(s) as .xlsx.
3. Upload the xlsx file in the Streamlit UI.

Features:
- Auto-detects sheet type: RR, Latency, or Cache Hit
- Filters by client, page type (RR), and server(s)
- Timeframe selection
- Charts with searchable labels
- Pattern Analysis tab (beta-v2):
  - Abnormal patterns and abrupt spikes/downs
  - Per-group analysis:
    - RR: marketplace_client_id + page_type
    - Latency: marketplace_client_id + f_pt
    - Cache Hit: mcid + f_pt + c_type
  - Server-by-server comparison vs beta-v2
  - Summary metrics, top deltas, consecutive windows
  - CSV export for flagged rows
  - Overlay plots (beta-v2 vs each server)
  - Delta ignore threshold to suppress small differences
