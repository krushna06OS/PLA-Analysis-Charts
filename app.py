import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def load_excel_sheet(file_bytes, sheet_name):
    df = pd.read_excel(file_bytes, sheet_name=sheet_name)
    return df


@st.cache_data
def get_sheet_names(file_bytes):
    xls = pd.ExcelFile(file_bytes)
    return xls.sheet_names


def detect_sheet_type_from_columns(columns):
    cols = set(columns)
    if "response_ratio" in cols:
        return "RR"
    if "latency_p99" in cols:
        return "Latency"
    if "hit_ratio" in cols:
        return "Cache Hit"
    return "Unknown"


@st.cache_data
def detect_sheet_type_in_workbook(file_bytes, sheet_name):
    df = pd.read_excel(file_bytes, sheet_name=sheet_name, nrows=5)
    return detect_sheet_type_from_columns(df.columns)


# ---------------------------
# RR Chart Generator
# ---------------------------
def generate_rr_charts(df, start_time=None, end_time=None):
    st.subheader("📊 Response Ratio Charts")

    # Convert time column
    df['interval_15_min_str'] = df['interval_15_min'].astype(str)
    df['interval_15_min'] = pd.to_datetime(df['interval_15_min'], utc=True).dt.tz_convert(None)

    # Keep only strict 15-minute boundaries
    df = df[
        (df['interval_15_min'].dt.minute % 15 == 0)
        & (df['interval_15_min'].dt.second == 0)
        & (df['interval_15_min'].dt.microsecond == 0)
    ]

    if start_time and end_time:
        df = df[(df['interval_15_min'] >= start_time) & (df['interval_15_min'] <= end_time)]

    if df.empty:
        st.warning("No data points found on strict 15-minute boundaries for the selected range.")
        return

    clients = df['marketplace_client_id'].unique()

    for client in clients:
        client_df = df[df['marketplace_client_id'] == client]

        page_types = client_df['page_type'].unique()

        for page in page_types:
            subset = client_df[client_df['page_type'] == page]

            servers = ", ".join(map(str, subset['server'].unique()))
            st.markdown(f"**Chart Label:** `RR | client={client} | page={page} | servers={servers}`")

            plt.figure(figsize=(14, 5))

            for server in subset['server'].unique():
                server_df = subset[subset['server'] == server]

                server_df = server_df.sort_values('interval_15_min')

                plt.plot(
                    server_df['interval_15_min_str'],
                    server_df['response_ratio'],
                    marker='o',
                    label=server
                )

            plt.title(f"Client: {client} | Page: {page}")
            plt.xlabel("Interval (15 min)")
            plt.ylabel("Response Ratio")
            plt.legend()
            ax = plt.gca()
            ticks = ax.get_xticks()
            if len(ticks) > 12:
                step = max(1, len(ticks) // 12)
                ax.set_xticks(ticks[::step])
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()

            st.pyplot(plt)
            plt.clf()


# ---------------------------
# Latency Chart Generator
# ---------------------------
def generate_latency_charts(df, start_time=None, end_time=None):
    st.subheader("⏱️ Latency Charts (p95 / p99)")

    df['interval_15_min_str'] = df['interval_15_min'].astype(str)
    df['interval_15_min'] = pd.to_datetime(df['interval_15_min'], utc=True).dt.tz_convert(None)

    # Keep only strict 15-minute boundaries
    df = df[
        (df['interval_15_min'].dt.minute % 15 == 0)
        & (df['interval_15_min'].dt.second == 0)
        & (df['interval_15_min'].dt.microsecond == 0)
    ]

    if start_time and end_time:
        df = df[(df['interval_15_min'] >= start_time) & (df['interval_15_min'] <= end_time)]

    if df.empty:
        st.warning("No data points found on strict 15-minute boundaries for the selected range.")
        return

    clients = df['marketplace_client_id'].unique()

    for client in clients:
        client_df = df[df['marketplace_client_id'] == client]
        f_pts = client_df['f_pt'].unique()

        for fpt in f_pts:
            subset = client_df[client_df['f_pt'] == fpt]
            servers = ", ".join(map(str, subset['server'].unique()))
            st.markdown(f"**Chart Label:** `Latency | client={client} | f_pt={fpt} | servers={servers}`")
            plt.figure(figsize=(14, 5))

            for server in subset['server'].unique():
                server_df = subset[subset['server'] == server].sort_values('interval_15_min')
                plt.plot(
                    server_df['interval_15_min_str'],
                    server_df['latency_p95'],
                    marker='o',
                    label=f"{server} p95"
                )
                plt.plot(
                    server_df['interval_15_min_str'],
                    server_df['latency_p99'],
                    marker='x',
                    linestyle='--',
                    label=f"{server} p99"
                )

            plt.title(f"Client: {client} | f_pt: {fpt}")
            plt.xlabel("Interval (15 min)")
            plt.ylabel("Latency (ms)")
            plt.legend()
            ax = plt.gca()
            ticks = ax.get_xticks()
            if len(ticks) > 12:
                step = max(1, len(ticks) // 12)
                ax.set_xticks(ticks[::step])
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()

            st.pyplot(plt)
            plt.clf()


# ---------------------------
# Cache Hit Chart Generator
# ---------------------------
def generate_cache_hit_charts(df, start_time=None, end_time=None):
    st.subheader("🧩 Cache Hit Charts (Hit Ratio)")

    # Build a timestamp from date + hour
    df['date_str'] = df['date'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)
    df['timestamp'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    df['timestamp_str'] = df['date_str'] + " " + df['hour'].astype(str).str.zfill(2) + ":00:00"

    if start_time and end_time:
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    if df.empty:
        st.warning("No data points found for the selected range.")
        return

    clients = df['mcid'].unique()

    for client in clients:
        client_df = df[df['mcid'] == client]
        f_pts = client_df['f_pt'].unique()

        for fpt in f_pts:
            fpt_df = client_df[client_df['f_pt'] == fpt]
            c_types = fpt_df['c_type'].unique()

            for ctype in c_types:
                subset = fpt_df[fpt_df['c_type'] == ctype]

                servers = ", ".join(map(str, subset['server'].unique()))
                st.markdown(
                    f"**Chart Label:** `CacheHit | mcid={client} | f_pt={fpt} | c_type={ctype} | servers={servers}`"
                )

                plt.figure(figsize=(14, 5))

                for server in subset['server'].unique():
                    server_df = subset[subset['server'] == server].sort_values('timestamp')
                    plt.plot(
                        server_df['timestamp_str'],
                        server_df['hit_ratio'],
                        marker='o',
                        label=server
                    )

                plt.title(f"MCID: {client} | f_pt: {fpt} | c_type: {ctype}")
                plt.xlabel("Time (hour)")
                plt.ylabel("Hit Ratio")
                plt.legend()
                ax = plt.gca()
                ticks = ax.get_xticks()
                if len(ticks) > 12:
                    step = max(1, len(ticks) // 12)
                    ax.set_xticks(ticks[::step])
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.tight_layout()

                st.pyplot(plt)
                plt.clf()


# ---------------------------
# Main App
# ---------------------------
st.title("📈 Excel Analytics Dashboard")

uploaded_file = st.file_uploader(
    "Upload sheet.xlsx",
    type=["xlsx", "xls"]
)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    try:
        sheet_names = get_sheet_names(file_bytes)
    except Exception as e:
        st.error(f"Error reading workbook: {e}")
        sheet_names = []

    if sheet_names:
        # Try to auto-detect the best sheet
        detected_types = {}
        for name in sheet_names:
            try:
                detected_types[name] = detect_sheet_type_in_workbook(file_bytes, name)
            except Exception:
                detected_types[name] = "Unknown"

        preferred = [n for n, t in detected_types.items() if t != "Unknown"]
        default_index = 0
        if len(preferred) == 1:
            default_index = sheet_names.index(preferred[0])

        sheet_name = st.selectbox("Select Sheet", sheet_names, index=default_index)

        try:
            df = load_excel_sheet(file_bytes, sheet_name)

            st.write("Preview Data:", df.head())

            # Detect type
            sheet_type = detect_sheet_type_from_columns(df.columns)

            st.info(f"Detected Sheet Type: {sheet_type}")

            if sheet_type == "RR":
                client_options = ["All"] + sorted(df['marketplace_client_id'].unique().tolist())
                selected_client = st.selectbox("Client ID", client_options, key="rr_client")
                if selected_client != "All":
                    df = df[df['marketplace_client_id'] == selected_client]

                page_options = ["All"] + sorted(df['page_type'].unique().tolist())
                selected_page = st.selectbox("Page Type", page_options, key="rr_page")
                if selected_page != "All":
                    df = df[df['page_type'] == selected_page]

                server_options = sorted(df['server'].unique().tolist())
                selected_servers = st.multiselect(
                    "Servers",
                    server_options,
                    default=server_options,
                    key="rr_server"
                )
                if selected_servers:
                    df = df[df['server'].isin(selected_servers)]

                df['interval_15_min_str'] = df['interval_15_min'].astype(str)
                df['interval_15_min'] = pd.to_datetime(df['interval_15_min'], utc=True).dt.tz_convert(None)
                min_ts = df['interval_15_min'].min()
                max_ts = df['interval_15_min'].max()

                if pd.notna(min_ts) and pd.notna(max_ts):
                    default_start = min_ts
                    default_end = max_ts

                    start_date = st.date_input("Start Date", value=default_start.date())
                    start_time = st.time_input("Start Time", value=default_start.time())
                    end_date = st.date_input("End Date", value=default_end.date())
                    end_time = st.time_input("End Time", value=default_end.time())

                    start_dt = pd.to_datetime(f"{start_date} {start_time}")
                    end_dt = pd.to_datetime(f"{end_date} {end_time}")

                    if start_dt > end_dt:
                        st.error("Start must be before end.")
                    else:
                        generate_rr_charts(df, start_dt, end_dt)
                else:
                    generate_rr_charts(df)
            elif sheet_type == "Latency":
                client_options = ["All"] + sorted(df['marketplace_client_id'].unique().tolist())
                selected_client = st.selectbox("Client ID", client_options, key="lat_client")
                if selected_client != "All":
                    df = df[df['marketplace_client_id'] == selected_client]

                server_options = sorted(df['server'].unique().tolist())
                selected_servers = st.multiselect(
                    "Servers",
                    server_options,
                    default=server_options,
                    key="lat_server"
                )
                if selected_servers:
                    df = df[df['server'].isin(selected_servers)]

                df['interval_15_min_str'] = df['interval_15_min'].astype(str)
                df['interval_15_min'] = pd.to_datetime(df['interval_15_min'], utc=True).dt.tz_convert(None)
                min_ts = df['interval_15_min'].min()
                max_ts = df['interval_15_min'].max()

                if pd.notna(min_ts) and pd.notna(max_ts):
                    default_start = min_ts
                    default_end = max_ts

                    start_date = st.date_input("Start Date", value=default_start.date(), key="lat_start_date")
                    start_time = st.time_input("Start Time", value=default_start.time(), key="lat_start_time")
                    end_date = st.date_input("End Date", value=default_end.date(), key="lat_end_date")
                    end_time = st.time_input("End Time", value=default_end.time(), key="lat_end_time")

                    start_dt = pd.to_datetime(f"{start_date} {start_time}")
                    end_dt = pd.to_datetime(f"{end_date} {end_time}")

                    if start_dt > end_dt:
                        st.error("Start must be before end.")
                    else:
                        generate_latency_charts(df, start_dt, end_dt)
                else:
                    generate_latency_charts(df)
            elif sheet_type == "Cache Hit":
                client_options = ["All"] + sorted(df['mcid'].unique().tolist())
                selected_client = st.selectbox("Client ID", client_options, key="ch_client")
                if selected_client != "All":
                    df = df[df['mcid'] == selected_client]

                server_options = sorted(df['server'].unique().tolist())
                selected_servers = st.multiselect(
                    "Servers",
                    server_options,
                    default=server_options,
                    key="ch_server"
                )
                if selected_servers:
                    df = df[df['server'].isin(selected_servers)]

                df['date_str'] = df['date'].astype(str)
                df['date'] = pd.to_datetime(df['date'])
                df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)
                df['timestamp'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
                df['timestamp_str'] = df['date_str'] + " " + df['hour'].astype(str).str.zfill(2) + ":00:00"
                min_ts = df['timestamp'].min()
                max_ts = df['timestamp'].max()

                if pd.notna(min_ts) and pd.notna(max_ts):
                    default_start = min_ts
                    default_end = max_ts

                    start_date = st.date_input("Start Date", value=default_start.date(), key="ch_start_date")
                    start_time = st.time_input("Start Time", value=default_start.time(), key="ch_start_time")
                    end_date = st.date_input("End Date", value=default_end.date(), key="ch_end_date")
                    end_time = st.time_input("End Time", value=default_end.time(), key="ch_end_time")

                    start_dt = pd.to_datetime(f"{start_date} {start_time}")
                    end_dt = pd.to_datetime(f"{end_date} {end_time}")

                    if start_dt > end_dt:
                        st.error("Start must be before end.")
                    else:
                        generate_cache_hit_charts(df, start_dt, end_dt)
                else:
                    generate_cache_hit_charts(df)

            else:
                st.warning("Unknown sheet type. Expected RR, Latency, or Cache Hit columns.")

        except Exception as e:
            st.error(f"Error loading sheet: {e}")
