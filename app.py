import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def normalize_server_name(value):
    return str(value).strip().lower().replace("-", "_")


def detect_patterns(series, time_series, window=7, z_thresh=3.0):
    s = pd.to_numeric(series, errors='coerce')
    t = pd.to_datetime(time_series, errors='coerce')
    df = pd.DataFrame({"time": t, "value": s}).dropna().sort_values("time")
    if df.empty or len(df) < 3:
        return df.assign(
            abnormal=False,
            spike_up=False,
            spike_down=False
        )

    rolling_median = df["value"].rolling(window=window, min_periods=3).median()
    rolling_mad = (df["value"] - rolling_median).abs().rolling(window=window, min_periods=3).median()
    rolling_std = df["value"].rolling(window=window, min_periods=3).std()
    rolling_mad = rolling_mad.fillna(0)
    rolling_mad = rolling_mad.where(rolling_mad != 0, rolling_std)

    abnormal = (df["value"] - rolling_median).abs() > (z_thresh * rolling_mad)

    diff = df["value"].diff()
    diff_median = diff.rolling(window=window, min_periods=3).median()
    diff_mad = (diff - diff_median).abs().rolling(window=window, min_periods=3).median()
    diff_std = diff.rolling(window=window, min_periods=3).std()
    diff_mad = diff_mad.fillna(0)
    diff_mad = diff_mad.where(diff_mad != 0, diff_std)

    spike_up = diff > (z_thresh * diff_mad)
    spike_down = diff < (-z_thresh * diff_mad)

    return df.assign(
        abnormal=abnormal.fillna(False),
        spike_up=spike_up.fillna(False),
        spike_down=spike_down.fillna(False)
    )


def compare_with_others(beta_df, other_df, time_col, metric_col, server_col="server", delta_ignore=0.0):
    beta = beta_df[[time_col, metric_col]].dropna()
    others = other_df[[time_col, metric_col, server_col]].dropna()
    if beta.empty or others.empty:
        return {}

    beta_agg = beta.groupby(time_col)[metric_col].median().reset_index()
    results = {}

    for server, group in others.groupby(server_col):
        others_agg = group.groupby(time_col)[metric_col].median().reset_index()
        merged = beta_agg.merge(others_agg, on=time_col, how="inner", suffixes=("_beta", "_other"))
        if merged.empty:
            results[server] = pd.DataFrame(columns=["time", "beta", "other", "delta", "flag"])
            continue

        merged["delta"] = merged[f"{metric_col}_beta"] - merged[f"{metric_col}_other"]
        delta_median = merged["delta"].rolling(window=7, min_periods=3).median()
        delta_mad = (merged["delta"] - delta_median).abs().rolling(window=7, min_periods=3).median()
        delta_std = merged["delta"].rolling(window=7, min_periods=3).std()
        delta_mad = delta_mad.fillna(0)
        delta_mad = delta_mad.where(delta_mad != 0, delta_std)

        baseline = merged[f"{metric_col}_other"].abs().replace(0, 1)
        flag = merged["delta"].abs() > (3 * delta_mad)
        min_delta = pd.Series([delta_ignore] * len(merged))
        flag = flag & (merged["delta"].abs() > min_delta)
        flag = flag & (merged["delta"].abs() > 0.05 * baseline)

        results[server] = pd.DataFrame({
            "time": merged[time_col],
            "beta": merged[f"{metric_col}_beta"],
            "other": merged[f"{metric_col}_other"],
            "delta": merged["delta"],
            "flag": flag.fillna(False)
        })

    return results


def summarize_compare_results(compare_results):
    rows = []
    for server, df in compare_results.items():
        if df.empty:
            continue
        total = len(df)
        flagged = int(df["flag"].sum())
        rel = (df["delta"].abs() / (df["other"].abs() + 1e-9)).median()
        score = 100 - min(100, (flagged / max(total, 1)) * 100 * 0.6 + rel * 100 * 0.4)
        rows.append({
            "server": server,
            "total_points": total,
            "flagged_points": flagged,
            "median_relative_delta": round(rel, 4),
            "health_score_0_100": round(score, 2)
        })
    return pd.DataFrame(rows).sort_values("health_score_0_100", ascending=False)


def top_n_deltas(compare_results, n=5):
    rows = []
    for server, df in compare_results.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["abs_delta"] = tmp["delta"].abs()
        tmp = tmp.sort_values("abs_delta", ascending=False).head(n)
        tmp["server"] = server
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["time", "beta", "other", "delta", "flag", "abs_delta", "server"])
    return pd.concat(rows, ignore_index=True)


def find_flag_windows(df, min_len=3):
    windows = []
    if df.empty:
        return windows
    flags = df["flag"].fillna(False).astype(bool).tolist()
    times = df["time"].tolist()
    deltas = df["delta"].tolist()
    start = None
    for i, is_flag in enumerate(flags):
        if is_flag and start is None:
            start = i
        if (not is_flag or i == len(flags) - 1) and start is not None:
            end = i if not is_flag else i
            length = end - start + 1
            if length >= min_len:
                window_deltas = [abs(x) for x in deltas[start:end + 1]]
                windows.append({
                    "start": times[start],
                    "end": times[end],
                    "length": length,
                    "max_abs_delta": max(window_deltas) if window_deltas else None
                })
            start = None
    return windows

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
    df['interval_15_min'] = pd.to_datetime(df['interval_15_min'], utc=True).dt.tz_convert(None)
    df['interval_15_min_str'] = df['interval_15_min'].dt.strftime('%Y-%m-%d %H:%M')

    # Keep only strict 15-minute boundaries
    df = df[
        (df['interval_15_min'].dt.minute % 15 == 0)
        & (df['interval_15_min'].dt.second == 0)
        & (df['interval_15_min'].dt.microsecond == 0)
    ]

    if start_time and end_time:
        df = df[(df['interval_15_min'] >= start_time) & (df['interval_15_min'] <= end_time)]

    # Remove exact duplicate rows, then dedupe same timestamp per server to avoid zigzag lines
    df = df.drop_duplicates()
    df = df.sort_values('interval_15_min').drop_duplicates(
        subset=['marketplace_client_id', 'page_type', 'server', 'interval_15_min'],
        keep='last'
    )

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
                    server_df['interval_15_min'],
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

    df['interval_15_min'] = pd.to_datetime(df['interval_15_min'], utc=True).dt.tz_convert(None)
    df['interval_15_min_str'] = df['interval_15_min'].dt.strftime('%Y-%m-%d %H:%M')

    # Keep only strict 15-minute boundaries
    df = df[
        (df['interval_15_min'].dt.minute % 15 == 0)
        & (df['interval_15_min'].dt.second == 0)
        & (df['interval_15_min'].dt.microsecond == 0)
    ]

    if start_time and end_time:
        df = df[(df['interval_15_min'] >= start_time) & (df['interval_15_min'] <= end_time)]

    # Remove exact duplicate rows, then dedupe same timestamp per server to avoid zigzag lines
    df = df.drop_duplicates()
    df = df.sort_values('interval_15_min').drop_duplicates(
        subset=['marketplace_client_id', 'f_pt', 'server', 'interval_15_min'],
        keep='last'
    )

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
                    server_df['interval_15_min'],
                    server_df['latency_p95'],
                    marker='o',
                    label=f"{server} p95"
                )
                plt.plot(
                    server_df['interval_15_min'],
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
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)
    df['timestamp'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

    if start_time and end_time:
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    # Remove exact duplicate rows, then dedupe same timestamp per server to avoid zigzag lines
    df = df.drop_duplicates()
    df = df.sort_values('timestamp').drop_duplicates(
        subset=['mcid', 'f_pt', 'c_type', 'server', 'timestamp'],
        keep='last'
    )

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
                        server_df['timestamp'],
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
st.title("📈 PLA AdServer RR, Latency and Cache Hit Analytics Dashboard")

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
            df = df.drop_duplicates()

            st.write("Preview Data:", df.head())

            # Detect type
            sheet_type = detect_sheet_type_from_columns(df.columns)

            st.info(f"Detected Sheet Type: {sheet_type}")
            tabs = st.tabs(["Charts", "Pattern Analysis"])

            with tabs[0]:
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

            with tabs[1]:
                st.subheader("🔎 Pattern Analysis (beta-v2)")
                delta_ignore = st.number_input(
                    "Delta ignore threshold (absolute value)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )

                df['server_norm'] = df['server'].apply(normalize_server_name)
                beta_df = df[df['server_norm'] == "beta_v2"]
                other_df = df[df['server_norm'] != "beta_v2"]

                if beta_df.empty:
                    st.warning("No beta-v2 server data found after filters.")
                else:
                    if sheet_type == "RR":
                        beta_df['interval_15_min'] = pd.to_datetime(beta_df['interval_15_min'], utc=True).dt.tz_convert(None)
                        other_df['interval_15_min'] = pd.to_datetime(other_df['interval_15_min'], utc=True).dt.tz_convert(None)

                        for (client, page), group in beta_df.groupby(["marketplace_client_id", "page_type"]):
                            st.markdown(f"**Client {client} | Page {page}**")
                            patterns = detect_patterns(
                                group['response_ratio'],
                                group['interval_15_min']
                            )
                            st.write("RR: Abnormal points / spikes / drops")
                            st.write(patterns[patterns["abnormal"] | patterns["spike_up"] | patterns["spike_down"]])

                            other_group = other_df[
                                (other_df["marketplace_client_id"] == client)
                                & (other_df["page_type"] == page)
                            ]
                            compare = compare_with_others(
                                group, other_group, "interval_15_min", "response_ratio", delta_ignore=delta_ignore
                            )
                            st.write("RR: beta-v2 vs other servers (flagged deltas)")
                            for server, cmp_df in compare.items():
                                flagged = cmp_df[cmp_df["flag"]]
                                st.write(f"Server: {server}")
                                st.write(flagged)

                            summary = summarize_compare_results(compare)
                            if not summary.empty:
                                st.write("RR: Summary (per server)")
                                st.write(summary)

                            top_deltas = top_n_deltas(compare, n=5)
                            if not top_deltas.empty:
                                st.write("RR: Top 5 deltas per server")
                                st.write(top_deltas)

                            windows_rows = []
                            for server, cmp_df in compare.items():
                                for w in find_flag_windows(cmp_df, min_len=3):
                                    windows_rows.append({
                                        "server": server,
                                        "start": w["start"],
                                        "end": w["end"],
                                        "length": w["length"],
                                        "max_abs_delta": w["max_abs_delta"]
                                    })
                            if windows_rows:
                                st.write("RR: Consecutive flagged windows (len >= 3)")
                                st.write(pd.DataFrame(windows_rows))

                            flagged_rows = []
                            for server, cmp_df in compare.items():
                                tmp = cmp_df[cmp_df["flag"]].copy()
                                if not tmp.empty:
                                    tmp["server"] = server
                                    flagged_rows.append(tmp)
                            if flagged_rows:
                                flagged_df = pd.concat(flagged_rows, ignore_index=True)
                                st.download_button(
                                    "Download RR flagged rows (CSV)",
                                    flagged_df.to_csv(index=False),
                                    file_name="rr_beta_v2_flagged.csv",
                                    mime="text/csv",
                                    key=f"rr_download_{client}_{page}"
                                )

                            st.write("RR: Overlay plots (beta-v2 vs each server)")
                            for server, cmp_df in compare.items():
                                if cmp_df.empty:
                                    continue
                                with st.expander(f"Overlay: {server}"):
                                    plt.figure(figsize=(14, 5))
                                    plt.plot(cmp_df["time"], cmp_df["beta"], label="beta-v2", marker="o")
                                    plt.plot(cmp_df["time"], cmp_df["other"], label=server, marker="x")
                                    plt.xlabel("Time")
                                    plt.ylabel("Response Ratio")
                                    plt.legend()
                                    plt.xticks(rotation=45, ha='right', fontsize=8)
                                    plt.tight_layout()
                                    st.pyplot(plt)
                                    plt.clf()

                    elif sheet_type == "Latency":
                        beta_df['interval_15_min'] = pd.to_datetime(beta_df['interval_15_min'], utc=True).dt.tz_convert(None)
                        other_df['interval_15_min'] = pd.to_datetime(other_df['interval_15_min'], utc=True).dt.tz_convert(None)

                        for (client, fpt), group in beta_df.groupby(["marketplace_client_id", "f_pt"]):
                            st.markdown(f"**Client {client} | f_pt {fpt}**")
                            other_group = other_df[
                                (other_df["marketplace_client_id"] == client)
                                & (other_df["f_pt"] == fpt)
                            ]
                            for metric in ["latency_p95", "latency_p99"]:
                                if metric in group.columns:
                                    patterns = detect_patterns(
                                        group[metric],
                                        group['interval_15_min']
                                    )
                                    st.write(f"{metric}: Abnormal points / spikes / drops")
                                    st.write(patterns[patterns["abnormal"] | patterns["spike_up"] | patterns["spike_down"]])

                                    compare = compare_with_others(
                                        group, other_group, "interval_15_min", metric, delta_ignore=delta_ignore
                                    )
                                    st.write(f"{metric}: beta-v2 vs other servers (flagged deltas)")
                                    for server, cmp_df in compare.items():
                                        flagged = cmp_df[cmp_df["flag"]]
                                        st.write(f"Server: {server}")
                                        st.write(flagged)

                                    summary = summarize_compare_results(compare)
                                    if not summary.empty:
                                        st.write(f"{metric}: Summary (per server)")
                                        st.write(summary)

                                    top_deltas = top_n_deltas(compare, n=5)
                                    if not top_deltas.empty:
                                        st.write(f"{metric}: Top 5 deltas per server")
                                        st.write(top_deltas)

                                    windows_rows = []
                                    for server, cmp_df in compare.items():
                                        for w in find_flag_windows(cmp_df, min_len=3):
                                            windows_rows.append({
                                                "server": server,
                                                "start": w["start"],
                                                "end": w["end"],
                                                "length": w["length"],
                                                "max_abs_delta": w["max_abs_delta"]
                                            })
                                    if windows_rows:
                                        st.write(f"{metric}: Consecutive flagged windows (len >= 3)")
                                        st.write(pd.DataFrame(windows_rows))

                                    flagged_rows = []
                                    for server, cmp_df in compare.items():
                                        tmp = cmp_df[cmp_df["flag"]].copy()
                                        if not tmp.empty:
                                            tmp["server"] = server
                                            flagged_rows.append(tmp)
                                    if flagged_rows:
                                        flagged_df = pd.concat(flagged_rows, ignore_index=True)
                                        st.download_button(
                                            f"Download {metric} flagged rows (CSV)",
                                            flagged_df.to_csv(index=False),
                                            file_name=f"{metric}_beta_v2_flagged.csv",
                                            mime="text/csv",
                                            key=f"{metric}_download_{client}_{fpt}"
                                        )

                                    st.write(f"{metric}: Overlay plots (beta-v2 vs each server)")
                                    for server, cmp_df in compare.items():
                                        if cmp_df.empty:
                                            continue
                                        with st.expander(f"Overlay: {server}"):
                                            plt.figure(figsize=(14, 5))
                                            plt.plot(cmp_df["time"], cmp_df["beta"], label="beta-v2", marker="o")
                                            plt.plot(cmp_df["time"], cmp_df["other"], label=server, marker="x")
                                            plt.xlabel("Time")
                                            plt.ylabel(metric)
                                            plt.legend()
                                            plt.xticks(rotation=45, ha='right', fontsize=8)
                                            plt.tight_layout()
                                            st.pyplot(plt)
                                            plt.clf()

                    elif sheet_type == "Cache Hit":
                        beta_df['date'] = pd.to_datetime(beta_df['date'])
                        beta_df['hour'] = pd.to_numeric(beta_df['hour'], errors='coerce').fillna(0).astype(int)
                        beta_df['timestamp'] = beta_df['date'] + pd.to_timedelta(beta_df['hour'], unit='h')

                        other_df['date'] = pd.to_datetime(other_df['date'])
                        other_df['hour'] = pd.to_numeric(other_df['hour'], errors='coerce').fillna(0).astype(int)
                        other_df['timestamp'] = other_df['date'] + pd.to_timedelta(other_df['hour'], unit='h')

                        for (mcid, fpt, ctype), group in beta_df.groupby(["mcid", "f_pt", "c_type"]):
                            st.markdown(f"**MCID {mcid} | f_pt {fpt} | c_type {ctype}**")
                            other_group = other_df[
                                (other_df["mcid"] == mcid)
                                & (other_df["f_pt"] == fpt)
                                & (other_df["c_type"] == ctype)
                            ]
                            patterns = detect_patterns(
                                group['hit_ratio'],
                                group['timestamp']
                            )
                            st.write("Hit Ratio: Abnormal points / spikes / drops")
                            st.write(patterns[patterns["abnormal"] | patterns["spike_up"] | patterns["spike_down"]])

                            compare = compare_with_others(
                                group, other_group, "timestamp", "hit_ratio", delta_ignore=delta_ignore
                            )
                            st.write("Hit Ratio: beta-v2 vs other servers (flagged deltas)")
                            for server, cmp_df in compare.items():
                                flagged = cmp_df[cmp_df["flag"]]
                                st.write(f"Server: {server}")
                                st.write(flagged)

                            summary = summarize_compare_results(compare)
                            if not summary.empty:
                                st.write("Hit Ratio: Summary (per server)")
                                st.write(summary)

                            top_deltas = top_n_deltas(compare, n=5)
                            if not top_deltas.empty:
                                st.write("Hit Ratio: Top 5 deltas per server")
                                st.write(top_deltas)

                            windows_rows = []
                            for server, cmp_df in compare.items():
                                for w in find_flag_windows(cmp_df, min_len=3):
                                    windows_rows.append({
                                        "server": server,
                                        "start": w["start"],
                                        "end": w["end"],
                                        "length": w["length"],
                                        "max_abs_delta": w["max_abs_delta"]
                                    })
                            if windows_rows:
                                st.write("Hit Ratio: Consecutive flagged windows (len >= 3)")
                                st.write(pd.DataFrame(windows_rows))

                            flagged_rows = []
                            for server, cmp_df in compare.items():
                                tmp = cmp_df[cmp_df["flag"]].copy()
                                if not tmp.empty:
                                    tmp["server"] = server
                                    flagged_rows.append(tmp)
                            if flagged_rows:
                                flagged_df = pd.concat(flagged_rows, ignore_index=True)
                                st.download_button(
                                    "Download hit_ratio flagged rows (CSV)",
                                    flagged_df.to_csv(index=False),
                                    file_name="hit_ratio_beta_v2_flagged.csv",
                                    mime="text/csv",
                                    key=f"hit_ratio_download_{mcid}_{fpt}_{ctype}"
                                )

                            st.write("Hit Ratio: Overlay plots (beta-v2 vs each server)")
                            for server, cmp_df in compare.items():
                                if cmp_df.empty:
                                    continue
                                with st.expander(f"Overlay: {server}"):
                                    plt.figure(figsize=(14, 5))
                                    plt.plot(cmp_df["time"], cmp_df["beta"], label="beta-v2", marker="o")
                                    plt.plot(cmp_df["time"], cmp_df["other"], label=server, marker="x")
                                    plt.xlabel("Time")
                                    plt.ylabel("Hit Ratio")
                                    plt.legend()
                                    plt.xticks(rotation=45, ha='right', fontsize=8)
                                    plt.tight_layout()
                                    st.pyplot(plt)
                                    plt.clf()

        except Exception as e:
            st.error(f"Error loading sheet: {e}")
