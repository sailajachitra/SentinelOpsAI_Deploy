"""
SentinelOps AI - Attack pattern graphs (parameter based, one model, no model comparison)

Same visual style as plot_attack_activity():
  left axis  = blue, solid line, circle markers
  right axis = red, dashed line, square markers
  figsize (14, 6), dashed grid, rotated time ticks, combined legend.

The graphs are built from the LIVE event table (see app.py -> events_dataframe),
so every event you add or import is included the next time they are drawn.
Events that were entered without a ground-truth label (ML-scored events) are
drawn as gold stars so you can see exactly where a new event landed.

Patterns and parameters
  1. Brute force            : failed_login_count, event_type, event_hour, src_ip
  2. Port scanning          : dst_port, protocol, connection_count_5min
  3. Unusual data transfer  : bytes_sent, total_bytes_received, bytes_per_second
  4. Suspicious privilege   : event_type, protocol, connection_count_5min
  5. Malware-like activity  : event_type, protocol, bytes_sent, flow_duration_ms
"""
from datetime import datetime
from hashlib import blake2b
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "graphs"
OUT_DIR.mkdir(exist_ok=True)

BLUE, RED, GREY, GREEN, GOLD = "blue", "red", "#888888", "#2e8b57", "#ffc107"
PATTERNS = ["brute_force", "port_scan_like", "unusual_transfer", "suspicious_privilege", "malware_like"]


# ----------------------------------------------------------------------------
# 1. Data preparation
# ----------------------------------------------------------------------------
def _u(ids, salt):
    """Deterministic uniform(0,1) per event id - the same event always gets the same value,
    so adding new events never moves the points that were already on the graph."""
    out = np.empty(len(ids))
    for i, event_id in enumerate(ids):
        digest = blake2b(f"{event_id}|{salt}".encode(), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "big") / 2 ** 64
    return out


def _uniform(ids, salt, low, high):
    return low + (high - low) * _u(ids, salt)


def _lognormal(ids, salt, median, sigma):
    z = np.sqrt(-2 * np.log(np.clip(_u(ids, salt + "a"), 1e-9, 1))) * np.cos(2 * np.pi * _u(ids, salt + "b"))
    return median * np.exp(sigma * z)


def _real_rolling_count(df, minutes=5):
    """Real connection count: events from the same source IP in the previous 5 minutes (incl. itself)."""
    counts = pd.Series(1, index=df.index, dtype=int)
    for _, group in df.groupby("source_ip"):
        group = group.sort_values("timestamp")
        t = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        start = np.searchsorted(t, t - np.timedelta64(minutes, "m"), side="left")
        counts.loc[group.index] = np.arange(1, len(t) + 1) - start
    return counts


def prepare_data(df):
    """
    Standardise columns and make sure every plotted parameter exists.

    * Uses the effective label: the ground-truth label when there is one, otherwise the
      ML prediction (events entered through the form have no ground truth).
    * `is_new` is True for those ML-scored events.
    * Columns your data already has are used as they are. connection_count_5min,
      total_bytes_received, flow_duration_ms and bytes_per_second are not part of the
      security_events table, so they are SIMULATED when missing - deterministically per
      event_id. connection_count_5min is never lower than the real number of events
      seen from the same source IP in the previous 5 minutes.
    """
    df = df.copy()
    df = df.rename(columns={"src_ip": "source_ip", "event_time": "timestamp"})
    for col in ("predicted_label", "predicted_attack_type", "model_score", "ingested_at"):
        if col not in df:
            df[col] = None
    if "event_id" not in df:
        df["event_id"] = [f"row{i}" for i in range(len(df))]
    df["event_id"] = df["event_id"].astype(str)

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, format="mixed", errors="coerce")
    fallback = pd.to_datetime(df["ingested_at"], errors="coerce")
    df["timestamp"] = df["timestamp"].fillna(fallback)
    df = df.dropna(subset=["timestamp"])

    has_truth = df["label"].isin(["Benign", "Malicious"])
    scored = df["predicted_label"].isin(["Benign", "Malicious"])
    df = df[has_truth | scored].copy()            # unscored, unlabelled rows cannot be drawn
    has_truth = df["label"].isin(["Benign", "Malicious"])

    df["is_new"] = ~has_truth
    eff_label = df["label"].where(has_truth, df["predicted_label"])
    eff_type = df["attack_type"].where(has_truth, df["predicted_attack_type"])
    df["is_malicious"] = eff_label.eq("Malicious")
    df["attack_type"] = eff_type.fillna("benign").astype(str).str.strip().str.lower()
    df["event_hour"] = df["timestamp"].dt.hour
    for col in ("dst_port", "failed_login_count", "bytes_sent"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["protocol"] = df["protocol"].fillna("N/A").astype(str).str.upper()
    df = df.reset_index(drop=True)

    ids, kind, n = df["event_id"].tolist(), df["attack_type"], len(df)
    simulated = []

    if "connection_count_5min" not in df:
        sim = 1 + np.floor(-np.log(1 - _u(ids, "base") * 0.999) * 3.5)     # normal: a few connections
        bursts = {"brute_force": (40, 120), "port_scan_like": (150, 600), "unusual_transfer": (8, 25),
                  "suspicious_privilege": (15, 45), "malware_like": (20, 70)}
        for k, (low, high) in bursts.items():
            sim = np.where(kind == k, np.floor(_uniform(ids, "burst", low, high)), sim)
        df["connection_count_5min"] = np.maximum(sim, _real_rolling_count(df)).astype(int)
        simulated.append("connection_count_5min")

    if "flow_duration_ms" not in df:
        dur = _lognormal(ids, "dur", 2500, 0.8)
        dur = np.where(kind == "port_scan_like", _uniform(ids, "dur2", 5, 80), dur)
        dur = np.where(kind == "unusual_transfer", _uniform(ids, "dur2", 30_000, 300_000), dur)
        dur = np.where(kind == "malware_like", _uniform(ids, "dur2", 15_000, 90_000), dur)
        df["flow_duration_ms"] = np.round(dur, 1)
        simulated.append("flow_duration_ms")

    if "total_bytes_received" not in df:
        ratio = _uniform(ids, "ratio", 0.5, 1.5)
        ratio = np.where(kind == "unusual_transfer", _uniform(ids, "ratio", 0.005, 0.05), ratio)
        df["total_bytes_received"] = np.round(df["bytes_sent"] * ratio, 0)
        simulated.append("total_bytes_received")

    if "bytes_per_second" not in df:
        df["bytes_per_second"] = np.round(df["bytes_sent"] / (df["flow_duration_ms"].clip(lower=1) / 1000), 1)
        simulated.append("bytes_per_second")

    df.attrs["simulated"] = simulated
    return df


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _footer(df):
    return (f"{len(df)} events · {int(df['is_new'].sum())} new ML-scored (gold stars) · "
            f"updated {datetime.now():%d %b %Y %H:%M:%S}")


def _combine_legend(ax1, ax2=None, loc="lower center"):
    """All axes in one legend, placed above the plot so it never hides data."""
    handles, labels = ax1.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax1.legend(handles, labels, loc=loc, bbox_to_anchor=(0.5, 1.0),
               ncol=min(5, max(1, len(handles))), fontsize=10, frameon=False)


def _style_axes(ax1, ax2, left_label, right_label):
    ax1.set_ylabel(left_label, color=BLUE, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylabel(right_label, color=RED, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=RED)


def _time_axis(ax):
    ax.set_xlabel("Time", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _mark_new(ax, xs, ys, ids):
    """Gold stars = events that were scored by the ML model and have no ground-truth label."""
    xs, ys, ids = list(xs), list(ys), list(ids)
    if not xs:
        return
    ax.scatter(xs, ys, marker="*", s=480, color=GOLD, edgecolor="black", linewidth=1.2, zorder=10,
               label=f"New event, ML-scored ({len(xs)})")
    for x, y, event_id in list(zip(xs, ys, ids))[-8:]:
        ax.annotate(event_id, (x, y), textcoords="offset points", xytext=(9, 11), fontsize=8,
                    fontweight="bold", zorder=11)


def _finish(fig, title, name, show, df):
    plt.title(title, fontsize=15, pad=38)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.01, 0.008, _footer(df), fontsize=8, color=GREY)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _blank(df, title, name, message, show):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color=GREY, transform=ax.transAxes)
    ax.set_axis_off()
    _finish(fig, title, name, show, df)


# ----------------------------------------------------------------------------
# 2. One graph per attack pattern
# ----------------------------------------------------------------------------
def plot_brute_force(df, title="Brute Force Activity", show=True):
    """failed_login_count + event_type + event_hour + src_ip."""
    d = df[df["event_type"] == "login_failure"]
    mal, ben = d[d["attack_type"] == "brute_force"], d[~d["is_malicious"]]
    if mal.empty:
        return _blank(df, title, "1_brute_force", "No brute-force events yet", show)
    hours = pd.Index(range(24), name="event_hour")
    mal_fail = mal.groupby("event_hour")["failed_login_count"].sum().reindex(hours, fill_value=0)
    ben_fail = ben.groupby("event_hour")["failed_login_count"].sum().reindex(hours, fill_value=0)
    mal_ips = mal.groupby("event_hour")["source_ip"].nunique().reindex(hours, fill_value=0)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(hours, mal_fail, color=BLUE, marker="o", linewidth=2.5, markersize=7,
             label="Failed Logins - brute force")
    ax1.plot(hours, ben_fail, color=GREY, marker="o", linewidth=1.8, markersize=5, alpha=0.8,
             label="Failed Logins - benign")
    ax1.axhline(10, color=BLUE, linestyle=":", alpha=0.6)
    ax1.text(23.4, 10, " alert threshold", color=BLUE, va="bottom", ha="right", fontsize=9)
    new = d[d["is_new"]]
    _mark_new(ax1, new["event_hour"],
              [(mal_fail if m else ben_fail)[h] for h, m in zip(new["event_hour"], new["is_malicious"])],
              new["event_id"])

    ax2 = ax1.twinx()
    ax2.plot(hours, mal_ips, color=RED, marker="s", linestyle="--", linewidth=2.5, markersize=7,
             label="Distinct Source IPs (brute force)")
    ax1.set_xlabel("Event Hour (0-23)", fontsize=12)
    ax1.set_xticks(range(24))
    _style_axes(ax1, ax2, "Failed Login Count (sum per hour)", "Distinct Source IPs")
    _combine_legend(ax1, ax2)
    _finish(fig, title + "  (event_type = login_failure)", "1_brute_force", show, df)


def plot_port_scan(df, title="Port Scanning Activity", show=True):
    """dst_port + protocol + connection_count_5min."""
    mal = df[df["attack_type"] == "port_scan_like"].sort_values("timestamp")
    if mal.empty:
        return _blank(df, title, "2_port_scan", "No port-scan events yet", show)
    ben = df[(df["event_type"] == "network_connection") & ~df["is_malicious"] & ~df["is_new"]]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.scatter(ben["timestamp"], ben["dst_port"].clip(lower=1), color=GREY, alpha=0.25, s=18,
                label="Benign connections (dst_port)")
    ax1.plot(mal["timestamp"], mal["dst_port"].clip(lower=1), color=BLUE, marker="o", linestyle="none",
             markersize=8, label=f"Destination Port ({', '.join(mal['protocol'].unique())})")
    ax1.set_yscale("log")
    new = df[df["is_new"] & ((df["attack_type"] == "port_scan_like") | (df["event_type"] == "network_connection"))]
    _mark_new(ax1, new["timestamp"], new["dst_port"].clip(lower=1), new["event_id"])

    ax2 = ax1.twinx()
    ax2.plot(mal["timestamp"], mal["connection_count_5min"], color=RED, marker="s", linestyle="--",
             linewidth=2.5, markersize=7, label="Connections (Rolling 5 min)")
    ax2.set_ylim(0, max(650, mal["connection_count_5min"].max() * 1.1))
    ax2.axhline(100, color=RED, linestyle=":", alpha=0.6)
    ax2.text(mal["timestamp"].iloc[-1], 100, "scan threshold (100)  ", color=RED, va="bottom", ha="right", fontsize=9)

    _style_axes(ax1, ax2, "Destination Port (log scale)", "Connection Count (Rolling 5 min)")
    _time_axis(ax1)
    _combine_legend(ax1, ax2)
    _finish(fig, title, "2_port_scan", show, df)


def plot_unusual_transfer(df, title="Unusual Data Transfer Activity", show=True):
    """bytes_sent + total_bytes_received + bytes_per_second."""
    mal = df[df["attack_type"] == "unusual_transfer"].sort_values("timestamp")
    if mal.empty:
        return _blank(df, title, "3_unusual_transfer", "No unusual-transfer events yet", show)
    ben = df[(df["event_type"] == "data_transfer") & ~df["is_malicious"] & ~df["is_new"]]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(mal["timestamp"], mal["bytes_sent"].clip(lower=1), color=BLUE, marker="o", linewidth=2.5,
             markersize=7, label="Bytes Sent")
    ax1.plot(mal["timestamp"], mal["total_bytes_received"].clip(lower=1), color=GREEN, marker="^",
             linewidth=2.0, markersize=6, label="Total Bytes Received")
    if len(ben):
        p99 = ben["bytes_sent"].quantile(0.99)
        ax1.axhline(p99, color=BLUE, linestyle=":", alpha=0.6)
        ax1.text(mal["timestamp"].iloc[-1], p99, "benign 99th percentile  ", color=BLUE, va="top",
                 ha="right", fontsize=9)
    ax1.set_yscale("log")
    new = df[df["is_new"] & ((df["attack_type"] == "unusual_transfer") | (df["event_type"] == "data_transfer"))]
    _mark_new(ax1, new["timestamp"], new["bytes_sent"].clip(lower=1), new["event_id"])

    ax2 = ax1.twinx()
    ax2.plot(mal["timestamp"], mal["bytes_per_second"].clip(lower=1), color=RED, marker="s", linestyle="--",
             linewidth=2.5, markersize=7, label="Bytes per Second")
    ax2.set_yscale("log")
    _style_axes(ax1, ax2, "Bytes Sent / Received (log)", "Bytes per Second (log)")
    _time_axis(ax1)
    _combine_legend(ax1, ax2)
    _finish(fig, title + "  (exfiltration: sent >> received)", "3_unusual_transfer", show, df)


def plot_suspicious_privilege(df, title="Suspicious Privilege Activity", show=True):
    """event_type + protocol + connection_count_5min (daily view)."""
    d = df[df["event_type"] == "privilege_change"].copy()
    mal = d[d["attack_type"] == "suspicious_privilege"]
    if mal.empty:
        return _blank(df, title, "4_suspicious_privilege", "No suspicious privilege events yet", show)
    d["day"] = d["timestamp"].dt.floor("D")
    mal, ben = d[d["attack_type"] == "suspicious_privilege"], d[~d["is_malicious"]]
    days = pd.date_range(d["day"].min(), d["day"].max(), freq="D")
    mal_cnt = mal.groupby("day").size().reindex(days, fill_value=0)
    ben_cnt = ben.groupby("day").size().reindex(days, fill_value=0)
    conn = mal.groupby("day")["connection_count_5min"].mean().reindex(days)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(days, mal_cnt, color=BLUE, marker="o", linewidth=2.5, markersize=7,
             label=f"Privilege Changes - suspicious ({', '.join(mal['protocol'].unique())})")
    ax1.plot(days, ben_cnt, color=GREY, marker="o", linewidth=1.8, markersize=5, alpha=0.8,
             label="Privilege Changes - benign")
    new = d[d["is_new"]]
    _mark_new(ax1, new["day"], [(mal_cnt if m else ben_cnt)[t] for t, m in zip(new["day"], new["is_malicious"])],
              new["event_id"])

    ax2 = ax1.twinx()
    ax2.plot(days, conn, color=RED, marker="s", linestyle="--", linewidth=2.5, markersize=7,
             label="Avg Connections (Rolling 5 min)")
    _style_axes(ax1, ax2, "privilege_change Events per Day", "Avg Connection Count (5 min)")
    ax1.set_xlabel("Day", fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    _combine_legend(ax1, ax2)
    _finish(fig, title, "4_suspicious_privilege", show, df)


def plot_malware_like(df, title="Malware-like Activity", show=True):
    """event_type + protocol + bytes_sent + flow_duration_ms."""
    d = df[df["event_type"] == "process_execution"].sort_values("timestamp")
    mal, ben = d[d["attack_type"] == "malware_like"], d[~d["is_malicious"] & ~d["is_new"]]
    if mal.empty:
        return _blank(df, title, "5_malware_like", "No malware-like events yet", show)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(mal["timestamp"], mal["bytes_sent"], color=BLUE, marker="o", linewidth=2.5, markersize=7,
             label="Bytes Sent")
    if len(ben):
        ax1.axhline(ben["bytes_sent"].median(), color=GREY, linestyle=":", alpha=0.8)
        ax1.text(mal["timestamp"].iloc[-1], ben["bytes_sent"].median(), "benign median bytes  ", color=GREY,
                 va="bottom", ha="right", fontsize=9)
    new = d[d["is_new"]]
    _mark_new(ax1, new["timestamp"], new["bytes_sent"], new["event_id"])

    ax2 = ax1.twinx()
    ax2.plot(mal["timestamp"], mal["flow_duration_ms"], color=RED, marker="s", linestyle="--", linewidth=2.5,
             markersize=7, label="Flow Duration (ms)")
    if len(ben):
        ax2.axhline(ben["flow_duration_ms"].median(), color=RED, linestyle=":", alpha=0.4)
    for proto, mk in (("TCP", "^"), ("UDP", "v")):
        p = mal[mal["protocol"] == proto]
        if len(p):
            ax1.scatter(p["timestamp"], p["bytes_sent"], marker=mk, s=70, color="navy", zorder=5,
                        label=f"{proto} flow")
    _style_axes(ax1, ax2, "Bytes Sent", "Flow Duration (ms)")
    _time_axis(ax1)
    _combine_legend(ax1, ax2)
    _finish(fig, title + "  (event_type = process_execution)", "5_malware_like", show, df)


# ----------------------------------------------------------------------------
# 3. Parameter curves - malicious vs benign (cumulative distribution)
# ----------------------------------------------------------------------------
def _ecdf(ax, values, color, label, ls="-"):
    v = np.sort(values.dropna().to_numpy())
    if len(v):
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, linewidth=2.5, linestyle=ls, label=label)


def plot_parameter_curves(df, show=True):
    """One curve panel per pattern: key parameter, malicious (red) vs benign (blue). Gold line = a new event."""
    panels = [
        ("brute_force", "failed_login_count", "Brute force - Failed Login Count", False),
        ("port_scan_like", "connection_count_5min", "Port scan - Connections / 5 min", True),
        ("unusual_transfer", "bytes_per_second", "Unusual transfer - Bytes / second", True),
        ("suspicious_privilege", "connection_count_5min", "Suspicious privilege - Connections / 5 min", True),
        ("malware_like", "flow_duration_ms", "Malware-like - Flow Duration (ms)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, (atk, col, ttl, logx) in zip(axes, panels):
        _ecdf(ax, df.loc[~df["is_malicious"], col], BLUE, "Benign")
        _ecdf(ax, df.loc[df["attack_type"] == atk, col], RED, "Malicious", "--")
        # every new event is drawn: solid gold = predicted as THIS pattern, dashed gold = predicted something else
        # (so a new benign event is visible too). Only the latest 10 are drawn to keep the panel readable.
        new = df[df["is_new"]].tail(10)
        shown = set()
        for _, row in new.iterrows():
            mine = row["attack_type"] == atk
            tag = "New event (this pattern)" if mine else "New event (other / benign)"
            ax.axvline(max(row[col], 1e-9) if logx else row[col], color=GOLD, linewidth=2.5 if mine else 1.6,
                       linestyle="-" if mine else (0, (4, 3)), zorder=5, label=None if tag in shown else tag)
            shown.add(tag)
        ax.set_title(ttl, fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Cumulative share of events")
        if logx:
            ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")
    axes[-1].axis("off")
    fig.suptitle("Parameter Curves - Malicious vs Benign (cumulative distribution)", fontsize=15)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.text(0.01, 0.008, _footer(df), fontsize=8, color=GREY)
    fig.savefig(OUT_DIR / "6_parameter_curves.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def generate_all(df, show=False):
    """Draw all six graphs from an (already prepared) dataframe."""
    plot_brute_force(df, "Brute Force Activity", show=show)
    plot_port_scan(df, "Port Scanning Activity", show=show)
    plot_unusual_transfer(df, "Unusual Data Transfer Activity", show=show)
    plot_suspicious_privilege(df, "Suspicious Privilege Activity", show=show)
    plot_malware_like(df, "Malware-like Activity", show=show)
    plot_parameter_curves(df, show=show)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    prepared = prepare_data(pd.read_csv(ROOT / "data" / "security_events.csv"))
    if prepared.attrs.get("simulated"):
        print("NOTE - simulated columns:", ", ".join(prepared.attrs["simulated"]))
    generate_all(prepared)
    print("Saved graphs to", OUT_DIR)
