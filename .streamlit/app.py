import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as dtparser
import pytz
import streamlit as st

st.set_page_config(page_title="NCNC 25/26", layout="wide")

# -----------------------------------
# Settings
# -----------------------------------
DEFAULT_LEAGUE_ID = 497796
BRISBANE_TZ = pytz.timezone("Australia/Brisbane")

# -----------------------------------
# Cached API helpers
# -----------------------------------
@st.cache_data(ttl=300)
def get_bootstrap():
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_events():
    bs = get_bootstrap()
    return pd.DataFrame(bs["events"])

@st.cache_data(ttl=300)
def get_elements_df():
    bs = get_bootstrap()
    els = pd.DataFrame(bs["elements"])
    teams = pd.DataFrame(bs["teams"])
    types = pd.DataFrame(bs["element_types"])
    els = els.merge(
        types[["id", "singular_name_short"]],
        left_on="element_type", right_on="id",
        how="left", suffixes=("", "_pos")
    )
    els = els.merge(
        teams[["id", "short_name", "name"]],
        left_on="team", right_on="id",
        how="left", suffixes=("", "_team")
    )
    els.rename(columns={
        "singular_name_short": "position",   # FPL playing position (GKP/DEF/MID/FWD)
        "short_name": "team_short",
        "name": "team_name"
    }, inplace=True)
    els["price_m"] = els["now_cost"] / 10.0
    return els

@st.cache_data(ttl=300)
def get_league_standings_all_pages(league_id: int):
    """Fetch all pages of classic league standings."""
    page = 1
    results = []
    league_obj = None
    while True:
        url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page}"
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            raise ValueError("League not found or private.")
        r.raise_for_status()
        data = r.json()
        if league_obj is None:
            league_obj = data.get("league", {})
        st_obj = data.get("standings", {})
        page_results = st_obj.get("results", [])
        results.extend(page_results)
        if not st_obj.get("has_next"):
            break
        page += 1
        time.sleep(0.2)  # be gentle to API
    df = pd.DataFrame(results)
    return league_obj, df

@st.cache_data(ttl=120)
def get_event_live(event_id: int):
    url = f"https://fantasy.premierleague.com/api/event/{event_id}/live/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_entry_history(entry_id: int):
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/history/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120)
def get_entry_picks(entry_id: int, event_id: int):
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event_id}/picks/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_entry_transfers(entry_id: int):
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/transfers/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

# New helper: bench points for each entry in a GW
@st.cache_data(ttl=180)
def get_league_bench_points(entries_list, gw_id: int) -> pd.DataFrame:
    rows = []
    for entry in entries_list:
        try:
            resp = get_entry_picks(int(entry), int(gw_id))
            eh = resp.get("entry_history") or {}
            bp = int(eh.get("points_on_bench") or 0)
            rows.append({"entry": entry, "bench_points": bp})
        except Exception:
            rows.append({"entry": entry, "bench_points": None})
    return pd.DataFrame(rows)

# -----------------------------------
# Utils
# -----------------------------------
def au_time(s):
    try:
        return dtparser.isoparse(str(s)).astimezone(BRISBANE_TZ).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return s

def current_gw(events_df):
    cur = events_df[events_df["is_current"] == True]
    if cur.empty:
        nxt = events_df[events_df["is_next"] == True]
        if not nxt.empty:
            return int(nxt.iloc[0]["id"])
        return int(events_df["id"].max())
    return int(cur.iloc[0]["id"])

def live_points_for_gw(gw_id: int) -> pd.DataFrame:
    """Return DataFrame mapping element id to that GW's points and stats."""
    live = get_event_live(gw_id)
    els = pd.DataFrame(live.get("elements", []))
    if els.empty:
        return pd.DataFrame(columns=["id", "total_points"])
    stats = pd.json_normalize(els["stats"])
    stats["element"] = els["id"]
    return stats.rename(columns={"element": "id"})

def draw_bar(series, title, xlabel, ylabel):
    import pandas as pd
    fig, ax = plt.subplots()
    if isinstance(series.index, pd.MultiIndex):
        labels = []
        for idx in series.index:
            if len(idx) >= 3:
                label = f"{idx[1]} ({idx[2]})"
            elif len(idx) == 2:
                label = str(idx[1])
            else:
                label = str(idx[0])
            labels.append(label)
    else:
        labels = series.index.astype(str)
    x = range(len(series))
    ax.bar(x, series.values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    st.pyplot(fig)

def chip_friendly(name: str) -> str:
    mapping = {"3xc": "Triple Captain", "bboost": "Bench Boost", "freehit": "Free Hit", "wildcard": "Wildcard"}
    return mapping.get(name, str(name).title())

# -----------------------------------
# Sidebar (nav + controls)
# -----------------------------------
events_df = get_events()
els_df = get_elements_df()

with st.sidebar:
    st.header("League")
    league_id = st.number_input("Classic League ID", min_value=1, value=DEFAULT_LEAGUE_ID, step=1)
    st.caption("Default set to 497796 as you requested.")

    st.header("Section")
    section = st.radio(
        "Go to",
        ["League Table", "GW Overview", "Chips & Hits (Season)", "Transfers", "Manager Lineups"],
        index=0,
    )

    st.header("Gameweek")
    max_gw = int(events_df["id"].max())
    cur_gw = current_gw(events_df)
    gw = st.number_input("Select GW", min_value=1, max_value=max_gw, value=int(cur_gw), step=1)

    st.header("Differential threshold")
    diff_thresh = st.slider("Max % owned in league", min_value=1, max_value=50, value=10, step=1)

    top_n = st.number_input("Top N managers (deep dive)", min_value=5, max_value=50, value=10, step=1)

    if st.button("Refresh data"):
        st.cache_data.clear()
        try:
            st.rerun()
        except Exception:
            pass

# -----------------------------------
# Load League + Live points for selected GW
# -----------------------------------
try:
    league_meta, league_standings = get_league_standings_all_pages(int(league_id))
except Exception as e:
    st.error(f"Failed to load league {league_id}: {e}")
    st.stop()

st.title(f"League: {league_meta.get('name','(unknown)')}")
st.caption(f"League ID: {league_id}")

live_df = live_points_for_gw(int(gw))

# Convenience maps
entries = league_standings["entry"].tolist()
entry_to_team = dict(zip(league_standings["entry"], league_standings["entry_name"]))
entry_to_manager = dict(zip(league_standings["entry"], league_standings["player_name"]))

# -----------------------------------
# SECTION: League Table
# -----------------------------------
if section == "League Table":
    st.subheader("League Table (Standings)")
    if not league_standings.empty:
        df = league_standings.copy()
        df["Current Pos"] = df["rank"].astype(int)
        prev_total = df["total"] - df["event_total"]
        prev_rank_all = prev_total.rank(ascending=False, method="min").astype(int)
        if "last_rank" in df.columns:
            df["Previous Pos"] = df["last_rank"].fillna(prev_rank_all).astype(int)
        else:
            df["Previous Pos"] = prev_rank_all

        # NEW: bench points per entry for the selected GW
        bench_df = get_league_bench_points(entries, int(gw))
        df = df.merge(bench_df, on="entry", how="left")

        show = (
            df[["Current Pos", "Previous Pos", "player_name", "event_total", "bench_points", "total"]]
            .rename(columns={
                "player_name": "Manager",
                "event_total": "GW Points",
                "bench_points": "Bench Pts",
                "total": "Total Points",
            })
            .sort_values(["Total Points", "GW Points"], ascending=[False, False])
            .reset_index(drop=True)
        )
        show["Δ Pos"] = show["Previous Pos"] - show["Current Pos"]  # positive = climbed
        show = show[["Current Pos", "Previous Pos", "Δ Pos", "Manager", "GW Points", "Bench Pts", "Total Points"]]
        st.dataframe(show, use_container_width=True)
    else:
        st.info("No standings data. Is this a valid league?")

# -----------------------------------
# SECTION: GW Overview
# -----------------------------------
if section == "GW Overview":
    st.subheader(f"GW {gw} Overview (across league)")

    # Fetch all managers' picks for the selected GW
    pbar = st.progress(0.0, text="Fetching manager picks...")
    all_picks = []
    meta_rows = []  # per-entry meta for adjusted points
    for i, entry in enumerate(entries):
        try:
            resp = get_entry_picks(int(entry), int(gw))
            dfp = pd.DataFrame(resp.get("picks", []))

            # Keep squad slot 1..15 separate from FPL playing "position"
            if "position" in dfp.columns:
                dfp = dfp.rename(columns={"position": "slot"})
            else:
                dfp["slot"] = None

            dfp["entry"] = entry
            dfp["team_name"] = entry_to_team.get(entry, "")
            dfp["manager"] = entry_to_manager.get(entry, "")
            all_picks.append(dfp)

            eh = resp.get("entry_history", {}) or {}
            meta_rows.append({
                "entry": entry,
                "manager": entry_to_manager.get(entry, ""),
                "team_name": entry_to_team.get(entry, ""),
                "active_chip": resp.get("active_chip"),
                "hits_points": int(eh.get("event_transfers_cost") or 0),
                "overall_rank": eh.get("overall_rank"),
            })
        except Exception:
            pass
        if len(entries) > 0:
            pbar.progress((i + 1) / len(entries), text=f"Fetching manager picks... ({i+1}/{len(entries)})")
    pbar.empty()

    if not all_picks:
        st.warning("Couldn't load picks for any entries — the league may be private or API is throttling.")
    else:
        picks_df = pd.concat(all_picks, ignore_index=True)
        gw_meta = pd.DataFrame(meta_rows)

        # Join with player meta and live points
        picks_df = picks_df.merge(
            els_df[["id", "second_name", "team_short", "position"]],
            left_on="element", right_on="id", how="left"
        ).merge(
            live_df[["id", "total_points"]],
            left_on="element", right_on="id", how="left", suffixes=("", "_live")
        )
        picks_df["total_points"].fillna(0, inplace=True)
        if ("position" not in picks_df.columns) or (picks_df["position"].isna().all()):
            pos_map = dict(zip(els_df["id"], els_df["position"])) if "position" in els_df.columns else {}
            picks_df["position"] = picks_df["element"].map(pos_map).fillna("UNK")

        # Ensure key flags exist
        if "is_captain" not in picks_df.columns:
            picks_df["is_captain"] = False
        picks_df["is_captain"] = picks_df["is_captain"].fillna(False).astype(bool)
        if "multiplier" not in picks_df.columns:
            picks_df["multiplier"] = 1

        # ---------------- Popular Picks ----------------
        st.markdown("### Popular Picks and Captains")
        league_size = picks_df["entry"].nunique()

        for col in ["second_name", "team_short"]:
            if col not in picks_df.columns:
                m = dict(zip(els_df["id"], els_df[col])) if col in els_df.columns else {}
                picks_df[col] = picks_df["element"].map(m)

        picks_ok = picks_df[picks_df["second_name"].notna()].copy()

        if league_size == 0 or picks_ok.empty:
            st.info("No picks available for this GW.")
            pop = pd.DataFrame()
        else:
            group_cols = ["element", "second_name", "team_short"]
            if "position" in picks_ok.columns:
                group_cols.append("position")

            pop = (picks_ok.groupby(group_cols)["entry"]
                   .nunique()
                   .reset_index(name="picked_by"))
            pop["owned_%"] = (pop["picked_by"] / league_size * 100).round(1)
            pop_top = pop.sort_values(["picked_by", "second_name"], ascending=[False, True]).head(30)

            st.dataframe(pop_top.rename(columns={
                "second_name": "Player", "team_short": "Team", "position": "Pos", "picked_by": "Managers"
            }), use_container_width=True)

        # ---------------- Captains chart + owner list ----------------
        cap = picks_ok[picks_ok["is_captain"] == True].copy()
        if not cap.empty:
            cap_counts = (
                cap.groupby(["element", "second_name", "team_short"])["entry"]
                .nunique()
                .reset_index(name="manager_count")
                .sort_values("manager_count", ascending=False)
                .head(15)
            )
            cap_counts["label"] = cap_counts["second_name"] + " (" + cap_counts["team_short"] + ")"

            c1, c2 = st.columns([2, 1])

            # Chart with value labels
            with c1:
                fig, ax = plt.subplots()
                x = range(len(cap_counts))
                vals = cap_counts["manager_count"].values
                ax.bar(x, vals)
                ax.set_xticks(x)
                ax.set_xticklabels(cap_counts["label"].tolist(), rotation=45, ha="right")
                ax.set_title(f"Top Captains — GW {gw}")
                ax.set_xlabel("Player")
                ax.set_ylabel("Managers")
                for xi, val in zip(x, vals):
                    ax.text(xi, val, str(int(val)), ha="center", va="bottom")
                fig.tight_layout()
                st.pyplot(fig)

            # Manager list for selected captain
            with c2:
                st.markdown("**Managers for selected captain**")
                selected_label = st.selectbox(
                    "Choose captain",
                    options=cap_counts["label"].tolist(),
                    index=0
                )
                sel_row = cap_counts[cap_counts["label"] == selected_label].iloc[0]
                sel_el = int(sel_row["element"])

                owners = (
                    cap[cap["element"] == sel_el]
                    .groupby("entry")
                    .agg(manager=("manager", "first"))
                    .reset_index(drop=True)
                    .sort_values("manager")
                )
                st.write(f"{selected_label} — picked by **{len(owners)}** manager(s):")
                st.dataframe(
                    owners.rename(columns={"manager": "Manager"}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No captain data available for this GW.")

        # ---------------- Adjusted GW Points ----------------
        st.markdown("### Adjusted GW Points ")
        raw_gw = (picks_df.assign(raw=picks_df["total_points"] * picks_df["multiplier"])
                  .groupby("entry")["raw"].sum().rename("Raw GW"))

        meta_map = gw_meta.set_index("entry")[["active_chip", "hits_points"]].to_dict(orient="index")
        adj_rows = []
        for entry_id, g in picks_df.groupby("entry"):
            info = meta_map.get(entry_id, {})
            chip = info.get("active_chip")
            mult = g["multiplier"].astype(float).copy()

            # triple captain -> treat as normal captain (2x)
            if chip == "3xc":
                mult = mult.where(~g["is_captain"], 2.0)

            # bench boost -> remove bench contribution (slots 12..15)
            if chip == "bboost" and "slot" in g.columns:
                mult = mult.where(~g["slot"].between(12, 15, inclusive="both"), 0.0)

            adj_points = float((g["total_points"] * mult).sum())
            hits = float(info.get("hits_points") or 0.0)
            adj_minus_hits = adj_points - hits
            adj_rows.append({"entry": entry_id, "Adj GW": adj_points, "Hits": hits, "Adj (− hits)": adj_minus_hits})

        adj_df = pd.DataFrame(adj_rows).set_index("entry")

        scoreboard = (pd.concat([raw_gw, adj_df], axis=1)
                      .reset_index()
                      .merge(
                          gw_meta[["entry", "manager", "team_name", "overall_rank"]],
                          on="entry", how="left"
                      )
                      .sort_values(["Adj (− hits)", "Adj GW", "Raw GW"], ascending=[False, False, False])
                     )

        st.dataframe(
            scoreboard.head(int(top_n))[["manager", "team_name", "Raw GW", "Adj GW", "Hits", "Adj (− hits)", "overall_rank"]]
            .rename(columns={"manager": "Manager", "team_name": "Team", "overall_rank": "Overall Rank"}),
            use_container_width=True
        )
        best_adj = scoreboard.iloc[0]
        st.caption(f"Best (Adjusted − hits): **{best_adj['manager']}** — {int(round(best_adj['Adj (− hits)']))} pts")

        # ---------------- Best (raw) Manager lineup ----------------
        st.markdown("### Best Manager in League (this GW — raw)")
        gw_points_by_entry = (
            picks_df.assign(points=picks_df["total_points"] * picks_df["multiplier"])
            .groupby("entry")["points"].sum()
            .sort_values(ascending=False)
        )
        best_entry = int(gw_points_by_entry.index[0])
        best_points = int(gw_points_by_entry.iloc[0])
        st.write(f"**{entry_to_team.get(best_entry,'?')}** ({entry_to_manager.get(best_entry,'?')}) — **{best_points} pts**")

        best_picks = picks_df[picks_df["entry"] == best_entry].copy()
        best_picks["C"] = best_picks["is_captain"].map({True: "(C)", False: ""})
        best_picks["VC"] = best_picks["is_vice_captain"].map({True: "(VC)", False: ""})
        best_view = best_picks[["position", "second_name", "team_short", "multiplier", "C", "VC", "total_points"]].rename(columns={
            "position": "Pos", "second_name": "Player", "team_short": "Team", "multiplier": "x", "total_points": "GW Pts"
        }).sort_values(["Pos", "Player"])
        st.dataframe(best_view, use_container_width=True)

        # ---------------- Differentials ----------------
        st.markdown("### Best Differentials")
        if pop is None or pop.empty:
            st.info("Differentials unavailable because no popular picks were computed.")
        else:
            pop_points = pop.merge(live_df[["id", "total_points"]], left_on="element", right_on="id", how="left")
            pop_points["total_points"].fillna(0, inplace=True)
            diff = pop_points[pop_points["owned_%"] <= diff_thresh].copy()

            owners = (picks_df.groupby(["element", "manager"])
                      .agg(is_captain=("is_captain", "max"))
                      .reset_index())
            owners = owners[owners["manager"].notna() & (owners["manager"] != "")]
            owners["manager_label"] = owners.apply(
                lambda r: f"{r['manager']} (C)" if r["is_captain"] else r["manager"], axis=1
            )
            owners_list = (owners.groupby("element")["manager_label"]
                           .apply(lambda s: " • ".join(sorted(s.unique())))
                           .reset_index(name="Owners"))

            diff = diff.merge(owners_list, on="element", how="left")
            diff = diff.sort_values(["total_points", "owned_%"], ascending=[False, True]).head(30)

            show_cols = ["second_name", "team_short", "owned_%", "total_points", "Owners"]
            if "position" in diff.columns:
                show_cols.insert(2, "position")

            st.dataframe(diff[show_cols].rename(columns={
                "second_name": "Player", "team_short": "Team", "position": "Pos",
                "owned_%": "Owned %", "total_points": "GW Pts"
            }), use_container_width=True)

# -----------------------------------
# SECTION: Chips & Hits (Season)
# -----------------------------------
if section == "Chips & Hits (Season)":
    st.subheader("Chips, Hits & Bench Points (Season)")

    chips_rows = []
    hist_rows = []

    pbar = st.progress(0.0, text="Loading manager histories...")
    for i, entry in enumerate(entries):
        try:
            hist = get_entry_history(int(entry))
            cur = pd.DataFrame(hist.get("current", []))
            chips = pd.DataFrame(hist.get("chips", []))

            hits_points = int(pd.to_numeric(cur.get("event_transfers_cost", pd.Series(dtype="float")), errors="coerce").fillna(0).sum())
            bench_points = int(pd.to_numeric(cur.get("points_on_bench", pd.Series(dtype="float")), errors="coerce").fillna(0).sum())
            hits_count = hits_points // 4

            hist_rows.append({
                "entry": entry,
                "manager": entry_to_manager.get(entry, ""),
                "hits_points": hits_points,
                "hits_count": int(hits_count),
                "bench_points": bench_points,
            })

            if not chips.empty:
                chips["chip"] = chips["name"].map(chip_friendly)
                chips["entry"] = entry
                chips["manager"] = entry_to_manager.get(entry, "")
                chips_rows.append(chips[["entry", "manager", "event", "chip", "time"]])
        except Exception:
            pass
        pbar.progress((i + 1) / max(1, len(entries)), text=f"Loading manager histories... ({i+1}/{len(entries)})")
    pbar.empty()

    hist_df = pd.DataFrame(hist_rows)
    chips_df = pd.concat(chips_rows, ignore_index=True) if chips_rows else pd.DataFrame(columns=["entry","manager","event","chip","time"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### League Chip Usage (count)")
        if not chips_df.empty:
            chip_counts = chips_df["chip"].value_counts()
            draw_bar(chip_counts, "Chip usage", "Chip", "Times used")
        else:
            st.info("No chip usage found yet.")
    with c2:
        st.markdown("#### Per-Manager Chip Log")
        if not chips_df.empty:
            log = (chips_df.sort_values(["manager", "event"])
                   .groupby("manager")
                   .apply(lambda d: ", ".join(f"{row.chip} (GW{int(row.event)})" for _, row in d.iterrows()))
                   .reset_index(name="Chips Used"))
            st.dataframe(log, use_container_width=True)
        else:
            st.info("No chip usage found yet.")

    st.markdown("#### Hits Taken (points & count)")
    if not hist_df.empty:
        hits_tbl = (hist_df[["manager", "hits_points", "hits_count"]]
                    .sort_values(["hits_points", "hits_count"], ascending=False)
                    .reset_index(drop=True)
                    .rename(columns={"manager": "Manager", "hits_points": "Hit Points", "hits_count": "Hit Count"}))
        st.dataframe(hits_tbl, use_container_width=True)
    else:
        st.info("No hits data available.")

    st.markdown("#### Top 5 — Bench Points (season)")
    if not hist_df.empty:
        bench_top = (hist_df[["manager", "bench_points"]]
                     .sort_values("bench_points", ascending=False)
                     .head(5)
                     .reset_index(drop=True)
                     .rename(columns={"manager": "Manager", "bench_points": "Bench Points"}))
        st.dataframe(bench_top, use_container_width=True)
    else:
        st.info("No bench points data available.")

# -----------------------------------
# SECTION: Transfers
# -----------------------------------
if section == "Transfers":
    st.subheader("Transfers")
    pbar = st.progress(0.0, text="Loading transfers...")
    transfers_rows = []
    for i, entry in enumerate(entries):
        try:
            # transfers for this manager (all season)
            tr = get_entry_transfers(int(entry))

            # per-GW transfer counts & hit points from entry history
            hist = get_entry_history(int(entry))
            cur = pd.DataFrame(hist.get("current", []))

            # make robust numeric columns
            if "event" not in cur.columns:
                cur["event"] = []
            cur["event_transfers"] = pd.to_numeric(cur.get("event_transfers", pd.Series(dtype="float")), errors="coerce").fillna(0).astype(int)
            cur["event_transfers_cost"] = pd.to_numeric(cur.get("event_transfers_cost", pd.Series(dtype="float")), errors="coerce").fillna(0).astype(int)

            gw_cost = dict(zip(cur["event"], cur["event_transfers_cost"]))
            gw_cnt  = dict(zip(cur["event"], cur["event_transfers"]))

            for t in tr:
                ev = t.get("event")
                cost = int(gw_cost.get(ev, 0))
                cnt = int(gw_cnt.get(ev, 0))
                transfers_rows.append({
                    "entry": entry,
                    "team_name": entry_to_team.get(entry, ""),
                    "manager": entry_to_manager.get(entry, ""),
                    "event": ev,
                    "element_in": t.get("element_in"),
                    "element_out": t.get("element_out"),
                    "time": t.get("time"),
                    "gw_transfers": cnt,          # NEW
                    "gw_hit_points": cost,        # NEW
                    "hit?": cost > 0              # NEW
                })
        except Exception:
            pass
        pbar.progress((i + 1) / max(1, len(entries)), text=f"Loading transfers... ({i+1}/{len(entries)})")
    pbar.empty()

    if transfers_rows:
        tr_df = pd.DataFrame(transfers_rows)
        tr_df["time_local"] = tr_df["time"].apply(au_time)

        # join player names/teams
        id_to_name = dict(zip(els_df["id"], els_df["second_name"]))
        id_to_team = dict(zip(els_df["id"], els_df["team_short"]))
        tr_df["in_name"]  = tr_df["element_in"].map(id_to_name)
        tr_df["out_name"] = tr_df["element_out"].map(id_to_name)
        tr_df["in_team"]  = tr_df["element_in"].map(id_to_team)
        tr_df["out_team"] = tr_df["element_out"].map(id_to_team)

        # filters
        c1, c2 = st.columns(2)
        with c1:
            gwf = st.number_input("Filter transfers by GW", min_value=1, max_value=int(events_df["id"].max()), value=int(gw), step=1)
        with c2:
            mgr = st.selectbox("Filter by manager (optional)", options=["(All)"] + [entry_to_manager[e] for e in entries], index=0)

        df_show = tr_df[tr_df["event"] == gwf]
        if mgr != "(All)":
            df_show = df_show[df_show["manager"] == mgr]

        st.dataframe(
            df_show[[
                "event","time_local","team_name","manager",
                "in_name","in_team","out_name","out_team",
                "gw_transfers","gw_hit_points","hit?"
            ]].rename(columns={
                "event":"GW",
                "time_local":"Time (AEST)",
                "team_name":"Team",
                "in_name":"In",
                "in_team":"In Tm",
                "out_name":"Out",
                "out_team":"Out Tm",
                "gw_transfers":"GW Transfers",
                "gw_hit_points":"GW Hit (pts)",
                "hit?":"Hit?"
            }),
            use_container_width=True
        )
    else:
        st.info("No transfers could be retrieved (entries may be private).")

# -----------------------------------
# SECTION: Manager Lineups
# -----------------------------------
if section == "Manager Lineups":
    st.subheader("Manager Lineups (per GW)")
    mgr_entry_opt = st.selectbox(
        "Choose manager",
        options=entries,
        format_func=lambda e: entry_to_manager.get(e, "?")
    )
    gw_opt = st.number_input(
        "Gameweek",
        min_value=1,
        max_value=int(events_df["id"].max()),
        value=int(gw),
        step=1
    )
    try:
        # Pull picks + entry_history for this manager/GW
        resp = get_entry_picks(int(mgr_entry_opt), int(gw_opt))
        mdf = pd.DataFrame(resp.get("picks", []))
        eh = resp.get("entry_history") or {}
        bench_total = int(eh.get("points_on_bench") or 0)

        # keep squad slot (1..15)
        if "position" in mdf.columns:  # 'position' here is the squad slot from FPL API
            mdf = mdf.rename(columns={"position": "slot"})
        else:
            mdf["slot"] = None

        # Join metadata and GW live points
        live_mgr = live_points_for_gw(int(gw_opt))
        meta_cols = ["id", "second_name", "team_short"]
        if "position" in els_df.columns:  # real playing position (GKP/DEF/MID/FWD)
            meta_cols.append("position")
        mdf = mdf.merge(els_df[meta_cols], left_on="element", right_on="id", how="left")
        if ("position" not in mdf.columns) or (mdf["position"].isna().all()):
            if "position" in els_df.columns:
                pos_map = dict(zip(els_df["id"], els_df["position"]))
                mdf["position"] = mdf["element"].map(pos_map)
            else:
                mdf["position"] = "UNK"
        mdf = mdf.merge(live_mgr[["id", "total_points"]], left_on="element", right_on="id", how="left")
        mdf["total_points"].fillna(0, inplace=True)

        # Flags & calc
        mdf["C"] = mdf["is_captain"].map({True: "(C)", False: ""})
        mdf["VC"] = mdf["is_vice_captain"].map({True: "(VC)", False: ""})
        if "multiplier" not in mdf.columns:
            mdf["multiplier"] = 1
        mdf["gw_pts"] = mdf["total_points"] * mdf["multiplier"]

        # --- Robust slot handling ---
        mdf["slot"] = pd.to_numeric(mdf["slot"], errors="coerce")  # numeric slots
        sel_start = mdf["slot"].between(1, 11, inclusive="both").fillna(False)
        sel_bench = mdf["slot"].between(12, 15, inclusive="both").fillna(False)

        # STARTING XI (slots 1..11), sorted by slot
        xi = mdf.loc[sel_start].sort_values("slot").copy()
        xi_view = (
            xi[["slot", "position", "second_name", "team_short", "multiplier", "C", "VC", "gw_pts", "total_points"]]
            .rename(columns={
                "slot": "Slot",
                "position": "Pos",
                "second_name": "Player",
                "team_short": "Team",
                "multiplier": "x",
                "gw_pts": "GW Pts",
                "total_points": "Base Pts"
            })
            .reset_index(drop=True)
        )
        xi_view.columns = [str(c) for c in xi_view.columns]  # ensure plain string col names

        st.markdown("### Starting XI")
        if xi_view.empty:
            st.info("No starting XI data for this GW.")
        else:
            st.dataframe(xi_view, use_container_width=True)

        # BENCH (slots 12..15)
        bench = mdf.loc[sel_bench].sort_values("slot").copy()
        bench_view = (
            bench[["slot", "position", "second_name", "team_short", "total_points"]]
            .rename(columns={
                "slot": "Slot",
                "position": "Pos",
                "second_name": "Player",
                "team_short": "Team",
                "total_points": "Base Pts"
            })
            .reset_index(drop=True)
        )
        bench_view.columns = [str(c) for c in bench_view.columns]

        if not bench_view.empty:
            st.markdown("### Bench")
            st.dataframe(bench_view, use_container_width=True)

        st.metric("Total Bench Points", bench_total)

    except Exception as e:
        st.warning(f"Could not load picks: {e}")
