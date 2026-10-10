"""
WFM Command Center - Streamlit dashboard
Run:  pip install streamlit pandas numpy openpyxl
      streamlit run wfm_dashboard.py

Modules: 1) Roster & shifts  2) Breaks & buffer health  3) 30-min heatmap & OT  4) Attendance & allowances
Assumption: the weekly roster repeats, so overnight shifts on the last day spill into the first day.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="WFM Command Center", page_icon="📊", layout="wide")
st.markdown(
    """<style>
div[data-testid="stMetric"]{background:#f6f8fb;border:1px solid #e3e8ef;border-radius:10px;padding:10px 14px}
@media (prefers-color-scheme: dark){div[data-testid="stMetric"]{background:#1e2530;border-color:#303a49}}
</style>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- constants
META = ["Employee ID", "Name", "SF Name", "LOB", "TL"]
SAMPLE_DATES = ["27-Sep", "28-Sep", "29-Sep", "30-Sep", "1-Oct", "2-Oct", "3-Oct"]
SHIFTS = ["14:00 - 23:00", "15:00 - 00:00", "17:00 - 02:00", "19:00 - 04:00",
          "20:00 - 05:00", "21:00 - 06:00", "23:00 - 08:00"]
EXC_TYPES = ["None", "Late Login", "Early Leave", "Unplanned Absence"]
BREAK_COLS = ["First Short Break", "Lunch Break", "Last Short Break"]
EDIT_COLS = BREAK_COLS + ["Exception", "Exception Minutes"]
SLOTS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
G96 = np.arange(96) * 30  # slot start minutes over 2 days (shift day + next day)
UNIT_F = {"Minutes": 30, "Hours": 0.5, "FTE": 1}  # 1 FTE = 30 min in a 30-min interval
UNIT_SFX = {"Minutes": "(min)", "Hours": "(hrs)", "FTE": "(FTE)"}
UNIT_FMT = {"Minutes": "{:.0f}", "Hours": "{:.2f}", "FTE": "{:.2f}"}
SHIFT_COLORS = ["#dbeafe", "#e0e7ff", "#ede9fe", "#fce7f3", "#ffedd5", "#fef9c3", "#d1fae5"]


# ----------------------------------------------------------------------------- helpers
def to_min(t):
    try:
        h, m = str(t).strip().split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception:
        return None


def fmt(m):
    return f"{(int(m) // 60) % 24:02d}:{int(m) % 60:02d}"


def parse_shift(s):
    """'17:00 - 02:00' -> (1020, 1560). Returns None for OFF/invalid. End <= start means overnight."""
    if not isinstance(s, str) or "-" not in s:
        return None
    a, b = s.split("-", 1)
    sm, em = to_min(a), to_min(b)
    if sm is None or em is None:
        return None
    return sm, (em + 1440 if em <= sm else em)


def clean_shift(x):
    s = str(x).strip()
    return "OFF" if s.upper() in ("OFF", "", "NAN", "NONE") else re.sub(r"\s*-\s*", " - ", s)


def norm_col(c):
    return f"{c.day}-{c.strftime('%b')}" if hasattr(c, "strftime") else str(c).strip()


def frac(a0, a1):
    """Fraction (0-1) of each 30-min slot in the 96-slot span covered by [a0, a1) minutes."""
    return np.clip(np.minimum(G96 + 30, a1) - np.maximum(G96, a0), 0, 30) / 30


def night_tier(shift):
    p = parse_shift(shift)
    if not p:
        return "None"
    h = p[0] // 60
    return "Overnight" if h >= 19 else "Late" if h >= 15 else "None"


def smap(styler, fn, **kw):
    return (styler.map if hasattr(styler, "map") else styler.applymap)(fn, **kw)


def read_any(f):
    return pd.read_excel(f) if f.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(f)


# ----------------------------------------------------------------------------- core engine
def build_vectors(roster, breaks, dates, lunch_min, short_min):
    """Per shift-day array (agents x [sched, absent, lunch, short] x 96 half-hour slots)."""
    out = {}
    for d in dates:
        bk = breaks.get(d)
        bmap = bk.set_index("Employee ID").to_dict("index") if bk is not None and len(bk) else {}
        V = np.zeros((len(roster), 4, 96))
        for i, (eid, sh) in enumerate(zip(roster["Employee ID"], roster[d])):
            p = parse_shift(sh)
            if not p:
                continue
            s, e = p
            sched = frac(s, e)
            r = bmap.get(eid, {})
            exc = r.get("Exception", "None")
            mins = pd.to_numeric(r.get("Exception Minutes", 0), errors="coerce")
            mins = 0 if pd.isna(mins) else float(mins)
            if exc == "Unplanned Absence":
                ab = sched.copy()
            elif exc == "Late Login":
                ab = frac(s, s + mins)
            elif exc == "Early Leave":
                ab = frac(e - mins, e)
            else:
                ab = np.zeros(96)
            ab = np.minimum(ab, sched)
            present = sched - ab

            def bstart(t):
                m = to_min(t)
                return None if m is None else (m + 1440 if m < s else m)

            lu, sb = np.zeros(96), np.zeros(96)
            m = bstart(r.get("Lunch Break"))
            if m is not None:
                lu = frac(m, m + lunch_min)
            for c in ("First Short Break", "Last Short Break"):
                m = bstart(r.get(c))
                if m is not None:
                    sb += frac(m, m + short_min)
            lu = np.minimum(lu, present)
            sb = np.minimum(sb, np.clip(present - lu, 0, None))
            V[i] = [sched, ab, lu, sb]
        out[d] = V
    return out


def agg(vecs, dates, di, mask_fn):
    """Sum of (sched, absent, lunch, short) per 30-min slot of day `di`, incl. spill-over from previous day."""
    cur = vecs[dates[di]][mask_fn(dates[di])][:, :, :48].sum(axis=0)
    prv = dates[di - 1]  # di=0 wraps to last day (weekly pattern repeats)
    return cur + vecs[prv][mask_fn(prv)][:, :, 48:].sum(axis=0)


def tgt_for(targets, lobs):
    t = targets.set_index("Interval")
    cols = [c for c in lobs if c in t.columns]
    if not cols:
        return np.zeros(48)
    return t[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).reindex(SLOTS).fillna(0).to_numpy()


def basis_fte(a, basis):
    sched, ab, lu, sb = a
    return sched if basis == "Scheduled FTE" else sched - lu - sb - ab


def status(buf, tgt, sched):
    if tgt == 0 and sched == 0:
        return "⚪ Closed"
    return "🟢 Optimal Buffer" if buf > 1 else "🟡 Fair Buffer" if buf >= 0 else "🔴 Interval Failure"


def interval_table(a, tgt):
    sched, ab, lu, sb = a
    net = sched - lu - sb - ab
    buf = np.round(net - tgt, 2)
    return pd.DataFrame({
        "Interval": SLOTS, "Scheduled FTE": sched.round(2), "On Lunch": lu.round(2),
        "On Short Break": sb.round(2), "Absent/Late": ab.round(2), "Net Floor FTE": net.round(2),
        "Client Target": tgt, "Buffered FTE": buf,
        "Status": [status(b, t, s) for b, t, s in zip(buf, tgt, sched)],
    })


def heat_color(v):
    if pd.isna(v):
        return "background-color:#f1f1f1;color:#aaa"
    if abs(v) < 1e-9:
        return "background-color:#ffeb9c;color:#7a5c00"
    return "background-color:#c6efce;color:#006100" if v > 0 else "background-color:#ffc7ce;color:#9c0006"


def status_color(v):
    return {"🟢": "background-color:#c6efce;color:#006100", "🟡": "background-color:#ffeb9c;color:#7a5c00",
            "🔴": "background-color:#ffc7ce;color:#9c0006"}.get(str(v)[:1], "color:#999")


# ----------------------------------------------------------------------------- sample data
def default_breaks(roster, d):
    rows = []
    for i, r in roster.reset_index(drop=True).iterrows():
        p = parse_shift(r[d])
        if not p:
            continue
        s, k = p[0], i % 4
        rows.append({"Employee ID": r["Employee ID"], "Name": r["Name"], "LOB": r["LOB"], "TL": r["TL"],
                     "Shift": r[d], "First Short Break": fmt(s + 120 + 15 * k), "Lunch Break": fmt(s + 270 + 30 * k),
                     "Last Short Break": fmt(s + 420 + 15 * (k % 2)), "Exception": "None", "Exception Minutes": 0})
    return pd.DataFrame(rows)


def init_sample():
    rng = np.random.default_rng(7)
    first = ["Omar", "Mona", "Youssef", "Salma", "Hady", "Nour", "Karim", "Dina", "Mostafa", "Heba", "Tarek", "Rana",
             "Amr", "Yara", "Khaled", "Laila", "Ziad", "Farida", "Mahmoud", "Aya", "Sherif", "Mariam", "Hossam",
             "Nada", "Ibrahim", "Jana", "Walid", "Reem"]
    last = ["Fathy", "Saad", "Nabil", "Gamal", "Mansour", "Ashraf", "Helmy", "Zaki", "Farouk", "Lotfy"]
    rows = []
    for i in range(28):
        eng = i < 16
        row = {"Employee ID": f"E{1001 + i}", "Name": f"{first[i]} {last[i % 10]}",
               "SF Name": f"{first[i].lower()}.{last[i % 10].lower()}", "LOB": "NMG-ENG" if eng else "NMG-SP",
               "TL": ["Ahmed Hassan", "Mona Samir"][i % 2] if eng else "Karim Adel"}
        off = int(rng.integers(0, 7))
        for j, d in enumerate(SAMPLE_DATES):
            if j in (off, (off + 1) % 7):
                row[d] = "OFF"
            else:
                row[d] = SHIFTS[i % 7] if rng.random() > 0.15 else SHIFTS[int(rng.integers(0, 7))]
        rows.append(row)
    roster = pd.DataFrame(rows)
    breaks = {}
    for d in SAMPLE_DATES:
        b = default_breaks(roster, d)
        for idx in b.index:
            x = rng.random()
            if x < 0.05:
                b.loc[idx, "Exception"] = "Unplanned Absence"
            elif x < 0.10:
                b.loc[idx, ["Exception", "Exception Minutes"]] = ["Late Login", int(rng.choice([15, 30, 60]))]
            elif x < 0.13:
                b.loc[idx, ["Exception", "Exception Minutes"]] = ["Early Leave", 60]
        breaks[d] = b
    vecs = build_vectors(roster, breaks, SAMPLE_DATES, 30, 15)
    tg = {"Interval": SLOTS}
    for lob in ("NMG-ENG", "NMG-SP"):
        acc = sum(agg(vecs, SAMPLE_DATES, di, lambda d, lob=lob: (roster["LOB"] == lob).to_numpy())[0]
                  for di in range(7)) / 7
        tg[lob] = np.floor(acc * 0.8 + 0.3).astype(int)
    st.session_state.update(roster=roster, breaks=breaks, targets=pd.DataFrame(tg))


def apply_uploads(f_r, f_t, f_b):
    if f_r:
        r = read_any(f_r)
        r.columns = [norm_col(c) for c in r.columns]
        for c in META:
            if c not in r:
                r[c] = ""
        dates = [c for c in r.columns if c not in META]
        for d in dates:
            r[d] = r[d].map(clean_shift)
        r = r[META + dates]
        st.session_state.roster = r
        st.session_state.breaks = {d: default_breaks(r, d) for d in dates}
    if f_t:
        t = read_any(f_t)
        t["Interval"] = t["Interval"].astype(str).str[:5]
        st.session_state.targets = t
    if f_b:
        b = read_any(f_b)
        b["Date"] = b["Date"].map(norm_col)
        for d, g in b.groupby("Date"):
            if d in st.session_state.breaks:
                base = st.session_state.breaks[d].set_index("Employee ID")
                base.update(g.set_index("Employee ID")[[c for c in EDIT_COLS if c in g]])
                st.session_state.breaks[d] = base.reset_index()


if "roster" not in st.session_state:
    init_sample()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📊 WFM Command Center")
    with st.expander("📂 Data inputs (optional)"):
        f_r = st.file_uploader("Master roster (CSV/XLSX)", type=["csv", "xlsx"], key="u_r")
        f_t = st.file_uploader("Client FTE targets: Interval + one column per LOB", type=["csv", "xlsx"], key="u_t")
        f_b = st.file_uploader("Break/exception log: Date, Employee ID, breaks, Exception, Exception Minutes",
                               type=["csv", "xlsx"], key="u_b")
        c1, c2 = st.columns(2)
        if c1.button("Apply"):
            apply_uploads(f_r, f_t, f_b)
            st.rerun()
        if c2.button("Reset sample"):
            init_sample()
            st.rerun()

    roster, targets, breaks = st.session_state.roster, st.session_state.targets, st.session_state.breaks
    DATES = [c for c in roster.columns if c not in META]
    all_shifts = sorted({s for d in DATES for s in roster[d].unique()},
                        key=lambda s: (SHIFTS.index(s) if s in SHIFTS else 99, s))

    st.header("🎛️ Global filters")
    sel_date = st.selectbox("Date", DATES)
    sel_lobs = st.multiselect("LOB", sorted(roster["LOB"].unique()), default=sorted(roster["LOB"].unique()))
    sel_tls = st.multiselect("Team Leader", sorted(roster["TL"].unique()), default=sorted(roster["TL"].unique()))
    sel_shifts = st.multiselect("Shift type", all_shifts, default=all_shifts)

    st.header("⚙️ Settings")
    lunch_min = st.select_slider("Lunch duration (min)", [30, 60], 30)
    short_min = st.select_slider("Short break duration (min)", [15, 30], 15)
    basis = st.radio("Heatmap/OT basis", ["Scheduled FTE", "Net Floor FTE"], help="Net = after breaks and exceptions")
    unit = st.radio("Display unit", ["Minutes", "Hours", "FTE"], help="1 FTE = 30 minutes of staffed time per 30-min interval")
    late_rate = st.number_input("Late shift allowance, start 15:00-18:59 (EGP)", 0, 1000, 75, 5)
    night_rate = st.number_input("Overnight allowance, start 19:00+ (EGP)", 0, 1000, 150, 5)


def mask_fn(d, lobs=None):
    lobs = sel_lobs if lobs is None else lobs
    return (roster["LOB"].isin(lobs) & roster["TL"].isin(sel_tls) & roster[d].isin(sel_shifts)).to_numpy()


vecs = build_vectors(roster, breaks, DATES, lunch_min, short_min)
di = DATES.index(sel_date)
UF = UNIT_F[unit]
tgt = tgt_for(targets, sel_lobs)
a_day = agg(vecs, DATES, di, mask_fn)
it = interval_table(a_day, tgt)

st.title("Workforce Management Dashboard")
st.caption(f"Date: **{sel_date}** | LOB: {', '.join(sel_lobs) or '-'} | {len(sel_tls)} TL(s) | {len(sel_shifts)} shift type(s)")
if not sel_lobs:
    st.warning("Select at least one LOB.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Roster & Shifts", "⏱️ Breaks & Buffer Health",
                                  "🔥 Interval Heatmap & OT", "📈 Attendance & Allowances"])

# ----------------------------------------------------------------------------- Module 1
with tab1:
    row_mask = roster["LOB"].isin(sel_lobs) & roster["TL"].isin(sel_tls) & roster[sel_date].isin(sel_shifts)
    view = roster[row_mask]
    daily = {"Working": [], "OFF": []}
    for d in DATES:
        col = roster.loc[roster["LOB"].isin(sel_lobs) & roster["TL"].isin(sel_tls), d]
        col = col[col.isin(sel_shifts)]
        daily["Working"].append(int((col != "OFF").sum()))
        daily["OFF"].append(int((col == "OFF").sum()))
    daily = pd.DataFrame(daily, index=DATES)

    k = st.columns(5)
    k[0].metric("Agents shown", len(view))
    k[1].metric("Scheduled shifts (week)", int(daily["Working"].sum()))
    k[2].metric("OFF days (week)", int(daily["OFF"].sum()))
    k[3].metric(f"Working {sel_date}", int(daily.loc[sel_date, "Working"]))
    k[4].metric(f"OFF {sel_date}", int(daily.loc[sel_date, "OFF"]))

    colmap = {s: SHIFT_COLORS[i % 7] for i, s in enumerate(SHIFTS)}
    sty = smap(view.style, lambda v: "background-color:#e5e7eb;color:#9ca3af" if v == "OFF"
               else f"background-color:{colmap.get(v, '#fff')};color:#111", subset=DATES)
    sty = sty.set_properties(subset=[sel_date], **{"border": "2px solid #3b82f6", "font-weight": "600"})
    st.dataframe(sty, hide_index=True, height=min(35 * len(view) + 40, 600))

    c1, c2 = st.columns(2)
    c1.subheader("Working vs OFF per day")
    c1.bar_chart(daily)
    c2.subheader(f"Shift mix on {sel_date}")
    wk = view[view[sel_date] != "OFF"]
    if len(wk):
        mix = pd.crosstab(wk[sel_date], wk["LOB"]).reindex([s for s in SHIFTS if s in set(wk[sel_date])] +
                                                            sorted(set(wk[sel_date]) - set(SHIFTS)))
        c2.bar_chart(mix)
    else:
        c2.info("No working agents for this selection.")

# ----------------------------------------------------------------------------- Module 2
with tab2:
    st.subheader(f"Break schedule & exceptions - {sel_date}")
    base = breaks[sel_date]
    ids = roster.loc[mask_fn(sel_date), "Employee ID"]
    bview = base[base["Employee ID"].isin(ids)].reset_index(drop=True)
    sig = hash((tuple(sel_lobs), tuple(sel_tls), tuple(sel_shifts)))
    time_cfg = lambda lbl: st.column_config.TextColumn(lbl, validate=r"^([01]\d|2[0-3]):[0-5]\d$", help="HH:MM")
    edited = st.data_editor(
        bview, key=f"ed_{sel_date}_{sig}", hide_index=True, height=min(35 * len(bview) + 40, 420),
        disabled=["Employee ID", "Name", "LOB", "TL", "Shift"],
        column_config={
            "First Short Break": time_cfg("First Short Break"), "Lunch Break": time_cfg("Lunch Break"),
            "Last Short Break": time_cfg("Last Short Break"),
            "Exception": st.column_config.SelectboxColumn("Exception", options=EXC_TYPES, required=True),
            "Exception Minutes": st.column_config.NumberColumn(
                "Exception Minutes", min_value=0, max_value=540, step=5, help="Minutes late / minutes left early"),
        },
    )

    def _norm(df):
        x = df[EDIT_COLS].copy()
        x["Exception Minutes"] = pd.to_numeric(x["Exception Minutes"], errors="coerce").fillna(0).astype(int)
        return x.fillna("").astype(str).reset_index(drop=True)

    if not _norm(edited).equals(_norm(bview)):  # write edits back, then recompute everything
        new = base.set_index("Employee ID")
        ed = edited.set_index("Employee ID")
        for c in EDIT_COLS:
            new.loc[ed.index, c] = ed[c]
        new["Exception Minutes"] = pd.to_numeric(new["Exception Minutes"], errors="coerce").fillna(0).astype(int)
        st.session_state.breaks[sel_date] = new.reset_index()
        st.rerun()

    st.subheader("Live buffer health")
    cnt = it["Status"].value_counts()
    peak = it.loc[it["On Lunch"].idxmax()]
    k = st.columns(4)
    k[0].metric("🟢 Optimal intervals", int(cnt.get("🟢 Optimal Buffer", 0)))
    k[1].metric("🟡 Fair intervals", int(cnt.get("🟡 Fair Buffer", 0)))
    k[2].metric("🔴 Failed intervals", int(cnt.get("🔴 Interval Failure", 0)))
    k[3].metric("Peak lunch concurrency", f"{peak['On Lunch']:.0f}", help=f"At {peak['Interval']}")

    risk = it[(it["On Lunch"] >= 2) & (it["Buffered FTE"] <= 1)]
    if len(risk):
        st.warning(f"⚠️ 2+ agents on lunch while buffer ≤ 1 at: {', '.join(risk['Interval'])}. Consider staggering lunches.")
    else:
        st.success("No lunch-concurrency risk in low-coverage windows.")

    num = ["Scheduled FTE", "On Lunch", "On Short Break", "Absent/Late", "Net Floor FTE", "Client Target", "Buffered FTE"]
    ren = {c: f"{c.replace(' FTE', '')} {UNIT_SFX[unit]}" for c in num} if unit != "FTE" else {}
    itd = it.copy()
    itd[num] = (it[num] * UF).round(2)
    itd = itd.rename(columns=ren)
    ncols = [ren.get(c, c) for c in num]
    bcol = ren.get("Buffered FTE", "Buffered FTE")
    sty = smap(itd.style, status_color, subset=["Status"])
    sty = smap(sty, heat_color, subset=[bcol]).format({c: UNIT_FMT[unit] for c in ncols})
    st.dataframe(sty, hide_index=True, height=520)
    st.line_chart(itd.set_index("Interval")[[ren.get("Net Floor FTE", "Net Floor FTE"), ren.get("Client Target", "Client Target")]])

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Interval buffer report (CSV)", itd.to_csv(index=False).encode("utf-8-sig"),
                       f"buffer_report_{sel_date}.csv", "text/csv")
    d2.download_button("⬇️ Break schedule & exceptions (CSV)", base.to_csv(index=False).encode("utf-8-sig"),
                       f"break_schedule_{sel_date}.csv", "text/csv")

# ----------------------------------------------------------------------------- Module 3
with tab3:
    st.caption(f"Basis: **{basis}** · Unit: **{unit}**. 🟩 surplus · 🟨 exact target · 🟥 understaffed · grey = closed")
    week, ot = {}, {}
    for j, d in enumerate(DATES):
        a = agg(vecs, DATES, j, mask_fn)
        f = basis_fte(a, basis)
        diff = np.round(f - tgt, 2)
        week[d] = np.where((tgt == 0) & (f == 0), np.nan, diff)
        ot[d] = float(np.clip(tgt - f, 0, None).sum() * 0.5)
    week = pd.DataFrame(week, index=SLOTS)
    ot = pd.Series(ot)
    f_sel = basis_fte(a_day, basis)
    deficit = np.clip(tgt - f_sel, 0, None)

    k = st.columns(4)
    k[0].metric("OT hours needed (week)", f"{ot.sum():.1f}")
    k[1].metric(f"OT hours needed ({sel_date})", f"{ot[sel_date]:.1f}")
    k[2].metric(f"Understaffed intervals ({sel_date})", int((deficit > 0).sum()))
    k[3].metric("Largest deficit (FTE)", f"{deficit.max():.1f}")

    st.subheader("Weekly 30-min heatmap (FTE vs target)")
    HFMT = "{:+.0f}" if unit == "Minutes" else "{:+.2f}"
    st.dataframe(smap((week * UF).style, heat_color).format(HFMT, na_rep="·"), height=620)

    st.subheader(f"{sel_date}: heatmap by LOB")
    cols = {}
    for lob in sel_lobs:
        a = agg(vecs, DATES, di, lambda d, lob=lob: mask_fn(d, [lob]))
        t, f = tgt_for(targets, [lob]), basis_fte(a, basis)
        cols[lob] = np.where((t == 0) & (f == 0), np.nan, np.round(f - t, 2))
    lob_heat = pd.DataFrame(cols, index=SLOTS)
    st.dataframe(smap((lob_heat * UF).style, heat_color).format(HFMT, na_rep="·"), height=400)

    c1, c2 = st.columns(2)
    c1.subheader("OT hours by day")
    c1.bar_chart(ot)
    c2.subheader(f"Understaffed intervals - {sel_date}")
    ot_df = pd.DataFrame({"Interval": SLOTS, "Target": tgt, "FTE": f_sel.round(2),
                          "Deficit": deficit.round(2), "OT Hours": (deficit * 0.5).round(2)})
    c2.dataframe(ot_df[ot_df["Deficit"] > 0], hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Weekly interval matrix (CSV)", (week * UF).rename_axis("Interval").reset_index().to_csv(index=False)
                       .encode("utf-8-sig"), "weekly_interval_matrix.csv", "text/csv")
    d2.download_button("⬇️ OT analysis for date (CSV)", ot_df.to_csv(index=False).encode("utf-8-sig"),
                       f"ot_{sel_date}.csv", "text/csv")

# ----------------------------------------------------------------------------- Module 4
with tab4:
    ad = roster.loc[mask_fn(sel_date), META + [sel_date]].rename(columns={sel_date: "Shift"})
    ad = ad[ad["Shift"].map(lambda s: parse_shift(s) is not None)]
    if ad.empty:
        st.info("No scheduled agents for this selection.")
    else:
        ad = ad.merge(breaks[sel_date][["Employee ID", "Exception", "Exception Minutes"]], on="Employee ID", how="left")
        ad["Exception"] = ad["Exception"].fillna("None")
        ad["Exception Minutes"] = pd.to_numeric(ad["Exception Minutes"], errors="coerce").fillna(0)
        ad["Hours"] = ad["Shift"].map(lambda s: (parse_shift(s)[1] - parse_shift(s)[0]) / 60)
        ad["Absent"] = (ad["Exception"] == "Unplanned Absence").astype(int)
        ad["Lost"] = np.where(ad["Absent"] == 1, ad["Hours"],
                              np.where(ad["Exception"].isin(["Late Login", "Early Leave"]),
                                       np.minimum(ad["Exception Minutes"] / 60, ad["Hours"]), 0))
        ad["Attended"] = 1 - ad["Absent"]
        ad["Tier"] = ad["Shift"].map(night_tier)
        ad["Late Shifts"] = ((ad["Tier"] == "Late") & (ad["Attended"] == 1)).astype(int)
        ad["Overnight Shifts"] = ((ad["Tier"] == "Overnight") & (ad["Attended"] == 1)).astype(int)
        ad["Night Allowance (EGP)"] = ad["Late Shifts"] * late_rate + ad["Overnight Shifts"] * night_rate

        g = ad.groupby("LOB").agg(**{
            "Total Scheduled": ("Employee ID", "count"), "Attended": ("Attended", "sum"),
            "Unplanned Absences": ("Absent", "sum"), "Lost Hours": ("Lost", "sum"), "Sched Hours": ("Hours", "sum"),
            "Late Shifts": ("Late Shifts", "sum"), "Overnight Shifts": ("Overnight Shifts", "sum"),
            "Night Allowance (EGP)": ("Night Allowance (EGP)", "sum")}).reset_index()
        tot = g.drop(columns="LOB").sum()
        tot["LOB"] = "TOTAL"
        g = pd.concat([g, tot.to_frame().T], ignore_index=True)
        for c in g.columns[1:]:
            g[c] = pd.to_numeric(g[c])
        g["Attendance %"] = (g["Attended"] / g["Total Scheduled"] * 100).round(1)
        g["Shrinkage %"] = (g["Lost Hours"] / g["Sched Hours"] * 100).round(1)
        T = g.iloc[-1]

        k = st.columns(4)
        k[0].metric("Scheduled FTE", int(T["Total Scheduled"]))
        k[1].metric("Attended FTE", int(T["Attended"]), delta=int(T["Attended"] - T["Total Scheduled"]))
        k[2].metric("Attendance %", f"{T['Attendance %']:.1f}%")
        k[3].metric("Shrinkage %", f"{T['Shrinkage %']:.1f}%", help="(Absence + late + early-leave hours) / scheduled hours")

        st.subheader(f"Executive summary by LOB - {sel_date}")
        show = g[["LOB", "Total Scheduled", "Attended", "Unplanned Absences", "Shrinkage %", "Attendance %",
                  "Late Shifts", "Overnight Shifts", "Night Allowance (EGP)"]]
        st.dataframe(show.style.format({"Shrinkage %": "{:.1f}%", "Attendance %": "{:.1f}%",
                                        "Night Allowance (EGP)": "{:,.0f}", "Total Scheduled": "{:.0f}",
                                        "Attended": "{:.0f}", "Unplanned Absences": "{:.0f}",
                                        "Late Shifts": "{:.0f}", "Overnight Shifts": "{:.0f}"}), hide_index=True)
        st.caption("Allowance applies to attended agents only; rates are configurable in the sidebar.")
        st.subheader("Exceptions logged")
        exc = ad[ad["Exception"] != "None"][["Employee ID", "Name", "LOB", "TL", "Shift", "Exception", "Exception Minutes"]]
        st.dataframe(exc, hide_index=True) if len(exc) else st.success("No exceptions logged.")
        st.download_button("⬇️ Executive summary (CSV)", show.to_csv(index=False).encode("utf-8-sig"),
                           f"attendance_summary_{sel_date}.csv", "text/csv")
