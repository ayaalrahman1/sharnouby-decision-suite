"""El-Sharnouby Group – 7 Decision Apps + Final Mission (Streamlit, one navigation sidebar).
Run:  pip install -r requirements.txt  ;  streamlit run app.py
Put the unzipped CSV files in ./data (or set DATA_DIR)."""
import os, numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA = os.environ.get("DATA_DIR", "data")
st.set_page_config(page_title="El-Sharnouby Decision Suite", layout="wide")
EGP = lambda x: f"EGP {x/1e6:,.1f}M"
NAVY, BLUE, RED, GREEN, ORG, GRAY = "#1F3864", "#2E75B6", "#C0392B", "#27AE60", "#E67E22", "#95A5A6"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True, "font.size": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})
def en(name, pid):
    return pid if any("\u0600" <= ch <= "\u06FF" for ch in str(name)) else f"{pid} {str(name)[:30]}"
def new(w=7, h=3.2): return plt.subplots(figsize=(w, h))
def show(f): f.tight_layout(); st.pyplot(f, use_container_width=True); plt.close(f)
def bars(ser, title, colors=BLUE, fmt="{:,.0f}", horizontal=False, w=7, h=3.2, ylabel=""):
    f, a = new(w, h); ser = ser.dropna()
    if horizontal:
        a.barh([str(i) for i in ser.index], ser.values, color=colors); a.invert_yaxis(); a.grid(axis="y", alpha=0)
        for i, v in enumerate(ser.values): a.text(v, i, " " + fmt.format(v), va="center", fontsize=8)
    else:
        a.bar([str(i) for i in ser.index], ser.values, color=colors); a.grid(axis="x", alpha=0); a.set_ylabel(ylabel)
        for i, v in enumerate(ser.values): a.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    a.set_title(title); return f, a
def heat(df, title, cmap="RdYlGn", vmin=None, vmax=None, fmt="{:.0f}", w=7, h=3.6):
    f, a = new(w, h); im = a.imshow(df.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"); a.set_xticks(range(df.shape[1])); a.set_xticklabels([str(c) for c in df.columns], rotation=30, ha="right"); a.set_yticks(range(df.shape[0])); a.set_yticklabels([str(i) for i in df.index]); a.grid(False)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]; lo, hi = np.nanmin(df.values), np.nanmax(df.values); dark = (not pd.isna(v)) and cmap in ("Reds", "Blues") and (v - lo) / (hi - lo + 1e-9) > .6; a.text(j, i, "" if pd.isna(v) else fmt.format(v), ha="center", va="center", fontsize=8, color="white" if dark else "black")
    a.set_title(title); return f, a

@st.cache_data
def load():
    r = lambda f, **k: pd.read_csv(os.path.join(DATA, f if os.path.exists(os.path.join(DATA, f)) else f + ".gz"), encoding="utf-8-sig", **k)
    s = r("Sales_Transactions.csv", parse_dates=["Order_Date"]); s["FY"] = np.where(s.Order_Date.dt.month >= 10, s.Order_Date.dt.year + 1, s.Order_Date.dt.year)
    return dict(s=s, iv=r("Inventory_History.csv", parse_dates=["Snapshot_Month"]), m=r("Marketing.csv"), c=r("Customers.csv"), p=r("Production_Orders.csv", parse_dates=["Start_Date"]),
                q=r("Quality.csv"), pb=r("Printing_Batches.csv"), dt=r("Machine_Downtime.csv", parse_dates=["Date"]), po=r("Purchase_Orders.csv", parse_dates=["Order_Date", "Promised_Date", "Actual_Receipt_Date"]),
                mt=r("Materials.csv"), su=r("Suppliers.csv"), bom=r("Product_BOM.csv"), pm=r("Product_Master.csv").set_index("Product_ID"))
D = load(); s, iv, m, c, p, q, pb, dt, po, mt, su, bom, pm = (D[k] for k in "s iv m c p q pb dt po mt su bom pm".split())

# ---------- shared logic
@st.cache_data
def wholesale():
    w = s[s.Channel == "Wholesale"].copy(); w["std_c"] = w.Product_ID.map((w.Estimated_COGS_EGP / w.Quantity).groupby(w.Product_ID).mean()); return w
W = wholesale()
def max_disc(cost, lst, floor): return np.maximum(0, 1 - cost / lst / (1 - floor))
def simulate(df, floor, vol_loss, ceiling):
    cap = np.minimum(ceiling, max_disc(df.std_c, df.List_Price_EGP, floor)); nd = np.minimum(df.Discount_Pct, cap); rp = nd < df.Discount_Pct - 1e-9
    qty = np.where(rp, df.Quantity * (1 - vol_loss), df.Quantity); gp = ((df.List_Price_EGP * (1 - nd) - df.std_c) * qty).sum()
    return int(rp.sum()), (df.Net_Sales_EGP - df.std_c * df.Quantity).sum(), gp
def _X(idx, t0=0): return np.column_stack([np.ones(len(idx)), np.arange(t0, t0 + len(idx))] + [(idx.month == k).astype(float) for k in range(2, 13)])
def forecast(y, h=3):
    X = _X(y.index); b = np.linalg.solve(X.T @ X + np.diag([0, 0] + [2] * 11), X.T @ np.log1p(y.values))
    f = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS"); mod = np.expm1(_X(f, len(y)) @ b)
    g = min(max((y.iloc[-12:].sum() + 1) / (y.iloc[-24:-12].sum() + 1), .8), 1.25); sn = np.array([y.loc[d - pd.DateOffset(years=1)] * g for d in f])
    return pd.Series((mod + sn) / 2 / 0.961, index=f)  # 0.961: bias correction from rolling-origin backtest
@st.cache_data
def fc_table():
    loc = {"LOC001": "Retail", "LOC002": "Online", "LOC003": "Wholesale"}; rows = []
    for (L, P), g in iv.groupby(["Location_ID", "Product_ID"]):
        y = g.set_index("Snapshot_Month").Sold_Qty.astype(float); f = forecast(y); l = g.sort_values("Snapshot_Month").iloc[-1]
        rows.append(dict(Channel=loc[L], Product_ID=P, Product=pm.loc[P, "Product_Name"], F30=f.iloc[0], F60=f.iloc[:2].sum(), F90=f.sum(), Stock=l.Closing_Stock, Lead=l.Supplier_Lead_Time_Days, m1=f.iloc[0], m2=f.iloc[1], m3=f.iloc[2]))
    F = pd.DataFrame(rows); F["Daily"] = F.F90 / 92; return F

nav = st.sidebar.radio("Apps", ["01 Executive Command Center", "02 Smart Inventory & Replenishment", "03 Customer 360 & Retention", "04 Factory Digital Twin", "05 Marketing Attribution & Growth", "06 Branch Intelligence & Expansion", "07 Procurement, Cost & Supplier Risk", "★ Final Mission: Wholesale Margin Guardrail"])
st.sidebar.caption("Oct-2023 → Sep-2026 · FY = Oct–Sep · modeled training data")

# =============== APP 01
if nav.startswith("01"):
    st.title("Executive Command Center"); st.caption("If the CEO had one screen, what should it know?")
    fy = s.groupby("FY").agg(Net=("Net_Sales_EGP", "sum"), GP=("Gross_Profit_EGP", "sum"), Units=("Quantity", "sum")); fy["GM"] = fy.GP / fy.Net; fy["Growth"] = fy.Net.pct_change(); L = fy.iloc[-1]
    k = st.columns(5); k[0].metric("FY26 net sales", EGP(L.Net), f"{L.Growth:.1%}"); k[1].metric("Gross profit", EGP(L.GP)); k[2].metric("Gross margin", f"{L.GM:.1%}"); k[3].metric("Units", f"{L.Units:,.0f}"); k[4].metric("Refunds % of sales", f"{s[s.FY==2026].Refund_Value_EGP.sum()/L.Net:.1%}")
    ch = s[s.FY == 2026].groupby("Channel").agg(Net=("Net_Sales_EGP", "sum"), GP=("Gross_Profit_EGP", "sum")); ch["Margin %"] = ch.GP / ch.Net * 100; ch["Rev share %"] = ch.Net / ch.Net.sum() * 100; ch["GP share %"] = ch.GP / ch.GP.sum() * 100
    po["ot"] = po.Actual_Receipt_Date <= po.Promised_Date; bad = po[po.Supplier_ID.isin(po[po.Order_Date.dt.year == 2026].groupby("Supplier_ID").ot.mean().nsmallest(2).index) & (po.Order_Date.dt.year == 2026)].ot.mean()
    p["late"] = p.Delay_Days > 0; so = iv[(iv.Snapshot_Month >= "2025-10-01") & (iv.Location_ID == "LOC003")].Stockout_Days_In_Month.sum()
    st.subheader("What management must know now")
    st.error(f"Wholesale = {ch.loc['Wholesale','Rev share %']:.0f}% of revenue but only {ch.loc['Wholesale','Margin %']:.0f}% margin (retail/online {ch.loc[['Retail','Online'],'Margin %'].mean():.0f}%). {int((W.Gross_Profit_EGP<0).sum()):,} wholesale lines sold below cost.")
    st.warning(f"Wholesale stockout days in FY26: {so:,.0f}. Two weakest suppliers deliver on time only {bad:.0%} of the time in 2026. Late production orders: {p[p.Start_Date.dt.year==2026].late.mean():.0%}.")
    a, b = st.columns(2)
    d = s[s.FY == 2026]; g_, n_, cg = d.Gross_Sales_EGP.sum() / 1e6, d.Net_Sales_EGP.sum() / 1e6, d.Estimated_COGS_EGP.sum() / 1e6; gp_ = n_ - cg
    f, ax = new(); steps = [("Gross sales", 0, g_, NAVY), ("Discounts", n_, g_ - n_, RED), ("Net sales", 0, n_, NAVY), ("COGS", gp_, cg, RED), ("Gross profit", 0, gp_, GREEN)]
    for i, (lab, bot, h_, c_) in enumerate(steps): ax.bar(i, h_, bottom=bot, color=c_); ax.text(i, bot + h_, f"{h_:,.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels([x[0] for x in steps]); ax.set_title("Where FY26 revenue goes (EGP M): discounts are 2.7× gross profit"); ax.grid(axis="x", alpha=0)
    with a: show(f)
    x = np.arange(3); f, ax = new(); ax.bar(x - .2, ch["Rev share %"], .4, color=NAVY, label="% of revenue"); ax.bar(x + .2, ch["GP share %"], .4, color=GREEN, label="% of gross profit"); ax.set_xticks(x); ax.set_xticklabels(ch.index); ax.legend(frameon=False); ax.set_title("Revenue share vs profit share (FY26)")
    for i, (u, v) in enumerate(zip(ch["Rev share %"], ch["GP share %"])): ax.text(i - .2, u, f"{u:.0f}%", ha="center", va="bottom", fontsize=8); ax.text(i + .2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    with b: show(f)
    mo = s.groupby("Month_Key").agg(n=("Net_Sales_EGP", "sum"), g=("Gross_Profit_EGP", "sum")); f, ax = new(7.2, 3); ax.bar(range(len(mo)), mo.n / 1e6, color=BLUE); ax.set_xticks(range(0, len(mo), 3)); ax.set_xticklabels(mo.index[::3], rotation=45, fontsize=7); ax.set_ylabel("Net sales (EGP M)"); ax2 = ax.twinx(); ax2.plot(range(len(mo)), mo.g / mo.n * 100, color=RED, lw=2); ax2.set_ylim(10, 22); ax2.set_ylabel("Gross margin %", color=RED); ax2.grid(False); ax2.spines["right"].set_visible(True); ax.set_title("Sales grow, margin stays flat near 15%")
    a, b = st.columns(2)
    with a: show(f)
    s["band"] = pd.cut(s.Discount_Pct, [-.01, 0, .1, .2, .3, .4], labels=["0", "0-10", "10-20", "20-30", "30-40"]); bm = s.groupby("band").apply(lambda x: x.Gross_Profit_EGP.sum() / x.Net_Sales_EGP.sum() * 100)
    f, ax = bars(bm, "Gross margin % by discount band", colors=[GREEN, GREEN, BLUE, ORG, RED], fmt="{:.0f}%", w=7, h=3)
    with b: show(f)
    h = s.groupby(s.Order_Date.dt.to_period("M").dt.to_timestamp()).Quantity.sum().iloc[-18:]; Fq = fc_table(); fm = Fq[["m1", "m2", "m3"]].sum(); fi = pd.date_range("2026-10-01", periods=3, freq="MS")
    f, ax = new(7, 3); ax.plot(h.index, h.values / 1e3, color=NAVY, lw=2, label="Actual"); ax.plot([h.index[-1]] + list(fi), [h.values[-1] / 1e3] + list(fm.values / 1e3), color=ORG, lw=2, ls="--", marker="o", label="Forecast (blend model)"); ax.set_ylabel("Units (000)"); ax.legend(frameon=False); ax.set_title("Units: last 18 months and 90-day forecast")
    c1, c2 = st.columns(2)
    with c1: show(f)
    tc = d.groupby("Customer_ID").agg(Net=("Net_Sales_EGP", "sum")).sort_values("Net", ascending=False).head(10); f, ax = bars(tc.Net / 1e6, "Top 10 customers FY26 (EGP M)", horizontal=True, fmt="{:.2f}", w=7, h=3)
    with c2: show(f)
    with st.expander("Data tables"):
        st.dataframe(ch.round(1)); st.dataframe(fy.assign(Net=lambda z: z.Net / 1e6, GP=lambda z: z.GP / 1e6).round(3))

# =============== APP 02
elif nav.startswith("02"):
    st.title("Smart Inventory & Replenishment"); st.caption("What should move, where, and when?")
    F = fc_table(); z = st.slider("Service level z", 1.0, 2.33, 1.65, 0.05); tw = st.slider("Wholesale target cover (days)", 15, 60, 35); tr = st.slider("Retail/Online target cover (days)", 30, 120, 60)
    F["Safety"] = z * np.sqrt(F.Lead) * F.Daily * 0.5 * 0.29; F["ROP"] = F.Daily * F.Lead + F.Safety; F["Cover d"] = F.Stock / F.Daily.replace(0, np.nan)
    F["Target"] = F.Daily * np.where(F.Channel == "Wholesale", tw, tr) + F.Safety; F["Order now"] = (F.F30 + F.Safety - F.Stock).clip(lower=0); F["Excess"] = (F.Stock - F.Target).clip(lower=0)
    F["Status"] = np.where(F.Stock < F.ROP, "REORDER NOW", np.where(F["Cover d"] > 150, "OVERSTOCK", "OK"))
    k = st.columns(4); k[0].metric("Reorder now (SKU-locations)", int((F.Status == "REORDER NOW").sum())); k[1].metric("Overstocked", int((F.Status == "OVERSTOCK").sum())); k[2].metric("Units to order", f"{F['Order now'].sum():,.0f}"); k[3].metric("Excess units (retail/online)", f"{F[F.Channel!='Wholesale'].Excess.sum():,.0f}")
    a, b = st.columns(2)
    lm = {"LOC001": "Retail", "LOC002": "Online", "LOC003": "Wholesale"}; cv = iv[iv.Snapshot_Month >= "2026-04-01"].groupby("Location_ID").Days_of_Cover.median().rename(index=lm)
    f, ax = bars(cv, "Median days of cover (Apr–Sep 2026)", colors=[BLUE, BLUE, RED], fmt="{:.0f}d", w=7, h=3)
    with a: show(f)
    stt = F.groupby(["Channel", "Status"]).size().unstack(fill_value=0).reindex(columns=["REORDER NOW", "OK", "OVERSTOCK"], fill_value=0); f, ax = new(7, 3); bot = np.zeros(len(stt))
    for col_, c_ in zip(stt.columns, [RED, GREEN, ORG]): ax.bar(stt.index, stt[col_], bottom=bot, color=c_, label=col_); bot += stt[col_].values
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3); ax.set_title("SKU-locations by stock status"); ax.grid(axis="x", alpha=0)
    with b: show(f)
    wo = F[F.Channel == "Wholesale"].nlargest(12, "Order now"); wo = pd.Series(wo["Order now"].values, index=[en(n, i) for n, i in zip(wo.Product, wo.Product_ID)]); f, ax = bars(wo, "Top 12 wholesale SKUs to order now (units)", colors=RED, horizontal=True, h=3.6)
    a, b = st.columns(2)
    with a: show(f)
    st.subheader("Move: transfer excess retail/online stock to cover wholesale shortage (assumes shared warehouse – verify)")
    ex = F[F.Channel != "Wholesale"].groupby("Product_ID").Excess.sum(); nd = F[F.Channel == "Wholesale"].set_index("Product_ID")["Order now"]
    T = pd.DataFrame({"Excess retail+online": ex, "Wholesale shortage": nd}).fillna(0); T["Transfer units"] = np.minimum(T.iloc[:, 0], T.iloc[:, 1]); T["Product"] = T.index.map(pm.Product_Name); T = T[T["Transfer units"] > 0].sort_values("Transfer units", ascending=False)
    if len(T): TT = T.head(10); f, ax = bars(pd.Series(TT["Transfer units"].values, index=[en(n, i) for n, i in zip(TT.Product, TT.index)]), "Top transfers (units)", colors=GREEN, horizontal=True, h=3.6)
    with b:
        if len(T): show(f)
    st.caption(f"Transfers could fill {T['Transfer units'].sum():,.0f} of {nd.sum():,.0f} wholesale units needed before placing new purchase orders.")
    ch_ = st.selectbox("Channel", ["All"] + sorted(F.Channel.unique())); V = F if ch_ == "All" else F[F.Channel == ch_]; st.subheader("30/60/90-day forecast & actions"); st.dataframe(V.sort_values("F90", ascending=False)[["Channel", "Product", "F30", "F60", "F90", "Stock", "Cover d", "ROP", "Order now", "Status"]].round(1))
    st.download_button("Download replenishment plan (CSV)", F[["Channel", "Product_ID", "Product", "F30", "F60", "F90", "Stock", "Lead", "Safety", "ROP", "Order now", "Excess", "Status"]].round(1).to_csv(index=False).encode("utf-8-sig"), "replenishment_plan.csv", "text/csv")
    rev = s.groupby("Product_ID").Net_Sales_EGP.sum().sort_values(ascending=False); abc = pd.cut(rev.cumsum() / rev.sum(), [0, .8, .95, 1.01], labels=list("ABC"))
    dm = iv.groupby(["Product_ID", "Snapshot_Month"]).Sold_Qty.sum().unstack(fill_value=0); cvv = dm.std(axis=1) / dm.mean(axis=1); xyz = pd.cut(cvv, [0, .25, .5, 9], labels=list("XYZ"))
    ct = pd.crosstab(abc, xyz.reindex(abc.index)).reindex(columns=list("XYZ"), fill_value=0); f, ax = heat(ct, "ABC / XYZ: number of SKUs", cmap="Blues", fmt="{:.0f}", h=2.6)
    with st.expander("ABC/XYZ & method"):
        show(f); st.caption("Backtest: 90-day WAPE ≈29% per SKU×channel, ≈4% total. Demand CV 0.44–0.91 for every SKU → safety-stock driven replenishment.")

# =============== APP 03
elif nav.startswith("03"):
    st.title("Customer 360 & Retention Engine"); st.caption("Who buys, why, and what happens next?")
    r_ = s[s.Customer_Type == "Retail"]; ref = pd.Timestamp("2026-09-30")
    inv = r_.groupby(["Customer_ID", "Invoice_ID"]).agg(d=("Order_Date", "first"), n=("Net_Sales_EGP", "sum")).reset_index()
    cu = inv.groupby("Customer_ID").agg(first=("d", "min"), last=("d", "max"), F=("Invoice_ID", "nunique"), M=("n", "sum")); cu["R"] = (ref - cu["last"]).dt.days
    cu["Rs"] = pd.qcut(cu.R, 5, labels=[5, 4, 3, 2, 1]).astype(int); cu["Fs"] = pd.cut(cu.F, [0, 1, 2, 3, 5, 999], labels=[1, 2, 3, 4, 5]).astype(int)
    def seg(x):
        if x.Rs >= 4 and x.Fs >= 4: return "Champions"
        if x.Fs >= 3 and x.Rs >= 3: return "Loyal"
        if x.Rs >= 4 and x.Fs <= 2: return "New / Promising"
        if x.Rs <= 2 and x.Fs >= 3: return "At Risk (high value)"
        if x.Rs <= 2: return "Lapsed"
        return "Needs Attention"
    cu["Segment"] = cu.apply(seg, axis=1); SC = {"Champions": GREEN, "Loyal": BLUE, "New / Promising": "#8E44AD", "Needs Attention": GRAY, "At Risk (high value)": RED, "Lapsed": ORG}
    k = st.columns(4); k[0].metric("Customers", f"{len(cu):,}"); k[1].metric("Repeat rate", f"{(cu.F>1).mean():.1%}"); k[2].metric("AOV", f"EGP {cu.M.sum()/cu.F.sum():,.0f}"); k[3].metric("Avg orders", f"{cu.F.mean():.1f}")
    g = cu.groupby("Segment").agg(Customers=("M", "size"), Revenue=("M", "sum"), Orders=("F", "mean"), Recency=("R", "mean")); g["% revenue"] = g.Revenue / g.Revenue.sum() * 100; g["% customers"] = g.Customers / g.Customers.sum() * 100; g = g.sort_values("% revenue", ascending=False)
    a, b = st.columns(2)
    f, ax = new(7, 3.2); y = np.arange(len(g)); ax.barh(y + .2, g["% customers"], .4, color=GRAY, label="% of customers"); ax.barh(y - .2, g["% revenue"], .4, color=NAVY, label="% of revenue"); ax.set_yticks(y); ax.set_yticklabels(g.index); ax.invert_yaxis(); ax.legend(frameon=False, fontsize=8); ax.set_title("Segments: customers vs revenue"); ax.grid(axis="y", alpha=0)
    with a: show(f)
    sm = cu.sample(min(3500, len(cu)), random_state=1); f, ax = new(7, 3.2); rng = np.random.default_rng(0)
    for sg, c_ in SC.items():
        q_ = sm[sm.Segment == sg]; ax.scatter(q_.R, q_.F + rng.uniform(-.3, .3, len(q_)), s=6, alpha=.5, color=c_, label=sg)
    ax.set_xlabel("Days since last order"); ax.set_ylabel("Orders"); ax.legend(frameon=False, fontsize=7, markerscale=2); ax.set_title("RFM map (sample of customers)")
    with b: show(f)
    risk = cu[(cu.F >= 2) & (cu.R > 180)]; st.warning(f"Win-back pool: {len(risk):,} repeat customers silent >180 days, {EGP(risk.M.sum())} historic revenue.")
    inv2 = inv.merge(cu[["first"]], left_on="Customer_ID", right_index=True); inv2["dt"] = (inv2.d - inv2["first"]).dt.days; ks = [30, 60, 90, 180, 365]; ret = {}
    for kk in ks:
        el = cu[(ref - cu["first"]).dt.days >= kk].index; back = inv2[(inv2.dt > 0) & (inv2.dt <= kk)].Customer_ID.unique(); ret[f"{kk}d"] = np.isin(el, back).mean() * 100
    a, b = st.columns(2)
    f, ax = bars(pd.Series(ret), "% of customers who return within N days of first order", colors=BLUE, fmt="{:.0f}%", w=7, h=3)
    with a: show(f)
    f, ax = bars(g["Revenue"] / 1e6, "Revenue by segment (EGP M)", colors=[SC[i] for i in g.index], fmt="{:.1f}", w=7, h=3); ax.tick_params(axis="x", labelsize=7); plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    with b: show(f)
    st.download_button("Download win-back list: test vs control (CSV)", risk.assign(Group=np.where(np.random.default_rng(42).random(len(risk)) < .5, "Test", "Control"))[["Segment", "F", "M", "R", "Group"]].reset_index().to_csv(index=False).encode("utf-8-sig"), "winback_test_control.csv", "text/csv")
    st.subheader("Customer profile & Next Best Offer"); cid = st.selectbox("Customer", cu.sort_values("M", ascending=False).index[:500])
    mine = r_[r_.Customer_ID == cid]; row = cu.loc[cid]; fam = mine.groupby("Product_Family").Net_Sales_EGP.sum().sort_values(ascending=False)
    a = st.columns(4); a[0].metric("Segment", row.Segment); a[1].metric("Orders", int(row.F)); a[2].metric("Lifetime spend", f"EGP {row.M:,.0f}"); a[3].metric("Days since last order", int(row.R))
    c1, c2 = st.columns(2)
    f, ax = bars(fam, "Spend by product family (EGP)", colors=NAVY, fmt="{:,.0f}", w=7, h=2.8)
    with c1: show(f)
    f, ax = bars(mine.groupby("Channel").Net_Sales_EGP.sum(), "Spend by channel (EGP)", colors=BLUE, fmt="{:,.0f}", w=7, h=2.8)
    with c2: show(f)
    st.write("Sizes bought:", mine.Size.value_counts().head(3).to_dict())
    peers = r_[r_.Customer_ID.isin(cu[cu.Segment == row.Segment].index)]; pop = peers.groupby("Product_ID").agg(buyers=("Customer_ID", "nunique"), gp=("Gross_Profit_EGP", "sum"), n=("Net_Sales_EGP", "sum")); pop["score"] = pop.buyers * (pop.gp / pop.n)
    nbo = pop[~pop.index.isin(mine.Product_ID.unique())].sort_values("score", ascending=False).head(3); nbo["Product"] = nbo.index.map(pm.Product_Name)
    act = {"At Risk (high value)": "WhatsApp win-back with these items + small incentive", "Lapsed": "low-cost reactivation message", "New / Promising": "second-order nudge within 90 days", "Champions": "early access / referral", "Loyal": "cross-sell", "Needs Attention": "seasonal reminder"}[row.Segment]
    st.success(f"Recommended action: {act}"); st.dataframe(nbo[["Product", "buyers", "score"]].round(2)); st.caption("NBO = products popular in the same segment (weighted by margin) that this customer has not bought. Not a trained model – a transparent baseline.")
    cc = c.set_index("Customer_ID").join(cu[["F", "M"]], how="inner"); ga = cc.groupby("Acquisition_Channel").agg(Customers=("M", "size"), Repeat=("F", lambda x: (x > 1).mean()), Revenue_per_customer=("M", "mean")); st.subheader("Acquisition channel quality"); st.dataframe(ga.round(2))

# =============== APP 04
elif nav.startswith("04"):
    st.title("Factory Digital Twin & Production Control"); st.caption("Where is throughput lost before delivery?")
    p["late"] = p.Delay_Days > 0; k = st.columns(5); k[0].metric("Plan attainment", f"{p.Produced_Qty.sum()/p.Planned_Qty.sum():.1%}"); k[1].metric("Reject rate", f"{p.Rejected_Qty.sum()/p.Produced_Qty.sum():.1%}"); k[2].metric("Efficiency", f"{p.Efficiency_Pct.mean():.1f}%"); k[3].metric("Late orders", f"{p.late.mean():.1%}"); k[4].metric("Overtime hours", f"{p.Overtime_Hours.sum():,.0f}")
    ln = p.groupby("Line_ID").agg(Orders=("Line_ID", "size"), Produced=("Produced_Qty", "sum"), Planned=("Planned_Qty", "sum"), Late=("late", "mean"), Eff=("Efficiency_Pct", "mean"), OT=("Overtime_Hours", "sum")); ln["Attainment"] = ln.Produced / ln.Planned
    a, b = st.columns(2)
    f, ax = bars(ln.Late * 100, "% late orders by line (red = above average)", colors=[RED if v > p.late.mean() * 100 else BLUE for v in ln.Late * 100], fmt="{:.1f}%", w=7, h=3); ax.axhline(p.late.mean() * 100, color=GRAY, ls="--")
    with a: show(f)
    f, ax = bars(ln.Eff, "Average efficiency by line (%)", colors=NAVY, fmt="{:.1f}", w=7, h=3)
    with b: show(f)
    a, b = st.columns(2)
    bd = p[p.late].groupby("Primary_Bottleneck").Delay_Days.sum().sort_values(ascending=False); f, ax = bars(bd, "Delay-days by bottleneck", colors=[RED] + [BLUE] * (len(bd) - 1), w=7, h=3)
    with a: show(f)
    ly = p.groupby(p.Start_Date.dt.year).late.mean() * 100; f, ax = new(7, 3); ax.plot(ly.index.astype(str), ly.values, marker="o", color=RED, lw=2); ax.set_ylim(0, 25); ax.set_title("% of production orders late, by year"); [ax.text(i, v + 1, f"{v:.0f}%", ha="center") for i, v in enumerate(ly.values)]
    with b: show(f)
    st.subheader("Printing machines")
    pm_ = pb.groupby("Machine_ID").agg(Printed=("Printed_Qty", "sum"), Good=("Good_Qty", "sum"), Reject=("Rejected_Qty", "sum"), Rework=("Rework_Cost_EGP", "sum")); pm_["Reject %"] = pm_.Reject / pm_.Printed * 100
    a, b = st.columns(2)
    f, ax = bars(pm_["Reject %"], "Print reject rate by machine (%)", colors=[RED if v == pm_["Reject %"].max() else BLUE for v in pm_["Reject %"]], fmt="{:.2f}", w=7, h=3)
    with a: show(f)
    f, ax = bars(pm_.Rework / 1e3, "Rework cost by machine (EGP thousand)", colors=ORG, fmt="{:.0f}", w=7, h=3)
    with b: show(f)
    st.subheader("Downtime")
    rs = dt.groupby("Reason").Downtime_Hours.sum().sort_values(ascending=False); f, ax = new(7, 3.2); ax.bar(rs.index, rs.values, color=BLUE); ax.set_ylabel("Hours"); plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8); ax2 = ax.twinx(); ax2.plot(rs.index, rs.cumsum() / rs.sum() * 100, color=RED, marker="o"); ax2.set_ylim(0, 105); ax2.grid(False); ax2.spines["right"].set_visible(True); ax.set_title("Downtime Pareto: hours by reason")
    a, b = st.columns(2)
    with a: show(f)
    top = dt.groupby("Machine_ID").Downtime_Hours.sum().nlargest(10).index; hm = dt[dt.Machine_ID.isin(top)].pivot_table(index="Machine_ID", columns="Reason", values="Downtime_Hours", aggfunc="sum").fillna(0).loc[top]; f, ax = heat(hm, "Top-10 machines × reason (hours)", cmap="Reds", fmt="{:.0f}", h=3.4)
    with b: show(f)
    qs = q.groupby("Stage").Estimated_Loss_EGP.sum() / 1e6; f, ax = bars(qs, "QC loss by stage (EGP M)", colors=NAVY, fmt="{:.2f}", w=7, h=2.8)
    show(f)
    with st.expander("Machine table"):
        st.dataframe(dt.groupby("Machine_ID").agg(Hours=("Downtime_Hours", "sum"), Events=("Downtime_Hours", "size"), Unplanned=("Planned", lambda x: (x == "No").mean())).sort_values("Hours", ascending=False).round(2).head(12)); st.dataframe(ln.round(3))
    st.info("Reconciliation: production units are ~3.8× units sold in the sales ledger – confirm what else the factory serves before judging capacity.")

# =============== APP 05
elif nav.startswith("05"):
    st.title("Marketing Attribution & Growth Engine"); st.caption("Which spend creates profitable customers?")
    pl = m.groupby("Platform").agg(Spend=("Spend_EGP", "sum"), Rev=("Attributed_Revenue_EGP", "sum"), Orders=("Orders_Attributed", "sum"), Clicks=("Clicks", "sum"), Imp=("Impressions", "sum"), Sessions=("Landing_Page_Sessions", "sum")); pl["ROAS"] = pl.Rev / pl.Spend; pl["CAC"] = pl.Spend / pl.Orders; pl["CTR %"] = pl.Clicks / pl.Imp * 100; pl["Conv %"] = pl.Orders / pl.Sessions * 100
    on = s[s.Channel == "Online"]; k = st.columns(4); k[0].metric("Total spend", EGP(pl.Spend.sum())); k[1].metric("Reported ROAS", f"{pl.Rev.sum()/pl.Spend.sum():.1f}×"); k[2].metric("Claimed revenue", EGP(pl.Rev.sum())); k[3].metric("Actual online sales", EGP(on.Net_Sales_EGP.sum()))
    be = st.slider("Break-even ROAS (1 ÷ gross margin)", 2.0, 10.0, 6.3, 0.1); a, b = st.columns(2)
    sr = pl.ROAS.sort_values(); f, ax = new(7, 3); ax.barh(sr.index, sr.values, color=[RED if v < be else GREEN for v in sr.values]); ax.set_xscale("log"); ax.axvline(be, color="k", ls="--"); ax.text(be, -.45, f" break-even {be:.1f}×", fontsize=8); ax.set_title("Reported ROAS by platform (log scale)"); ax.grid(axis="y", alpha=0)
    for i, v in enumerate(sr.values): ax.text(v, i, f" {v:.1f}×", va="center", fontsize=8)
    with a: show(f)
    f, ax = bars(pd.Series({"Marketing file\nclaims": m.Attributed_Revenue_EGP.sum() / 1e6, "Sales ledger\n(online)": on.Net_Sales_EGP.sum() / 1e6}), "Claimed vs actual revenue (EGP M)", colors=[RED, NAVY], fmt="{:.1f}", w=7, h=3)
    with b: show(f)
    st.error(f"Reconciliation: marketing claims {EGP(m.Attributed_Revenue_EGP.sum())} / {m.Orders_Attributed.sum():,} orders; ledger online = {EGP(on.Net_Sales_EGP.sum())} / {on.Invoice_ID.nunique():,} orders. Treat ROAS as directional.")
    m["ROAS"] = m.Attributed_Revenue_EGP / m.Spend_EGP; a, b = st.columns(2)
    PC = {"Google": BLUE, "Meta": NAVY, "TikTok": "#8E44AD", "Influencer": RED, "WhatsApp / CRM": GREEN}; f, ax = new(7, 3.2)
    for pf_, g_ in m.groupby("Platform"): ax.scatter(g_.Spend_EGP / 1e3, g_.ROAS, s=14, alpha=.6, color=PC.get(pf_, GRAY), label=pf_)
    ax.set_yscale("log"); ax.axhline(be, color="k", ls="--", lw=1); ax.set_xlabel("Campaign spend (EGP thousand)"); ax.set_ylabel("ROAS"); ax.legend(frameon=False, fontsize=7); ax.set_title("Every campaign: spend vs ROAS")
    with a: show(f)
    f, ax = bars(pl.CAC.sort_values(), "Cost per order (CAC, EGP)", colors=BLUE, fmt="{:,.0f}", horizontal=True, w=7, h=3.2)
    with b: show(f)
    with st.expander("Platform table & campaign outliers"):
        st.dataframe(pl.round(2)); c1, c2 = st.columns(2); c1.write("Worst ROAS"); c1.dataframe(m.nsmallest(8, "ROAS")[["Campaign_Name", "Platform", "Spend_EGP", "ROAS"]].round(2)); c2.write("Best ROAS"); c2.dataframe(m.nlargest(8, "ROAS")[["Campaign_Name", "Platform", "Spend_EGP", "ROAS"]].round(2))
    st.subheader("Budget reallocation what-if"); fr = st.selectbox("Cut from", pl.index, index=list(pl.index).index("Influencer") if "Influencer" in pl.index else 0); to = st.selectbox("Move to", pl.index); pc = st.slider("% of source budget moved", 0, 100, 50); eff = st.slider("Efficiency of new spend vs average ROAS", 0.3, 1.0, 0.6, 0.05)
    mv = pl.loc[fr, "Spend"] * pc / 100; dRev = -mv * pl.loc[fr, "ROAS"] + mv * pl.loc[to, "ROAS"] * eff; st.metric("Change in attributed revenue", EGP(dRev)); st.caption("Marginal returns diminish; efficiency slider assumes new spend earns less than the platform average. Validate with a holdout test.")
    st.subheader("Profitable customers by acquisition channel"); r_ = s[s.Customer_Type == "Retail"].groupby("Customer_ID").agg(GP=("Gross_Profit_EGP", "sum"), Net=("Net_Sales_EGP", "sum"), F=("Invoice_ID", "nunique")); ac = c.set_index("Customer_ID").join(r_, how="inner").groupby("Acquisition_Channel").agg(Customers=("GP", "size"), GP_per_customer=("GP", "mean"), Repeat=("F", lambda x: (x > 1).mean()))
    f, ax = bars(ac.GP_per_customer, "Gross profit per customer by acquisition channel (EGP)", colors=GREEN, fmt="{:,.0f}", w=7, h=2.8); plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); show(f); st.caption("Finding: gross profit per customer is almost identical across acquisition channels (≈EGP 1,000) – channel choice is not what separates good customers."); st.dataframe(ac.round(2))

# =============== APP 06
elif nav.startswith("06"):
    st.title("Branch Intelligence & Expansion"); st.caption("Which branch wins — and where should the next one open?")
    st.info("Finding: the data contains ONE physical store (LEO MODA Sidi Bishr, Alexandria), one online store and one wholesale desk. A branch league table is therefore not possible; the useful question is where demand is un-served.")
    nr = s[s.Channel != "Wholesale"]; fyg = nr.groupby(["Customer_Governorate", "FY"]).Net_Sales_EGP.sum().unstack(); gov = pd.DataFrame({"FY26 retail+online": fyg[2026], "CAGR": (fyg[2026] / fyg[2024]) ** .5 - 1})
    gov["Customers"] = nr[nr.FY == 2026].groupby("Customer_Governorate").Customer_ID.nunique(); gov["GM"] = nr.groupby("Customer_Governorate").apply(lambda x: x.Gross_Profit_EGP.sum() / x.Net_Sales_EGP.sum())
    st_ = s[s.Channel == "Retail"].groupby("Customer_Governorate").Net_Sales_EGP.sum(); gov["Store sales (36m)"] = st_; gov["Wholesale FY26"] = s[(s.Channel == "Wholesale") & (s.FY == 2026)].groupby("Customer_Governorate").Net_Sales_EGP.sum()
    km = {"Alexandria": 0, "Cairo": 220, "Giza": 210, "Beheira": 60, "Dakahlia": 190, "Gharbia": 130, "Sharqia": 200, "Kafr El Sheikh": 100, "Monufia": 150, "Port Said": 340, "Damietta": 270, "Suez": 350}; gov["km from store (approx.)"] = pd.Series(km)
    gov = gov.drop(index="Other", errors="ignore"); a = st.columns(3); wd = a[0].slider("Weight: demand", 0.0, 1.0, 0.5); wg = a[1].slider("Weight: growth", 0.0, 1.0, 0.3); wk = a[2].slider("Weight: distance from existing store", 0.0, 1.0, 0.2)
    nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9); gov["Expansion score"] = wd * nz(gov["FY26 retail+online"]) + wg * nz(gov.CAGR) + wk * nz(gov["km from store (approx.)"].clip(upper=250)); gov.loc["Alexandria", "Expansion score"] = np.nan
    gc = gov.loc[["Cairo", "Giza"]]["FY26 retail+online"].sum(); st.success(f"Greater Cairo (Cairo+Giza) buys {EGP(gc)} (FY26, retail+online) – {gc/gov.loc['Alexandria','FY26 retail+online']:.2f}× Alexandria's demand – with no store. Recommended pilot: one Cairo/Giza location or pop-up, measured against online baseline.")
    d_ = gov["FY26 retail+online"].sort_values(ascending=False); f, ax = new(7, 3.4); ax.bar(d_.index, d_.values / 1e6, color=[NAVY if i == "Alexandria" else (RED if i in ("Cairo", "Giza") else BLUE) for i in d_.index]); ax.set_ylabel("EGP M"); plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8); ax.grid(axis="x", alpha=0); ax.set_title("FY26 retail + online demand by governorate (navy = existing store)")
    for i, (gname, v) in enumerate(d_.items()): ax.text(i, v / 1e6, f"{gov.loc[gname,'CAGR']:+.0%}", ha="center", va="bottom", fontsize=7)
    a, b = st.columns(2)
    with a: show(f)
    sc_ = gov["Expansion score"].dropna().sort_values(ascending=False); f, ax = bars(sc_, "Expansion score (Alexandria excluded – already served)", colors=[RED if i in ("Cairo", "Giza") else BLUE for i in sc_.index], fmt="{:.2f}", horizontal=True, w=7, h=3.4)
    with b: show(f)
    st.caption("Labels above bars: growth (CAGR FY24→FY26).")
    ll = {"Alexandria": (31.2, 29.92), "Cairo": (30.04, 31.24), "Giza": (29.99, 31.13), "Beheira": (31.04, 30.47), "Dakahlia": (31.04, 31.38), "Gharbia": (30.79, 31.0), "Sharqia": (30.59, 31.5), "Kafr El Sheikh": (31.11, 30.94), "Monufia": (30.47, 30.93), "Port Said": (31.26, 32.3), "Damietta": (31.42, 31.81), "Suez": (29.97, 32.55)}
    mp = pd.DataFrame([(k, *ll[k], gov.loc[k, "FY26 retail+online"]) for k in gov.index if k in ll], columns=["gov", "lat", "lon", "size"]); mp["size"] = mp["size"] / mp["size"].max() * 20000; st.map(mp, size="size")
    with st.expander("Governorate table"): st.dataframe(gov.sort_values("Expansion score", ascending=False).round(3))
    st.warning("Limits: customer governorate is home location, not store catchment; no footfall, rent or branch-level costs in the data; distances are approximate road km. Validate with real catchment data before committing capex.")

# =============== APP 07
elif nav.startswith("07"):
    st.title("Procurement, Cost & Supplier Risk"); st.caption("Where do cost, delay, and quality risk begin?")
    po["lead"] = (po.Actual_Receipt_Date - po.Order_Date).dt.days; po["plan"] = (po.Promised_Date - po.Order_Date).dt.days; po["ot"] = po.Actual_Receipt_Date <= po.Promised_Date; po["val"] = po.Ordered_Qty * po.Unit_Cost_EGP; po["Year"] = po.Order_Date.dt.year
    k = st.columns(4); k[0].metric("POs", f"{len(po):,}"); k[1].metric("On-time", f"{po.ot.mean():.0%}"); k[2].metric("Reject rate", f"{po.Rejected_Qty.sum()/po.Ordered_Qty.sum():.2%}"); k[3].metric("Avg lead (d)", f"{po.lead.mean():.1f}")
    sc = po.groupby("Supplier_ID").agg(POs=("ot", "size"), OnTime=("ot", "mean"), Lead=("lead", "mean"), Reject=("Rejected_Qty", "sum"), Ord=("Ordered_Qty", "sum"), Spend=("val", "sum")); sc["Reject"] = sc.Reject / sc.Ord; sc = sc.drop(columns="Ord").join(su.set_index("Supplier_ID")[["Primary_Category", "Contracted", "Payment_Terms_Days"]])
    t = po.pivot_table(index="Supplier_ID", columns="Year", values="ot", aggfunc="mean"); sc["Trend 24→26 (pts)"] = (t[2026] - t[2024]) * 100
    st.subheader("Risk score weights (you decide)"); w1, w2, w3 = st.columns(3); a1 = w1.slider("Weight: late delivery", 0, 100, 60); a2 = w2.slider("Weight: rejects", 0, 100, 20); a3 = w3.slider("Weight: worsening trend", 0, 100, 20)
    rj = sc.Reject / sc.Reject.max(); tr = (sc["Trend 24→26 (pts)"].clip(upper=0).abs() / 100); sc["Risk score"] = (a1 * (1 - sc.OnTime) + a2 * rj + a3 * tr) / max(a1 + a2 + a3, 1) * 100
    a, b = st.columns(2)
    rk = sc["Risk score"].sort_values(ascending=False); f, ax = bars(rk, "Supplier risk score (higher = riskier)", colors=[RED if v >= rk.quantile(.85) else BLUE for v in rk.values], fmt="{:.0f}", w=7, h=3.4)
    with a: show(f)
    f, ax = heat(t * 100, "On-time delivery % by supplier and year", vmin=0, vmax=100, fmt="{:.0f}", h=3.4)
    with b: show(f)
    pt = po.groupby(["Material_ID", "Year"]).apply(lambda g: g.val.sum() / g.Ordered_Qty.sum()).unstack(); idx = pt.div(pt[2023], axis=0) * 100; units = s.groupby("Product_ID").Quantity.sum(); bm = bom.assign(need=lambda d: d.Product_ID.map(units) * d.Required_Per_Unit)
    cr = bm.groupby("Material_ID").agg(Products=("Product_ID", "nunique"), BOM_need=("need", "sum")); idx = idx.join(cr).join(mt.set_index("Material_ID")[["Material_Name"]])
    chg = (idx[2026] - 100).set_axis([en(n, i) for n, i in zip(idx.Material_Name, idx.index)]).sort_values(ascending=False); bought = po.groupby("Material_ID").Accepted_Qty.sum(); rc = pd.DataFrame({"Bought": bought, "BOM need": cr.BOM_need}); rc["Ratio"] = rc.Bought / rc["BOM need"]
    a, b = st.columns(2)
    f, ax = bars(chg, "Unit price change 2023 → 2026 (%)", colors=[RED if v > 10 else BLUE for v in chg.values], fmt="{:+.0f}%", horizontal=True, w=7, h=3.6)
    with a: show(f)
    rr = rc.Ratio.dropna().sort_values(ascending=False); rr.index = [en(mt.set_index("Material_ID").Material_Name.get(i, i), i) for i in rr.index]; f, ax = bars(rr, "PO quantity ÷ BOM-implied need (1 = matched)", colors=ORG, fmt="{:.1f}×", horizontal=True, w=7, h=3.6); ax.set_xscale("log"); ax.axvline(1, color="k", ls="--", lw=1)
    with b: show(f)
    with st.expander("Supplier scorecard & tables"):
        st.dataframe(sc.sort_values("Risk score", ascending=False).round(3)); st.dataframe(idx.round(1)); st.dataframe(rc.round(1)); st.caption("Ratios far above 1 (or missing BOM) mean PO control or BOM master needs an audit. PO spend is 2.5× total COGS in this dataset.")
    st.warning("Action: dual-source and add service-level penalties for the suppliers at the top of the risk table; refresh standard costs for material inflation.")

# =============== FINAL MISSION
else:
    st.title("★ Final Mission – Wholesale Margin Guardrail"); st.write("Hidden problem: wholesale discounts (28–38%) are unrelated to quantity or cost. Set a margin floor → each SKU gets a maximum discount; reps check deals before approval.")
    a, b_, c_ = st.columns(3); floor = a.slider("Minimum margin floor", 0.0, 0.20, 0.08, 0.01); ceil = b_.slider("Max discount ceiling", 0.20, 0.45, 0.38, 0.01); vl = c_.slider("Volume lost on repriced lines", 0.0, 0.6, 0.25, 0.05)
    k = st.columns(2)
    for col_, (lab, d) in zip(k, (("36 months", W), ("FY2026", W[W.FY == 2026]))):
        n, base, gp = simulate(d, floor, vl, ceil); col_.metric(f"{lab}: GP uplift", EGP(gp - base), f"{n:,} lines repriced ({n/len(d):.0%})")
    W26 = W[W.FY == 2026]; fl = np.linspace(0, .12, 13); f, ax = new(7, 3.2)
    for v, c0 in ((0, GREEN), (.25, BLUE), (.5, ORG)): ax.plot(fl * 100, [(simulate(W26, x, v, ceil)[2] - simulate(W26, x, v, ceil)[1]) / 1e6 for x in fl], color=c0, lw=2, label=f"{int(v*100)}% volume loss")
    ax.axvline(floor * 100, color="k", ls=":"); ax.set_xlabel("Margin floor (%)"); ax.set_ylabel("FY26 GP uplift (EGP M)"); ax.legend(frameon=False, fontsize=8); ax.set_title("Guardrail value vs margin floor (even harsh volume loss still gains)")
    sk = W.groupby("Product_ID").agg(list=("List_Price_EGP", "mean"), cost=("std_c", "first"), avg_disc=("Discount_Pct", "mean"), net=("Net_Sales_EGP", "sum")); sk["Product"] = sk.index.map(pm.Product_Name)
    sk["Break-even disc"] = 1 - sk.cost / sk.list; sk["Approved cap"] = np.minimum(ceil, max_disc(sk.cost, sk.list, floor))
    a, b = st.columns(2)
    with a: show(f)
    tp = sk.nlargest(15, "net").sort_values("net"); f, ax = new(7, 3.8); yy = np.arange(len(tp))
    for i, (_, r0) in enumerate(tp.iterrows()): ax.plot([r0["Approved cap"] * 100, r0.avg_disc * 100], [i, i], color=GRAY, lw=1)
    ax.scatter(tp["Approved cap"] * 100, yy, color=GREEN, label="Approved cap", zorder=3); ax.scatter(tp.avg_disc * 100, yy, color=[RED if x > y else BLUE for x, y in zip(tp.avg_disc, tp["Approved cap"])], label="Avg discount given", zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels([en(n, i) for n, i in zip(tp.Product, tp.index)], fontsize=7); ax.set_xlabel("Discount %"); ax.legend(frameon=False, fontsize=7); ax.set_title("Top 15 SKUs: avg discount vs cap (red = over cap)")
    with b: show(f)
    with st.expander("SKU caps table"): st.dataframe(sk[["Product", "list", "cost", "Break-even disc", "Approved cap", "avg_disc", "net"]].round(3))
    st.markdown("**Deal checker**"); x = st.columns(3); pid = x[0].selectbox("SKU", sk.index, format_func=lambda i: f"{i} – {sk.loc[i,'Product']}"); qty = x[1].number_input("Quantity", 1, 500, 24); dd = x[2].slider("Requested discount", 0.0, 0.45, 0.35, 0.01)
    r = sk.loc[pid]; gpd = qty * (r.list * (1 - dd) - r.cost); cap = r["Approved cap"]
    (st.success if dd <= cap else st.error)(f"{'APPROVE' if dd <= cap else 'ESCALATE'} · deal margin {1 - r.cost/(r.list*(1-dd)):.1%} · GP EGP {gpd:,.0f} · max allowed {cap:.1%} (GP if capped EGP {qty*(r.list*(1-min(dd,cap))-r.cost):,.0f})")
    st.caption("No price-elasticity data exists, so volume loss is a stress assumption. Recommended: 60-day pilot on half of wholesale customers.")
