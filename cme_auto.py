"""Luv CME auto levels — pulls CME options OI + settlements from Databento (licensed CME data),
computes dealer GEX levels and straddle-based SD vols, and sends ready paste lines to Telegram.

Modes:
  python cme_auto.py --discover   → finds the option product codes for 6E / NQ / GC and prints the cost (cheap, run once)
  python cme_auto.py              → daily run (needs products.json from discover)
Env: DATABENTO_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, MAX_COST (USD, default 3), BASIS_EUR/BASIS_NAS/BASIS_GOLD (optional overrides)
"""
import argparse, json, math, os, re, sys, datetime as dt
import numpy as np, pandas as pd
import databento as db

DATASET = "GLBX.MDP3"
PRODUCTS = {
    "EUR":  {"tv": "CME", "settle_utc": "19:00", "root": "6E", "mult": 125000, "spot": ["EURUSD=X"], "dec": 5},
    "NAS":  {"tv": "CME_MINI", "settle_utc": "20:15", "root": "NQ", "mult": 20,     "spot": ["^NDX"],     "dec": 2},
    "GOLD": {"tv": "COMEX", "settle_utc": "17:30", "root": "GC", "mult": 100,    "spot": ["XAUUSD=X"], "dec": 2},
}
GEX_DAYS = 35           # expiries inside this window count for GEX
MAX_COST = float(os.environ.get("MAX_COST", "3"))
OUT = []

def say(s=""):
    print(s)
    OUT.append(s)

def client():
    k = os.environ.get("DATABENTO_API_KEY")
    if not k:
        sys.exit("DATABENTO_API_KEY missing (add it as a GitHub secret)")
    return db.Historical(k)

def data_end(c):
    r = c.metadata.get_dataset_range(dataset=DATASET)
    return pd.Timestamp(r.get("end") or r.get("end_date"))

def fetch(c, **kw):
    import time
    for i in range(5):
        try:
            return c.timeseries.get_range(dataset=DATASET, **kw).to_df()
        except Exception as e:
            if i == 4 or not isinstance(e, db.BentoServerError):
                raise
            print(f"retry {i+1} after: {e}")
            time.sleep(30 * (i + 1))

def guarded(c, label, **kw):
    cost = c.metadata.get_cost(dataset=DATASET, **kw)
    print(f"[cost] {label}: ${cost:.4f}")
    if cost > MAX_COST:
        sys.exit(f"Stopped: {label} would cost ${cost:.2f} (> MAX_COST ${MAX_COST}). Raise MAX_COST if you're sure.")
    return fetch(c, **kw)

# ───────── Discover: which option product codes belong to 6E / NQ / GC ─────────
def discover(c):
    end = data_end(c)
    start = end - pd.Timedelta(hours=26)
    df = guarded(c, "all definitions (1 day)", schema="definition", symbols="ALL_SYMBOLS", start=start, end=end)
    opt = df[df["instrument_class"].isin(["C", "P"])].copy()
    found = {}
    for name, p in PRODUCTS.items():
        sub = opt[opt["underlying"].astype(str).str.match(rf"^{p['root']}[FGHJKMNQUVXZ]\d{{1,2}}$")]
        codes = sorted(sub["asset"].astype(str).unique())
        found[name] = codes
        say(f"{name}: option products {codes}")
        for a in codes:
            s = sub[sub["asset"] == a]
            say(f"   {a}: {len(s)} strikes, next expiry {pd.to_datetime(s['expiration']).min()}")
    json.dump(found, open("products.json", "w"), indent=1)
    say("Saved products.json — commit it to the repo.")

# ───────── Maths ─────────
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

def b76(F, K, s, T, call):
    if s <= 0 or T <= 0:
        return max(F - K, 0) if call else max(K - F, 0)
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    return F * N(d1) - K * N(d2) if call else K * N(-d2) - F * N(-d1)

def gamma(F, K, s, T):
    if s <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (F * s * math.sqrt(T))

def implied(target, F, K, T, legs):
    lo, hi = 0.005, 3.0
    for _ in range(80):
        m = (lo + hi) / 2
        v = sum(b76(F, K, m, T, cl) for cl in legs)
        lo, hi = (m, hi) if v < target else (lo, m)
    return (lo + hi) / 2

def trading_years(now, exp):
    days = (exp - now).total_seconds() / 86400
    wk = (exp.date() - now.date()).days - np.busday_count(now.date(), exp.date())
    return max(days - wk, 0.05) / 252

def spot_basis(name, fut_last):
    env = os.environ.get(f"BASIS_{name}")
    if env:
        return float(env), "manual"
    try:
        import yfinance as yf
        for s in PRODUCTS[name]["spot"]:
            h = yf.Ticker(s).history(period="5d", interval="1m")
            if len(h):
                return fut_last - float(h["Close"].iloc[-1]), f"{s} {float(h['Close'].iloc[-1]):.{PRODUCTS[name]['dec']}f}"
    except Exception as e:
        print("spot fetch failed:", e)
    return 0.0, "none (levels in FUTURES prices)"

# ───────── Skew / term / vanna / charm ─────────
phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def d12(F, K, s, T):
    d1 = (math.log(F / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return d1, d1 - s * math.sqrt(T)

def rr25(g, F, T):
    """25-delta risk reversal (call vol − put vol, in vol points) from OTM settlements of one expiry."""
    pts = []
    for r in g.itertuples():
        if pd.isna(r.settle) or r.settle <= 0:
            continue
        call = r.instrument_class == "C"
        if (call and r.strike_price <= F) or ((not call) and r.strike_price >= F):
            continue                                   # OTM only (cleaner prices)
        iv = implied(float(r.settle), F, r.strike_price, T, [call])
        if iv <= 0.006 or iv >= 2.9:
            continue
        d1, _ = d12(F, r.strike_price, iv, T)
        dl = N(d1) if call else N(d1) - 1
        pts.append((call, dl, iv))
    c = [x for x in pts if x[0]]
    pu = [x for x in pts if not x[0]]
    if not c or not pu:
        return None
    ivc = min(c, key=lambda x: abs(x[1] - 0.25))
    ivp = min(pu, key=lambda x: abs(x[1] + 0.25))
    if abs(ivc[1] - 0.25) > 0.12 or abs(ivp[1] + 0.25) > 0.12:
        return None
    return (ivc[2] - ivp[2]) * 100

def flows(o, vmap, mult_unused=None):
    """Dealer hedge flow in contracts (dealers long calls / short puts):
       vanna: if implied vol FALLS 1 point; charm: per trading day that passes."""
    van = cha = 0.0
    for r in o.itertuples():
        e = vmap.get((r.expiration, r.underlying))
        if e is None or r.oi <= 0 or e.ty <= 0:
            continue
        d1, d2 = d12(e.F, r.strike_price, e.vol, e.ty)
        w = r.oi if r.instrument_class == "C" else -r.oi
        vanna = phi(d1) * (-d2 / e.vol)                 # dDelta / dVol (same for calls and puts)
        charm = phi(d1) * d2 / (2 * e.ty) / 252        # dDelta per trading day passing
        van += w * vanna * (-0.01)                      # dealer delta change if vol drops 1pt
        cha += w * charm
    return -van, -cha                                  # dealers hedge the opposite way

# ───────── COT (free CFTC data, weekly) ─────────
COT = {"EUR": ("gpe5-46if", "099741", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds"),
       "NAS": ("gpe5-46if", "209742", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds"),
       "GOLD": ("72hh-3qpy", "088691", "m_money_positions_long_all", "m_money_positions_short_all", "Managed money")}

def cot_line(name):
    try:
        import requests
        ds, code, fl, fs, who = COT[name]
        url = f"https://publicreporting.cftc.gov/resource/{ds}.json"
        rows = requests.get(url, params={"cftc_contract_market_code": code, "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 160}, timeout=30).json()
        if not rows:
            return None
        net = [int(float(r[fl])) - int(float(r[fs])) for r in rows]
        d = pd.Timestamp(rows[0]["report_date_as_yyyy_mm_dd"])
        ch = net[0] - net[1] if len(net) > 1 else 0
        def idx(n):
            w = net[:n]
            lo, hi = min(w), max(w)
            return 100 * (net[0] - lo) / (hi - lo) if hi > lo else 50
        i6, i36 = idx(26), idx(156)
        ext = " → most bullish positioning in 3y (crowded, reversal risk)" if i36 >= 90 else " → most bearish positioning in 3y (crowded, squeeze risk)" if i36 <= 10 else ""
        side = "net LONG" if net[0] > 0 else "net SHORT"
        return f"COT ({d:%d %b}): {who} {side} {abs(net[0]):,} ({ch:+,} w/w) · COT index 6m {i6:.0f} / 3y {i36:.0f}{ext}"
    except Exception as e:
        print("cot failed:", e)
        return None

# ───────── Daily run ─────────
def run(c):
    prods = json.load(open("products.json"))
    try:
        state = json.load(open("state.json"))
    except Exception:
        state = {}
    end = data_end(c)
    now = pd.Timestamp.now(tz="UTC")
    day0 = end.normalize()                       # definitions snapshot = UTC midnight
    if end - day0 < pd.Timedelta(hours=2):
        day0 -= pd.Timedelta(days=1)
    # definitions per product (smaller requests = no gateway timeouts)
    dparts = []
    for name, p in PRODUCTS.items():
        par = [f"{a}.OPT" for a in prods.get(name, [])] + [f"{p['root']}.FUT"]
        dparts.append(guarded(c, f"definitions {name}", schema="definition", symbols=par, stype_in="parent", start=day0, end=end))
    defs = pd.concat(dparts).sort_index().groupby("instrument_id").last()
    sstart = day0 - pd.Timedelta(days=3)         # covers the last OI publication (weekends too)
    # 1) futures first (tiny) → reference price per product
    fids = [int(i) for i in defs.index[defs["instrument_class"] == "F"]]
    fst = fetch(c, schema="statistics", symbols=fids, stype_in="instrument_id", start=sstart, end=end)
    fset = fst[fst.stat_type == 3].sort_index().groupby("instrument_id").last()["price"]
    ref = {}
    for name, p in PRODUCTS.items():
        fr = defs[(defs["instrument_class"] == "F") & defs["raw_symbol"].astype(str).str.match(rf"^{p['root']}[FGHJKMNQUVXZ]\d$")]
        px = fset.reindex(fr.index).dropna()
        if len(px):
            ref[name] = float(px.loc[pd.to_datetime(fr.loc[px.index, "expiration"], utc=True).idxmin()]) if False else float(px.median())
    # 2) only options near price and inside the expiry window
    BAND = {"EUR": 0.05, "NAS": 0.12, "GOLD": 0.12}
    exp_ = pd.to_datetime(defs["expiration"], utc=True)
    near = pd.Series(False, index=defs.index)
    for name, p in PRODUCTS.items():
        if name not in ref:
            continue
        m = defs["asset"].isin(prods.get(name, [])) & defs["instrument_class"].isin(["C", "P"])
        k = defs["strike_price"]
        near |= m & (k > ref[name] * (1 - BAND[name])) & (k < ref[name] * (1 + BAND[name]))
    keep = defs[near & (exp_ > now) & (exp_ < now + pd.Timedelta(days=GEX_DAYS))]
    ids = [int(i) for i in keep.index]
    print(f"{len(ids)} options to price · ref prices {ref}")
    parts, total = [fst], 0.0
    for i in range(0, len(ids), 400):
        kw = dict(schema="statistics", symbols=ids[i:i + 400], stype_in="instrument_id", start=sstart, end=end)
        total += c.metadata.get_cost(dataset=DATASET, **kw)
        if total > MAX_COST:
            sys.exit(f"Stopped: statistics would cost ${total:.2f} (> MAX_COST ${MAX_COST}).")
        parts.append(fetch(c, **kw))
    print(f"[cost] statistics: ${total:.4f}")
    stats = pd.concat(parts).sort_index()
    oi = stats[stats.stat_type == 9].groupby("instrument_id").last()
    st = stats[stats.stat_type == 3].groupby("instrument_id").last()
    oi_date = pd.to_datetime(oi["ts_ref"]).max() if len(oi) else None
    say(f"CME levels · {now:%a %d %b %H:%M} UTC · OI as of {oi_date:%a %d %b} (CME official)" if oi_date is not None else "CME levels · no OI found")

    defs["oi"] = oi["quantity"].reindex(defs.index)
    defs["settle"] = st["price"].reindex(defs.index)
    defs["sdate"] = pd.to_datetime(st["ts_ref"], utc=True).reindex(defs.index)
    fut = defs[defs["instrument_class"] == "F"].set_index("raw_symbol")

    for name, p in PRODUCTS.items():
        o = defs[defs["asset"].isin(prods.get(name, [])) & defs["instrument_class"].isin(["C", "P"])].copy()
        o["expiration"] = pd.to_datetime(o["expiration"], utc=True)
        o = o[(o.expiration > now) & (o.expiration < now + pd.Timedelta(days=GEX_DAYS))]
        o["oi"] = pd.to_numeric(o["oi"], errors="coerce").fillna(0)
        o.loc[o["oi"] > 1e9, "oi"] = 0
        if o.empty:
            say(f"\n{name}: no options found"); continue
        # per-expiry ATM straddle vol
        exps = []
        for (exp, und), g in o.groupby(["expiration", "underlying"]):
            if und not in fut.index or pd.isna(fut.loc[und, "settle"]):
                continue
            F = float(fut.loc[und, "settle"])
            # time to expiry measured from WHEN the settlement prices were set (not from now),
            # otherwise stale settlements make the vol look too high
            sdt = fut.loc[und, "sdate"]
            tset = (sdt.normalize() + pd.Timedelta(p["settle_utc"] + ":00")) if not pd.isna(sdt) else now
            T = trading_years(tset, exp)
            g = g.dropna(subset=["settle"])
            if g.empty:
                continue
            K = g.iloc[(g.strike_price - F).abs().argsort()].strike_price.iloc[0]
            c_ = g[(g.strike_price == K) & (g.instrument_class == "C")].settle
            p_ = g[(g.strike_price == K) & (g.instrument_class == "P")].settle
            legs, tgt = [], 0.0
            if len(c_): legs.append(True); tgt += float(c_.iloc[0])
            if len(p_): legs.append(False); tgt += float(p_.iloc[0])
            if not legs or tgt <= 0:
                continue
            s = implied(tgt, F, K, T, legs)
            sd = F * s * math.sqrt(T)
            exps.append(dict(exp=exp, und=und, F=F, ty=trading_years(now, exp), vol=s, sd=sd, code=g.raw_symbol.iloc[0][:5]))
        if not exps:
            say(f"\n{name}: no settlement prices yet"); continue
        ex = pd.DataFrame(exps).sort_values("exp")
        front = ex.iloc[0]
        fri = ex[ex.exp.dt.weekday == 4]
        weekly = fri.iloc[0] if len(fri) else front
        # monthly: expiry closest to the end of this month
        mend = (now + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=21)
        monthly = ex.iloc[(ex.exp - mend).abs().argsort()].iloc[0]
        # lock weekly vol for the whole week and monthly vol for the whole month (bands must not move)
        wk, mk = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]}", f"{now:%Y-%m}"
        stp = state.setdefault(name, {})
        if stp.get("week") != wk:
            stp.update(week=wk, volw=round(float(weekly.vol) * 100, 2), wexp=f"{weekly.exp:%a %d %b}")
        if stp.get("month") != mk:
            stp.update(month=mk, volm=round(float(monthly.vol) * 100, 2), mexp=f"{monthly.exp:%a %d %b}")
        volw, volm = stp["volw"], stp["volm"]
        # GEX (all expiries) + this-week-only
        vmap = {(r.exp, r.und): r for r in ex.itertuples()}
        F0 = front.F
        def gex(oo):
            rows = []
            for r in oo.itertuples():
                e = vmap.get((r.expiration, r.underlying))
                if e is None or r.oi <= 0:
                    continue
                k = gamma(e.F, r.strike_price, e.vol, e.ty) * p["mult"] * e.F * e.F * 0.01
                rows.append((r.strike_price, r.oi * k if r.instrument_class == "C" else 0.0, -r.oi * k if r.instrument_class == "P" else 0.0))
            if not rows:
                return None, [], lambda S: 0.0
            g = pd.DataFrame(rows, columns=["strike", "c", "p"]).groupby("strike").sum()
            g["net"] = g.c + g.p
            K, W, FE, VO, TY = [], [], [], [], []
            for r in oo.itertuples():
                e = vmap.get((r.expiration, r.underlying))
                if e is None or r.oi <= 0:
                    continue
                K.append(r.strike_price); W.append(r.oi if r.instrument_class == "C" else -r.oi)
                FE.append(e.F); VO.append(e.vol); TY.append(e.ty)
            K, W, FE, VO, TY = map(np.array, (K, W, FE, VO, TY))
            def tot(S):
                Se = S + (FE - F0)
                sq = VO * np.sqrt(TY)
                d1 = (np.log(Se / K) + 0.5 * sq * sq) / sq
                gm = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (Se * sq)
                return float(np.sum(W * gm * p["mult"] * Se * Se * 0.01))
            xs = np.linspace(F0 * 0.97, F0 * 1.03, 241)
            ys = [tot(x) for x in xs]
            fl = [xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]) for i in range(len(xs) - 1) if np.sign(ys[i]) != np.sign(ys[i + 1])]
            return g, sorted(fl, key=lambda q: abs(q - F0))[:2], tot
        g, flips, tot = gex(o)
        gw, flw, _ = gex(o[o.expiration <= weekly.exp])
        # levels stay in FUTURES prices; the TradingView indicator subtracts the live basis itself
        m_ = re.match(r"^([A-Z0-9]+?)([FGHJKMNQUVXZ])(\d)$", str(front.und))
        tv = f"{p['tv']}:{m_.group(1)}{m_.group(2)}202{m_.group(3)}" if m_ else "AUTO"
        d = p["dec"]
        tod = o[(o.expiration > now) & (o.expiration <= now + pd.Timedelta(hours=24)) & (o.oi > 0)]
        big = tod.groupby("strike_price").oi.sum().sort_values(ascending=False).head(3) if len(tod) else pd.Series(dtype=float)
        exp_txt = ("EXP " + " ".join(f"{k:.{d}f}" for k in big.index) + f" @{(tod.expiration.min() + pd.Timedelta(hours=4)):%H:%M}") if len(big) else ""
        f = lambda x: f"{x:.{d}f}"
        pos = g.sort_values("net", ascending=False).head(3).index
        neg = g.sort_values("net").head(3).index
        paste = "; ".join(x for x in [
            ("FLIP " + " ".join(f(x) for x in flips)) if flips else "",
            "PUT " + f(g.p.idxmin()), "CALL " + f(g.c.idxmax()),
            "POS " + " ".join(f(x) for x in pos), "NEG " + " ".join(f(x) for x in neg),
            f"VOLD {front.vol*100:.2f}", f"VOLW {volw:.2f}", f"VOLM {volm:.2f}", exp_txt, f"FUT {tv}"] if x)
        sd_ = fut.loc[front.und, "sdate"] if front.und in fut.index else None
        say(f"\n— {name} (futures {front.und} settle {F0:.{d}f}" + (f" from {sd_:%a %d %b}" if sd_ is not None and not pd.isna(sd_) else "") + " · levels in futures prices, indicator converts them live)")
        say(f"Gamma now: {'POSITIVE (pinning, fade levels)' if tot(F0) > 0 else 'NEGATIVE (moves run)'}")
        say(f"Daily 1SD ±{F0 * front.vol / math.sqrt(252):.{d}f} (expiry {front.exp:%a %d %b}, vol {front.vol*100:.2f}%)")
        say(f"Weekly vol {volw:.2f}% (locked {stp['week']}, expiry {stp['wexp']}) · Monthly vol {volm:.2f}% (locked {stp['month']}, expiry {stp['mexp']})")
        # ── positioning extras: skew, term structure, vanna / charm ──
        flip_note = " (DXY: opposite)" if name == "EUR" else ""
        up_word = {"EUR": "EUR UP / USD down", "NAS": "NAS UP", "GOLD": "GOLD UP"}[name]
        dn_word = {"EUR": "EUR DOWN / USD up", "NAS": "NAS DOWN", "GOLD": "GOLD DOWN"}[name]
        try:
            gwk = o[(o.expiration == weekly.exp) & (o.underlying == weekly.und)]
            rr = rr25(gwk, weekly.F, weekly.ty if False else trading_years(now, weekly.exp))
        except Exception as e_:
            print("rr failed:", e_); rr = None
        if rr is not None:
            prev = stp.get("rr")
            ch = f", {rr - prev:+.2f} vs last run" if prev is not None else ""
            stp["rr"] = round(rr, 3)
            lean = f"calls dearer → market paying for {up_word}" if rr > 0.05 else f"puts dearer → market paying for {dn_word}" if rr < -0.05 else "balanced"
            say(f"Skew (25Δ RR, {weekly.exp:%a %d %b}): {rr:+.2f} vol{ch} → {lean}{flip_note}")
        mon = ex.iloc[(ex.exp - (now + pd.Timedelta(days=30))).abs().argsort()].iloc[0]
        if mon.exp > front.exp:
            tr = front.vol / mon.vol
            tnote = "front > 1-month → EVENT PREMIUM, big move priced soon" if tr > 1.1 else "front < 1-month → calm near term" if tr < 0.9 else "flat → nothing special priced"
            say(f"Term: front {front.vol*100:.2f}% vs ~1-month {mon.vol*100:.2f}% ({mon.exp:%d %b}) → {tnote}")
        try:
            vn, chm = flows(o, vmap)
            say(f"Vanna: if vol DROPS 1pt (e.g. after news) dealers {'BUY' if vn > 0 else 'SELL'} ~{abs(vn):,.0f} contracts → {'supports' if vn > 0 else 'pressures'} price{flip_note}")
            say(f"Charm: each day into expiry dealers {'BUY' if chm > 0 else 'SELL'} ~{abs(chm):,.0f} contracts{flip_note}")
        except Exception as e_:
            print("flows failed:", e_)
        # today's CME expiries (next 24h) with the biggest OI → pin candidates into the cut
        if len(big):
            ex_t = tod.expiration.min() + pd.Timedelta(hours=4)
            say(f"Expiring today {ex_t:%H:%M} MU: " + ", ".join(f"{k:.{d}f} fut ({v:,.0f} OI)" for k, v in big.items()) + " → pin candidates into the cut")
        cl = cot_line(name)
        if cl:
            say(cl)
        say("PASTE:\n" + paste)
        if gw is not None:
            say(f"This week only (to {weekly.exp:%a %d %b}): " + "; ".join(x for x in [
                ("FLIP " + " ".join(f(x) for x in flw)) if flw else "",
                "PUT " + f(gw.p.idxmin()), "CALL " + f(gw.c.idxmax()),
                "POS " + " ".join(f(x) for x in gw.sort_values("net", ascending=False).head(3).index), f"FUT {tv}"] if x))
    json.dump(state, open("state.json", "w"), indent=1)

def save_state(state):
    json.dump(state, open("state.json", "w"), indent=1)

def oi_times(c):
    """When does CME's OI for each trading day arrive? Uses the futures only (very cheap)."""
    end = data_end(c)
    kw = dict(schema="statistics", symbols=["6E.FUT", "NQ.FUT", "GC.FUT"], stype_in="parent", start=end - pd.Timedelta(days=12), end=end)
    df = guarded(c, "OI timing check", **kw).reset_index()
    oi = df[df.stat_type == 9].copy()
    oi["trade_date"] = pd.to_datetime(oi["ts_ref"], utc=True).dt.date
    oi["arrived_MU"] = pd.to_datetime(oi["ts_recv"], utc=True) + pd.Timedelta(hours=4)
    say("When CME OI arrived (Mauritius time):")
    for d, g in oi.groupby("trade_date"):
        first = g["arrived_MU"].min()
        say(f"  OI for {pd.Timestamp(d):%a %d %b} → first arrived {first:%a %d %b %H:%M}")

def telegram():
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        print("(no Telegram secrets, skipped)"); return
    import requests
    # Telegram limit is 4096 chars → send in parts, split on product blocks
    text, parts, cur = "\n".join(OUT), [], ""
    for blk in text.split("\n\n"):
        if len(cur) + len(blk) + 2 > 3900 and cur:
            parts.append(cur); cur = ""
        cur += ("\n\n" if cur else "") + blk
    if cur:
        parts.append(cur)
    for part in parts:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": part[:4000]}, timeout=20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--oitime", action="store_true")
    a = ap.parse_args()
    c = client()
    try:
        oi_times(c) if a.oitime else discover(c) if a.discover else run(c)
    except SystemExit as e:
        say(f"CME levels stopped: {e}"); open("levels.txt", "w").write("\n".join(OUT)); telegram(); raise
    except Exception as e:
        import traceback; traceback.print_exc()
        say(f"CME levels FAILED: {type(e).__name__}: {str(e)[:300]}"); open("levels.txt", "w").write("\n".join(OUT)); telegram(); raise
    open("levels.txt", "w").write("\n".join(OUT))
    telegram()
