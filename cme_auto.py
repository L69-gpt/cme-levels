"""Luv CME auto levels — pulls CME options OI + settlements from Databento (licensed CME data),
computes dealer GEX levels and straddle-based SD vols, and sends ready paste lines to Telegram.

Modes:
  python cme_auto.py --refresh    → Sunday job: saves the contract list (defs_cache.csv.gz) so Monday needs only small downloads
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
    # Treasury note futures (price levels; the indicator maps them onto the US02Y / US10Y yield charts)
    # dur = approx. modified duration of the futures (CTD) → converts price moves to yield bp
    "ZT":   {"tv": "CBOT", "settle_utc": "19:00", "root": "ZT", "mult": 2000,   "spot": [], "dec": 4, "dur": 1.9, "yld": "US 2y"},
    "ZN":   {"tv": "CBOT", "settle_utc": "19:00", "root": "ZN", "mult": 1000,   "spot": [], "dec": 4, "dur": 6.3, "yld": "US 10y"},
}
GEX_DAYS = 35           # expiries inside this window count for GEX
MAX_COST = float(os.environ.get("MAX_COST", "3"))
RUN_BUDGET_MIN = float(os.environ.get("RUN_BUDGET_MIN", "10"))   # hard cap per run, then fall back to last good levels
# scheduled tries per weekday (UTC) — must match the workflow crons. Mauritius = UTC+4.
TRIES_UTC = ["04:17", "05:02", "06:02", "08:02"]
CACHE = "defs_cache.csv.gz"     # saved contract list (definitions) per market
LAST = "last_levels.txt"        # last message that worked, re-sent as STALE when Databento is down
DEF_COLS = ["instrument_class", "raw_symbol", "asset", "underlying", "strike_price", "expiration"]
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
    r = retry(lambda: c.metadata.get_dataset_range(dataset=DATASET), "dataset_range")
    return pd.Timestamp(r.get("end") or r.get("end_date"))

def retry(fn, label="databento", waits=None):
    # Databento sometimes drops the stream ("Response ended prematurely") or times out (504).
    # Retry ANY Databento / network error with growing waits (~13 min total) before giving up.
    import time
    waits = waits or [20, 40, 80, 120]
    for i in range(len(waits) + 1):
        try:
            return fn()
        except Exception as e:
            if isinstance(e, getattr(db, "BentoClientError", ())):
                raise                                   # 4xx = bad request (e.g. expired code) → retrying won't help
            transient = isinstance(e, getattr(db, "BentoError", Exception)) or any(w in str(e) for w in ("prematurely", "504", "timed out", "Timeout", "Connection", "502", "503"))
            if i == len(waits) or not transient:
                raise
            print(f"[{label}] retry {i+1} in {waits[i]}s after: {e}")
            time.sleep(waits[i])

def fetch(c, _waits=None, **kw):
    return retry(lambda: c.timeseries.get_range(dataset=DATASET, **kw).to_df(), "get_range", _waits)

def guarded(c, label, _waits=None, **kw):
    cost = retry(lambda: c.metadata.get_cost(dataset=DATASET, **kw), "cost")
    print(f"[cost] {label}: ${cost:.4f}")
    if cost > MAX_COST:
        sys.exit(f"Stopped: {label} would cost ${cost:.2f} (> MAX_COST ${MAX_COST}). Raise MAX_COST if you're sure.")
    return fetch(c, _waits=_waits, **kw)

# ───────── Discover: which option product codes belong to 6E / NQ / GC ─────────
def discover(c):
    end = data_end(c)
    start = end - pd.Timedelta(hours=26)
    df = guarded(c, "all definitions (1 day)", schema="definition", symbols="ALL_SYMBOLS", start=start, end=end)
    opt = df[df["instrument_class"].isin(["C", "P"])].copy()
    try:
        found = json.load(open("products.json"))
    except Exception:
        found = {}
    for name, p in PRODUCTS.items():
        sub = opt[opt["underlying"].astype(str).str.match(rf"^{p['root']}[FGHJKMNQUVXZ]\d{{1,2}}$")]
        codes = sorted(sub["asset"].astype(str).unique())
        if codes:
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
       "GOLD": ("72hh-3qpy", "088691", "m_money_positions_long_all", "m_money_positions_short_all", "Managed money"),
       "ZT": ("gpe5-46if", "042601", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds"),
       "ZN": ("gpe5-46if", "043602", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds")}

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

# ───────── Level scorecard: log each day's levels, score them after the day ends ─────────
# Futures prices (same units as the levels) from Yahoo 15m bars. Free.
YF = {"EUR": "6E=F", "NAS": "NQ=F", "GOLD": "GC=F", "ZT": "ZT=F", "ZN": "ZN=F"}
KIND_NAME = {"FLIP": "Flip", "PUT": "Put wall", "CALL": "Call wall", "POS": "+γ pin", "NEG": "−γ accel",
             "SD05": "±0.5SD", "SD1": "±1SD", "EXP": "EXP pin"}

def day_window(now):
    """Trading day ends 17:00 New York. Returns (day start, day end) in UTC for the day 'now' is in."""
    ny = now.tz_convert("America/New_York")
    end = ny.normalize() + pd.Timedelta(hours=17)
    if end <= ny:
        end += pd.Timedelta(days=1)
    while end.weekday() >= 5:
        end += pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=1)
    if start.weekday() == 4 or start.weekday() >= 5:     # Monday's session starts Sunday 17:00
        start = end - pd.Timedelta(days=1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")

def log_levels(state, name, now, lv, sd1, regime, exp_cut):
    """Keep one entry per product per trading day (the latest run of the day wins: freshest OI)."""
    st, en = day_window(now)
    lg = state.setdefault("log", [])
    old = [e for e in lg if e["p"] == name and e["end"] == en.isoformat() and not e.get("done")]
    t0 = min([pd.Timestamp(e["t"]) for e in old] + [now])      # score from the FIRST run of the day
    lg[:] = [e for e in lg if e not in old]
    lg.append(dict(p=name, start=st.isoformat(), end=en.isoformat(), t=t0.isoformat(), lv=lv,
                   sd=float(sd1), reg=regime, cut=exp_cut))
    state["log"] = lg[-400:]

def _score_one(px, L, sd, from_t):
    """held = after the first touch price moved 0.5SD away on the side it came from, before closing 0.3SD through.
       broke = closed 0.3SD through first. chop = neither within 4h. miss = never touched."""
    tol = 0.05 * sd
    p = px[px.index >= from_t]
    hit = p[(p.Low <= L + tol) & (p.High >= L - tol)]
    if hit.empty:
        return "miss"
    i = p.index.get_loc(hit.index[0])
    before = p.Close.iloc[i - 1] if i > 0 else p.Open.iloc[i]
    above = before > L
    for _, b in p.iloc[i:i + 16].iterrows():
        if above:
            if b.Close < L - 0.3 * sd: return "broke"
            if b.High >= L + 0.5 * sd: return "held"
        else:
            if b.Close > L + 0.3 * sd: return "broke"
            if b.Low <= L - 0.5 * sd: return "held"
    return "chop"

def score_logs(state, now):
    lg = state.get("log", [])
    todo = [e for e in lg if not e.get("done") and pd.Timestamp(e["end"]) < now]
    if not todo:
        return
    try:
        import yfinance as yf
    except Exception as e:
        print("scorecard: yfinance missing", e); return
    cache = {}
    sc = state.setdefault("score", {})
    for e in todo:
        sym = YF.get(e["p"])
        if not sym:
            e["done"] = True; continue
        if sym not in cache:
            try:
                h = yf.Ticker(sym).history(period="30d", interval="15m")
                h.index = h.index.tz_convert("UTC")
                cache[sym] = h
            except Exception as ex:
                print("scorecard fetch failed", sym, ex); cache[sym] = None
        h = cache[sym]
        if h is None or h.empty:
            continue
        st, en, t0 = pd.Timestamp(e["start"]), pd.Timestamp(e["end"]), pd.Timestamp(e["t"])
        px = h[(h.index >= st) & (h.index < en)]
        if px.empty:
            e["done"] = True; continue
        sd, reg = e["sd"], e["reg"]
        lv = dict(e["lv"])
        o = float(px.Open.iloc[0])                          # daily open (futures) → daily SD bands
        lv["SD05"] = [o - 0.5 * sd, o + 0.5 * sd]
        lv["SD1"] = [o - sd, o + sd]
        pp = sc.setdefault(e["p"], {})
        for kind, vals in lv.items():
            if kind == "EXP":
                continue
            for L in vals:
                r = _score_one(px, float(L), sd, t0 if kind not in ("SD05", "SD1") else st)
                c = pp.setdefault(f"{kind}|{reg}", {"held": 0, "broke": 0, "chop": 0, "miss": 0})
                c[r] += 1
        # EXP pins: pinned = price within 0.25SD of the strike at the cut
        if e.get("cut") and lv.get("EXP"):
            cut = pd.Timestamp(e["cut"])
            atc = h[h.index <= cut]
            if len(atc) and cut <= h.index[-1]:
                pc = float(atc.Close.iloc[-1])
                c = pp.setdefault(f"EXP|{reg}", {"pinned": 0, "not": 0})
                L = float(lv["EXP"][0])                      # the biggest expiring strike
                c["pinned" if abs(pc - L) <= 0.25 * sd else "not"] += 1
        e["done"] = True
    state["log"] = lg

def score_line(state, name):
    pp = state.get("score", {}).get(name)
    if not pp:
        return None
    days = len({e["end"] for e in state.get("log", []) if e["p"] == name and e.get("done")})
    out = []
    for kind in ["PUT", "CALL", "FLIP", "POS", "NEG", "SD05", "SD1", "EXP"]:
        rows = {k.split("|")[1]: v for k, v in pp.items() if k.split("|")[0] == kind}
        if not rows:
            continue
        def rate(v):
            if kind == "EXP":
                n = v.get("pinned", 0) + v.get("not", 0); return v.get("pinned", 0), n
            n = v["held"] + v["broke"] + v["chop"]
            return (v["broke"] if kind == "NEG" else v["held"]), n
        tot_k = sum(rate(v)[0] for v in rows.values()); tot_n = sum(rate(v)[1] for v in rows.values())
        if tot_n == 0:
            continue
        word = "broke" if kind == "NEG" else "pinned" if kind == "EXP" else "held"
        txt = f"{KIND_NAME[kind]} {word} {tot_k}/{tot_n} ({100 * tot_k / tot_n:.0f}%)"
        if kind in ("SD05", "SD1", "PUT", "CALL") and len(rows) > 1:
            txt += " [" + " · ".join(f"{r}γ {rate(v)[0]}/{rate(v)[1]}" for r, v in rows.items() if rate(v)[1]) + "]"
        out.append(txt)
    if not out:
        return None
    return f"Scorecard ({days} day{'s' if days != 1 else ''} scored, levels touched only): " + " · ".join(out)

# ───────── Saved contract list (definitions) — so daily runs only need the small OI download ─────────
def load_cache():
    try:
        d = pd.read_csv(CACHE)
    except Exception:
        return pd.DataFrame()
    d["expiration"] = pd.to_datetime(d["expiration"], utc=True)
    age = pd.Timestamp.now(tz="UTC") - pd.to_datetime(d["cached_at"], utc=True)
    d = d[age < pd.Timedelta(days=10)]                     # too old → ignore (new expiries missing)
    d.index = pd.DatetimeIndex([pd.Timestamp("2000-01-01", tz="UTC")] * len(d), name="ts_recv")
    return d

def put_cache(cache, name, df, now):
    if df is None or not len(df):
        return cache
    d = df.reset_index(drop=True)[["instrument_id"] + DEF_COLS].copy()
    d["expiration"] = pd.to_datetime(d["expiration"], utc=True)
    fut = d["instrument_class"] == "F"
    d = d[fut | ((d["expiration"] > now) & (d["expiration"] < now + pd.Timedelta(days=GEX_DAYS + 14)))]
    d = d.drop_duplicates("instrument_id", keep="last")
    d["prod"], d["cached_at"] = name, now.isoformat()
    d.index = pd.DatetimeIndex([pd.Timestamp("2000-01-01", tz="UTC")] * len(d), name="ts_recv")
    keep = cache[cache["prod"] != name] if len(cache) else cache
    return pd.concat([keep, d]) if len(keep) else d

def save_cache(cache):
    if len(cache):
        cache.reset_index(drop=True).to_csv(CACHE, index=False)

def refresh(c):
    """Sunday job: save Friday's full contract list while Databento is quiet (warms Monday's run)."""
    now = pd.Timestamp.now(tz="UTC")
    end = data_end(c)
    day0 = end.normalize()
    while day0.weekday() >= 5:
        day0 -= pd.Timedelta(days=1)
    prods = json.load(open("products.json"))
    cache = load_cache()
    for name, p in PRODUCTS.items():
        if not prods.get(name):
            continue
        par = [f"{a}.OPT" for a in prods[name]] + [f"{p['root']}.FUT"]
        try:
            df_ = guarded(c, f"definitions {name}", schema="definition", symbols=par, stype_in="parent", start=day0, end=day0 + pd.Timedelta(hours=2), _waits=[30, 60, 120])
            cache = put_cache(cache, name, df_, now)
            print(f"{name}: saved {len(df_)} contracts")
        except Exception as e:
            print(f"{name}: refresh failed ({e}), keeping the old list")
    save_cache(cache)

# ───────── Daily run ─────────
def run(c):
    prods = json.load(open("products.json"))
    try:
        state = json.load(open("state.json"))
    except Exception:
        state = {}
    end = data_end(c)
    now = pd.Timestamp.now(tz="UTC")
    try:
        score_logs(state, now)
    except Exception as e_:
        print("scorecard scoring failed:", e_)
    day0 = end.normalize()                       # definitions snapshot = UTC midnight
    if end - day0 < pd.Timedelta(hours=2):
        day0 -= pd.Timedelta(days=1)
    # definitions per product (smaller requests = no gateway timeouts)
    dend = min(end, day0 + pd.Timedelta(hours=2))   # definitions snapshot is at UTC midnight → small window = small request
    dparts, skipped, cached = [], [], []
    cache = load_cache()
    for name, p in PRODUCTS.items():
        if not prods.get(name):
            print(f"{name}: no option codes in products.json yet (run discover once)"); continue
        par = [f"{a}.OPT" for a in prods.get(name, [])] + [f"{p['root']}.FUT"]
        try:
            # the contract list is the big request that times out → one quick try only
            df_ = guarded(c, f"definitions {name}", schema="definition", symbols=par, stype_in="parent", start=day0, end=dend, _waits=[20, 40])
            dparts.append(df_)
            cache = put_cache(cache, name, df_, now); save_cache(cache)     # save right away (survives the 10-min cap)
            continue
        except Exception as e_:
            print(f"{name}: fresh definitions failed ({e_})")
        cd = cache[cache["prod"] == name] if len(cache) else cache
        if len(cd):
            print(f"{name}: using saved contract list ({len(cd)} contracts)")
            dparts.append(cd.drop(columns=["prod"]))
            cached.append(name)
            continue
        # no saved list yet → ask for each option code on its own (small requests)
        got = []
        for sym in par:
            try:
                got.append(fetch(c, schema="definition", symbols=[sym], stype_in="parent", start=day0, end=dend, _waits=[10]))
            except Exception as e2:
                print(f"  {sym}: failed ({e2})")
        if got:
            dparts.extend(got)
            cache = put_cache(cache, name, pd.concat(got), now); save_cache(cache)
        else:
            skipped.append(name)
            prods[name] = []          # no data for this market today → skip it, keep the others
            print(f"{name}: skipped today (Databento timeout)")
    save_cache(cache)
    if not dparts:
        raise RuntimeError("no contract data from Databento for any market")
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
    BAND = {"EUR": 0.05, "NAS": 0.12, "GOLD": 0.12, "ZT": 0.02, "ZN": 0.05}
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
    for i in range(0, len(ids), 200):
        kw = dict(schema="statistics", symbols=ids[i:i + 200], stype_in="instrument_id", start=sstart, end=end)
        total += retry(lambda: c.metadata.get_cost(dataset=DATASET, **kw), "cost")
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
        if not prods.get(name):
            if name in skipped:
                say(f"\n— {name}: skipped today (Databento timed out). Use yesterday's {name} levels.")
            continue
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
        wk_fri = (now + pd.Timedelta(days=(4 - now.weekday()) % 7)).normalize() + pd.Timedelta(hours=20)
        weekly = ex.iloc[(ex.exp - wk_fri).abs().argsort()].iloc[0]
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
        exp_txt = ("EXP " + " ".join(f"{k:.{d}f}" for k in big.index) + f" @{(tod.expiration.min() + pd.Timedelta(hours=4)):%Y-%m-%dT%H:%M}") if len(big) else ""
        f = lambda x: f"{x:.{d}f}"
        pos = g.sort_values("net", ascending=False).head(3).index
        neg = g.sort_values("net").head(3).index
        paste = "; ".join(x for x in [
            ("FLIP " + " ".join(f(x) for x in flips)) if flips else "",
            "PUT " + f(g.p.idxmin()), "CALL " + f(g.c.idxmax()),
            "POS " + " ".join(f(x) for x in pos), "NEG " + " ".join(f(x) for x in neg),
            f"VOLD {front.vol*100:.2f}", f"VOLW {volw:.2f}", f"VOLM {volm:.2f}", exp_txt, f"FUT {tv}"] if x)
        sd_ = fut.loc[front.und, "sdate"] if front.und in fut.index else None
        say(f"\n— {name}" + (f" = {p['yld']} note futures, levels show on the {'US02Y' if name == 'ZT' else 'US10Y'} chart" if "dur" in p else "") + f" (futures {front.und} settle {F0:.{d}f}" + (f" from {sd_:%a %d %b}" if sd_ is not None and not pd.isna(sd_) else "") + " · levels in futures prices, indicator converts them live)")
        say(f"Gamma now: {'POSITIVE (pinning, fade levels)' if tot(F0) > 0 else 'NEGATIVE (moves run)'}")
        bp = (lambda pts: f" ≈ {pts / (F0 * p['dur']) * 1e4:.1f}bp of {p['yld']} yield") if "dur" in p else (lambda pts: "")
        say(f"Daily 1SD ±{F0 * front.vol / math.sqrt(252):.{d}f}{bp(F0 * front.vol / math.sqrt(252))} (expiry {front.exp:%a %d %b}, vol {front.vol*100:.2f}%)")
        say(f"Weekly vol {volw:.2f}% (locked {stp['week']}, expiry {stp['wexp']}) · Monthly vol {volm:.2f}% (locked {stp['month']}, expiry {stp['mexp']})")
        # ── positioning extras: skew, term structure, vanna / charm ──
        flip_note = " (DXY: opposite)" if name == "EUR" else " (yield: opposite)" if "dur" in p else ""
        up_word = {"EUR": "EUR UP / USD down", "NAS": "NAS UP", "GOLD": "GOLD UP"}.get(name, f"bonds UP = {p.get('yld', '')} YIELD DOWN")
        dn_word = {"EUR": "EUR DOWN / USD up", "NAS": "NAS DOWN", "GOLD": "GOLD DOWN"}.get(name, f"bonds DOWN = {p.get('yld', '')} YIELD UP")
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
        try:
            reg = "+" if tot(F0) > 0 else "−"
            lv = {"FLIP": [float(x) for x in flips], "PUT": [float(g.p.idxmin())], "CALL": [float(g.c.idxmax())],
                  "POS": [float(x) for x in pos], "NEG": [float(x) for x in neg], "EXP": [float(x) for x in big.index]}
            log_levels(state, name, now, lv, F0 * front.vol / math.sqrt(252), reg,
                       tod.expiration.min().isoformat() if len(big) else None)
            sl = score_line(state, name)
            if sl:
                say(sl)
        except Exception as e_:
            print("scorecard log failed:", e_)
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
        if len(blk) > 3900:                       # one huge block (e.g. discover) → split it by lines
            for ln in blk.split("\n"):
                if len(cur) + len(ln) + 1 > 3900 and cur:
                    parts.append(cur); cur = ""
                cur += ("\n" if cur else "") + ln
            continue
        cur += ("\n\n" if cur else "") + blk
    if cur:
        parts.append(cur)
    for part in parts:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": part[:4000]}, timeout=20)

def today_key(now):
    return day_window(now)[1].strftime("%Y-%m-%d")

def load_state():
    try:
        return json.load(open("state.json"))
    except Exception:
        return {}

def next_try(now):
    if now.weekday() >= 5:
        return None
    for t in TRIES_UTC:
        ts = now.normalize() + pd.Timedelta(t + ":00")
        if ts > now + pd.Timedelta(minutes=5):
            return ts
    return None

def on_success(now):
    st = load_state()
    st["last_ok"], st["last_ok_at"] = today_key(now), now.isoformat()
    st.pop("fail_day", None); st.pop("fails", None)
    save_state(st)
    open(LAST, "w").write("\n".join(OUT))

def on_fail(now, err, manual):
    """Databento down: send the last good levels once (marked STALE), then retry quietly on schedule."""
    st = load_state()
    day = today_key(now)
    n = st.get("fails", 0) + 1 if st.get("fail_day") == day else 1
    st["fail_day"], st["fails"] = day, n
    save_state(st)
    nt = next_try(now)
    when = f"{(nt + pd.Timedelta(hours=4)):%H:%M} Mauritius" if nt is not None else None
    OUT.clear()
    if st.get("last_ok") == day:
        print("levels already sent today; nothing to send"); return
    if n == 1 or manual:
        try:
            old = open(LAST).read()
        except Exception:
            old = ""
        say(f"⚠️ CME levels: Databento is down ({err}).")
        say(f"Auto-retrying at {when}, nothing for you to do." if when else "No more tries today; tomorrow's 08:17 run tries again.")
        if old and n == 1:
            say(f"\nBelow are the LAST GOOD levels (from {st.get('last_ok', '?')}). STALE, use with care:\n")
            say(old)
    elif nt is None:
        say(f"CME levels: Databento still down after {n} tries today. Keep using the STALE levels sent earlier; tomorrow's 08:17 run tries again.")
    else:
        print(f"try {n} failed, staying quiet; next try {when}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--oitime", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    c = client()
    now = pd.Timestamp.now(tz="UTC")
    if a.refresh:
        refresh(c); sys.exit(0)
    if a.discover or a.oitime:
        oi_times(c) if a.oitime else discover(c)
        open("levels.txt", "w").write("\n".join(OUT)); telegram(); sys.exit(0)
    manual = os.environ.get("EVENT") == "workflow_dispatch"
    if not manual and load_state().get("last_ok") == today_key(now):
        print("Today's levels already sent; retry slot not needed."); sys.exit(0)
    import signal
    class RunCap(BaseException):      # BaseException so per-market "except Exception" handlers can't swallow it
        pass
    def _cap(*_):
        raise RunCap(f"no answer within {RUN_BUDGET_MIN:g} min")
    signal.signal(signal.SIGALRM, _cap)
    signal.alarm(int(RUN_BUDGET_MIN * 60))
    try:
        run(c)
        signal.alarm(0)
    except SystemExit as e:
        signal.alarm(0)
        say(f"CME levels stopped: {e}"); open("levels.txt", "w").write("\n".join(OUT)); telegram(); raise
    except BaseException as e:
        signal.alarm(0)
        import traceback; traceback.print_exc()
        on_fail(now, str(e)[:120] if type(e).__name__ == "RunCap" else f"{type(e).__name__}: {str(e)[:120]}", manual)
        open("levels.txt", "w").write("\n".join(OUT)); telegram(); sys.exit(0)
    on_success(now)
    open("levels.txt", "w").write("\n".join(OUT))
    telegram()
