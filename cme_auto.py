"""Luv CME auto levels — pulls CME options OI + settlements from Databento (licensed CME data),
computes dealer GEX levels and straddle-based SD vols, and sends ready paste lines to Telegram.

Modes:
  python cme_auto.py --discover   → finds the option product codes for 6E / NQ / GC and prints the cost (cheap, run once)
  python cme_auto.py              → daily run (needs products.json from discover)
Env: DATABENTO_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, MAX_COST (USD, default 3), BASIS_EUR/BASIS_NAS/BASIS_GOLD (optional overrides)
"""
import argparse, json, math, os, sys, datetime as dt
import numpy as np, pandas as pd
import databento as db

DATASET = "GLBX.MDP3"
PRODUCTS = {
    "EUR":  {"root": "6E", "mult": 125000, "spot": ["EURUSD=X"], "dec": 5},
    "NAS":  {"root": "NQ", "mult": 20,     "spot": ["^NDX"],     "dec": 2},
    "GOLD": {"root": "GC", "mult": 100,    "spot": ["XAUUSD=X"], "dec": 2},
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

def guarded(c, label, **kw):
    cost = c.metadata.get_cost(dataset=DATASET, **kw)
    print(f"[cost] {label}: ${cost:.4f}")
    if cost > MAX_COST:
        sys.exit(f"Stopped: {label} would cost ${cost:.2f} (> MAX_COST ${MAX_COST}). Raise MAX_COST if you're sure.")
    return c.timeseries.get_range(dataset=DATASET, **kw).to_df()

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

# ───────── Daily run ─────────
def run(c):
    prods = json.load(open("products.json"))
    end = data_end(c)
    start = end - pd.Timedelta(days=3)
    parents = [f"{a}.OPT" for v in prods.values() for a in v] + [f"{p['root']}.FUT" for p in PRODUCTS.values()]
    kw = dict(symbols=parents, stype_in="parent", start=start, end=end)
    defs = guarded(c, "definitions", schema="definition", **kw)
    stats = guarded(c, "statistics", schema="statistics", **kw)
    defs = defs.sort_index().groupby("instrument_id").last()
    stats = stats.sort_index()
    oi = stats[stats.stat_type == 9].groupby("instrument_id").last()
    st = stats[stats.stat_type == 3].groupby("instrument_id").last()
    oi_date = pd.to_datetime(oi["ts_ref"]).max() if len(oi) else None
    now = pd.Timestamp.now(tz="UTC")
    say(f"CME levels · {now:%a %d %b %H:%M} UTC · OI as of {oi_date:%a %d %b} (CME official)" if oi_date is not None else "CME levels · no OI found")

    defs["oi"] = oi["quantity"].reindex(defs.index)
    defs["settle"] = st["price"].reindex(defs.index)
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
            T = trading_years(now, exp)
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
            exps.append(dict(exp=exp, und=und, F=F, T=T, vol=s, sd=sd, code=g.raw_symbol.iloc[0][:5]))
        if not exps:
            say(f"\n{name}: no settlement prices yet"); continue
        ex = pd.DataFrame(exps).sort_values("exp")
        front = ex.iloc[0]
        fri = ex[ex.exp.dt.weekday == 4]
        weekly = fri.iloc[0] if len(fri) else front
        # GEX (all expiries) + this-week-only
        vmap = {(r.exp, r.und): r for r in ex.itertuples()}
        F0 = front.F
        def gex(oo):
            rows = []
            for r in oo.itertuples():
                e = vmap.get((r.expiration, r.underlying))
                if e is None or r.oi <= 0:
                    continue
                k = gamma(e.F, r.strike_price, e.vol, e.T) * p["mult"] * e.F * e.F * 0.01
                rows.append((r.strike_price, r.oi * k if r.instrument_class == "C" else 0.0, -r.oi * k if r.instrument_class == "P" else 0.0))
            if not rows:
                return None, [], lambda S: 0.0
            g = pd.DataFrame(rows, columns=["strike", "c", "p"]).groupby("strike").sum()
            g["net"] = g.c + g.p
            def tot(S):
                t = 0.0
                for r in oo.itertuples():
                    e = vmap.get((r.expiration, r.underlying))
                    if e is None or r.oi <= 0:
                        continue
                    Se = S + (e.F - F0)
                    t += (r.oi if r.instrument_class == "C" else -r.oi) * gamma(Se, r.strike_price, e.vol, e.T) * p["mult"] * Se * Se * 0.01
                return t
            xs = np.linspace(F0 * 0.97, F0 * 1.03, 241)
            ys = [tot(x) for x in xs]
            fl = [xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]) for i in range(len(xs) - 1) if np.sign(ys[i]) != np.sign(ys[i + 1])]
            return g, sorted(fl, key=lambda q: abs(q - F0))[:2], tot
        g, flips, tot = gex(o)
        gw, flw, _ = gex(o[o.expiration <= weekly.exp])
        # map to the spot chart
        bars = None
        try:
            bars = c.timeseries.get_range(dataset=DATASET, schema="ohlcv-1m", symbols=[front.und], start=end - pd.Timedelta(days=3), end=end).to_df()
        except Exception as e:
            print("fut last failed:", e)
        fut_last = float(bars["close"].iloc[-1]) if bars is not None and len(bars) else F0
        basis, bsrc = spot_basis(name, fut_last)
        d = p["dec"]
        f = lambda x: f"{x - basis:.{d}f}"
        pos = g.sort_values("net", ascending=False).head(3).index
        neg = g.sort_values("net").head(3).index
        paste = "; ".join(x for x in [
            ("FLIP " + " ".join(f(x) for x in flips)) if flips else "",
            "PUT " + f(g.p.idxmin()), "CALL " + f(g.c.idxmax()),
            "POS " + " ".join(f(x) for x in pos), "NEG " + " ".join(f(x) for x in neg),
            f"VOLD {front.vol*100:.2f}", f"VOLW {weekly.vol*100:.2f}"] if x)
        say(f"\n— {name} (futures {front.und} {F0:.{d}f}, basis {basis:+.{d}f} from {bsrc})")
        say(f"Gamma now: {'POSITIVE (pinning, fade levels)' if tot(F0) > 0 else 'NEGATIVE (moves run)'}")
        say(f"Daily 1SD ±{front.sd / math.sqrt(max(front.T * 252, 1)):.{d}f} (expiry {front.exp:%a %d %b}, vol {front.vol*100:.2f}%)")
        say(f"Weekly vol {weekly.vol*100:.2f}% (expiry {weekly.exp:%a %d %b})")
        say("PASTE:\n" + paste)
        if gw is not None:
            say(f"This week only (to {weekly.exp:%a %d %b}): " + "; ".join(x for x in [
                ("FLIP " + " ".join(f(x) for x in flw)) if flw else "",
                "PUT " + f(gw.p.idxmin()), "CALL " + f(gw.c.idxmax()),
                "POS " + " ".join(f(x) for x in gw.sort_values("net", ascending=False).head(3).index)] if x))

def telegram():
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        print("(no Telegram secrets, skipped)"); return
    import requests
    requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": "\n".join(OUT)[:4000]}, timeout=20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    a = ap.parse_args()
    c = client()
    discover(c) if a.discover else run(c)
    open("levels.txt", "w").write("\n".join(OUT))
    telegram()
