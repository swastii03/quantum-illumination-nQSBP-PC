import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

kappa=0.01
nB=1.0
truncOrder=2
gain=1.5
outDir="qiFigures"
os.makedirs(outDir, exist_ok=True)

families=["pcSingle", "pcBiside", "nqSingle", "nqBiside"]
labels={"pcSingle": "PC single-side", "pcBiside": "PC two-side",
          "nqSingle": "nQSBP single-side", "nqBiside": "nQSBP two-side"}
styles={"pcSingle": ("tab:blue", "o", "-"), "pcBiside": ("tab:orange", "s", "-"),
          "nqSingle": ("tab:green", "^","--"), "nqBiside": ("tab:red", "v", "--")}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titlesize": 10, "legend.fontsize": 8.5})


def pcFactor(j, T):
    j=np.asarray(j, dtype=float)
    return T ** ((j-1.0)/2.0)*(T-j*(1.0-T))

def buildState(family, lam, T=0.2, g=gain, n=truncOrder, jMax=400):
    j=np.arange(jMax)
    amp = np.sqrt(1-lam**2)*lam**j
    if family in ("pcSingle", "nqSingle"):
        amp= amp * pcFactor(j, T)
    elif family in ("pcBiside", "nqBiside"):
        amp=amp * pcFactor(j, T) ** 2
    pPc= float(np.sum(amp ** 2))
    c = amp / math.sqrt(pPc)
    pNla = 1.0
    if family.startswith("nq"):
        modes = 1 if family == "nqSingle" else 2
        pref = (n + 1) * math.factorial(n) / ((n + 1) ** n * (g ** 2 + 1) ** n)
        w = g ** (modes * j[: n + 1])
        pNla = pref ** modes * float(np.sum((w * c[: n + 1]) ** 2))
        cNew = np.zeros_like(c)
        cNew[: n + 1] = w * c[: n + 1]
        c = cNew / np.linalg.norm(cNew)
    p = c ** 2
    meanPhotons = float(np.sum(j * p))
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    return {"c": c, "pPc": pPc, "pNla": pNla, "pSucc": pPc * pNla,
            "ns": meanPhotons, "entropy": entropy}


def lamForNs(family, nsTarget, **kw):
    f = lambda lam: buildState(family, lam, **kw)["ns"] - nsTarget
    grid = np.linspace(1e-3, 0.9, 400)
    vals = np.array([f(x) for x in grid])
    idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if len(idx) == 0:
        return None
    return brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-14)

krausCache = {}

def krausOps(k, nb, dIn, kEnv, dOut):
    key = (k, nb, dIn, kEnv, dOut)
    if key in krausCache:
        return krausCache[key]
    t, r = math.sqrt(k), math.sqrt(1 - k)
    nEnv = nb / (1 - k)
    pk = nEnv ** np.arange(kEnv) / (1 + nEnv) ** (np.arange(kEnv) + 1)
    kq = dIn + kEnv
    ops = np.zeros((kEnv, kq, dOut, dIn))
    lg = lambda x: math.lgamma(x + 1)
    for m in range(dIn):
        for e in range(kEnv):
            for i in range(m + 1):
                ci = math.comb(m, i) * t ** i * r ** (m - i)
                for jj in range(e + 1):
                    out = i + jj
                    if out >= dOut:
                        continue
                    q = (m-i)+(e-jj)
                    coef = ci * math.comb(e, jj) * (-r) ** jj * t ** (e - jj)
                    coef *= math.exp(0.5 * (lg(out) + lg(q) - lg(m) - lg(e)))
                    ops[e, q, out, m] += coef
    krausCache[key] = (ops, pk)
    return krausCache[key]


def dimsFor(c, k, nb, tol=1e-13, tail=1e-11):
    p=c ** 2
    cum=np.cumsum(p[::-1])[::-1]
    dIn=int(max(6, np.argmax(cum < tol) if np.any(cum < tol) else len(c)))
    dIn=min(dIn, 45)
    nEnv=nb/(1 - k)
    x=nEnv/(1+ nEnv)
    kEnv=int(math.ceil(math.log(tail) / math.log(x))) + 1
    return dIn, kEnv, dIn + kEnv

def stateRho(c, k, nb, dims=None):
    dIn, kEnv, dOut=dims or dimsFor(c, k, nb)
    ops,pk=krausOps(k, nb, dIn, kEnv, dOut)
    cc=np.zeros(dIn)
    cc[: min(dIn, len(c))]=c[:dIn]
    v=np.zeros((kEnv, ops.shape[1], dIn, dOut))
    for i in range(dIn):
        v[:, :, i, :] = cc[i] * ops[:, :, :, i] * np.sqrt(pk)[:, None, None]
    v=v.reshape(-1, dIn * dOut).T
    rho1=v @ v.T
    nth=nb / (1 + nb)
    tau=nth**np.arange(dOut) / (1 + nb)
    p0=np.outer(cc ** 2, tau).reshape(-1)
    return p0, rho1

def chernoff(c, k=kappa, nb=nB, dims=None):
    p0, rho1 = stateRho(c, k, nb, dims)
    eigVals, eigVecs = np.linalg.eigh(rho1)
    eigVals = np.clip(eigVals, 0.0, None)
    w = eigVecs ** 2

    def qOfS(s):
        a= np.where(p0 > 0, p0 ** s, 0.0) if s > 0 else (p0 > 0).astype(float)
        b= eigVals ** (1 - s) if s < 1 else (eigVals > 0).astype(float)
        return float(a @ w @ b)

    grid= np.linspace(0.0, 1.0, 101)
    vals= [qOfS(s) for s in grid]
    s0 =grid[int(np.argmin(vals))]
    res= minimize_scalar(qOfS, bounds=(max(0, s0 - 0.01), min(1, s0 + 0.01)), method="bounded", options={"xatol":1e-12})
    q= min(res.fun, min(vals))
    qHalf= qOfS(0.5)
    return {"xi": -math.log(q), "xiHalf": -math.log(qHalf), "trace": float(np.trace(rho1))}

def helstromError(c, dims):
    p0, rho1= stateRho(c, kappa, nB, dims)
    return 0.5 * (1 - 0.5 * np.abs(np.linalg.eigvalsh(rho1 - np.diag(p0))).sum())


def evaluate(state):
    dims= dimsFor(state["c"], kappa, nB)
    ch= chernoff(state["c"], kappa, nB, dims)
    assert abs(ch["trace"] - 1) < 1e-7
    return {"xi": ch["xi"], "xiHalf": ch["xiHalf"], "hel": helstromError(state["c"], dims)}


def l1Coherence(c):
    return float(np.sum(np.abs(c))**2-1)


def tmsvAtNs(ns):
    lam =lamForNs("tmsv", ns)
    s =buildState("tmsv", lam)
    return s, evaluate(s)

def runPoints(points, cachePath):
    if os.path.exists(cachePath):
        return pd.read_csv(cachePath)
    rows=[]
    for lab, fam, lam, kw in points:
        s=buildState(fam, lam, **kw)
        e=evaluate(s)
        b=buildState("tmsv", lam)
        be=evaluate(b)
        t,te=tmsvAtNs(s["ns"])
        row=dict(lab)
        row.update({
            "family": fam, "lam": lam, "ns": s["ns"], "nsRatio": s["ns"] / b["ns"],
            "xiRatio": e["xi"]/be["xi"], "bhRatio": e["xiHalf"] / be["xiHalf"],
            "xiRatioEqualNs": e["xi"] / te["xi"], "bhRatioEqualNs": e["xiHalf"] / te["xiHalf"],
            "helAdv": be["hel"] - e["hel"], "helAdvEqualNs": te["hel"] - e["hel"],
            "entropyRatio": s["entropy"] / b["entropy"],
            "entropyRatioEqualNs": s["entropy"] / t["entropy"],
            "l1Ratio": l1Coherence(s["c"]) / l1Coherence(b["c"]),
            "pSucc": s["pSucc"], "rateRatio": s["pSucc"] * e["xi"] / be["xi"],
        })
        rows.append(row)
        print(fam, lab, round(row["xiRatio"], 4), round(row["xiRatioEqualNs"], 5), flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(cachePath, index=False)
    return df


tList=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ns0List=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
nList=[1, 2,3,4,5,6]
gList =[0.5,0.8,1.0, 1.25, 1.5, 2.0, 3.0]
lamRef=lamForNs("tmsv", 0.05)

tPoints=[({"T": T}, f, lamRef, {"T": T}) for f in families for T in tList]
nsPoints=[({"nS0": x}, f, lamForNs("tmsv", x), {"T": 0.2}) for f in families for x in ns0List]
truncPoints=[({"n": 0}, f, lamRef, {"T": 0.2}) for f in ["pcSingle", "pcBiside"]]
truncPoints+=[({"n": n}, f, lamRef, {"T": 0.2, "n": n}) for f in ["nqSingle", "nqBiside"] for n in nList]
gainPoints=[({"g": g}, f, lamRef, {"T": 0.2, "g": g}) for f in ["nqSingle", "nqBiside"] for g in gList]

dfT=runPoints(tPoints, os.path.join(outDir, "sweepT.csv"))
dfNs=runPoints(nsPoints, os.path.join(outDir, "sweepSqueezing.csv"))
dfTrunc=runPoints(truncPoints, os.path.join(outDir, "sweepTruncation.csv"))
dfGain=runPoints(gainPoints, os.path.join(outDir, "sweepGain.csv"))

subtitle=f"$\\kappa$={kappa}, $N_B$={nB}"
tLabel="catalysis transmissivity $T$"
nsLabel="mean photon number $N_{S0}$"


def saveFigure(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outDir,name + ".png"), dpi=200)
    fig.savefig(os.path.join(outDir,name + ".pdf"))
    plt.show()
    plt.close(fig)

def drawLines(ax, df, x, y, fams=families):
    for f in fams:
        s = df[df.family== f].sort_values(x)
        color, marker, ls = styles[f]
        ax.plot(s[x], s[y], color=color, marker=marker, ls=ls, ms=4, label=labels[f])

def refLine(ax, value=1.0):
    ax.axhline(value, color="k", lw=1.2)

for name, df, x, xLabel, logX, extra, yLow in [
    ("fig1ExponentVsT", dfT, "T", tLabel, False, f"$N_{{S0}}$=0.05, $g$={gain}, $n$={truncOrder}", 0.0),
    ("fig2ExponentVsSqueezing", dfNs, "nS0", nsLabel, True, f"$T$=0.2, $g$={gain}, $n$={truncOrder}", 0.6),
]:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    drawLines(ax[0], df, x, "xiRatio")
    refLine(ax[0])
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\xi/\xi_{\rm TMSV}$(Chernoff exponent ratio)")
    ax[0].set_title("(a)Equal parent squeezing $\\lambda$\n" + extra + ", " + subtitle)
    drawLines(ax[1], df, x, "xiRatioEqualNs")
    refLine(ax[1])
    ax[1].set_ylabel(r"$\xi/\xi_{\rm TMSV}$ at same $N_S$")
    ax[1].set_title("(b)Equal signal energy $N_S$\n" + extra + ", " + subtitle)
    ax[1].set_ylim(yLow, 1.05)
    for a in ax:
        a.set_xlabel(xLabel)
        if logX:
            a.set_xscale("log")
    ax[0].legend()
    saveFigure(fig, name)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
drawLines(ax[0], dfT, "T", "pSucc")
ax[0].set_yscale("log")
ax[0].set_xlabel(tLabel)
ax[0].set_ylabel("success probability $P_{\\rm succ}$")
ax[0].set_title("(a)vs $T$ ($N_{S0}$=0.05)")
drawLines(ax[1], dfNs, "nS0", "pSucc")
ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[1].set_xlabel(nsLabel)
ax[1].set_ylabel("$P_{\\rm succ}$")
ax[1].set_title("(b)vs parent squeezing ($T$=0.2)")
ax[0].legend()
saveFigure(fig, "fig3SuccessProbability")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
panels = [(ax[0], "xiRatio", "(a) EQUAL $\\lambda$: exponent ratio vs $P_{\\rm succ}$"),
          (ax[1], "xiRatioEqualNs","(b) EQUAL $N_S$: exponent ratio vs $P_{\\rm succ}$")]
for a, y, title in panels:
    for f in families:
        s = dfT[dfT.family == f].sort_values("T")
        color, marker, ls = styles[f]
        a.plot(s.pSucc, s[y], color=color, marker=marker, ls=ls, ms=4, label=labels[f])
        if f in ("pcSingle", "nqBiside"):
            for _, r in s.iterrows():
                if round(r["T"], 2) in (0.1, 0.2, 0.9):
                    a.annotate(f"T={r['T']:.1f}", (r.pSucc, r[y]), fontsize=6.5,
                               xytext=(3, 3), textcoords="offset points", color=color)
    a.plot([1], [1], "k*", ms=12, label="plain TMSV ($P$=1)")
    a.axhline(1, color="k", lw=1)
    a.set_xscale("log")
    a.set_xlabel("heralding success probability $P_{\\rm succ}$")
    a.set_title(title + f"\n$N_{{S0}}$=0.05, $g$={gain}, $n$={truncOrder}, {subtitle}")
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$\xi/\xi_{\rm TMSV}$")
ax[1].set_ylabel(r"$\xi/\xi_{\rm TMSV}$ (same $N_S$)")
ax[1].set_ylim(0, 1.05)
ax[0].legend(loc="lower left")
saveFigure(fig,"fig4TradeoffSuccessVsExponent")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
drawLines(ax[0], dfT, "T", "rateRatio")
refLine(ax[0])
ax[0].set_yscale("log")
ax[0].set_xlabel(tLabel)
ax[0].set_ylabel(r"$P_{\rm succ}\,\xi/\xi_{\rm TMSV}$ (per heralding attempt)")
ax[0].set_title("(a) Equal $\\lambda$-rate vs $T$($N_{S0}$=0.05)\n"+subtitle)
drawLines(ax[1], dfNs, "nS0", "rateRatio")
refLine(ax[1])
ax[1].set_yscale("log")
ax[1].set_xscale("log")
ax[1].set_xlabel(nsLabel)
ax[1].set_ylabel(r"$P_{\rm succ}\,\xi/\xi_{\rm TMSV}$")
ax[1].set_title("(b)Equal $\\lambda$-rate vs squeezing($T$=0.2)")
ax[0].legend()
saveFigure(fig,"fig5RatePerAttempt")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
drawLines(ax[0], dfT, "T", "nsRatio")
refLine(ax[0])
ax[0].set_yscale("log")
ax[0].set_xlabel(tLabel)
ax[0].set_ylabel(r"$N_S/N_S^{\rm TMSV}$ (brightness ratio)")
ax[0].set_title("(a)brightness of heralded probe at equal $\\lambda$($N_{S0}$=0.05)")
ax[0].legend()
for f in families:
    s = pd.concat([dfT[dfT.family == f], dfNs[dfNs.family == f]])
    color, marker, ls = styles[f]
    ax[1].scatter(s.nsRatio, s.xiRatio, color=color, marker=marker, s=18, label=labels[f], alpha=0.8)
xx = np.logspace(-2.5, 1.5, 50)
ax[1].plot(xx, xx, "k--", lw=1, label=r"$\xi$ ratio = $N_S$ ratio")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"$N_S/N_S^{\rm TMSV}$")
ax[1].set_ylabel(r"$\xi/\xi_{\rm TMSV}$ (equal $\lambda$)")
ax[1].set_title("(b)exponent gain tracks brightness")
ax[1].legend()
saveFigure(fig, "fig6Brightness")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
drawLines(ax[0], dfT, "T", "bhRatio")
refLine(ax[0])
ax[0].set_yscale("log")
ax[0].set_xlabel(tLabel)
ax[0].set_ylabel(r"$\xi_{\rm B}/\xi_{\rm B,TMSV}$ (Bhattacharyya, $s$=1/2)")
ax[0].set_title("(a)Equal $\\lambda$($N_{S0}$=0.05)\n"+subtitle)
ax[0].legend()
drawLines(ax[1], dfT, "T", "bhRatioEqualNs")
refLine(ax[1])
ax[1].set_ylim(0, 1.05)
ax[1].set_xlabel(tLabel)
ax[1].set_ylabel(r"$\xi_{\rm B}/\xi_{\rm B,TMSV}$ at the same $N_S$")
ax[1].set_title("(b)Equal $N_S$")
saveFigure(fig, "fig7Bhattacharyya")

fig, ax=plt.subplots(1, 3, figsize=(15, 4.2))
drawLines(ax[0], dfT, "T", "entropyRatio")
refLine(ax[0])
ax[0].set_yscale("log")
ax[0].set_title("(a)entanglement entropy/TMSV, Equal $\\lambda$")
ax[0].set_ylabel("$E/E_{\\rm TMSV}$")
ax[0].legend()
drawLines(ax[1], dfT, "T", "entropyRatioEqualNs")
refLine(ax[1])
ax[1].set_title("(b)entanglement entropy/TMSV,Equal $N_S$")
ax[1].set_ylabel("$E/E_{\\rm TMSV}$ at same $N_S$")
drawLines(ax[2], dfT, "T", "l1Ratio")
refLine(ax[2])
ax[2].set_yscale("log")
ax[2].set_title("(c) $\\ell_1$ coherence/TMSV,Equal $\\lambda$")
ax[2].set_ylabel("$C_{\\ell_1}/C_{\\ell_1,{\\rm TMSV}}$")
for a in ax:
    a.set_xlabel(tLabel)
saveFigure(fig, "fig8EntanglementCoherence")

fig, ax=plt.subplots(1, 2, figsize=(11, 4.2))
for a, y, title in [(ax[0], "xiRatio", "(a)Equal $\\lambda$"), (ax[1], "xiRatioEqualNs", "(b)Equal $N_S$")]:
    for f in ("nqSingle", "nqBiside"):
        s=dfTrunc[dfTrunc.family == f].sort_values("n")
        color, marker, ls = styles[f]
        a.plot(s.n, s[y], color=color, marker=marker, ls=ls, label=labels[f])
    for f in ("pcSingle", "pcBiside"):
        v=dfTrunc[dfTrunc.family == f][y].iloc[0]
        a.axhline(v, color=styles[f][0], ls=":", label=labels[f] +"(no truncation)")
    a.axhline(1, color="k", lw=1.2)
    a.set_xlabel("nQSBP truncation order $n$")
    a.set_title(title + f": $T$=0.2, $g$={gain}, $N_{{S0}}$=0.05\n" + subtitle)
ax[0].set_ylabel(r"$\xi/\xi_{\rm TMSV}$")
ax[1].set_ylabel(r"$\xi/\xi_{\rm TMSV}$ at the same $N_S$")
ax[0].legend(fontsize=7.5)
saveFigure(fig, "fig9TruncationOrder")

fig, ax=plt.subplots(1, 3, figsize=(15, 4.2))
for f in ("nqSingle", "nqBiside"):
    s = dfGain[dfGain.family == f].sort_values("g")
    color, marker, ls = styles[f]
    for a, y in zip(ax, ("xiRatio", "xiRatioEqualNs", "pSucc")):
        a.plot(s.g, s[y], color=color, marker=marker, ls=ls, label=labels[f])
ax[0].set_ylabel(r"$\xi/\xi_{\rm TMSV}$")
ax[0].set_title("(a)Equal $\\lambda$")
ax[0].set_yscale("log")
refLine(ax[0])
ax[1].set_ylabel(r"$\xi/\xi_{\rm TMSV}$ at same $N_S$")
ax[1].set_title("(b)Equal $N_S$")
ax[1].set_ylim(0, 1.05)
refLine(ax[1])
ax[2].set_ylabel("$P_{\\rm succ}$")
ax[2].set_yscale("log")
ax[2].set_title("(c)heralding probability")
for a in ax:
    a.set_xlabel("nQSBP gain $g$")
ax[0].legend()
fig.suptitle(f"$T$=0.2, $n$={truncOrder}, $N_{{S0}}$=0.05, {subtitle}", y=1.0, fontsize=9)
saveFigure(fig, "fig10GainSweep")

fig, ax=plt.subplots(1, 2, figsize=(11, 4.2))
drawLines(ax[0], dfT, "T", "helAdv")
refLine(ax[0], 0)
ax[0].set_title("(a)Equal $\\lambda$ ($N_{S0}$=0.05)")
ax[0].set_ylabel("single copy Helstrom advantage $P_e^{\\rm TMSV}-P_e$ ($>0$ better)")
ax[0].legend()
drawLines(ax[1], dfT, "T", "helAdvEqualNs")
refLine(ax[1], 0)
ax[1].set_title("(b)Equal $N_S$")
ax[1].set_ylabel("$P_e^{\\rm TMSV}-P_e$ at same $N_S$")
for a in ax:
    a.set_xlabel(tLabel)
saveFigure(fig, "fig11HelstromSingleCopy")

try:
    import shutil
    from google.colab import files
    shutil.make_archive("qiFigures", "zip", outDir)
    files.download("qiFigures.zip")
except ImportError:
    pass
