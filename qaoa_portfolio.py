"""
Quantum Portfolio Optimization for Renewable-Energy Stocks using QAOA
=====================================================================
Pipeline
  1. Market data      : 12 stocks (solar / wind / hydrogen). Simulated with a sector-factor
                        model by default (offline, reproducible). Swap in real data via load_real_data().
  2. QUBO model       : binary x_i (1 = hold asset i)
        minimize  C(x) = -mu.x                          (maximize return)
                         + q * x' Sigma x               (minimize risk)
                         + A * (cost.x - B)^2           (budget limit)
                         + Kpen * (sum(x) - K)^2        (number of holdings)
                         + D * sum_s (n_s - K/3)^2      (sector diversification)
  3. Ising mapping    : x_i = (1 - Z_i)/2  ->  cost Hamiltonian H_C (diagonal)
  4. QAOA             : |+>^n -> [exp(-i g H_C) exp(-i b sum X)]^p, classical optimizer on (g, b)
  5. Benchmark        : brute force (exact) + greedy classical heuristic
  6. Plots            : saved to ./qaoa_results/

Run:  python qaoa_portfolio.py
"""
import itertools, os, time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7
rng = np.random.default_rng(SEED)
OUT = "qaoa_results"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. MARKET DATA
# ----------------------------------------------------------------------------
TICKERS = ["FSLR", "ENPH", "RUN", "SEDG",      # solar
           "VWS", "ORSTED", "TPIC", "NEE",      # wind
           "PLUG", "BE", "BLDP", "NEL"]         # hydrogen
SECTORS = np.array([0]*4 + [1]*4 + [2]*4)
SECTOR_NAMES = ["Solar", "Wind", "Hydrogen"]
N = len(TICKERS)


def simulate_market(days=750):
    """Sector-factor model: r_i = beta_i*market + gamma_i*sector + idio. (Illustrative data!)"""
    ann_drift = np.array([0.20, 0.12, 0.10, 0.08,    # solar
                          0.11, 0.09, 0.10, 0.13,    # wind (NEE = stable utility)
                          0.18, 0.22, 0.15, 0.17])   # hydrogen (high return, high risk)
    idio_vol  = np.array([0.30, 0.38, 0.45, 0.42,
                          0.20, 0.22, 0.40, 0.14,
                          0.60, 0.50, 0.55, 0.50])
    beta   = rng.uniform(0.8, 1.4, N)
    gamma  = rng.uniform(0.6, 1.0, N)
    mkt    = rng.normal(0, 0.15/np.sqrt(252), days)
    sec    = rng.normal(0, 0.20/np.sqrt(252), (3, days))
    r = (ann_drift/252)[:, None] + beta[:, None]*mkt + gamma[:, None]*sec[SECTORS] \
        + rng.normal(0, 1, (N, days))*(idio_vol/np.sqrt(252))[:, None]
    return r.T                                       # (days, N)


def load_real_data(period="3y"):
    """Optional: real prices (needs internet + `pip install yfinance`)."""
    import yfinance as yf
    px = yf.download(["FSLR", "ENPH", "RUN", "SEDG", "VWDRY", "DNNGY", "TPIC",
                     "NEE", "PLUG", "BE", "BLDP", "NLLSF"], period=period)["Close"].dropna()
    return px.pct_change().dropna().values


def market_stats(returns, prices):
    mu = returns.mean(0) * 252                       # annualised expected return
    Sigma = np.cov(returns.T) * 252                  # annualised covariance
    return mu, Sigma


# ----------------------------------------------------------------------------
# 2. QUBO FORMULATION
# ----------------------------------------------------------------------------
def build_qubo(mu, Sigma, prices, budget, K, q=2.0, A=0.0005, Kpen=1.0, D=0.4):
    """Return symmetric Q (n x n) and constant c0 so that C(x) = x'Qx + c0 for binary x."""
    n = len(mu)
    Q = np.zeros((n, n))
    c0 = 0.0
    Q += q * Sigma                                   # risk
    Q -= np.diag(mu)                                 # return (x_i^2 = x_i)
    # budget  A (p.x - B)^2
    Q += A * np.outer(prices, prices)
    Q -= 2 * A * budget * np.diag(prices)
    c0 += A * budget**2
    # cardinality  Kpen (1.x - K)^2
    Q += Kpen * np.ones((n, n))
    Q -= 2 * Kpen * K * np.eye(n)
    c0 += Kpen * K**2
    # sector diversification  D * sum_s (n_s - K/3)^2
    for s in range(3):
        m = (SECTORS == s).astype(float)
        Q += D * np.outer(m, m)
        Q -= 2 * D * (K/3) * np.diag(m)
        c0 += D * (K/3)**2
    return Q, c0


def all_bitstrings(n):
    """(2^n, n) array; row index k has bit i = (k >> (n-1-i)) & 1  (qubit 0 = most significant)."""
    k = np.arange(2**n)[:, None]
    return ((k >> np.arange(n-1, -1, -1)) & 1).astype(np.int8)


def cost_vector(Q, c0, X):
    return np.einsum("ki,ij,kj->k", X, Q, X) + c0


def qubo_to_ising(Q, c0):
    """x=(1-Z)/2  ->  H = sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j + const  (for the report / hardware)."""
    n = len(Q)
    h = np.zeros(n); J = np.zeros((n, n)); const = c0
    for i in range(n):
        h[i] += -Q[i, i]/2; const += Q[i, i]/2
        for j in range(i+1, n):
            qij = Q[i, j] + Q[j, i]
            J[i, j] = qij/4
            h[i] += -qij/4; h[j] += -qij/4; const += qij/4
    return h, J, const


# ----------------------------------------------------------------------------
# 3. QAOA STATEVECTOR SIMULATOR  (two variants)
#    (A) Standard QAOA  : X-mixer on full 2^n space, constraints enforced by penalties
#    (B) Constrained    : Dicke start state + ring-XY mixer -> only portfolios with exactly K assets
#                         are ever explored (hard cardinality constraint, much smaller search space)
# ----------------------------------------------------------------------------
class XMixer:
    def __init__(self, n): self.n = n
    def initial(self): return np.full(2**self.n, 1/np.sqrt(2**self.n), dtype=complex)  # H on all qubits
    def apply(self, psi, beta):
        n = self.n
        c, s = np.cos(beta), -1j*np.sin(beta)
        psi = psi.reshape([2]*n)
        for ax in range(n):
            a = np.take(psi, 0, axis=ax); b = np.take(psi, 1, axis=ax)
            psi = np.stack([c*a + s*b, s*a + c*b], axis=ax)
        return psi.reshape(-1)


class XYMixer:
    """H_M = sum_i (X_i X_{i+1} + Y_i Y_{i+1})/2 on a ring; conserves Hamming weight."""
    def __init__(self, Xs):
        m, n = Xs.shape
        pos = {tuple(r): k for k, r in enumerate(Xs)}
        A = np.zeros((m, m))
        for k, r in enumerate(Xs):
            for i in range(n):
                j = (i + 1) % n
                if r[i] != r[j]:
                    t = r.copy(); t[i], t[j] = r[j], r[i]
                    A[k, pos[tuple(t)]] = 1.0
        self.w, self.V = np.linalg.eigh(A)
        self.m = m
    def initial(self): return np.full(self.m, 1/np.sqrt(self.m), dtype=complex)        # Dicke state
    def apply(self, psi, beta): return self.V @ (np.exp(-1j*beta*self.w) * (self.V.T @ psi))


def qaoa_state(params, cost, mixer):
    p = len(params)//2
    gammas, betas = params[:p], params[p:]
    psi = mixer.initial()
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j*g*cost)                    # cost layer  e^{-i g H_C}
        psi = mixer.apply(psi, b)                         # mixer layer
    return psi


def qaoa_energy(params, cost, mixer):
    return float((np.abs(qaoa_state(params, cost, mixer))**2) @ cost)


def run_qaoa(cost, mixer, p, restarts=8, maxiter=250):
    """Multi-start COBYLA on standardised cost (so angles are well scaled)."""
    c_norm = (cost - cost.mean()) / cost.std()
    best = None
    for r in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 0.6, p), rng.uniform(0, 1.2, p)])
        trace = []
        def f(x):
            e = qaoa_energy(x, c_norm, mixer); trace.append(e); return e
        res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.25})
        if best is None or res.fun < best[0]: best = (res.fun, res.x, trace)
    _, x, trace = best
    prob = np.abs(qaoa_state(x, c_norm, mixer))**2
    return x, prob, trace


# ----------------------------------------------------------------------------
# 4. CLASSICAL BASELINES + METRICS
# ----------------------------------------------------------------------------
def greedy(mu, Sigma, prices, budget, K, q=2.0):
    """Classical heuristic: repeatedly add the asset with the best return-minus-risk score."""
    chosen = []
    for _ in range(K):
        best, bi = -1e9, None
        for i in range(N):
            if i in chosen or sum(prices[chosen]) + prices[i] > budget*1.05: continue
            t = chosen + [i]
            x = np.zeros(N); x[t] = 1
            score = mu @ x - q * x @ Sigma @ x
            if score > best: best, bi = score, i
        if bi is None: break
        chosen.append(bi)
    x = np.zeros(N, dtype=int); x[chosen] = 1
    return x


def describe(x, mu, Sigma, prices):
    xs = np.asarray(x, float)
    k = xs.sum()
    if k == 0: return dict(k=0, ret=0, vol=0, sharpe=0, cost=0)
    w = xs / k                                         # equal-weight across held assets
    ret = float(mu @ w); vol = float(np.sqrt(w @ Sigma @ w))
    return dict(k=int(k), ret=ret, vol=vol, sharpe=ret/vol, cost=float(prices @ xs))


def fmt(x):
    return ", ".join(t for t, b in zip(TICKERS, x) if b)


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    returns = simulate_market()
    # returns = load_real_data()             # <- use real data instead
    mu, Sigma = market_stats(returns, None)
    prices = np.array([180, 120, 15, 70, 25, 20, 12, 75, 9, 18, 3, 1.5])
    prices = np.round(prices / prices.mean() * 100, 1)          # normalised unit cost (lot price)
    K = 6                                                       # target number of holdings
    budget = float(np.sort(prices)[::-1][:K].mean() * K * 0.55)
    Q, c0 = build_qubo(mu, Sigma, prices, budget, K, A=40.0/budget**2)

    X = all_bitstrings(N)
    cost = cost_vector(Q, c0, X)
    h, J, const = qubo_to_ising(Q, c0)
    Z = 1 - 2*X                                                  # sanity check of Ising mapping
    ising = const + Z @ h + np.einsum("ki,ij,kj->k", Z, np.triu(J, 1), Z)
    assert np.allclose(ising, cost), "Ising mapping mismatch"

    opt_idx = int(np.argmin(cost)); x_opt = X[opt_idx]
    sub = np.where(X.sum(1) == K)[0]; Xs, cs = X[sub], cost[sub]
    opt_s = int(np.argmin(cs)); assert sub[opt_s] == opt_idx
    print(f"Assets/qubits: {N} | full space {2**N} | constrained space (exactly K={K}) {len(sub)} | budget={budget:.0f}")

    def ratio(prob, c):  # 1 = always optimal, 0 = no better than random guessing
        return (c.mean() - prob @ c) / (c.mean() - c.min())

    mixers = {"standard": (cost, XMixer(N), 2**N), "constrained": (cs, XYMixer(Xs), len(sub))}
    results = {}
    for name, (c, mix, size) in mixers.items():
        for p in (1, 2, 3, 4):
            x, prob, trace = run_qaoa(c, mix, p)
            o = opt_idx if name == "standard" else opt_s
            top5 = np.argsort(prob)[::-1][:5]
            results[(name, p)] = dict(prob=prob, trace=trace, ar=ratio(prob, c), p_opt=float(prob[o]),
                                      uniform=1/size, opt=o, top=int(top5[0]), in_top5=o in top5)
            r = results[(name, p)]
            print(f"{name:12s} p={p}: quality={r['ar']:.3f}  P(opt)={r['p_opt']:.3f} "
                  f"({r['p_opt']/r['uniform']:.0f}x random)  optimum in top-5 samples: {r['in_top5']}")

    p_star = 4
    R = results[("constrained", p_star)]
    x_q = Xs[R["top"]]
    x_g = greedy(mu, Sigma, prices, budget, K)

    print("\n=== PORTFOLIOS (equal-weight across selected assets) ===")
    rows = [("Exact optimum (brute force)", x_opt), (f"QAOA constrained p={p_star}", x_q),
            ("Greedy classical", x_g), ("Equal-weight all 12", np.ones(N, int))]
    print(f"{'Method':30s} {'k':>2s} {'Return':>7s} {'Vol':>7s} {'Sharpe':>7s} {'Cost':>6s} {'QUBO':>8s}  Holdings")
    for name, x in rows:
        d = describe(x, mu, Sigma, prices)
        cval = float(np.asarray(x) @ Q @ np.asarray(x) + c0)
        print(f"{name:30s} {d['k']:2d} {d['ret']:7.1%} {d['vol']:7.1%} {d['sharpe']:7.2f} "
              f"{d['cost']:6.0f} {cval:8.3f}  {fmt(x)}")
    nsec = [int(sum(np.asarray(x_q)[SECTORS == s])) for s in range(3)]
    print(f"\nQAOA sector split (solar/wind/hydrogen): {nsec}   budget used: {prices@x_q:.0f}/{budget:.0f}")
    print(f"Runtime: {time.time()-t0:.1f}s")

    # ------------------------------ PLOTS ------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    prob = R["prob"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    order = np.argsort(prob)[::-1][:12]
    ax[0].bar(range(12), prob[order], color=["#2ca02c" if i == opt_s else "#4c72b0" for i in order])
    ax[0].set_xticks(range(12))
    ax[0].set_xticklabels(["".join(map(str, Xs[i])) for i in order], rotation=70, fontsize=7)
    ax[0].axhline(1/len(sub), color="red", ls="--", lw=1, label="random guess")
    ax[0].set_title(f"QAOA p={p_star}: most likely portfolios (green = exact optimum)")
    ax[0].set_ylabel("Measurement probability"); ax[0].legend()
    for name, mk in [("standard", "o--"), ("constrained", "s-")]:
        ps = [1, 2, 3, 4]
        ax[1].plot(ps, [results[(name, p)]["ar"] for p in ps], mk, label=f"{name} mixer")
    ax[1].set_xticks(ps); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("QAOA depth p")
    ax[1].set_ylabel("Solution quality (0 = random, 1 = optimal)")
    ax[1].set_title("Constraint-preserving mixer wins"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/qaoa_results.png", dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pick = np.random.default_rng(0).choice(sub, 600, replace=False)
    pts = np.array([[describe(X[i], mu, Sigma, prices)["vol"], describe(X[i], mu, Sigma, prices)["ret"]] for i in pick])
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c="lightgrey", label="random 6-asset portfolios")
    for name, x, c, mk in [("Exact optimum", x_opt, "#2ca02c", "*"), ("QAOA", x_q, "#d62728", "o"),
                           ("Greedy", x_g, "#ff7f0e", "s")]:
        d = describe(x, mu, Sigma, prices)
        ax.scatter(d["vol"], d["ret"], s=260 if mk == "*" else 90, c=c, marker=mk, label=name, zorder=5)
    ax.set_xlabel("Annualised risk (volatility)"); ax.set_ylabel("Annualised expected return")
    ax.set_title("Risk / return of chosen portfolios"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/risk_return.png", dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in (1, 2, 3, 4): ax.plot(results[("constrained", p)]["trace"], label=f"p={p}")
    ax.set_xlabel("Optimiser iteration (last restart shown per p)"); ax.set_ylabel("<C> (standardised)")
    ax.set_title("Hybrid quantum-classical loop convergence"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/convergence.png", dpi=160); plt.close()
    print(f"Plots saved in ./{OUT}/")


if __name__ == "__main__":
    main()
