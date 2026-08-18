"""
EXPERIMENT_ID : 96a_tau_delay
VERSION       : 1.0
ФАЗА          : 5 — исследование аттрактора

Проверка гипотезы о задержке τ в embedding LWR.
Вместо [att[t], att[t-1], ..., att[t-p+1]] (τ=1)
используем [att[t], att[t-τ], ..., att[t-(p-1)τ]] (τ>1).

acc_τ[t] = att[t] − 2·att[t−τ] + att[t−2τ]  (τ-задержанное ускорение).
Cascade expansion = (p_lvl − p_next) × τ баров.
Физический диапазон p фиксирован: 2..80 баров → p_max(τ) = 80÷τ.

Запуск:
  python 96a_tau_delay.py --mode test
  python 96a_tau_delay.py --mode full
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import KDTree

EXPERIMENT_ID = "96a_tau_delay"
VERSION       = "1.0"

_HERE    = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR",    str(_HERE.parent.parent.parent / "data" / "candles")))
OUT_DIR  = Path(os.environ.get("RESULTS_DIR", str(_HERE / "results")))
FIG_DIR  = Path(os.environ.get("FIGURES_DIR", str(_HERE / "figures")))

# Источник origins (95a)
PATH_95A = Path(os.environ.get(
    "PATH_95A",
    str(_HERE.parent / "95_oracle_distribution" / "results" / "origins_SBER_full.jsonl"),
))

INTERVAL         = "1d"
P_MAX            = 300
LAMBDA           = 0.01
TAU_VALS         = [1, 2, 3]
PHYS_MAX_BARS    = 80     # фиксированный физический диапазон
LP_M, LP_D, LP_K, LP_N = 9, 3, 30, 3

MODES: dict[str, dict] = {
    "test": dict(n_origins=5,   stride=100, desc="ТЕСТ (5 origins)"),
    "full": dict(n_origins=200, stride=2,   desc="ПОЛНЫЙ (200 origins)"),
}


# ══════════════════════════════════════════════════════════════════════════════
# LP-пайплайн
# ══════════════════════════════════════════════════════════════════════════════

def _logtrend(close: np.ndarray) -> np.ndarray:
    n=len(close); lc=np.log(np.maximum(close,1e-10)); t=np.arange(n,dtype=float)
    cn=np.arange(1,n+1,dtype=float); ct=np.cumsum(t); ct2=np.cumsum(t**2)
    cy=np.cumsum(lc); cty=np.cumsum(t*lc); denom=cn*ct2-ct**2
    with np.errstate(invalid='ignore',divide='ignore'):
        b=np.where(denom>0,(cn*cty-ct*cy)/denom,0.)
    a=(cy-b*ct)/cn; tr=np.exp(a+b*t); tr[:2]=close[:2]; return tr


def _lp_proj(ratio: np.ndarray, m:int, d:int, k:int, n_iter:int) -> np.ndarray:
    s=ratio.copy().astype(float); N=len(s)
    k_eff=min(k,N-m); d_eff=min(d,m-1)
    for _ in range(n_iter):
        n_pts=N-m+1
        if n_pts<k_eff+1: break
        rows=np.arange(n_pts)[:,None]+np.arange(m)[None,:]
        X=s[rows]; tree=KDTree(X); _,inds=tree.query(X,k=k_eff+1)
        Xp=np.empty_like(X)
        for i in range(n_pts):
            nn=inds[i,1:]; Xnn=X[nn]; cen=Xnn.mean(0)
            _,_,Vt=np.linalg.svd(Xnn-cen,full_matrices=False)
            Vd=Vt[:d_eff].T; xc=X[i]-cen; Xp[i]=cen+Vd@(Vd.T@xc)
        res=np.zeros(N); cnt=np.zeros(N,int)
        for i in range(n_pts): res[i:i+m]+=Xp[i]; cnt[i:i+m]+=1
        s=res/np.maximum(cnt,1)
    return np.diff(s)


def load_att(ticker: str) -> np.ndarray:
    raw   = json.loads((DATA_DIR / ticker / f"{INTERVAL}.json").read_text())
    close = np.array([c["close"] for c in raw], dtype=float)
    lt    = _logtrend(close)
    ratio = close / np.maximum(lt, 1e-10)
    return _lp_proj(ratio, LP_M, LP_D, LP_K, LP_N)


# ══════════════════════════════════════════════════════════════════════════════
# τ-задержанное ускорение
# ══════════════════════════════════════════════════════════════════════════════

def make_acc_tau(att: np.ndarray, tau: int) -> np.ndarray:
    """acc_τ[t] = att[t] − 2·att[t−τ] + att[t−2τ]."""
    n   = len(att)
    acc = np.zeros(n)
    if tau == 1:
        acc[2:] = att[2:] - 2*att[1:-1] + att[:-2]
    else:
        start = 2 * tau
        acc[start:] = att[start:] - 2*att[tau:n-tau] + att[:n-2*tau]
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# Контекст origin
# ══════════════════════════════════════════════════════════════════════════════

class _Ctx:
    __slots__ = ("X_full", "X_acc", "y_base", "vec_full", "vec_acc", "n_lib", "ok")

    def __init__(self, att: np.ndarray, acc_tau: np.ndarray) -> None:
        self.ok = False
        n = len(att)
        if n - P_MAX - 1 < 3:
            return
        att_wins    = sliding_window_view(att[:-1], P_MAX)
        acc_wins    = sliding_window_view(acc_tau[:-1], P_MAX)
        self.X_full = np.asarray(att_wins[:n - P_MAX - 1])
        self.X_acc  = np.asarray(acc_wins[:n - P_MAX - 1])
        self.y_base = att[P_MAX + 1:n].copy()
        self.n_lib  = len(self.y_base)
        self.vec_full = att[n - P_MAX:n].copy()
        self.vec_acc  = acc_tau[n - P_MAX:n].copy()
        self.ok = True


# ══════════════════════════════════════════════════════════════════════════════
# τ-delayed вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def _tau_cols(p: int, tau: int) -> np.ndarray:
    """Индексы столбцов X_full для τ-delayed embedding размерности p.
    X_full хранит (oldest..newest), X_full[:,-1]=att[t], X_full[:,-k-1]=att[t-k].
    τ-delayed: att[t], att[t-τ], ..., att[t-(p-1)τ] → col = P_MAX-1-k*τ.
    """
    return P_MAX - 1 - np.arange(p) * tau


def _p_lvl_max(tau: int) -> int:
    """Максимальный p_lvl при данном τ, чтобы (p_lvl-1)*τ ≤ P_MAX-1."""
    return (P_MAX - 1) // tau + 1


def levels_aligned_tau(p_fit: int, tau: int) -> list[int]:
    """p-aligned каскад с учётом ограничения P_MAX для τ."""
    cap  = _p_lvl_max(tau)
    levs = [p_fit]
    p    = p_fit
    while p * 2 <= cap:
        p *= 2; levs.append(p)
    return list(reversed(levs))


# ══════════════════════════════════════════════════════════════════════════════
# LWR с τ
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_dist(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(A, axis=1); nb = float(np.linalg.norm(b))
    if nb < 1e-12: return np.ones(len(A))
    cos = np.where(nA > 1e-12, (A @ b) / (nA * nb), 0.)
    return 1. - np.clip(cos, -1., 1.)


def _lwr_predict(X_nn, y_nn, q):
    d=np.linalg.norm(X_nn-q,axis=1); h=max(float(d.max()),1e-10)
    w=np.exp(-0.5*(d/h)**2); A=np.hstack([np.ones((len(X_nn),1)),X_nn])
    sw=np.sqrt(np.maximum(w,1e-30)); c,*_=np.linalg.lstsq(sw[:,None]*A,sw*y_nn,rcond=None)
    return float(c[0]+q@c[1:])


def _lwr_loo(X_nn, y_nn, q):
    d=np.linalg.norm(X_nn-q,axis=1); h=max(float(d.max()),1e-10)
    w=np.maximum(np.exp(-0.5*(d/h)**2),1e-30)
    A=np.hstack([np.ones((len(X_nn),1)),X_nn]); sw=np.sqrt(w)
    c,*_=np.linalg.lstsq(sw[:,None]*A,sw*y_nn,rcond=None)
    resid=y_nn-A@c
    try:
        Aw=sw[:,None]*A; AtWA_inv=np.linalg.inv(Aw.T@Aw+1e-10*np.eye(A.shape[1]))
        h_diag=w*np.einsum("ij,ij->i",A@AtWA_inv,A)
        denom=1.-h_diag
        e_loo=np.where(np.abs(denom)>1e-6,resid/denom,resid)
    except np.linalg.LinAlgError:
        e_loo=resid
    return float(np.mean(np.abs(e_loo)))


def forecast_p_tau(ctx: _Ctx, p_reg: int, tau: int) -> tuple[float, float]:
    """Прогноз с τ-delayed embedding и τ-масштабированным каскадом."""
    xi     = 3 * (p_reg + 1) + 5
    levels = levels_aligned_tau(p_reg, tau)
    cands  = np.arange(ctx.n_lib)

    for k, p_lvl in enumerate(levels):
        if k == len(levels) - 1:
            break
        xi_lvl = min(xi, len(cands))
        if len(cands) > xi_lvl:
            cols = _tau_cols(p_lvl, tau)
            d = np.linalg.norm(
                ctx.X_full[cands][:, cols] - ctx.vec_full[cols], axis=1
            )
            cands = cands[np.argpartition(d, xi_lvl - 1)[:xi_lvl]]
        # Расширение в физических барах
        radius = (p_lvl - levels[k + 1]) * tau
        if radius > 0:
            exp   = cands[:, None] - np.arange(radius + 1)[None, :]
            cands = np.unique(np.clip(exp, 0, ctx.n_lib - 1))

    cols_p = _tau_cols(p_reg, tau)
    if len(cands) > xi:
        d_pos = np.linalg.norm(
            ctx.X_full[cands][:, cols_p] - ctx.vec_full[cols_p], axis=1
        )
        d_acc = _cosine_dist(ctx.X_acc[cands][:, cols_p], ctx.vec_acc[cols_p])
        idx   = np.argpartition(d_pos + LAMBDA * d_acc, xi - 1)[:xi]
        cands = cands[idx]

    if len(cands) < p_reg + 2:
        return np.nan, np.nan

    X_nn = ctx.X_full[cands][:, cols_p]
    y_nn = ctx.y_base[cands]
    q    = ctx.vec_full[cols_p]
    return _lwr_predict(X_nn, y_nn, q), _lwr_loo(X_nn, y_nn, q)


# ══════════════════════════════════════════════════════════════════════════════
# Origins из 95a
# ══════════════════════════════════════════════════════════════════════════════

def load_origins_95a(path: Path, n_origins: int, stride: int) -> list[int]:
    t_origs = []
    for line in path.read_text().splitlines():
        if line.strip():
            t_origs.append(json.loads(line)["t_orig"])
    return sorted(t_origs)[::stride][:n_origins]


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward для одного τ
# ══════════════════════════════════════════════════════════════════════════════

def run_tau(
    ticker: str,
    att_full: np.ndarray,
    acc_tau: np.ndarray,
    tau: int,
    t_origins: list[int],
    p_grid: list[int],
    std_att: float,
    out_file: Path,
) -> None:
    n_total = len(att_full)

    for idx_o, t_orig in enumerate(t_origins):
        if t_orig + 1 >= n_total:
            continue
        t0 = time.time()

        hist     = att_full[:t_orig + 1]
        hist_acc = acc_tau[:t_orig + 1]
        true     = float(att_full[t_orig + 1])

        ctx = _Ctx(hist, hist_acc)
        if not ctx.ok:
            continue

        per_p: dict[str, dict] = {}
        for p in p_grid:
            pred, loo = forecast_p_tau(ctx, p, tau)
            per_p[str(p)] = {
                "pred":   float(pred) if not np.isnan(pred) else None,
                "loo":    float(loo)  if not np.isnan(loo)  else None,
                "xi":     3 * (p + 1) + 5,
                "levels": levels_aligned_tau(p, tau),
            }

        record = {
            "exp_id":   EXPERIMENT_ID,
            "version":  VERSION,
            "ticker":   ticker,
            "tau":      tau,
            "t_orig":   int(t_orig),
            "true_att": true,
            "std_att":  float(std_att),
            "per_p":    per_p,
        }
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        print(
            f"    τ={tau}  origin {idx_o+1}/{len(t_origins)}"
            f"  t={t_orig}  true={true:+.5f}  [{time.time()-t0:.2f}с]",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Анализ и фигуры
# ══════════════════════════════════════════════════════════════════════════════

def analyse(results_by_tau: dict[int, list[dict]], mode: str) -> dict:
    summary: dict = {"tau": {}}
    colors = {1: "#42a5f5", 2: "#66bb6a", 3: "#ffa726"}

    fig_rmae_p,   ax_rp   = plt.subplots(figsize=(14, 5))
    fig_rmae_bar, ax_rb   = plt.subplots(figsize=(14, 5))
    fig_oracle,   ax_or   = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
    fig_bars,     ax_ob   = plt.subplots(1, 3, figsize=(18, 4), sharey=False)

    for tau_idx, (tau, rows) in enumerate(sorted(results_by_tau.items())):
        if not rows:
            continue
        std_mean = float(np.mean([r["std_att"] for r in rows]))

        # Собираем данные
        p_all      = sorted({int(p) for r in rows for p in r["per_p"]})
        errs_by_p: dict[int, list[float]] = {p: [] for p in p_all}
        loo_by_p:  dict[int, list[float]] = {p: [] for p in p_all}
        oracle_ps, oracle_bars_list = [], []
        ens_errs   = []

        for r in rows:
            true = r["true_att"]
            preds = {int(p): r["per_p"][p]["pred"] for p in r["per_p"]
                     if r["per_p"][p].get("pred") is not None}
            loos  = {int(p): r["per_p"][p]["loo"]  for p in r["per_p"]
                     if r["per_p"][p].get("loo")  is not None}
            if not preds:
                continue
            for p, v in preds.items():
                errs_by_p[p].append(v - true)
            for p, v in loos.items():
                loo_by_p[p].append(v)

            best_p = min(preds, key=lambda p: abs(preds[p] - true))
            oracle_ps.append(best_p)
            oracle_bars_list.append(best_p * tau)

            valid_loo = {p: loos[p] for p in loos if p in preds and loos[p] > 0}
            if valid_loo:
                inv = np.array([1.0/loos[p] for p in valid_loo])
                w   = inv/inv.sum()
                ens = float(w @ np.array([preds[p] for p in valid_loo]))
                ens_errs.append(ens - true)

        rmae_by_p = {p: float(np.mean(np.abs(errs_by_p[p])))/std_mean
                     for p in p_all if errs_by_p[p]}

        p_base  = min(p_all, key=lambda x: abs(x - 16))
        base    = rmae_by_p.get(p_base, np.nan)
        oracle  = float(np.mean([abs(errs_by_p[op][i])
                                 for i, op in enumerate(oracle_ps)
                                 if i < len(errs_by_p.get(op,[]))]))/std_mean \
                  if oracle_ps else np.nan
        ens_r   = float(np.mean(np.abs(ens_errs)))/std_mean if ens_errs else np.nan

        summary["tau"][tau] = {
            "base_rmae":    base,
            "oracle_rmae":  oracle,
            "ensemble_rmae": ens_r,
            "best_p_rmae":  float(min(rmae_by_p.values())) if rmae_by_p else np.nan,
            "best_p":       min(rmae_by_p, key=rmae_by_p.get) if rmae_by_p else None,
            "oracle_p_mean": float(np.mean(oracle_ps)) if oracle_ps else np.nan,
            "oracle_bars_mean": float(np.mean(oracle_bars_list)) if oracle_bars_list else np.nan,
        }

        col = colors[tau]
        # rMAE(p) кривая
        ps   = sorted(rmae_by_p); vals = [rmae_by_p[p] for p in ps]
        ax_rp.plot(ps, vals, color=col, lw=1.5, marker="o", ms=3, label=f"τ={tau}")
        # rMAE(bars) кривая
        bars = [p*tau for p in ps]
        ax_rb.plot(bars, vals, color=col, lw=1.5, marker="o", ms=3, label=f"τ={tau}")
        # Гистограмма oracle_p
        ax = ax_or[tau_idx]
        ax.hist(oracle_ps, bins=max(p_all)-min(p_all)+1, color=col, alpha=0.8)
        ax.axvline(np.mean(oracle_ps), color="white", ls="--", lw=1.5,
                   label=f"mean={np.mean(oracle_ps):.1f}")
        ax.set_title(f"τ={tau}  oracle_p  ({len(oracle_ps)} origins)")
        ax.set_xlabel("oracle_p"); ax.legend(fontsize=9)
        # Гистограмма oracle_bars
        ax2 = ax_ob[tau_idx]
        ax2.hist(oracle_bars_list, bins=20, color=col, alpha=0.8)
        ax2.axvline(np.mean(oracle_bars_list), color="white", ls="--", lw=1.5,
                    label=f"mean={np.mean(oracle_bars_list):.1f}")
        ax2.set_title(f"τ={tau}  oracle_bars = oracle_p×τ")
        ax2.set_xlabel("физ. баров"); ax2.legend(fontsize=9)

        print(f"\n  ── τ={tau} ──")
        print(f"    base (p≈16)   rMAE = {base:.4f}")
        print(f"    oracle        rMAE = {oracle:.4f}  ({(oracle/base-1)*100:+.1f}%)")
        print(f"    ensemble      rMAE = {ens_r:.4f}   ({(ens_r/base-1)*100:+.1f}%)")
        print(f"    best avg p    = {summary['tau'][tau]['best_p']}  rMAE={summary['tau'][tau]['best_p_rmae']:.4f}")
        print(f"    oracle_p mean = {summary['tau'][tau]['oracle_p_mean']:.1f}  "
              f"oracle_bars mean = {summary['tau'][tau]['oracle_bars_mean']:.1f}")

    for ax in [ax_rp, ax_rb]:
        ax.legend(); ax.set_ylabel("rMAE"); ax.grid(alpha=0.2)
    ax_rp.set_xlabel("p_reg"); ax_rp.set_title(f"96a  rMAE(p) по τ  (режим={mode})")
    ax_rb.set_xlabel("физ. баров (p×τ)"); ax_rb.set_title(f"96a  rMAE(физ. баров) — общая ось")

    for fig, name in [
        (fig_rmae_p,   f"96a_rmae_by_p_{mode}.png"),
        (fig_rmae_bar, f"96a_rmae_by_bars_{mode}.png"),
        (fig_oracle,   f"96a_oracle_dist_{mode}.png"),
        (fig_bars,     f"96a_oracle_bars_{mode}.png"),
    ]:
        fig.tight_layout()
        out = FIG_DIR / name
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  Рис. → {out.name}")

    out_path = OUT_DIR / f"summary_{mode}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Сводка → {out_path}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID} v{VERSION}")
    parser.add_argument("--mode", choices=["test", "full"], default="full")
    args = parser.parse_args()
    cfg  = MODES[args.mode]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  EXPERIMENT : {EXPERIMENT_ID}")
    print(f"  VERSION    : {VERSION}")
    print(f"  РЕЖИМ      : {cfg['desc']}")
    print(f"  TAU        : {TAU_VALS}")
    print(f"  PHYS_MAX   : {PHYS_MAX_BARS} баров")
    print("=" * 65)

    if not PATH_95A.exists():
        print(f"ОШИБКА: {PATH_95A} не найден. Запустите сначала 95a.")
        raise SystemExit(1)

    t_origins = load_origins_95a(PATH_95A, cfg["n_origins"], cfg["stride"])
    print(f"\nOrigins: {len(t_origins)}  [{min(t_origins)}..{max(t_origins)}]")

    ticker = "SBER"
    print(f"\nЗагрузка att ({ticker})...")
    t0_lp = time.time()
    att_full = load_att(ticker)
    std_att  = float(np.std(att_full))
    print(f"  att: {len(att_full)} баров  std={std_att:.5f}  [{time.time()-t0_lp:.1f}с]")

    t_global = time.time()
    results_by_tau: dict[int, list[dict]] = {}

    for tau in TAU_VALS:
        p_max   = PHYS_MAX_BARS // tau
        p_grid  = list(range(2, p_max + 1))
        print(f"\n{'─'*60}")
        print(f"  τ={tau}  p_grid=[2..{p_max}]  ({len(p_grid)} значений)")
        print(f"  p_lvl_max={_p_lvl_max(tau)}  каскад(p=9): {levels_aligned_tau(9, tau)}")

        acc_tau  = make_acc_tau(att_full, tau)
        out_file = OUT_DIR / f"origins_tau{tau}_{args.mode}.jsonl"
        if out_file.exists():
            out_file.unlink()

        run_tau(ticker, att_full, acc_tau, tau, t_origins, p_grid, std_att, out_file)

        rows = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
        results_by_tau[tau] = rows

    print(f"\n{'='*65}")
    print("  АНАЛИЗ РЕЗУЛЬТАТОВ")
    print(f"{'='*65}")
    analyse(results_by_tau, args.mode)

    print(f"\n{'='*65}")
    print(f"  Готово за {(time.time()-t_global)/60:.1f} мин")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
