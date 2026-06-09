from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the CETER pilot solver from this package.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from ceter_schur_pilot_core import Grid, Parameters, assemble_saddle_system, solve_full_saddle


def robust_normalize(q: np.ndarray) -> np.ndarray:
    med = np.median(q)
    q75, q25 = np.percentile(q, [75, 25])
    iqr = q75 - q25
    return (q - med) / (iqr + 1.0e-12)


def cubical_persistence_2d_superlevel(q: np.ndarray) -> dict[int, list[tuple[float, float, float, bool]]]:
    """Compute simple 0D and 1D cubical persistence diagrams for a 2D field.

    Superlevel filtration in q is implemented as a lower-star filtration in f=-q.
    Returns dictionary: dimension -> list of (birth_q, death_q, lifetime, essential).
    This compact implementation is intended for reproducible pilot figures.
    """
    ny, nx = q.shape
    f = -q
    cells = []

    # Vertices.
    for i in range(ny):
        for j in range(nx):
            cells.append((("v", i, j), 0, [(i, j)]))

    # Horizontal edges.
    for i in range(ny):
        for j in range(nx - 1):
            cells.append((("eh", i, j), 1, [(i, j), (i, j + 1)]))

    # Vertical edges.
    for i in range(ny - 1):
        for j in range(nx):
            cells.append((("ev", i, j), 1, [(i, j), (i + 1, j)]))

    # Squares.
    for i in range(ny - 1):
        for j in range(nx - 1):
            cells.append((("s", i, j), 2, [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]))

    filt = {}
    dim = {}
    for key, d, verts in cells:
        dim[key] = d
        filt[key] = max(f[i, j] for i, j in verts)

    order_keys = sorted([c[0] for c in cells], key=lambda k: (filt[k], dim[k], str(k)))
    index = {k: idx for idx, k in enumerate(order_keys)}

    def boundary_keys(k):
        typ = k[0]
        if typ == "v":
            return []
        if typ == "eh":
            _, i, j = k
            return [("v", i, j), ("v", i, j + 1)]
        if typ == "ev":
            _, i, j = k
            return [("v", i, j), ("v", i + 1, j)]
        if typ == "s":
            _, i, j = k
            return [("eh", i, j), ("eh", i + 1, j), ("ev", i, j), ("ev", i, j + 1)]
        raise ValueError(k)

    columns = [set(index[b] for b in boundary_keys(k)) for k in order_keys]
    low_to_col = {}
    birth_indices = set()
    paired_births = set()
    reduced_cols = []
    pairs = {0: [], 1: []}

    for j, col0 in enumerate(columns):
        col = set(col0)
        while col:
            low = max(col)
            if low in low_to_col:
                col ^= reduced_cols[low_to_col[low]]
            else:
                break
        reduced_cols.append(col)
        if not col:
            birth_indices.add(j)
        else:
            low = max(col)
            low_to_col[low] = j
            paired_births.add(low)
            birth_key = order_keys[low]
            death_key = order_keys[j]
            d = dim[birth_key]
            if d in (0, 1):
                birth_q = -filt[birth_key]
                death_q = -filt[death_key]
                lifetime = birth_q - death_q
                if lifetime > 1.0e-12:
                    pairs[d].append((birth_q, death_q, lifetime, False))

    min_q = float(np.min(q))
    for j in sorted(birth_indices - paired_births):
        key = order_keys[j]
        d = dim[key]
        if d in (0, 1):
            birth_q = -filt[key]
            pairs[d].append((birth_q, min_q, birth_q - min_q, True))

    return pairs


def solve_field(mu: float, grid: Grid, prm: Parameters):
    K, rhs, A, B, G, C = assemble_saddle_system(grid, prm, mu)
    z, elapsed = solve_full_saddle(K, rhs)
    ux = z[: grid.n].reshape(grid.ny, grid.nx)
    q = prm.recirculation_threshold - ux
    qhat = robust_normalize(q)
    return ux, qhat, elapsed


def pairs_to_dataframe(pairs: dict, mu: float) -> pd.DataFrame:
    rows = []
    for d, pts in pairs.items():
        for birth, death, lifetime, essential in pts:
            rows.append(
                {
                    "mu": mu,
                    "dimension": d,
                    "birth": birth,
                    "death": death,
                    "lifetime": lifetime,
                    "essential": essential,
                }
            )
    return pd.DataFrame(rows)


def run(output_dir: Path, fig_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid(nx=36, ny=18, lx=4.0, ly=1.0)
    prm = Parameters()
    mu_values = [0.20, 0.30]
    fields = []
    dfs = []

    for mu in mu_values:
        ux, qhat, elapsed = solve_field(mu, grid, prm)
        pairs = cubical_persistence_2d_superlevel(qhat)
        fields.append((mu, ux, qhat, pairs))
        dfs.append(pairs_to_dataframe(pairs, mu))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_dir / "stenosed_channel_persistence_pairs.csv", index=False)
    df[df["lifetime"] >= 0.25].to_csv(
        output_dir / "stenosed_channel_persistence_pairs_lifetime_ge_0p25.csv", index=False
    )

    X, Y = grid.coordinates()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), constrained_layout=True)
    for row, (mu, ux, qhat, pairs) in enumerate(fields):
        ax = axes[row, 0]
        cf = ax.contourf(X, Y, qhat, levels=25)
        ax.contour(X, Y, qhat, levels=[0.0], linewidths=1.0)
        ax.set_title(f"normalised recirculation indicator, $\\mu={mu:.1f}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(cf, ax=ax, shrink=0.86)

        ax = axes[row, 1]
        pts0 = [p for p in pairs.get(0, []) if p[2] >= 0.10]
        pts1 = [p for p in pairs.get(1, []) if p[2] >= 0.10]
        allpts = pts0 + pts1
        vals = [v for p in allpts for v in (p[0], p[1])] if allpts else [float(qhat.min()), float(qhat.max())]
        lo, hi = min(vals), max(vals)
        pad = 0.05 * (hi - lo + 1.0e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.0, label="diagonal")
        if pts0:
            ax.scatter([p[0] for p in pts0], [p[1] for p in pts0], marker="o", label="$H_0$", s=35)
        if pts1:
            ax.scatter([p[0] for p in pts1], [p[1] for p in pts1], marker="s", label="$H_1$", s=35)
        ax.set_title(f"0D/1D persistence diagram, $\\mu={mu:.1f}$")
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_dir / "stenosed_channel_persistence_diagrams.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "stenosed_channel_persistence_diagrams.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for mu, ux, qhat, pairs in fields:
        for d in [0, 1]:
            pts = [p for p in pairs.get(d, []) if p[2] >= 0.25]
            summary_rows.append(
                {
                    "mu": mu,
                    "dimension": d,
                    "features_lifetime_ge_0p25": len(pts),
                    "max_lifetime": max([p[2] for p in pairs.get(d, [])], default=0.0),
                }
            )
    pd.DataFrame(summary_rows).to_csv(output_dir / "persistence_summary.csv", index=False)
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = run(root / "outputs", root / "figures")
    print(df.head(20).to_string(index=False))
