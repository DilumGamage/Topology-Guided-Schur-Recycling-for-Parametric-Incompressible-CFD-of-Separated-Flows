from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import ndimage


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class Grid:
    nx: int = 36
    ny: int = 18
    lx: float = 4.0
    ly: float = 1.0

    @property
    def n(self) -> int:
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        return self.lx / (self.nx + 1)

    @property
    def dy(self) -> float:
        return self.ly / (self.ny + 1)

    def coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.dx, self.lx - self.dx, self.nx)
        y = np.linspace(self.dy, self.ly - self.dy, self.ny)
        return np.meshgrid(x, y)


@dataclass
class Parameters:
    viscosity: float = 2.5e-2
    pressure_stabilisation: float = 5.0e-4
    base_resistance: float = 2.0e-2
    obstacle_resistance: float = 8.0
    convection_x: float = 0.20
    convection_y: float = 0.00
    residual_tol: float = 2.0e-3
    divergence_tol: float = 1.0e-4
    topology_distance_tol: float = 1.10
    topo_cfl_tol: float = 8.0
    recirculation_threshold: float = 2.0
    random_seed: int = 7


@dataclass
class TopologySignature:
    components: int
    holes: int
    area_fraction: float

    def distance_to(self, other: "TopologySignature") -> float:
        """Simple topology distance used as an event gate.

        A full paper implementation can replace this with a Wasserstein or
        bottleneck distance between persistence diagrams. This self-contained
        proxy is enough to detect component/void events in thresholded
        recirculation fields.
        """
        return (
            abs(self.components - other.components)
            + abs(self.holes - other.holes)
            + 10.0 * abs(self.area_fraction - other.area_fraction)
        )


# -----------------------------------------------------------------------------
# Sparse finite-difference operators
# -----------------------------------------------------------------------------

def first_derivative_matrix(n: int, h: float) -> sparse.csr_matrix:
    """Central first derivative on interior nodes with simple boundary closure."""
    e = np.ones(n)
    D = sparse.diags([-0.5 * e, 0.5 * e], offsets=[-1, 1], shape=(n, n), format="lil") / h
    # One-sided closure near the artificial boundary.
    if n >= 2:
        D[0, 0] = -1.0 / h
        D[0, 1] = 1.0 / h
        D[-1, -2] = -1.0 / h
        D[-1, -1] = 1.0 / h
    return D.tocsr()


def laplacian_1d(n: int, h: float) -> sparse.csr_matrix:
    """Positive -d2/dx2 operator with homogeneous Dirichlet boundary."""
    e = np.ones(n)
    return sparse.diags([-e, 2.0 * e, -e], offsets=[-1, 0, 1], shape=(n, n), format="csr") / (h * h)


def build_2d_operators(grid: Grid) -> Dict[str, sparse.csr_matrix]:
    """Return sparse operators for a collocated demonstration grid."""
    Ix = sparse.eye(grid.nx, format="csr")
    Iy = sparse.eye(grid.ny, format="csr")

    Dx1 = first_derivative_matrix(grid.nx, grid.dx)
    Dy1 = first_derivative_matrix(grid.ny, grid.dy)
    Lx = laplacian_1d(grid.nx, grid.dx)
    Ly = laplacian_1d(grid.ny, grid.dy)

    Dx = sparse.kron(Iy, Dx1, format="csr")
    Dy = sparse.kron(Dy1, Ix, format="csr")
    Lap = sparse.kron(Iy, Lx, format="csr") + sparse.kron(Ly, Ix, format="csr")
    return {"Dx": Dx, "Dy": Dy, "Lap": Lap}


# -----------------------------------------------------------------------------
# Parametric CFD model
# -----------------------------------------------------------------------------

def brinkman_resistance(grid: Grid, prm: Parameters, mu: float) -> np.ndarray:
    """Local parameter-dependent resistance field beta(x,y; mu).

    The shape mimics a smooth internal stenosis/obstruction. Increasing mu
    strengthens and slightly widens the local resistance patch.
    """
    X, Y = grid.coordinates()
    xc = 1.75 + 0.25 * mu
    yc = 0.50
    sx = 0.18 + 0.06 * mu
    sy = 0.13 + 0.04 * mu
    obstacle = np.exp(-((X - xc) ** 2 / sx**2 + (Y - yc) ** 2 / sy**2))

    # Wall-proximity resistance makes the synthetic channel more physical.
    wall = 0.04 * (np.exp(-(Y / 0.08) ** 2) + np.exp(-((grid.ly - Y) / 0.08) ** 2))
    beta = prm.base_resistance + prm.obstacle_resistance * (mu**2) * obstacle + wall
    return beta.ravel()


def body_force(grid: Grid, mu: float) -> np.ndarray:
    """Streamwise drive plus a local adverse forcing behind the obstruction.

    The adverse forcing is deliberately used to create a detectable topology
    change in the recirculation indicator. It is a controlled demonstration,
    not experimental physics.
    """
    X, Y = grid.coordinates()
    n = grid.n

    drive = 1.0 + 0.10 * np.sin(np.pi * Y / grid.ly)

    # A single bubble at small mu gradually splits into two separated lobes.
    bubble_single = np.exp(-((X - 2.15) ** 2 / 0.20**2 + (Y - 0.50) ** 2 / 0.13**2))
    bubble_upper = np.exp(-((X - 2.35) ** 2 / 0.22**2 + (Y - 0.66) ** 2 / 0.10**2))
    bubble_lower = np.exp(-((X - 2.38) ** 2 / 0.22**2 + (Y - 0.34) ** 2 / 0.10**2))
    split_weight = 1.0 / (1.0 + np.exp(-18.0 * (mu - 0.55)))
    adverse = (1.0 - split_weight) * bubble_single + split_weight * (bubble_upper + bubble_lower)

    fu = drive - (1.10 + 2.50 * mu) * adverse

    # Small transverse swirl-like forcing to avoid overly one-dimensional fields.
    fv = 0.10 * mu * (Y - 0.50) * np.exp(-((X - 2.2) ** 2 / 0.35**2 + (Y - 0.5) ** 2 / 0.22**2))
    return np.r_[fu.ravel(), fv.ravel()]


def build_momentum_block(grid: Grid, prm: Parameters, beta: np.ndarray) -> sparse.csr_matrix:
    ops = build_2d_operators(grid)
    N = grid.n
    I = sparse.eye(N, format="csr")
    A_scalar = (
        prm.viscosity * ops["Lap"]
        + prm.convection_x * ops["Dx"]
        + prm.convection_y * ops["Dy"]
        + sparse.diags(beta, format="csr")
    )
    A = sparse.block_diag((A_scalar, A_scalar), format="csr")
    return A


def build_divergence_and_gradient(grid: Grid) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    ops = build_2d_operators(grid)
    B = sparse.hstack((ops["Dx"], ops["Dy"]), format="csr")
    G = B.T.tocsr()
    return B, G


def assemble_saddle_system(
    grid: Grid,
    prm: Parameters,
    mu: float,
) -> Tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    beta = brinkman_resistance(grid, prm, mu)
    A = build_momentum_block(grid, prm, beta)
    B, G = build_divergence_and_gradient(grid)
    C = prm.pressure_stabilisation * sparse.eye(grid.n, format="csr")
    f = body_force(grid, mu)
    g = np.zeros(grid.n)
    K = sparse.bmat([[A, G], [B, -C]], format="csr")
    rhs = np.r_[f, g]
    return K, rhs, A, B, G, C


# -----------------------------------------------------------------------------
# Full solve and Woodbury-recycled Schur solve
# -----------------------------------------------------------------------------

def solve_full_saddle(K: sparse.csr_matrix, rhs: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    z = spla.spsolve(K.tocsc(), rhs)
    elapsed = time.perf_counter() - t0
    return z, elapsed


class WoodburyMomentumInverse:
    """Apply (A0 + U diag(delta) U^T)^(-1) by Woodbury."""

    def __init__(self, A0: sparse.csr_matrix, update_indices: np.ndarray, deltas: np.ndarray):
        self.A0 = A0.tocsc()
        self.lu0 = spla.splu(self.A0)
        keep = np.where(np.abs(deltas) > 1.0e-13)[0]
        self.update_indices = update_indices[keep].astype(int)
        self.deltas = deltas[keep].astype(float)
        self.rank = len(self.deltas)

        if self.rank > 0:
            n = A0.shape[0]
            U = sparse.csc_matrix(
                (np.ones(self.rank), (self.update_indices, np.arange(self.rank))),
                shape=(n, self.rank),
            )
            self.U = U
            self.A0_inv_U = self.lu0.solve(U.toarray())
            selected = self.A0_inv_U[self.update_indices, :]
            self.small_matrix = np.diag(1.0 / self.deltas) + selected
        else:
            self.U = None
            self.A0_inv_U = None
            self.small_matrix = None

    def apply(self, y: np.ndarray) -> np.ndarray:
        was_vector = y.ndim == 1
        Y = y.reshape(-1, 1) if was_vector else y
        X0 = self.lu0.solve(Y)
        if self.rank == 0:
            return X0.ravel() if was_vector else X0
        selected_X0 = X0[self.update_indices, :]
        coeff = np.linalg.solve(self.small_matrix, selected_X0)
        X = X0 - self.A0_inv_U @ coeff
        return X.ravel() if was_vector else X


def diagonal_update_from_beta(
    beta0: np.ndarray,
    beta: np.ndarray,
    max_rank: int = 90,
    relative_drop_tol: float = 1.0e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    db = beta - beta0
    N = len(beta0)
    idx_u = np.arange(N)
    idx_v = np.arange(N, 2 * N)
    indices = np.r_[idx_u, idx_v]
    deltas = np.r_[db, db]

    absd = np.abs(deltas)
    if absd.max(initial=0.0) == 0.0:
        return np.array([], dtype=int), np.array([], dtype=float)
    mask = absd >= relative_drop_tol * absd.max()
    candidate_indices = indices[mask]
    candidate_deltas = deltas[mask]

    if len(candidate_deltas) > max_rank:
        order = np.argsort(np.abs(candidate_deltas))[::-1][:max_rank]
        candidate_indices = candidate_indices[order]
        candidate_deltas = candidate_deltas[order]
    return candidate_indices.astype(int), candidate_deltas.astype(float)


def solve_recycled_schur(
    Ainv: WoodburyMomentumInverse,
    B: sparse.csr_matrix,
    G: sparse.csr_matrix,
    C: sparse.csr_matrix,
    f: np.ndarray,
    g: np.ndarray,
    rtol: float = 1.0e-9,
) -> Tuple[np.ndarray, float, int, int]:
    """Pressure Schur solve with a Woodbury momentum inverse.

    For this compact research demo, the pressure Schur matrix is assembled
    explicitly by applying the Woodbury inverse to all pressure-gradient basis
    vectors. That makes the code simple and reproducible. In a large CFD solver,
    replace this dense solve with GMRES/FGMRES using the same matrix-vector action.
    """
    t0 = time.perf_counter()
    Ainv_f = Ainv.apply(f)
    Ainv_G = Ainv.apply(G.toarray())
    H = B @ Ainv_G + C.toarray()
    rhs_p = B @ Ainv_f - g
    p = np.linalg.solve(H, rhs_p)
    u = Ainv.apply(f - G @ p)
    elapsed = time.perf_counter() - t0
    z = np.r_[u, p]
    return z, elapsed, 0, 0


# -----------------------------------------------------------------------------
# Topology gate
# -----------------------------------------------------------------------------

def count_holes(binary: np.ndarray) -> int:
    """Count holes in a binary mask using background components."""
    background = ~binary
    structure = ndimage.generate_binary_structure(2, 1)
    labels, nlabels = ndimage.label(background, structure=structure)
    if nlabels == 0:
        return 0

    # Background components touching the boundary are exterior, not holes.
    boundary_labels = set(np.unique(labels[0, :]))
    boundary_labels.update(np.unique(labels[-1, :]))
    boundary_labels.update(np.unique(labels[:, 0]))
    boundary_labels.update(np.unique(labels[:, -1]))
    hole_labels = [lab for lab in range(1, nlabels + 1) if lab not in boundary_labels]
    return len(hole_labels)


def topology_signature(u_velocity: np.ndarray, grid: Grid, threshold: float) -> TopologySignature:
    ux = u_velocity[: grid.n].reshape(grid.ny, grid.nx)
    mask = ux < threshold
    structure = ndimage.generate_binary_structure(2, 1)
    _, comps = ndimage.label(mask, structure=structure)
    holes = count_holes(mask)
    area_fraction = float(mask.mean())
    return TopologySignature(components=int(comps), holes=int(holes), area_fraction=area_fraction)


def ceter_decision(
    K: sparse.csr_matrix,
    rhs: np.ndarray,
    z: np.ndarray,
    B: sparse.csr_matrix,
    grid: Grid,
    prm: Parameters,
    topo_ref: TopologySignature,
    topo_now: TopologySignature,
    delta_mu: float,
) -> Dict[str, float | bool]:
    rel_res = np.linalg.norm(K @ z - rhs) / max(np.linalg.norm(rhs), 1.0e-14)
    div_norm = np.linalg.norm(B @ z[: 2 * grid.n]) / max(np.linalg.norm(z[: 2 * grid.n]), 1.0e-14)
    topo_dist = topo_now.distance_to(topo_ref)
    topo_cfl = topo_dist / max(abs(delta_mu), 1.0e-14)
    accepted = (
        rel_res <= prm.residual_tol
        and div_norm <= prm.divergence_tol
        and topo_dist <= prm.topology_distance_tol
        and topo_cfl <= prm.topo_cfl_tol
    )
    return {
        "relative_residual": float(rel_res),
        "divergence_norm": float(div_norm),
        "topology_distance": float(topo_dist),
        "topological_cfl": float(topo_cfl),
        "accepted": bool(accepted),
    }


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------

def run_parameter_sweep(
    output_dir: Path,
    grid: Grid = Grid(),
    prm: Parameters = Parameters(),
    mu_values: np.ndarray | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if mu_values is None:
        mu_values = np.linspace(0.0, 0.90, 10)

    # Reference parameter and reference factorisation.
    mu0 = float(mu_values[0])
    K0, rhs0, A0, B, G, C = assemble_saddle_system(grid, prm, mu0)
    z0, t0_full = solve_full_saddle(K0, rhs0)
    beta0 = brinkman_resistance(grid, prm, mu0)
    topo0 = topology_signature(z0[: 2 * grid.n], grid, prm.recirculation_threshold)

    records: List[Dict[str, float | int | bool]] = []
    saved_fields: Dict[str, np.ndarray] = {}

    for j, mu in enumerate(mu_values):
        K, rhs, A, B, G, C = assemble_saddle_system(grid, prm, float(mu))
        beta = brinkman_resistance(grid, prm, float(mu))

        # Full direct solve for comparison.
        z_full, t_full = solve_full_saddle(K, rhs)

        # Low-rank diagonal update against reference A0.
        update_indices, deltas = diagonal_update_from_beta(beta0, beta)
        Ainv = WoodburyMomentumInverse(A0, update_indices, deltas)
        f = rhs[: 2 * grid.n]
        g = rhs[2 * grid.n :]
        z_rec, t_rec, gmres_info, gmres_iters = solve_recycled_schur(Ainv, B, G, C, f, g)

        topo = topology_signature(z_rec[: 2 * grid.n], grid, prm.recirculation_threshold)
        gate = ceter_decision(K, rhs, z_rec, B, grid, prm, topo0, topo, float(mu - mu0))
        sol_error = np.linalg.norm(z_rec - z_full) / max(np.linalg.norm(z_full), 1.0e-14)
        speed_ratio = t_full / max(t_rec, 1.0e-14)

        records.append(
            {
                "mu": float(mu),
                "rank_update": int(Ainv.rank),
                "full_solve_time_s": float(t_full),
                "recycled_schur_time_s": float(t_rec),
                "speed_ratio_full_over_recycled": float(speed_ratio),
                "gmres_info": int(gmres_info),
                "gmres_iterations": int(gmres_iters),
                "solution_error_vs_full": float(sol_error),
                "relative_residual": gate["relative_residual"],
                "divergence_norm": gate["divergence_norm"],
                "components": int(topo.components),
                "holes": int(topo.holes),
                "area_fraction": float(topo.area_fraction),
                "topology_distance": gate["topology_distance"],
                "topological_cfl": gate["topological_cfl"],
                "ceter_accept": bool(gate["accepted"]),
            }
        )

        if j in (0, len(mu_values) // 2, len(mu_values) - 1):
            saved_fields[f"mu={mu:.2f}"] = z_rec[: grid.n].reshape(grid.ny, grid.nx)

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_dir / "ceter_schur_results.csv", index=False)
    make_plots(df, saved_fields, grid, fig_dir)
    write_latex_table(df, output_dir / "ceter_schur_table.tex")
    return df


# -----------------------------------------------------------------------------
# Plotting and export
# -----------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, fields: Dict[str, np.ndarray], grid: Grid, fig_dir: Path) -> None:
    X, Y = grid.coordinates()

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogy(df["mu"], df["solution_error_vs_full"], marker="o", label="solution error")
    ax.semilogy(df["mu"], df["relative_residual"], marker="s", label="relative residual")
    ax.semilogy(df["mu"], df["divergence_norm"], marker="^", label="divergence norm")
    ax.set_xlabel("parameter $\\mu$")
    ax.set_ylabel("relative quantity")
    ax.set_title("CETER-Schur algebraic certificates")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "algebraic_certificates.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(df["mu"], df["components"], marker="o", label="components")
    ax.plot(df["mu"], df["holes"], marker="s", label="holes")
    ax.plot(df["mu"], df["topology_distance"], marker="^", label="topology distance")
    ax.set_xlabel("parameter $\\mu$")
    ax.set_ylabel("topology signal")
    ax.set_title("Topology gate along the parameter path")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "topology_gate.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(df["mu"], df["speed_ratio_full_over_recycled"], marker="o")
    ax.set_xlabel("parameter $\\mu$")
    ax.set_ylabel("full solve time / recycled time")
    ax.set_title("Measured speed ratio in the sparse demonstration")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "speed_ratio.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(fields), figsize=(4.2 * len(fields), 3.2), constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]
    for ax, (label, ux) in zip(axes, fields.items()):
        cf = ax.contourf(X, Y, ux, levels=20)
        ax.contour(X, Y, ux, levels=[2.0], linewidths=1.2)
        ax.set_title(f"streamwise velocity, {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(fig_dir / "recirculation_fields.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    accepted = df["ceter_accept"].astype(int)
    ax.step(df["mu"], accepted, where="mid")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["reject/rebuild", "accept/recycle"])
    ax.set_xlabel("parameter $\\mu$")
    ax.set_title("CETER accept/rebuild decision")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "accept_rebuild_signal.pdf", bbox_inches="tight")
    plt.close(fig)


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "mu",
        "rank_update",
        "solution_error_vs_full",
        "relative_residual",
        "divergence_norm",
        "components",
        "topology_distance",
        "ceter_accept",
    ]
    compact = df[cols].copy()
    with path.open("w", encoding="utf-8") as f:
        f.write("% Generated by ceter_schur_demo.py\n")
        f.write(compact.to_latex(index=False, float_format=lambda x: f"{x:.2e}"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    grid = Grid(nx=12, ny=6, lx=4.0, ly=1.0)
    prm = Parameters()
    mu_values = np.linspace(0.0, 0.90, 10)

    df = run_parameter_sweep(output_dir, grid, prm, mu_values)

    print("\nCETER-Schur demonstration completed.")
    print(f"Outputs written to: {output_dir}")
    print("\nMain results:")
    show_cols = [
        "mu",
        "rank_update",
        "solution_error_vs_full",
        "relative_residual",
        "divergence_norm",
        "components",
        "topology_distance",
        "topological_cfl",
        "ceter_accept",
    ]
    print(df[show_cols].to_string(index=False))
    


if __name__ == "__main__":
    main()
