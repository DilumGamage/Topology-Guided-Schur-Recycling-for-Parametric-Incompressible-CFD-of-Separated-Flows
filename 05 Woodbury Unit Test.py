from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def diffusion_matrix(n: int) -> np.ndarray:
    h = 1.0 / (n + 1)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2.0 / h**2
        if i > 0:
            A[i, i - 1] = -1.0 / h**2
        if i < n - 1:
            A[i, i + 1] = -1.0 / h**2
    return A


def woodbury_solve(A0: np.ndarray, U: np.ndarray, M: np.ndarray, V: np.ndarray, b: np.ndarray) -> np.ndarray:
    A0_inv_b = np.linalg.solve(A0, b)
    A0_inv_U = np.linalg.solve(A0, U)
    theta = np.linalg.inv(M) + V.T @ A0_inv_U
    return A0_inv_b - A0_inv_U @ np.linalg.solve(theta, V.T @ A0_inv_b)


def run(output_dir: Path, fig_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    n = 80
    A0 = diffusion_matrix(n) + 0.1 * np.eye(n)
    b = rng.normal(size=n)
    records = []

    for rank in [2, 4, 8, 12, 16]:
        U = rng.normal(size=(n, rank))
        V = rng.normal(size=(n, rank))
        M = np.diag(0.02 + 0.04 * rng.random(rank))
        Ar = A0 + U @ M @ V.T

        t0 = time.perf_counter()
        x_full = np.linalg.solve(Ar, b)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_update = woodbury_solve(A0, U, M, V, b)
        t_update = time.perf_counter() - t0

        rel_error = np.linalg.norm(x_update - x_full) / np.linalg.norm(x_full)
        rel_res = np.linalg.norm(Ar @ x_update - b) / np.linalg.norm(b)

        records.append(
            {
                "rank": rank,
                "relative_error": rel_error,
                "relative_residual": rel_res,
                "full_solve_s": t_full,
                "update_solve_s": t_update,
                "speed_ratio": t_full / max(t_update, 1e-15),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "woodbury_unit_test.csv", index=False)
    with (output_dir / "woodbury_unit_test_table.tex").open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.2e}"))

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.semilogy(df["rank"], df["relative_error"], marker="o", label="solution error")
    ax.semilogy(df["rank"], df["relative_residual"], marker="s", label="relative residual")
    ax.set_xlabel("update rank")
    ax.set_ylabel("relative quantity")
    ax.set_title("Woodbury update reproduces the full local solve")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "woodbury_error_residual.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.plot(df["rank"], df["speed_ratio"], marker="o")
    ax.set_xlabel("update rank")
    ax.set_ylabel("measured speed ratio")
    ax.set_title("Illustrative dense-matrix recycling speed ratio")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "woodbury_speed_ratio.pdf", bbox_inches="tight")
    plt.close(fig)

    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = run(root / "outputs", root / "figures")
    print(df.to_string(index=False))
