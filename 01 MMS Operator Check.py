from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def central_diff_x(f: np.ndarray, h: float) -> np.ndarray:
    out = np.empty_like(f)
    out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * h)
    out[:, 0] = (f[:, 1] - f[:, 0]) / h
    out[:, -1] = (f[:, -1] - f[:, -2]) / h
    return out


def central_diff_y(f: np.ndarray, h: float) -> np.ndarray:
    out = np.empty_like(f)
    out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * h)
    out[0, :] = (f[1, :] - f[0, :]) / h
    out[-1, :] = (f[-1, :] - f[-2, :]) / h
    return out


def laplacian_2d(f: np.ndarray, h: float) -> np.ndarray:
    out = np.empty_like(f)
    out[1:-1, 1:-1] = (
        f[1:-1, 2:] + f[1:-1, :-2] + f[2:, 1:-1] + f[:-2, 1:-1]
        - 4.0 * f[1:-1, 1:-1]
    ) / (h * h)
    # Fill boundaries with NaN because the convergence check uses interior points.
    out[0, :] = np.nan
    out[-1, :] = np.nan
    out[:, 0] = np.nan
    out[:, -1] = np.nan
    return out


def l2_error(num: np.ndarray, exact: np.ndarray) -> float:
    mask = np.isfinite(num) & np.isfinite(exact)
    return float(np.sqrt(np.mean((num[mask] - exact[mask]) ** 2)))


def run(output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    prev_grad = None
    prev_lap = None
    prev_h = None

    for n in [16, 32, 64, 128]:
        # Interior grid on (0,1)^2. h = 1/(n+1), matching the manuscript table.
        h = 1.0 / (n + 1)
        x = np.linspace(h, 1.0 - h, n)
        y = np.linspace(h, 1.0 - h, n)
        X, Y = np.meshgrid(x, y)

        u = np.sin(np.pi * X) ** 2 * np.sin(2.0 * np.pi * Y)
        v = -np.sin(np.pi * Y) ** 2 * np.sin(2.0 * np.pi * X)
        p = np.sin(np.pi * X) * np.cos(np.pi * Y)

        # Exact derivatives.
        px_exact = np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y)
        py_exact = -np.pi * np.sin(np.pi * X) * np.sin(np.pi * Y)

        lap_u_exact = (
            2.0 * np.pi**2 * np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
            - 4.0 * np.pi**2 * np.sin(np.pi * X) ** 2 * np.sin(2.0 * np.pi * Y)
        )
        lap_v_exact = (
            -2.0 * np.pi**2 * np.cos(2.0 * np.pi * Y) * np.sin(2.0 * np.pi * X)
            + 4.0 * np.pi**2 * np.sin(np.pi * Y) ** 2 * np.sin(2.0 * np.pi * X)
        )

        ux_exact = 2.0 * np.pi * np.sin(np.pi * X) * np.cos(np.pi * X) * np.sin(2.0 * np.pi * Y)
        vy_exact = -2.0 * np.pi * np.sin(np.pi * Y) * np.cos(np.pi * Y) * np.sin(2.0 * np.pi * X)

        px_num = central_diff_x(p, h)
        py_num = central_diff_y(p, h)
        lap_u_num = laplacian_2d(u, h)
        lap_v_num = laplacian_2d(v, h)
        div_num = central_diff_x(u, h) + central_diff_y(v, h)
        div_exact = ux_exact + vy_exact

        # Use interior points for convergence checks so that boundary closure does not
        # dominate the observed order.
        inner = np.s_[1:-1, 1:-1]
        e_grad = np.sqrt(
            l2_error(px_num[inner], px_exact[inner]) ** 2
            + l2_error(py_num[inner], py_exact[inner]) ** 2
        )
        e_lap = np.sqrt(
            l2_error(lap_u_num[inner], lap_u_exact[inner]) ** 2
            + l2_error(lap_v_num[inner], lap_v_exact[inner]) ** 2
        )
        # Discrete divergence check on the same interior stencil.
        e_div = l2_error(div_num[inner], div_exact[inner])

        grad_order = np.nan if prev_grad is None else np.log(prev_grad / e_grad) / np.log(prev_h / h)
        lap_order = np.nan if prev_lap is None else np.log(prev_lap / e_lap) / np.log(prev_h / h)

        records.append(
            {
                "grid": f"{n}x{n}",
                "h": h,
                "E_grad_p": e_grad,
                "grad_order": grad_order,
                "E_lap_u": e_lap,
                "lap_order": lap_order,
                "E_div": e_div,
            }
        )
        prev_grad, prev_lap, prev_h = e_grad, e_lap, h

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "mms_operator_check.csv", index=False)
    with (output_dir / "mms_operator_check_table.tex").open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.3e}"))
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = run(root / "outputs")
    print(df.to_string(index=False))
