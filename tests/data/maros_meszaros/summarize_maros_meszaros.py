"""
Summarize all Maros-Meszaros QP problems in tests/data/maros_meszaros/
and write the results to a markdown table.

The .mat files use PIQP format:
    min  0.5 x'Px + c'x
    s.t. Ax = b          (p equality constraints)
         h_l <= Gx <= h_u (m inequality constraints)
         x_l <= x  <= x_u (variable bounds)

Usage:
    python summarize_maros_meszaros.py
"""

import os
import glob
import scipy.io
import scipy.sparse as sp
import numpy as np


def load_piqp_mat(path):
    data = scipy.io.loadmat(path)
    P = sp.csc_matrix(data["P"])
    c = data["c"].ravel()
    A = sp.csc_matrix(data["A"])
    b = data["b"].ravel()
    G = sp.csc_matrix(data["G"])
    h_l = data["h_l"].ravel()
    h_u = data["h_u"].ravel()
    x_l = data["x_l"].ravel()
    x_u = data["x_u"].ravel()
    return P, c, A, b, G, h_l, h_u, x_l, x_u


def summarize(path):
    P, c, A, b, G, h_l, h_u, x_l, x_u = load_piqp_mat(path)
    n = P.shape[0]
    p = A.shape[0]  # equality constraints
    m = G.shape[0]  # inequality constraints

    n_hl_finite = int(np.sum(np.isfinite(h_l)))
    n_hu_finite = int(np.sum(np.isfinite(h_u)))
    n_xl_finite = int(np.sum(np.isfinite(x_l)))
    n_xu_finite = int(np.sum(np.isfinite(x_u)))

    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "n": n,
        "p": p,
        "m": m,
        "nnz_P": P.nnz,
        "nnz_A": A.nnz,
        "nnz_G": G.nnz,
        "n_hl": n_hl_finite,
        "n_hu": n_hu_finite,
        "n_xl": n_xl_finite,
        "n_xu": n_xu_finite,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir
    output_path = os.path.join(script_dir, "maros_meszaros_summary.md")

    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        print(f"No .mat files found in {data_dir}")
        return

    print(f"Found {len(mat_files)} problems. Summarizing ...")

    rows = []
    for f in mat_files:
        try:
            info = summarize(f)
            rows.append(info)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")

    # Sort by n (number of variables)
    rows.sort(key=lambda r: r["n"])

    # Write markdown
    headers = ["Problem", "n", "p", "m", "nnz(P)", "nnz(A)", "nnz(G)",
               "n(h_l)", "n(h_u)", "n(x_l)", "n(x_u)"]
    keys = ["name", "n", "p", "m", "nnz_P", "nnz_A", "nnz_G",
            "n_hl", "n_hu", "n_xl", "n_xu"]

    with open(output_path, "w") as f:
        f.write("# Maros-Meszaros QP Dataset Summary\n\n")
        f.write(f"Total problems: {len(rows)}\n\n")

        # Header
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        # Rows
        for row in rows:
            vals = [str(row[k]) for k in keys]
            f.write("| " + " | ".join(vals) + " |\n")

        # Summary stats
        f.write(f"\n## Summary Statistics\n\n")
        for stat_name, key in [("n (variables)", "n"), ("p (equalities)", "p"),
                                ("m (inequalities)", "m")]:
            values = [r[key] for r in rows]
            f.write(f"- **{stat_name}**: min={min(values)}, max={max(values)}, "
                    f"median={int(np.median(values))}, mean={int(np.mean(values))}\n")

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
