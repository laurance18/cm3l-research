"""
clt_gui.py - A Tkinter GUI for Classical Lamination Theory (CLT)

Dependencies:
    • Python >= 3.8 (standard library: tkinter, math)
    • numpy (matrix math)

Run with:
    python clt_gui.py

This tool lets you:
    1. Define a laminate by specifying ply orientation, thickness, and material properties per ply.
    2. Enter mechanical loads (Nx, Ny, Nxy, Mx, My, Mxy).
    3. Compute laminate [A], [B], [D] matrices, mid-plane strains/curvatures, and ply stresses/strains.
    4. Visualise results inside the application.

The code is intentionally verbose and modular to make it easier to follow the underlying
CLT mathematics.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple

import numpy as np
import tkinter.font as tkfont  # Added for custom fonts and styling

# --------------------------
# CLT CORE IMPLEMENTATION
# --------------------------

def compute_q_matrix(E1: float, E2: float, G12: float, v12: float) -> np.ndarray:
    """Return 3×3 reduced stiffness matrix [Q] for an orthotropic lamina in its principal axes."""
    v21 = v12 * E2 / E1  # reciprocal Poisson's ratio
    denom = 1 - v12 * v21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = v12 * E2 / denom
    Q66 = G12
    Q = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]], dtype=float)
    return Q


def transform_q(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """Return 3×3 transformed reduced stiffness matrix [\bar{Q}] for ply angle theta (deg)."""
    m = math.cos(math.radians(theta_deg))
    n = math.sin(math.radians(theta_deg))

    # Transformation coefficients
    m2, n2 = m * m, n * n
    mn = m * n
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

    # Engineering constants transformation (standard CLT formulas)
    Qbar11 = Q11 * m2**2 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * n2**2
    Qbar22 = Q11 * n2**2 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * m2**2
    Qbar12 = (Q11 + Q22 - 4 * Q66) * m2 * n2 + Q12 * (m2**2 + n2**2)
    Qbar16 = (Q11 - Q12 - 2 * Q66) * m2 * mn - (Q22 - Q12 - 2 * Q66) * n2 * mn
    Qbar26 = (Q11 - Q12 - 2 * Q66) * n2 * mn - (Q22 - Q12 - 2 * Q66) * m2 * mn
    Qbar66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m2 * n2 + Q66 * (m2**2 + n2**2)

    Qbar = np.array(
        [
            [Qbar11, Qbar12, Qbar16],
            [Qbar12, Qbar22, Qbar26],
            [Qbar16, Qbar26, Qbar66],
        ],
        dtype=float,
    )
    return Qbar


def laminate_abd(
    plies: List[Tuple[float, float, float, float, float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Compute laminate A, B, D matrices.

    Each ply is a tuple: (E1, E2, G12, v12, thickness, theta_deg)
    Returns (A, B, D, z_list) where z_list are the thickness coordinates.
    """
    # z-coordinates (bottom of laminate at z=-h/2, top at z=+h/2)
    h_total = sum(t for *_, t, _ in [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in plies])
    z = [-h_total / 2]
    for ply in plies:
        z.append(z[-1] + ply[4])

    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    for k, ply in enumerate(plies, start=1):
        E1, E2, G12, v12, t, theta = ply
        Q = compute_q_matrix(E1, E2, G12, v12)
        Qb = transform_q(Q, theta)

        zk = z[k]
        zk_1 = z[k - 1]
        dz = zk - zk_1

        A += Qb * dz
        B += 0.5 * Qb * (zk**2 - zk_1**2)
        D += (1.0 / 3.0) * Qb * (zk**3 - zk_1**3)

    return A, B, D, z


def midplane_strains(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    N: np.ndarray,
    M: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for mid-plane strains {ε0} and curvatures {κ}."""
    # Assemble ABD
    upper = np.hstack((A, B))
    lower = np.hstack((B, D))
    ABD = np.vstack((upper, lower))
    load = np.hstack((N, M))
    # Solve
    x = np.linalg.solve(ABD, load)
    eps0 = x[:3]
    kappa = x[3:]
    return eps0, kappa


def ply_strain_stress(
    eps0: np.ndarray,
    kappa: np.ndarray,
    z_bot: float,
    z_top: float,
    Qb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average ply strains/stresses (at mid-ply)."""
    z_mid = 0.5 * (z_bot + z_top)
    strain_global = eps0 + kappa * z_mid
    stress_global = Qb @ strain_global
    return strain_global, stress_global


# --------------------------
# GUI IMPLEMENTATION
# --------------------------

class LaminationGUI(tk.Tk):
    DEFAULT_PLY = {
        "E1": 135e9,  # Pa
        "E2": 10e9,
        "G12": 5e9,
        "v12": 0.3,
        "t": 0.000125,  # 0.125 mm
        "theta": 0.0,  # degrees
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("Classical Lamination Theory – GUI")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.geometry("1000x600")

        # ----- Custom fonts & styles -----
        self.font_mono = tkfont.Font(family="Consolas", size=10)
        self.style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
        self.style.configure("Treeview", rowheight=22)
        self.style.configure("TButton", padding=6)

        self.plies: List[dict[str, tk.Variable]] = []  # store tk variables for each ply

        self._create_widgets()

    # ----- GUI construction -----

    def _create_widgets(self) -> None:
        self.main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left frame – inputs
        self.frm_inputs = ttk.Frame(self.main_pane, padding=10)
        self.main_pane.add(self.frm_inputs, weight=1)

        # Right frame – results
        self.frm_results = ttk.Frame(self.main_pane, padding=10)
        self.main_pane.add(self.frm_results, weight=1)

        # -------- Inputs Side --------
        # Number of plies selector
        frm_n = ttk.Frame(self.frm_inputs)
        frm_n.pack(anchor=tk.W)
        ttk.Label(frm_n, text="Number of plies:").pack(side=tk.LEFT)
        self.var_nplies = tk.IntVar(value=4)
        spn = ttk.Spinbox(frm_n, from_=1, to=50, width=5, textvariable=self.var_nplies, command=self._update_plies)
        spn.pack(side=tk.LEFT, padx=5)

        # Ply table with scrollbars
        self.tbl_frame = ttk.Frame(self.frm_inputs)
        self.tbl_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scroll_y = ttk.Scrollbar(self.tbl_frame, orient=tk.VERTICAL)
        scroll_x = ttk.Scrollbar(self.tbl_frame, orient=tk.HORIZONTAL)

        self.tbl = ttk.Treeview(
            self.tbl_frame,
            columns=("E1", "E2", "G12", "v12", "t", "theta"),
            show="headings",
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
            height=15,
        )
        scroll_y.config(command=self.tbl.yview)
        scroll_x.config(command=self.tbl.xview)

        for col in self.tbl["columns"]:
            self.tbl.heading(col, text=col)
            self.tbl.column(col, width=80, anchor=tk.CENTER)

        self.tbl.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        self.tbl_frame.rowconfigure(0, weight=1)
        self.tbl_frame.columnconfigure(0, weight=1)

        # Alternate row colours
        self.tbl.tag_configure("oddrow", background="#f2f2f7")
        self.tbl.tag_configure("evenrow", background="#ffffff")

        # Edit selected row button
        btn_edit = ttk.Button(self.frm_inputs, text="Edit Selected Ply", command=self._edit_selected_ply)
        btn_edit.pack(pady=2)

        # Load inputs for N and M
        frm_loads = ttk.LabelFrame(self.frm_inputs, text="Applied Loads", padding=5)
        frm_loads.pack(fill=tk.X, pady=5)
        self.load_vars = {name: tk.DoubleVar(value=0.0) for name in ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy")}
        for i, (label, var) in enumerate(self.load_vars.items()):
            ttk.Label(frm_loads, text=label + " (N/m or N):").grid(row=i // 3, column=(i % 3) * 2, sticky=tk.E, pady=2)
            ttk.Entry(frm_loads, textvariable=var, width=12).grid(row=i // 3, column=(i % 3) * 2 + 1, sticky=tk.W, padx=3)

        # Compute button
        ttk.Button(self.frm_inputs, text="Compute", command=self._compute).pack(pady=8)

        # -------- Results Side --------
        self.txt_frame = ttk.Frame(self.frm_results)
        self.txt_frame.pack(fill=tk.BOTH, expand=True)

        txt_scroll_y = ttk.Scrollbar(self.txt_frame, orient=tk.VERTICAL)
        txt_scroll_x = ttk.Scrollbar(self.txt_frame, orient=tk.HORIZONTAL)

        self.txt_res = tk.Text(
            self.txt_frame,
            wrap=tk.NONE,
            state=tk.DISABLED,
            yscrollcommand=txt_scroll_y.set,
            xscrollcommand=txt_scroll_x.set,
            font=self.font_mono,
        )
        txt_scroll_y.config(command=self.txt_res.yview)
        txt_scroll_x.config(command=self.txt_res.xview)

        self.txt_res.grid(row=0, column=0, sticky="nsew")
        txt_scroll_y.grid(row=0, column=1, sticky="ns")
        txt_scroll_x.grid(row=1, column=0, sticky="ew")
        self.txt_frame.rowconfigure(0, weight=1)
        self.txt_frame.columnconfigure(0, weight=1)

        self._update_plies()

    # ----- Ply table management -----

    def _update_plies(self) -> None:
        n = self.var_nplies.get()
        current = len(self.plies)
        # Add new rows if needed
        for _ in range(current, n):
            vars_ = {key: tk.DoubleVar(value=val) for key, val in self.DEFAULT_PLY.items()}
            self.plies.append(vars_)
        # Trim extra
        if n < current:
            self.plies = self.plies[:n]
        # Refresh treeview
        for row in self.tbl.get_children():
            self.tbl.delete(row)
        for i, vars_ in enumerate(self.plies, start=1):
            values = [f"{vars_[c].get():.3g}" if c != "theta" else f"{vars_[c].get():.1f}" for c in self.tbl["columns"]]
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            self.tbl.insert("", tk.END, iid=str(i - 1), values=values, tags=(tag,))

    def _edit_selected_ply(self) -> None:
        sel = self.tbl.focus()
        if not sel:
            messagebox.showinfo("Edit Ply", "Select a ply to edit.")
            return
        idx = int(sel)
        vars_ = self.plies[idx]

        win = tk.Toplevel(self)
        win.title(f"Edit Ply #{idx + 1}")
        entries = {}
        for i, key in enumerate(self.tbl["columns"]):
            ttk.Label(win, text=key).grid(row=i, column=0, sticky=tk.E, padx=5, pady=2)
            v = vars_[key]
            e = ttk.Entry(win, textvariable=v)
            e.grid(row=i, column=1, padx=5, pady=2)
            entries[key] = e
        ttk.Button(win, text="Update", command=lambda: (self._update_plies(), win.destroy())).grid(
            row=len(self.tbl["columns"]), column=0, columnspan=2, pady=5
        )

    # ----- Computation -----

    def _compute(self) -> None:
        # Gather ply data
        try:
            plies_data = []
            for vars_ in self.plies:
                E1 = float(vars_["E1"].get())
                E2 = float(vars_["E2"].get())
                G12 = float(vars_["G12"].get())
                v12 = float(vars_["v12"].get())
                t = float(vars_["t"].get())
                theta = float(vars_["theta"].get())
                plies_data.append((E1, E2, G12, v12, t, theta))
        except Exception as exc:
            messagebox.showerror("Error", f"Invalid ply data: {exc}")
            return

        A, B, D, z = laminate_abd(plies_data)

        N = np.array([self.load_vars["Nx"].get(), self.load_vars["Ny"].get(), self.load_vars["Nxy"].get()])
        M = np.array([self.load_vars["Mx"].get(), self.load_vars["My"].get(), self.load_vars["Mxy"].get()])

        try:
            eps0, kappa = midplane_strains(A, B, D, N, M)
        except np.linalg.LinAlgError as exc:
            messagebox.showerror("Error", f"Singular laminate stiffness matrix: {exc}")
            return

        # Build results string
        out = []
        out.append("A matrix (N/m):\n" + np.array2string(A, precision=3, suppress_small=True))
        out.append("\nB matrix (N):\n" + np.array2string(B, precision=3, suppress_small=True))
        out.append("\nD matrix (N·m):\n" + np.array2string(D, precision=3, suppress_small=True))
        out.append("\nMid-plane strains ε0: " + np.array2string(eps0, precision=6))
        out.append("\nCurvatures κ: " + np.array2string(kappa, precision=6))

        # Ply stresses/strains
        out.append("\n\nPly-by-ply results (mid-ply):")
        for k, ply in enumerate(plies_data, start=1):
            E1, E2, G12, v12, t, theta = ply
            Q = compute_q_matrix(E1, E2, G12, v12)
            Qb = transform_q(Q, theta)
            strain, stress = ply_strain_stress(eps0, kappa, z[k - 1], z[k], Qb)
            out.append(
                f"\nPly {k} (θ={theta}°):\n  Strain: {np.array2string(strain, precision=6)}\n  Stress (Pa): {np.array2string(stress, precision=3)}"
            )

        self.txt_res.config(state=tk.NORMAL)
        self.txt_res.delete(1.0, tk.END)
        self.txt_res.insert(tk.END, "\n".join(out))
        self.txt_res.config(state=tk.DISABLED)


# --------------------------
# Entry point
# --------------------------

def main() -> None:
    app = LaminationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
