import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
import plotly.graph_objects as go
import time
import os
st.set_page_config(
    page_title="NanoScope",
    layout="wide",
    page_icon="🔬"
)

st.title("🔬 NanoScope")
st.caption("Size and Current Drop Predictor")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Size Calculator",
        "ΔI Range Explorer",
        "Live Animation"
    ]
)
if page == "Home":

    st.header("Welcome to NanoScope")

    st.write("""
NanoScope is a nanopore analysis tool for exploring pore geometry,
ionic current, and biomolecule blockade signals.

This tool allows you to:

• Estimate nanopore size from IV curves  
• Model blockade current (ΔI) for different biomarker geometries  
• Explore orientation effects for ellipsoids and rod-like proteins  

""")

    st.subheader("Modules")

    st.write("""
**Size Calculator**

Estimate pore diameter from conductance using CBD or conical models.

**ΔI Range Explorer**

Predict possible current blockade values for biomolecules entering
the nanopore with different orientations.

""")

    st.info("Developed for nanopore biosensing research.")
# =========================
# Common: TXT -> clean CSV
# =========================
def clean_voltage(v: str) -> float:
    v = str(v).strip()
    if v.endswith("m"):
        return float(v.replace("m", "")) * 1e-3
    return float(v)

def clean_current(i: str) -> float:
    i = str(i).strip()
    if i.endswith("n"):
        return float(i.replace("n", "")) * 1e-9
    if i.endswith("f"):
        return float(i.replace("f", "")) * 1e-15
    return float(i)

def txt_to_iv_clean_df(file_bytes: bytes) -> pd.DataFrame:
    s = file_bytes.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(s), skipinitialspace=True)
    df["Voltage_V"] = df["Voltage2[V]"].apply(clean_voltage)
    df["Current_A"] = df["Current[A]"].apply(clean_current)
    out = df[["Sweep #", "Voltage_V", "Current_A"]].copy()
    return out

# =========================
# Common helpers
# =========================
def sort_by_voltage(V, I):
    idx = np.argsort(V)
    return V[idx], I[idx]

def slope_through_origin(V, I):
    denom = np.sum(V**2)
    if denom <= 0:
        return np.nan
    return np.sum(V * I) / denom

def slope_with_intercept(V, I):
    G, b = np.polyfit(V, I, 1)
    return G, b

def plot_iv_line(V, I, title="I–V Curve", y_in_nA=True, show_fit=False, fit_G=None):
    V, I = sort_by_voltage(V, I)
    fig, ax = plt.subplots(figsize=(7,5))
    if y_in_nA:
        ax.plot(V, I*1e9, linewidth=2, label="IV data")
        ax.set_ylabel("Current (nA)")
        if show_fit and fit_G is not None:
            ax.plot(V, (fit_G*V)*1e9, linestyle="--", linewidth=2, label="Fit")
    else:
        ax.plot(V, I, linewidth=2, label="IV data")
        ax.set_ylabel("Current (A)")
        if show_fit and fit_G is not None:
            ax.plot(V, fit_G*V, linestyle="--", linewidth=2, label="Fit")

    ax.set_xlabel("Voltage (V)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

def pick_linear_region_auto(V, I, eps=0.005, window=0.05, min_points=6):
    w = window
    while True:
        mask = (np.abs(V) > eps) & (np.abs(V) < w)
        V_lin = V[mask]
        I_lin = I[mask]
        denom = np.sum(V_lin**2)

        if V_lin.size >= min_points and denom > 0:
            return V_lin, I_lin, w

        w *= 1.5
        if w > 0.5:
            raise ValueError(
                "Couldn't find enough points in a near-zero linear window.\n"
                "Try increasing window, lowering min_points, or use a global fit."
            )

# =========================
# CBD cylindrical size + MC uncertainty
# =========================
def diameter_cyl_no_access(G_S, sigma_Sm, L_m):
    if G_S <= 0 or sigma_Sm <= 0 or L_m <= 0:
        return np.nan
    return np.sqrt((4*L_m*G_S)/(sigma_Sm*np.pi))

def diameter_cyl_with_access(G_S, sigma_Sm, L_m):
    if G_S <= 0 or sigma_Sm <= 0 or L_m <= 0:
        return np.nan
    a = (np.pi * sigma_Sm) / G_S
    b = np.pi / 2.0
    disc = b*b + 4*a*L_m
    if disc <= 0 or a <= 0:
        return np.nan
    r = (b + np.sqrt(disc)) / (2*a)
    return 2*r

def mc_cyl_diameter(G_nS, dG_nS, sigma, dsigma, L_nm, dL_nm, include_access=True, N=100000, seed=7):
    rng = np.random.default_rng(seed)
    Gs = rng.normal(G_nS, dG_nS, N) * 1e-9
    ss = rng.normal(sigma, dsigma, N)
    Ls = rng.normal(L_nm, dL_nm, N) * 1e-9

    mask = (Gs > 0) & (ss > 0) & (Ls > 0)
    Gs, ss, Ls = Gs[mask], ss[mask], Ls[mask]
    if Gs.size < 2000:
        return np.nan, np.nan, (np.nan, np.nan), 0

    if include_access:
        d = np.array([diameter_cyl_with_access(g, s, l) for g, s, l in zip(Gs, ss, Ls)])
    else:
        d = np.sqrt((4*Ls*Gs)/(ss*np.pi))

    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 2000:
        return np.nan, np.nan, (np.nan, np.nan), 0

    d_nm = d * 1e9
    mean = float(np.mean(d_nm))
    std  = float(np.std(d_nm, ddof=1))
    lo, hi = np.percentile(d_nm, [2.5, 97.5])
    return mean, std, (float(lo), float(hi)), int(d_nm.size)

# =========================
# Conical model (your formula)
# =========================
def G_conical_single(r, K, L, theta):
    num = 4*np.pi*r*(r + L*np.tan(theta))
    den = 4*L + np.pi*(2*r + L*np.tan(theta))
    return K*(num/den)

def solve_tip_radius_brentq(G_single, K, L, theta, r_lo=0.5e-9, r_hi=300e-9):
    def f(r):
        return G_conical_single(r, K, L, theta) - G_single
    f_lo, f_hi = f(r_lo), f(r_hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            "Root not bracketed in [0.5 nm, 300 nm].\n"
            f"f(0.5 nm)={f_lo:.3e}, f(300 nm)={f_hi:.3e}\n"
            "Check n, K, L, theta, or your conductance fit."
        )
    return brentq(f, r_lo, r_hi, maxiter=2000)

# =========================
# TAB 1: Size (CBD / Conical)
# =========================
if page == "Size Calculator":
    
    st.subheader("1) Upload IV file")
    up = st.file_uploader("Upload .csv or .txt", type=["csv", "txt"], key="iv_upload")

    df_iv = None
    if up is not None:
        raw = up.read()
        name = up.name.lower()

        if name.endswith(".txt"):
            df_iv = txt_to_iv_clean_df(raw)
            st.success("Loaded TXT → converted to clean CSV columns (Voltage_V, Current_A).")
            out_csv = df_iv.to_csv(index=False).encode("utf-8")
            st.download_button("Download cleaned CSV", out_csv, file_name="IV_clean.csv", mime="text/csv")
        else:
            df_iv = pd.read_csv(io.BytesIO(raw))
            st.success("Loaded CSV.")

        st.write("Preview:")
        st.dataframe(df_iv.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("2) Choose analysis mode")
    mode = st.radio(
        "Mode",
        ["CBD (cylindrical)", "Conical"],
        horizontal=False
    )
    plot_nA = st.checkbox("Plot current in nA", value=True, key="plot_nA_tab1")

    # ---------- CBD ----------
    if mode.startswith("CBD"):
        st.subheader("CBD cylindrical pore diameter")

        col1, col2 = st.columns(2)
        with col1:
            use_iv = st.checkbox("Use uploaded IV to compute conductance (G)", value=True)
        with col2:
            st.caption("If you already know G, turn this off and enter G ± error below.")

        G_from_iv_nS = None
        if use_iv:
            if df_iv is None:
                st.warning("Upload an IV file first.")
            else:
                if "Voltage_V" in df_iv.columns and "Current_A" in df_iv.columns:
                    V = df_iv["Voltage_V"].to_numpy(dtype=float)
                    I = df_iv["Current_A"].to_numpy(dtype=float)
                elif "Voltage (V)" in df_iv.columns and "Current (A)" in df_iv.columns:
                    V = df_iv["Voltage (V)"].to_numpy(dtype=float)
                    I = df_iv["Current (A)"].to_numpy(dtype=float)
                else:
                    st.error("Need columns: (Voltage_V, Current_A) OR (Voltage (V), Current (A)).")
                    V, I = None, None

                if V is not None:
                    ok = np.isfinite(V) & np.isfinite(I)
                    V, I = V[ok], I[ok]

                    G_S = slope_through_origin(V, I)
                    G_from_iv_nS = G_S * 1e9

                    plot_iv_line(V, I, title="I–V (line) + global fit", y_in_nA=plot_nA, show_fit=True, fit_G=G_S)
                    st.info(f"Global slope (whole trace, through origin): **G = {G_from_iv_nS:.2f} nS**")

        st.markdown("### Inputs (with uncertainties)")
        default_G = float(G_from_iv_nS) if (G_from_iv_nS is not None and np.isfinite(G_from_iv_nS)) else 175.0

        G_nS   = st.number_input("Conductance G (nS)", value=default_G, step=1.0, format="%.3f")
        dG_nS  = st.number_input("± error in G (nS)", value=1.0, step=0.1, format="%.3f")

        sigma  = st.number_input("Conductivity σ (S/m)", value=11.5, step=0.1, format="%.4f")
        dsigma = st.number_input("± error in σ (S/m)", value=0.2, step=0.05, format="%.4f")

        L_nm   = st.number_input("Pore length L (nm)", value=7.0, step=0.5, format="%.3f")
        dL_nm  = st.number_input("± error in L (nm)", value=0.5, step=0.1, format="%.3f")

        include_access = st.checkbox("Include access resistance (recommended)", value=True)
        N = st.selectbox("Monte Carlo samples", [20000, 50000, 100000, 200000], index=2)

        if st.button("Calculate CBD diameter"):
            mean, std, (lo, hi), n_ok = mc_cyl_diameter(
                G_nS, dG_nS, sigma, dsigma, L_nm, dL_nm,
                include_access=include_access, N=N
            )
            if np.isnan(mean):
                st.error("Could not compute. Check inputs (must be >0) and uncertainties not too huge.")
            else:
                st.metric("Diameter (nm)", f"{mean:.2f} ± {std:.2f}")
                st.write(f"95% interval: **{lo:.2f} – {hi:.2f} nm**")
                st.caption(f"Valid MC samples used: {n_ok:,}")

    # ---------- Conical ----------
    else:
        st.subheader("Conical pore (tip radius) from IV")

        stage = st.selectbox(
            "Which conical scenario?",
            [
                "Just IV (10 mM NaCl) – auto near-zero window fit (brentq)",
                "Just IV (1 M NaCl) – polyfit with intercept (fsolve)",
                "After functionalization / antibody (10 mM) – window fit (through origin)",
                "Antibody/biosensing IV (10 mM) – polyfit with intercept (fsolve)"
            ]
        )

        st.markdown("### Geometry + experiment settings")
        n = st.number_input("Number of pores n", value=250, step=1)
        theta_deg = st.number_input("Half cone angle θ (deg)", value=12.6, step=0.1, format="%.3f")
        L_nm = st.number_input("Pore length L (nm)", value=750.0, step=5.0, format="%.2f")

        if "1 M" in stage:
            K_default = 8.97
        else:
            K_default = 0.14

        K = st.number_input("Conductivity K (S/m)", value=float(K_default), step=0.01, format="%.4f")

        if df_iv is None:
            st.warning("Upload an IV file first.")
        else:
            if "Voltage (V)" in df_iv.columns and "Current (A)" in df_iv.columns:
                V = df_iv["Voltage (V)"].to_numpy(dtype=float)
                I = df_iv["Current (A)"].to_numpy(dtype=float)
            elif "Voltage_V" in df_iv.columns and "Current_A" in df_iv.columns:
                V = df_iv["Voltage_V"].to_numpy(dtype=float)
                I = df_iv["Current_A"].to_numpy(dtype=float)
            else:
                st.error("Need columns: (Voltage (V), Current (A)) OR (Voltage_V, Current_A).")
                V, I = None, None

            if V is not None:
                ok = np.isfinite(V) & np.isfinite(I)
                V, I = V[ok], I[ok]
                plot_iv_line(V, I, title="I–V (line)", y_in_nA=plot_nA)

                theta = np.deg2rad(theta_deg)
                L = L_nm * 1e-9

                st.markdown("### Conductance extraction + solve")

                if stage.startswith("Just IV (10 mM"):
                    eps = st.number_input("Exclude |V| < eps (V)", value=0.005, step=0.001, format="%.4f")
                    window = st.number_input("Start window |V| < window (V)", value=0.05, step=0.01, format="%.3f")
                    min_points = st.number_input("Min points for fit", value=6, step=1)

                    if st.button("Compute r (10 mM, auto-window brentq)"):
                        V_lin, I_lin, w = pick_linear_region_auto(V, I, eps=eps, window=window, min_points=int(min_points))
                        G_total = slope_through_origin(V_lin, I_lin)
                        G_single = G_total / n

                        st.write(f"Using: eps={eps} V, window={w:.4f} V, points={V_lin.size}")
                        st.write(f"G_total  = {G_total:.6e} S")
                        st.write(f"G_single = {G_single:.6e} S")

                        r = solve_tip_radius_brentq(G_single, K, L, theta)
                        st.success(f"Estimated tip radius r ≈ {r*1e9:.2f} nm")

                elif stage.startswith("Just IV (1 M"):
                    eps = st.number_input("Ignore |V| < eps (V)", value=0.05, step=0.01, format="%.3f")
                    if st.button("Compute r (1 M, polyfit+intercept, fsolve)"):
                        mask = np.abs(V) > eps
                        G_total, intercept = slope_with_intercept(V[mask], I[mask])
                        G_single = G_total / n

                        st.write("Fit: I = G*V + b")
                        st.write(f"G_total: {G_total:.6e} S")
                        st.write(f"Intercept b: {intercept:.3e} A")
                        st.write(f"G_single: {G_single:.6e} S")

                        def eq_r(r):
                            return G_conical_single(r, K, L, theta) - G_single

                        r0 = 20e-9
                        r = fsolve(eq_r, r0)[0]
                        st.success(f"Estimated tip radius r ≈ {r*1e9:.2f} nm")

                elif stage.startswith("After functionalization"):
                    window = st.number_input("Fit window ±V (V)", value=0.1, step=0.01, format="%.3f")
                    if st.button("Compute r (functionalization, window fit through origin)"):
                        mask = (V > -window) & (V < window)
                        V_lin, I_lin = V[mask], I[mask]

                        G_total = slope_through_origin(V_lin, I_lin)
                        G_single = G_total / n

                        st.write(f"Points in window: {V_lin.size}")
                        st.write(f"G_total: {G_total:.6e} S")
                        st.write(f"G_single: {G_single:.6e} S")

                        r = solve_tip_radius_brentq(G_single, K, L, theta)
                        st.success(f"Estimated tip radius r ≈ {r*1e9:.2f} nm")

                else:  # Antibody/biosensing IV
                    window = st.number_input("Fit window ±V (V)", value=0.1, step=0.01, format="%.3f")
                    if st.button("Compute r (biosensing IV, polyfit+intercept)"):
                        mask = (V >= -window) & (V <= window)
                        V_lin, I_lin = V[mask], I[mask]

                        G_total, I0 = slope_with_intercept(V_lin, I_lin)
                        G_single = G_total / n

                        st.write(f"Points in window: {V_lin.size}")
                        st.write(f"G_total: {G_total:.6e} S")
                        st.write(f"Intercept: {I0:.3e} A")
                        st.write(f"G_single: {G_single:.6e} S")

                        def eq_r(r):
                            return G_conical_single(r, K, L, theta) - G_single

                        r0 = 60e-9
                        r = fsolve(eq_r, r0)[0]
                        st.success(f"Estimated tip radius r ≈ {r*1e9:.2f} nm")

# =========================
# TAB 2: Current Drop (ΔI)
# =========================
if page == "ΔI Range Explorer":

    st.subheader("ΔI Range Explorer")

    # ---------- Core equations ----------
    def pore_d_from_i0(i0_A, L_m, V_V, sigma_Sm):
        # d = (i0 + sqrt(i0*(i0 + 16 L V sigma/pi))) / (2 V sigma)
        term = i0_A + (16.0 * L_m * V_V * sigma_Sm) / np.pi
        if i0_A <= 0 or term <= 0 or V_V <= 0 or sigma_Sm <= 0:
            return np.nan
        return (i0_A + np.sqrt(i0_A * term)) / (2.0 * V_V * sigma_Sm)

    def i_from_d(d_m, L_m, V_V, sigma_Sm):
        denom = (4.0 * L_m) / (np.pi * d_m**2) + (1.0 / d_m)
        return sigma_Sm * V_V / denom

    def delta_i(i0_A, d_m, L_m, V_V, sigma_Sm, dbio_m):
        inside = d_m**2 - dbio_m**2
        if inside <= 0:
            return np.nan
        d_withbio = np.sqrt(inside)
        i_withbio = i_from_d(d_withbio, L_m, V_V, sigma_Sm)
        return i0_A - i_withbio  # A

    def circle_overlap_area(R, r, x):
        """
        Overlap area between two circles:
        - pore radius = R
        - biomolecule effective radius = r
        - center-to-center offset = x
        Returns area in m^2
        """
        # no overlap
        if x >= R + r:
            return 0.0

        # one fully inside the other
        if x <= abs(R - r):
            return np.pi * min(R, r)**2

        # partial overlap
        term1 = r**2 * np.arccos((x**2 + r**2 - R**2) / (2 * x * r))
        term2 = R**2 * np.arccos((x**2 + R**2 - r**2) / (2 * x * R))
        term3 = 0.5 * np.sqrt(
            (-x + r + R) *
            (x + r - R) *
            (x - r + R) *
            (x + r + R)
        )
        return term1 + term2 - term3

    def dbio_from_blocked_area(A_blocked):
        """
        Convert blocked area to equivalent circular blocking diameter.
        A = pi d^2 / 4  => d = 2 sqrt(A/pi)
        """
        if A_blocked <= 0:
            return 0.0
        return 2.0 * np.sqrt(A_blocked / np.pi)

    def delta_i_from_blocked_area(i0_A, d_m, L_m, V_V, sigma_Sm, A_blocked):
        """
        Compute ΔI from blocked overlap area directly.
        """
        dbio_eff = dbio_from_blocked_area(A_blocked)
        return delta_i(i0_A, d_m, L_m, V_V, sigma_Sm, dbio_eff)

    def summarize(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return {
            "count": int(values.size),
            "min": float(np.min(values)),
            "p5": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    st.markdown("### Inputs")
    col1, col2 = st.columns(2)

    with col1:
        i0_nA = st.number_input("Open pore current i0 (nA)", value=24.0, step=0.5)
        V = st.number_input("Voltage V (V)", value=0.300, step=0.010, format="%.3f")
        sigma = st.number_input("Conductivity σ (S/m)", value=11.51, step=0.01)

    with col2:
        L_nm = st.number_input("Pore length L (nm)", value=7.0, step=0.5)
        occupancy = st.slider("Occupancy factor (0.3–1.0)", 0.3, 1.0, 1.0, 0.05)
        st.caption("Occupancy scales the effective blocker size; overlap geometry models bumps and adsorption.")

    i0_A = i0_nA * 1e-9
    L_m = L_nm * 1e-9

    d_m = pore_d_from_i0(i0_A, L_m, V, sigma)
    if np.isfinite(d_m):
        st.info(f"Inferred pore diameter d ≈ **{d_m*1e9:.2f} nm**")
    else:
        st.error("Could not infer pore diameter. Check that i0, V, σ, L are > 0.")

    st.markdown("---")
    st.markdown("### Choose biomarker shape model")
    model = st.selectbox(
        "Model",
        ["Sphere", "Ellipsoid", "Rod / spherocylinder"]
    )

    # ---------- Model A: Sphere ----------
    if model.startswith("Sphere"):
        dbio_nm = st.number_input("Biomarker diameter d_bio (nm)", value=6.0, step=0.2)
        if st.button("Compute ΔI (sphere)"):
            dbio_m = dbio_nm * 1e-9 * occupancy
            di = delta_i(i0_A, d_m, L_m, V, sigma, dbio_m)
            if np.isfinite(di):
                st.success(f"ΔI ≈ **{di*1e12:.0f} pA**  (fractional blockade ≈ {di/i0_A:.3f})")
            else:
                st.error("This dbio is too large for the inferred pore diameter.")

    # ---------- Model B: Ellipsoid ----------
    elif model.startswith("Ellipsoid"):

        A_nm = st.number_input("Axis A (nm) (long)", value=14.0, step=0.5)
        B_nm = st.number_input("Axis B (nm)", value=4.0, step=0.5)
        C_nm = st.number_input("Axis C (nm)", value=4.0, step=0.5)
        N = int(st.number_input("Orientation samples", value=50000, step=5000))
        seed = int(st.number_input("Random seed", value=7, step=1))

        event_model = st.selectbox(
            "Event model",
            ["Centered translocation", "Bump / partial entry", "Adsorption / rim interaction"]
        )

        def random_unit_vectors(N, rng):
            u = rng.random(N)
            v = rng.random(N)
            theta = 2 * np.pi * u
            z = 2 * v - 1
            r = np.sqrt(1 - z**2)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.stack([x, y, z], axis=1)

        def projected_area_ellipsoid(a, b, c, nvec):
            nx, ny, nz = nvec[:, 0], nvec[:, 1], nvec[:, 2]
            denom = np.sqrt((a * nx)**2 + (b * ny)**2 + (c * nz)**2)
            return (np.pi * a * b * c) / denom  # m^2

        if st.button("Compute ΔI range (ellipsoid)"):

            rng = np.random.default_rng(seed)

            # semi-axes (m)
            a = (A_nm / 2) * 1e-9
            b = (B_nm / 2) * 1e-9
            c = (C_nm / 2) * 1e-9

            pore_radius = d_m / 2.0

            nvec = random_unit_vectors(N, rng)
            Aproj = projected_area_ellipsoid(a, b, c, nvec)  # m^2
            dbio_eff = 2 * np.sqrt(Aproj / np.pi)            # m
            rbio_eff = dbio_eff / 2.0

            di_list = []
            offset_list_nm = []
            blocked_area_list_nm2 = []

            for r_eff in rbio_eff:

                if event_model == "Centered translocation":
                    offset = 0.0

                elif event_model == "Bump / partial entry":
                    offset = rng.uniform(
                        max(0.0, pore_radius - 0.3 * r_eff),
                        pore_radius + 0.8 * r_eff
                    )

                else:  # Adsorption / rim interaction
                    offset = rng.uniform(
                        max(0.0, pore_radius - 0.8 * r_eff),
                        pore_radius + 0.2 * r_eff
                    )

                A_blocked = circle_overlap_area(pore_radius, r_eff * occupancy, offset)
                di_val = delta_i_from_blocked_area(i0_A, d_m, L_m, V, sigma, A_blocked)

                if np.isfinite(di_val):
                    di_list.append(di_val * 1e12)  # pA
                    offset_list_nm.append(offset * 1e9)
                    blocked_area_list_nm2.append(A_blocked * 1e18)  # nm²

            di_pA = np.array(di_list)
            stats = summarize(di_pA)

            if stats is None:
                st.error("No valid events were generated.")
            else:
                st.success(f"Possible ΔI range: **{stats['min']:.0f} – {stats['max']:.0f} pA**")
                st.info(f"Typical ΔI range (5–95%): **{stats['p5']:.0f} – {stats['p95']:.0f} pA**")
                st.caption(f"Valid simulated events: {stats['count']:,} | Median ΔI ≈ {stats['median']:.0f} pA")

                diag_df = pd.DataFrame({
                    "ΔI (pA)": di_pA,
                    "offset (nm)": offset_list_nm,
                    "blocked area (nm²)": blocked_area_list_nm2
                })
                st.dataframe(diag_df.describe().T, use_container_width=True)

                hist_counts, hist_edges = np.histogram(di_pA, bins=60)
                hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
                hist_df = pd.DataFrame({
                    "ΔI center (pA)": hist_centers,
                    "count": hist_counts
                })
                st.line_chart(hist_df.set_index("ΔI center (pA)"))

    # ---------- Model C: Rod / spherocylinder ----------
    else:

        Lrod_nm = st.number_input("Rod length L_rod (nm)", value=50.0, step=5.0)
        Drod_nm = st.number_input("Rod diameter D_rod (nm)", value=6.0, step=0.5)
        n_angles = int(st.number_input("Angle steps", value=361, step=60))

        if st.button("Compute ΔI range (rod)"):
            Lrod = Lrod_nm * 1e-9
            Drod = Drod_nm * 1e-9

            theta = np.linspace(0, np.pi/2, n_angles)

            # Approx projected area of spherocylinder
            Atheta = (Drod * Lrod * np.abs(np.sin(theta))) + (np.pi * Drod**2 / 4.0)

            dbio_eff = 2 * np.sqrt(Atheta / np.pi) * occupancy  # m
            di = np.array([delta_i(i0_A, d_m, L_m, V, sigma, x) for x in dbio_eff])
            di_pA = di * 1e12

            stats = summarize(di_pA)
            if stats is None:
                st.error("No valid angles (rod too large for pore).")
            else:
                st.success(f"Possible ΔI range: **{stats['min']:.0f} – {stats['max']:.0f} pA**")
                st.info(f"Typical ΔI range (5–95%): **{stats['p5']:.0f} – {stats['p95']:.0f} pA**")
                st.caption(f"Valid angles used: {stats['count']:,} | Typical ΔI ≈ {stats['median']:.0f} pA")

                st.write(f"Aligned (θ=0°) ΔI ≈ {di_pA[0]:.0f} pA")
                st.write(f"Side-on (θ=90°) ΔI ≈ {di_pA[-1]:.0f} pA")

# =========================
# TAB 3: Live Animation
# =========================
if page == "Live Animation":

    st.subheader("Live nanopore translocation animation")

    st.markdown("### Inputs")
    col1, col2 = st.columns(2)

    with col1:
        i0_nA = st.number_input("Open pore current i₀ (nA)", value=24.0, step=0.5, key="anim_i0")
        V = st.number_input("Voltage V (V)", value=0.300, step=0.010, format="%.3f", key="anim_V")
        sigma = st.number_input("Conductivity σ (S/m)", value=11.51, step=0.01, key="anim_sigma")

    with col2:
        L_nm = st.number_input("Pore length L (nm)", value=7.0, step=0.5, key="anim_L")
        occupancy = st.slider("Occupancy factor", 0.3, 1.0, 1.0, 0.05, key="anim_occ")
        speed = st.slider("Animation speed", 0.5, 3.0, 1.4, 0.1, key="anim_speed")

    st.markdown("### Biomolecule settings")
    A_nm = st.number_input("Axis A (nm)", value=14.0, step=0.5, key="anim_A")
    B_nm = st.number_input("Axis B (nm)", value=4.0, step=0.5, key="anim_B")
    C_nm = st.number_input("Axis C (nm)", value=4.0, step=0.5, key="anim_C")

    event_type = st.selectbox(
        "Event type",
        ["Centered translocation", "Bump", "Adsorption / rim interaction"],
        key="anim_event"
    )

    show_labels = st.checkbox("Show labels", value=True, key="anim_labels")

    i0_A = i0_nA * 1e-9
    L_m = L_nm * 1e-9
    d_m = pore_d_from_i0(i0_A, L_m, V, sigma)

    if not np.isfinite(d_m):
        st.error("Could not infer pore diameter from the given i₀, V, σ, and L.")
    else:
        st.info(f"Inferred pore diameter d ≈ **{d_m*1e9:.2f} nm**")

        pore_radius = d_m / 2.0
        a = (A_nm / 2) * 1e-9
        b = (B_nm / 2) * 1e-9
        c = (C_nm / 2) * 1e-9

        frames = 120

        if event_type == "Centered translocation":
            y_positions = np.linspace(1.18, -1.18, frames)
            x_positions = np.zeros(frames)
        elif event_type == "Bump":
            half = frames // 2
            y_in = np.linspace(1.18, 0.18, half)
            y_out = np.linspace(0.18, 1.18, frames - half)
            y_positions = np.concatenate([y_in, y_out])
            x_positions = np.linspace(0.28, 0.10, frames)
        else:
            y_positions = np.linspace(0.45, -0.05, frames)
            x_positions = np.ones(frames) * 0.16

        angles = np.linspace(0, 2*np.pi, frames)
        nvecs = np.array([
            [
                0.65*np.cos(t),
                0.30*np.sin(t),
                np.sqrt(max(0, 1 - (0.65*np.cos(t))**2 - (0.30*np.sin(t))**2))
            ]
            for t in angles
        ])

        currents = []
        deltaI_pA = []
        blocked_areas_nm2 = []

        for i in range(frames):
            nvec = nvecs[i]
            Aproj = projected_area_ellipsoid(a, b, c, nvec.reshape(1, 3))[0]
            dbio_eff = 2 * np.sqrt(Aproj / np.pi)
            rbio_eff = 0.5 * dbio_eff * occupancy

            offset_scene = abs(x_positions[i]) * pore_radius / 0.18

            if event_type == "Centered translocation":
                offset = 0.0
            elif event_type == "Bump":
                offset = min(pore_radius + rbio_eff * 0.7, offset_scene + 0.6 * rbio_eff)
            else:
                offset = min(pore_radius + rbio_eff * 0.2, max(pore_radius - 0.8 * rbio_eff, offset_scene))

            A_blocked = circle_overlap_area(pore_radius, rbio_eff, offset)
            di = delta_i_from_blocked_area(i0_A, d_m, L_m, V, sigma, A_blocked)

            if not np.isfinite(di):
                di = 0.0

            I_now = i0_A - di
            currents.append(I_now / i0_A)   # normalized current
            deltaI_pA.append(di * 1e12)
            blocked_areas_nm2.append(A_blocked * 1e18)

        currents = np.array(currents)
        deltaI_pA = np.array(deltaI_pA)
        blocked_areas_nm2 = np.array(blocked_areas_nm2)

        def build_scene(frame_idx):
            y = float(y_positions[frame_idx])
            x = float(x_positions[frame_idx])

            fig = go.Figure()
            fig.update_layout(
                paper_bgcolor="#0a1730",
                plot_bgcolor="#0a1730",
                margin=dict(l=10, r=10, t=10, b=10),
                height=620,
                xaxis=dict(range=[-1, 1], visible=False),
                yaxis=dict(range=[-1.35, 1.35], visible=False, scaleanchor="x", scaleratio=1),
                showlegend=False,
            )

            membrane_color = "rgba(52, 123, 219, 0.25)"
            glow_color = "rgba(56, 189, 248, 0.45)"

            fig.add_shape(type="rect", x0=-1, x1=1, y0=-0.10, y1=0.10,
                          fillcolor=membrane_color, line=dict(width=0))
            fig.add_shape(type="rect", x0=-1, x1=1, y0=-0.16, y1=-0.10,
                          fillcolor=glow_color, line=dict(width=0))
            fig.add_shape(type="rect", x0=-1, x1=1, y0=0.10, y1=0.16,
                          fillcolor=glow_color, line=dict(width=0))

            # pore opening
            fig.add_shape(type="rect", x0=-0.035, x1=0.035, y0=-0.16, y1=0.16,
                          fillcolor="#0a1730", line=dict(width=0))
            fig.add_shape(type="rect", x0=-0.020, x1=0.020, y0=-0.16, y1=0.16,
                          fillcolor="rgba(34,211,238,0.85)", line=dict(width=0))

            if show_labels:
                fig.add_annotation(x=0, y=1.25, text="cis", showarrow=False,
                                   font=dict(color="#94a3b8", size=14))
                fig.add_annotation(x=0, y=-1.25, text="trans", showarrow=False,
                                   font=dict(color="#94a3b8", size=14))
                fig.add_annotation(x=-0.88, y=1.05, text="+", showarrow=False,
                                   font=dict(color="#fb7185", size=18))
                fig.add_annotation(x=-0.88, y=-1.05, text="−", showarrow=False,
                                   font=dict(color="#60a5fa", size=18))

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=34, color="#8b5cf6", opacity=0.14),
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=20, color="#8b5cf6", opacity=0.90),
                hoverinfo="skip"
            ))

            return fig

        def build_trace(frame_idx):
            fig = go.Figure()
            t = np.arange(frame_idx + 1)
            y = currents[:frame_idx + 1]

            fig.add_trace(go.Scatter(
                x=t, y=y,
                mode="lines",
                line=dict(color="#38bdf8", width=3)
            ))

            fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.25)")
            fig.add_trace(go.Scatter(
                x=[frame_idx], y=[currents[frame_idx]],
                mode="markers",
                marker=dict(size=10, color="#f43f5e"),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[frame_idx, frame_idx],
                y=[currents[frame_idx], 1.0],
                mode="lines",
                line=dict(color="#f43f5e", width=2),
                showlegend=False
            ))

            fig.update_layout(
                paper_bgcolor="#0a1730",
                plot_bgcolor="#0a1730",
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                xaxis=dict(title="Time", color="#94a3b8", showgrid=False),
                yaxis=dict(title="Normalized current", color="#94a3b8", range=[0.75, 1.02],
                           showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                showlegend=False,
            )

            if show_labels:
                fig.add_annotation(x=max(4, frame_idx * 0.12), y=1.0 + 0.006,
                                   text="I₀", showarrow=False,
                                   font=dict(color="#cbd5e1", size=14))
                fig.add_annotation(x=frame_idx + 1, y=currents[frame_idx] - 0.012,
                                   text="I", showarrow=False,
                                   font=dict(color="#fda4af", size=14))
                fig.add_annotation(x=frame_idx + 4, y=(1.0 + currents[frame_idx]) / 2,
                                   text="ΔI", showarrow=False,
                                   font=dict(color="#fda4af", size=14))
            return fig

        left, right = st.columns([1.7, 1])

        with left:
            scene_slot = st.empty()
        with right:
            trace_slot = st.empty()
            metric_slot = st.empty()

        frame_idx = 0
        scene_fig = build_scene(frame_idx)
        trace_fig = build_trace(frame_idx)

        scene_slot.plotly_chart(scene_fig, use_container_width=True, config={"displayModeBar": False})
        trace_slot.plotly_chart(trace_fig, use_container_width=True, config={"displayModeBar": False})
        metric_slot.markdown(
            f"**I₀ = {i0_nA:.2f} nA**  |  "
            f"**I = {currents[0]*i0_nA:.2f} nA**  |  "
            f"**ΔI = {deltaI_pA[0]:.0f} pA**"
        )

        st.markdown("### Save / Export")
        col_save1, col_save2 = st.columns(2)

        with col_save1:
            save_current_png = st.button("Save current frame as PNG")
        with col_save2:
            run_anim = st.button("Run animation")

        if save_current_png:
            try:
                scene_fig.write_image("nanopore_scene_frame.png", scale=2)
                trace_fig.write_image("current_trace_frame.png", scale=2)
                st.success("Saved PNGs: nanopore_scene_frame.png and current_trace_frame.png")
            except Exception as e:
                st.error(f"Could not save PNG. Install kaleido with: pip install kaleido. Error: {e}")

        if run_anim:
            for i in range(frames):
                scene_fig = build_scene(i)
                trace_fig = build_trace(i)

                scene_slot.plotly_chart(scene_fig, use_container_width=True, config={"displayModeBar": False})
                trace_slot.plotly_chart(trace_fig, use_container_width=True, config={"displayModeBar": False})
                metric_slot.markdown(
                    f"**I₀ = {i0_nA:.2f} nA**  |  "
                    f"**I = {currents[i]*i0_nA:.2f} nA**  |  "
                    f"**ΔI = {deltaI_pA[i]:.0f} pA**"
                )
                time.sleep(max(0.01, 0.04 / speed))
