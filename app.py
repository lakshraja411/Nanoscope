import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from scipy.stats import gaussian_kde

st.set_page_config(
    page_title="NanoScope",
    layout="wide",
    page_icon="🔬"
)

st.title("🔬 NanoScope")
st.caption("Size and Current Drop Predictor")




# =========================
# Shared nanopore / geometry functions
# =========================
def pore_d_from_i0(i0_A, L_m, V_V, sigma_Sm):
    """
    Infer pore diameter from open pore current using:
    i0 = sigma*V / ( 4L/(pi d^2) + 1/d )
    """
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
    R = pore radius
    r = blocker radius
    x = center offset
    """
    if x >= R + r:
        return 0.0

    if x <= abs(R - r):
        return np.pi * min(R, r) ** 2

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
    if A_blocked <= 0:
        return 0.0
    return 2.0 * np.sqrt(A_blocked / np.pi)


def delta_i_from_blocked_area(i0_A, d_m, L_m, V_V, sigma_Sm, A_blocked):
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
    """
    nvec: shape (N,3) or (3,)
    returns projected area(s) in m^2
    """
    nvec = np.atleast_2d(nvec)
    nx, ny, nz = nvec[:, 0], nvec[:, 1], nvec[:, 2]
    denom = np.sqrt((a * nx)**2 + (b * ny)**2 + (c * nz)**2)
    return (np.pi * a * b * c) / denom


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

    fig, ax = plt.subplots(figsize=(7, 5))

    if y_in_nA:
        ax.plot(
            V,
            I * 1e9,
            linewidth=2,
            label="IV data",
        )
        ax.set_ylabel("Current (nA)")

        if show_fit and fit_G is not None:
            ax.plot(
                V,
                (fit_G * V) * 1e9,
                color="blue",
                linestyle="--",
                linewidth=2,
                label="Fit"
            )

    else:
        ax.plot(
            V,
            I,
            linewidth=2,
            label="IV data"
        )
        ax.set_ylabel("Current (A)")

        if show_fit and fit_G is not None:
            ax.plot(
                V,
                fit_G * V,
                color="blue",
                linestyle="-",
                linewidth=1.8,
                label="Fit"
            )

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
    return np.sqrt((4 * L_m * G_S) / (sigma_Sm * np.pi))


def diameter_cyl_with_access(G_S, sigma_Sm, L_m):
    if G_S <= 0 or sigma_Sm <= 0 or L_m <= 0:
        return np.nan
    a = (np.pi * sigma_Sm) / G_S
    b = np.pi / 2.0
    disc = b * b + 4 * a * L_m
    if disc <= 0 or a <= 0:
        return np.nan
    r = (b + np.sqrt(disc)) / (2 * a)
    return 2 * r


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
        d = np.sqrt((4 * Ls * Gs) / (ss * np.pi))

    d = d[np.isfinite(d) & (d > 0)]
    if d.size < 2000:
        return np.nan, np.nan, (np.nan, np.nan), 0

    d_nm = d * 1e9
    mean = float(np.mean(d_nm))
    std = float(np.std(d_nm, ddof=1))
    lo, hi = np.percentile(d_nm, [2.5, 97.5])
    return mean, std, (float(lo), float(hi)), int(d_nm.size)


# =========================
# Conical model
# =========================
def G_conical_single(r, K, L, theta):
    num = 4 * np.pi * r * (r + L * np.tan(theta))
    den = 4 * L + np.pi * (2 * r + L * np.tan(theta))
    return K * (num / den)


def solve_tip_radius_brentq(G_single, K, L, theta, r_lo=0.5e-9, r_hi=30000e-9):
    def f(r):
        return G_conical_single(r, K, L, theta) - G_single

    f_lo, f_hi = f(r_lo), f(r_hi)

    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        raise ValueError("Non-finite values in root bracketing. Check K, L, theta, and G_single.")

    if f_lo == 0:
        return r_lo
    if f_hi == 0:
        return r_hi

    if f_lo * f_hi > 0:
        G_lo = G_conical_single(r_lo, K, L, theta)
        G_hi = G_conical_single(r_hi, K, L, theta)
        raise ValueError(
            "Root not bracketed in [0.5 nm, 30000 nm].\n"
            f"G_single target = {G_single:.6e} S\n"
            f"G(0.5 nm) = {G_lo:.6e} S\n"
            f"G(30000 nm) = {G_hi:.6e} S\n"
            "Check n, K, L, theta, or your conductance fit."
        )

    return brentq(f, r_lo, r_hi, maxiter=2000)


def solve_tip_radius_fsolve(G_single, K, L, theta, r0=20e-9):
    def eq_r(r):
        return G_conical_single(r, K, L, theta) - G_single

    r = fsolve(eq_r, r0)[0]
    if not np.isfinite(r) or r <= 0:
        raise ValueError("fsolve returned an invalid radius.")
    return r


def compute_conical_radius(V, I, n, K, L, theta, method, eps=0.005, window=0.05, min_points=6):
    """
    method:
        - 'auto_window_brentq'
        - 'polyfit_fsolve'
    """
    if method == "auto_window_brentq":
        V_lin, I_lin, used_window = pick_linear_region_auto(V, I, eps=eps, window=window, min_points=min_points)
        G_total = slope_through_origin(V_lin, I_lin)
        intercept = 0.0
        meta = {
            "fit_method": "through-origin near-zero window",
            "used_window": used_window,
            "points_used": len(V_lin),
        }
        r = None

    elif method == "polyfit_fsolve":
        mask = np.abs(V) > eps
        if np.sum(mask) < 3:
            raise ValueError("Not enough points after excluding near-zero voltages.")
        G_total, intercept = slope_with_intercept(V[mask], I[mask])
        meta = {
            "fit_method": "polyfit with intercept",
            "used_window": None,
            "points_used": int(np.sum(mask)),
        }
        r = None

    else:
        raise ValueError("Unknown conical fit method.")

    if not np.isfinite(G_total) or G_total <= 0:
        raise ValueError(f"Invalid conductance extracted from IV: G_total = {G_total:.6e} S")

    G_single = G_total / n
    if not np.isfinite(G_single) or G_single <= 0:
        raise ValueError(f"Invalid single-pore conductance: G_single = {G_single:.6e} S")

    if method == "auto_window_brentq":
        r = solve_tip_radius_brentq(G_single, K, L, theta)
    elif method == "polyfit_fsolve":
        r = solve_tip_radius_fsolve(G_single, K, L, theta)

    return {
        "radius_m": r,
        "diameter_m": 2 * r,
        "G_total_S": G_total,
        "G_single_S": G_single,
        "intercept_A": intercept,
        "meta": meta,
    }


# =========================
# NaCl conductivity map
# =========================
K_MAP = {
    "1 mM NaCl": 0.02,
    "10 mM NaCl": 0.14,
    "100 mM NaCl": 1.4,
    "1 M NaCl": 8.97,
}

CONICAL_STAGE_CONFIG = {
    "1 mM NaCl (auto-window, brentq)": {
        "label": "1 mM NaCl",
        "K": K_MAP["1 mM NaCl"],
        "method": "auto_window_brentq",
    },
    "10 mM NaCl (auto-window, brentq)": {
        "label": "10 mM NaCl",
        "K": K_MAP["10 mM NaCl"],
        "method": "auto_window_brentq",
    },
    "100 mM NaCl (auto-window, brentq)": {
        "label": "100 mM NaCl",
        "K": K_MAP["100 mM NaCl"],
        "method": "auto_window_brentq",
    },
    "1 M NaCl (polyfit + intercept, fsolve)": {
        "label": "1 M NaCl",
        "K": K_MAP["1 M NaCl"],
        "method": "polyfit_fsolve",
    },
    "After functionalization / antibody (10 mM, auto-window, brentq)": {
        "label": "10 mM NaCl",
        "K": K_MAP["10 mM NaCl"],
        "method": "auto_window_brentq",
    },
    "Antibody/biosensing IV (10 mM, polyfit + intercept, fsolve)": {
        "label": "10 mM NaCl",
        "K": K_MAP["10 mM NaCl"],
        "method": "polyfit_fsolve",
    },
}

# =========================
# Navigation
# =========================
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Size Calculator",
        "ΔI Range Explorer",
    ]
)

# =========================
# Home
# =========================
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
# TAB 1: Size Calculator
# =========================
if page == "Size Calculator":
    st.subheader("1) Upload IV file")
    up = st.file_uploader("Upload .csv or .txt", type=["csv", "txt"], key="iv_upload")

    df_iv = None
    if up is not None:
        raw = up.read()
        name = up.name.lower()

        if name.endswith(".txt"):
            s = raw.decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(s), skipinitialspace=True)

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

            df["Voltage_V"] = df["Voltage2[V]"].apply(clean_voltage)
            df["Current_A"] = df["Current[A]"].apply(clean_current)
            df_iv = df[["Sweep #", "Voltage_V", "Current_A"]].copy()

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
    mode = st.radio("Mode", ["CBD (cylindrical)", "Conical"], horizontal=False)
    plot_nA = st.checkbox("Plot current in nA", value=True, key="plot_nA_tab1")

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

                    plot_iv_line(V, I, title="I–V Curve with Global Fit", y_in_nA=plot_nA, show_fit=True, fit_G=G_S)
                    st.info(f"Global slope (whole trace, through origin): **G = {G_from_iv_nS:.2f} nS**")

        st.markdown("### Inputs (with uncertainties)")
        default_G = float(G_from_iv_nS) if (G_from_iv_nS is not None and np.isfinite(G_from_iv_nS)) else 175.0

        G_nS = st.number_input("Conductance G (nS)", value=default_G, step=1.0, format="%.3f")
        dG_nS = st.number_input("± error in G (nS)", value=1.0, step=0.1, format="%.3f")

        sigma = st.number_input("Conductivity σ (S/m)", value=11.5, step=0.1, format="%.4f")
        dsigma = st.number_input("± error in σ (S/m)", value=0.2, step=0.05, format="%.4f")

        L_nm = st.number_input("Pore length L (nm)", value=7.0, step=0.5, format="%.3f")
        dL_nm = st.number_input("± error in L (nm)", value=0.5, step=0.1, format="%.3f")

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

    else:
        st.subheader("Conical pore (tip radius) from IV")

        stage = st.selectbox(
            "Which conical scenario?",
            list(CONICAL_STAGE_CONFIG.keys())
        )

        cfg = CONICAL_STAGE_CONFIG[stage]

        st.markdown("### Geometry + experiment settings")
        n = st.number_input("Number of pores n", value=250, step=1)
        theta_deg = st.number_input("Half cone angle θ (deg)", value=12.6, step=0.1, format="%.3f")
        L_nm = st.number_input("Pore length L (nm)", value=750.0, step=5.0, format="%.2f")

        K = st.number_input("Conductivity K (S/m)", value=float(cfg["K"]), step=0.001, format="%.4f")

        if "1 mM" in stage:
            st.warning(
                "At 1 mM ionic strength, double layer effects can become important, "
                "so the simple conical conductance model may give an apparent radius rather than a true geometric radius."
            )

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
                plot_iv_line(V, I, title="I–V Curve", y_in_nA=plot_nA)

                theta = np.deg2rad(theta_deg)
                L = L_nm * 1e-9

                st.markdown("### Conductance extraction + solve")

                if cfg["method"] == "auto_window_brentq":
                    eps = st.number_input("Exclude |V| < eps (V)", value=0.005, step=0.001, format="%.4f")
                    window = st.number_input("Start window |V| < window (V)", value=0.05, step=0.01, format="%.3f")
                    min_points = st.number_input("Min points for fit", value=6, step=1)

                    if st.button("Compute conical radius"):
                        try:
                            result = compute_conical_radius(
                                V, I, n, K, L, theta,
                                method="auto_window_brentq",
                                eps=float(eps),
                                window=float(window),
                                min_points=int(min_points),
                            )

                            st.write(f"Fit method: {result['meta']['fit_method']}")
                            st.write(f"Used window = {result['meta']['used_window']:.4f} V")
                            st.write(f"Points used = {result['meta']['points_used']}")
                            st.write(f"G_total  = {result['G_total_S']:.6e} S")
                            st.write(f"G_single = {result['G_single_S']:.6e} S")

                            st.success(f"Estimated tip radius r ≈ {result['radius_m'] * 1e9:.2f} nm")
                            st.info(f"Estimated tip diameter d ≈ {result['diameter_m'] * 1e9:.2f} nm")

                        except Exception as e:
                            st.error(str(e))

                elif cfg["method"] == "polyfit_fsolve":
                    eps = st.number_input("Ignore |V| < eps (V)", value=0.05, step=0.01, format="%.3f")

                    if st.button("Compute conical radius"):
                        try:
                            result = compute_conical_radius(
                                V, I, n, K, L, theta,
                                method="polyfit_fsolve",
                                eps=float(eps),
                            )

                            st.write(f"Fit method: {result['meta']['fit_method']}")
                            st.write(f"Points used = {result['meta']['points_used']}")
                            st.write(f"G_total  = {result['G_total_S']:.6e} S")
                            st.write(f"G_single = {result['G_single_S']:.6e} S")
                            st.write(f"Intercept = {result['intercept_A']:.3e} A")

                            st.success(f"Estimated tip radius r ≈ {result['radius_m'] * 1e9:.2f} nm")
                            st.info(f"Estimated tip diameter d ≈ {result['diameter_m'] * 1e9:.2f} nm")

                        except Exception as e:
                            st.error(str(e))

                st.markdown("---")
                st.subheader("Compare radius across NaCl concentrations")

                compare_eps = st.number_input(
                    "Comparison eps (exclude |V| < eps)", value=0.005, step=0.001, format="%.4f", key="compare_eps"
                )
                compare_window = st.number_input(
                    "Comparison start window", value=0.05, step=0.01, format="%.3f", key="compare_window"
                )
                compare_min_points = st.number_input(
                    "Comparison min points", value=6, step=1, key="compare_min_points"
                )

                if st.button("Compare radii across 1 mM / 10 mM / 100 mM / 1 M"):
                    rows = []
                    for label, K_val in K_MAP.items():
                        try:
                            if label == "1 M NaCl":
                                res = compute_conical_radius(
                                    V, I, n, K_val, L, theta,
                                    method="polyfit_fsolve",
                                    eps=max(float(compare_eps), 0.05),
                                )
                            else:
                                res = compute_conical_radius(
                                    V, I, n, K_val, L, theta,
                                    method="auto_window_brentq",
                                    eps=float(compare_eps),
                                    window=float(compare_window),
                                    min_points=int(compare_min_points),
                                )

                            rows.append({
                                "Condition": label,
                                "K (S/m)": K_val,
                                "G_total (S)": res["G_total_S"],
                                "G_single (S)": res["G_single_S"],
                                "Radius (nm)": res["radius_m"] * 1e9,
                                "Diameter (nm)": res["diameter_m"] * 1e9,
                                "Status": "OK",
                            })
                        except Exception as e:
                            rows.append({
                                "Condition": label,
                                "K (S/m)": K_val,
                                "G_total (S)": np.nan,
                                "G_single (S)": np.nan,
                                "Radius (nm)": np.nan,
                                "Diameter (nm)": np.nan,
                                "Status": str(e),
                            })

                    df_compare = pd.DataFrame(rows)
                    st.dataframe(df_compare, use_container_width=True)

# =========================
# TAB 2: ΔI Range Explorer
# =========================
if page == "ΔI Range Explorer":
    st.subheader("ΔI Range Explorer")

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
        st.info(f"Inferred pore diameter d ≈ **{d_m * 1e9:.2f} nm**")
    else:
        st.error("Could not infer pore diameter. Check that i0, V, σ, L are > 0.")

    st.markdown("---")
    st.markdown("### Choose biomarker shape model")
    model = st.selectbox("Model", ["Sphere", "Ellipsoid", "Rod / spherocylinder"])

    # ---------- Sphere ----------
    if model.startswith("Sphere"):
        dbio_nm = st.number_input("Biomarker diameter d_bio (nm)", value=6.0, step=0.2)
        if st.button("Compute ΔI (sphere)"):
            dbio_m = dbio_nm * 1e-9 * occupancy
            di = delta_i(i0_A, d_m, L_m, V, sigma, dbio_m)
            if np.isfinite(di):
                st.success(f"ΔI ≈ **{di * 1e12:.0f} pA**")
            else:
                st.error("This dbio is too large for the inferred pore diameter.")

    # ---------- Ellipsoid ----------
    elif model.startswith("Ellipsoid"):
        A_nm = st.number_input("Axis A (nm) (long)", value=14.0, step=0.5)
        B_nm = st.number_input("Axis B (nm)", value=4.0, step=0.5)
        C_nm = st.number_input("Axis C (nm)", value=4.0, step=0.5)
        N = int(st.number_input("Orientation samples", value=50000, step=5000))
        seed = int(st.number_input("Random seed", value=7, step=1))

        event_mode = st.selectbox(
            "Event model",
            [
                "Centered translocation",
                "Bump / partial entry",
                "Adsorption / rim interaction",
                "Combined mixture"
            ]
        )

        if event_mode == "Combined mixture":
            st.markdown("### Event mixture weights")
            mix_centered = st.slider("Centered fraction", 0.0, 1.0, 0.4, 0.05)
            mix_bump = st.slider("Bump fraction", 0.0, 1.0, 0.4, 0.05)
            mix_ads = st.slider("Adsorption fraction", 0.0, 1.0, 0.2, 0.05)

            mix_total = mix_centered + mix_bump + mix_ads
            if mix_total > 0:
                w_centered = mix_centered / mix_total
                w_bump = mix_bump / mix_total
                w_ads = mix_ads / mix_total
            else:
                w_centered, w_bump, w_ads = 1.0, 0.0, 0.0

            st.caption(
                f"Normalized weights: centered = {w_centered:.2f}, "
                f"bump = {w_bump:.2f}, adsorption = {w_ads:.2f}"
            )

        st.markdown("### Noise model")
        add_noise = st.checkbox("Add Gaussian measurement noise", value=True)
        noise_pA = st.slider("Noise SD (pA)", 0.0, 1000.0, 20.0, 1.0)

        hist_source = st.selectbox(
            "Histogram source",
            ["Noisy prediction", "Theoretical prediction", "Both"]
        )

        show_component_curves = st.checkbox("Show component smooth curves", value=True)
        kde_bandwidth = st.slider("Smoothness (KDE bandwidth factor)", 0.05, 2.0, 1.0, 0.05)

        if st.button("Compute ΔI range (ellipsoid)"):
            rng = np.random.default_rng(seed)
            a = (A_nm / 2) * 1e-9
            b = (B_nm / 2) * 1e-9
            c = (C_nm / 2) * 1e-9

            pore_radius = d_m / 2.0

            nvec = random_unit_vectors(N, rng)
            Aproj = projected_area_ellipsoid(a, b, c, nvec)
            dbio_eff = 2 * np.sqrt(Aproj / np.pi)
            rbio_eff = dbio_eff / 2.0

            di_centered = []
            di_bump = []
            di_ads = []

            offset_list_nm = []
            blocked_area_list_nm2 = []
            event_labels = []

            for r_eff in rbio_eff:
                if event_mode == "Combined mixture":
                    event_type = rng.choice(
                        ["Centered translocation", "Bump / partial entry", "Adsorption / rim interaction"],
                        p=[w_centered, w_bump, w_ads]
                    )
                else:
                    event_type = event_mode

                if event_type == "Centered translocation":
                    offset = 0.0
                elif event_type == "Bump / partial entry":
                    offset = rng.uniform(
                        max(0.0, pore_radius - 0.3 * r_eff),
                        pore_radius + 0.8 * r_eff
                    )
                else:
                    offset = rng.uniform(
                        max(0.0, pore_radius - 0.8 * r_eff),
                        pore_radius + 0.2 * r_eff
                    )

                A_blocked = circle_overlap_area(pore_radius, r_eff * occupancy, offset)
                di_val = delta_i_from_blocked_area(i0_A, d_m, L_m, V, sigma, A_blocked)

                if np.isfinite(di_val):
                    di_pA_val = di_val * 1e12

                    if event_type == "Centered translocation":
                        di_centered.append(di_pA_val)
                    elif event_type == "Bump / partial entry":
                        di_bump.append(di_pA_val)
                    else:
                        di_ads.append(di_pA_val)

                    offset_list_nm.append(offset * 1e9)
                    blocked_area_list_nm2.append(A_blocked * 1e18)
                    event_labels.append(event_type)

            di_centered = np.asarray(di_centered)
            di_bump = np.asarray(di_bump)
            di_ads = np.asarray(di_ads)

            di_theory_all = np.concatenate([
                di_centered if di_centered.size else np.array([]),
                di_bump if di_bump.size else np.array([]),
                di_ads if di_ads.size else np.array([])
            ])

            if add_noise and noise_pA > 0:
                di_centered_noisy = di_centered + rng.normal(0.0, noise_pA, size=len(di_centered)) if di_centered.size else np.array([])
                di_bump_noisy = di_bump + rng.normal(0.0, noise_pA, size=len(di_bump)) if di_bump.size else np.array([])
                di_ads_noisy = di_ads + rng.normal(0.0, noise_pA, size=len(di_ads)) if di_ads.size else np.array([])
            else:
                di_centered_noisy = di_centered.copy()
                di_bump_noisy = di_bump.copy()
                di_ads_noisy = di_ads.copy()

            di_noisy_all = np.concatenate([
                di_centered_noisy if di_centered_noisy.size else np.array([]),
                di_bump_noisy if di_bump_noisy.size else np.array([]),
                di_ads_noisy if di_ads_noisy.size else np.array([])
            ])

            if hist_source == "Theoretical prediction":
                di_for_stats = di_theory_all
            elif hist_source == "Noisy prediction":
                di_for_stats = di_noisy_all
            else:
                di_for_stats = di_noisy_all

            stats = summarize(di_for_stats)

            if stats is None:
                st.error("No valid events were generated.")
            else:
                st.success(f"Possible ΔI range: **{stats['min']:.0f} – {stats['max']:.0f} pA**")
                st.info(f"Typical ΔI range (5–95%): **{stats['p5']:.0f} – {stats['p95']:.0f} pA**")
                st.caption(f"Valid simulated events: {stats['count']:,} | Median ΔI ≈ {stats['median']:.0f} pA")

                diag_df = pd.DataFrame({
                    "offset (nm)": offset_list_nm,
                    "blocked area (nm²)": blocked_area_list_nm2,
                    "event type": event_labels
                })
                st.dataframe(diag_df.head(50), use_container_width=True)

                candidates = []
                if di_theory_all.size:
                    candidates.append(di_theory_all)
                if di_noisy_all.size:
                    candidates.append(di_noisy_all)

                global_min = min(np.min(x) for x in candidates)
                global_max = max(np.max(x) for x in candidates)
                x_grid = np.linspace(global_min, global_max, 1000)

                fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

                if hist_source in ["Theoretical prediction", "Both"] and di_theory_all.size:
                    ax.hist(
                        di_theory_all,
                        bins=60,
                        density=True,
                        alpha=0.35 if hist_source == "Both" else 0.50,
                        edgecolor="black",
                        linewidth=0.5,
                        label="Theoretical data"
                    )

                if hist_source in ["Noisy prediction", "Both"] and di_noisy_all.size:
                    ax.hist(
                        di_noisy_all,
                        bins=60,
                        density=True,
                        alpha=0.35 if hist_source == "Both" else 0.50,
                        edgecolor="black",
                        linewidth=0.5,
                        label="Noisy data"
                    )

                def kde_curve(data):
                    if len(data) < 2:
                        return None
                    kde = gaussian_kde(data, bw_method=kde_bandwidth)
                    return kde(x_grid)

                if hist_source in ["Theoretical prediction", "Both"] and di_theory_all.size > 1:
                    y_total_theory = kde_curve(di_theory_all)
                    ax.plot(
                        x_grid,
                        y_total_theory,
                        color="blue",
                        linewidth=2.0,
                        label="Total theoretical fit"
                    )

                if hist_source in ["Noisy prediction", "Both"] and di_noisy_all.size > 1:
                    y_total_noisy = kde_curve(di_noisy_all)
                    ax.plot(
                        x_grid,
                        y_total_noisy,
                        color="black",
                        linewidth=2.0,
                        label="Total noisy fit"
                    )

                if show_component_curves:
                    comp_map = []
                    if hist_source == "Theoretical prediction":
                        comp_map = [
                            ("Centered component", di_centered),
                            ("Bump component", di_bump),
                            ("Adsorption component", di_ads),
                        ]
                    elif hist_source == "Noisy prediction":
                        comp_map = [
                            ("Centered component", di_centered_noisy),
                            ("Bump component", di_bump_noisy),
                            ("Adsorption component", di_ads_noisy),
                        ]
                    else:
                        comp_map = [
                            ("Centered component", di_centered_noisy),
                            ("Bump component", di_bump_noisy),
                            ("Adsorption component", di_ads_noisy),
                        ]

                    for label, comp_data in comp_map:
                        if len(comp_data) > 1:
                            y_comp = kde_curve(comp_data)
                            ax.plot(
                                x_grid,
                                y_comp,
                                linestyle="--",
                                linewidth=2.0,
                                label=label
                            )

                ax.set_xlabel("ΔI (pA"))
                ax.set_ylabel("Density", fontsize=12)
                ax.set_title("Predicted ΔI Histogram")

                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                plt.close(fig)

                if event_mode == "Combined mixture":
                    comp_stats_df = pd.DataFrame([
                        {
                            "Component": "Centered",
                            "Count": len(di_centered),
                            "Median ΔI (pA)": np.median(di_centered) if len(di_centered) else np.nan,
                        },
                        {
                            "Component": "Bump",
                            "Count": len(di_bump),
                            "Median ΔI (pA)": np.median(di_bump) if len(di_bump) else np.nan,
                        },
                        {
                            "Component": "Adsorption",
                            "Count": len(di_ads),
                            "Median ΔI (pA)": np.median(di_ads) if len(di_ads) else np.nan,
                        },
                    ])
                    st.dataframe(comp_stats_df, use_container_width=True)

                if add_noise and noise_pA > 0:
                    st.caption(
                        "Gaussian noise is added after each physical event population is simulated, "
                        "so the combined histogram preserves the centered / bump / adsorption trends."
                    )

    # ---------- Rod / spherocylinder ----------
    else:
        Lrod_nm = st.number_input("Rod length L_rod (nm)", value=50.0, step=5.0)
        Drod_nm = st.number_input("Rod diameter D_rod (nm)", value=6.0, step=0.5)
        n_angles = int(st.number_input("Angle steps", value=361, step=60))

        if st.button("Compute ΔI range (rod)"):
            Lrod = Lrod_nm * 1e-9
            Drod = Drod_nm * 1e-9

            theta = np.linspace(0, np.pi / 2, n_angles)
            Atheta = (Drod * Lrod * np.abs(np.sin(theta))) + (np.pi * Drod**2 / 4.0)

            dbio_eff = 2 * np.sqrt(Atheta / np.pi) * occupancy
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

                
