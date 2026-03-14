import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


# ======================================
# Page config
# ======================================

st.set_page_config(
    page_title="Metaheuristics – Benchmark",
    page_icon=None,
    layout="wide",
)

# ======================================
# Custom CSS
# ======================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* ---- Hide Streamlit default header/footer ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- App header banner ---- */
.app-header {
    background: linear-gradient(135deg, #0d0f14 0%, #131722 60%, #0f1520 100%);
    border-bottom: 1px solid #1e2535;
    padding: 2.5rem 2rem 1.8rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
}
.app-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.04em;
    margin: 0 0 0.3rem 0;
}
.app-header .subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    color: #64748b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.accent-bar {
    width: 40px;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    border-radius: 2px;
    margin-bottom: 1rem;
}

/* ---- Section labels ---- */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e2535, transparent);
}

/* ---- Config panel card ---- */
.config-card {
    background: #131722;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* ---- Selectbox, number inputs ---- */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #94a3b8 !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #0d0f14 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}

div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
}

/* ---- Slider ---- */
div[data-testid="stSlider"] > div > div > div {
    background: #3b82f6 !important;
}

/* ---- Buttons ---- */
div[data-testid="stButton"] > button {
    background: transparent;
    border: 1px solid #3b82f6;
    color: #3b82f6;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.6rem 1.8rem;
    border-radius: 6px;
    transition: all 0.2s ease;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background: #3b82f6;
    color: #0d0f14;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
}

/* ---- File uploader ---- */
div[data-testid="stFileUploader"] {
    background: #131722;
    border: 1px dashed #1e2535;
    border-radius: 10px;
    padding: 1rem;
}
div[data-testid="stFileUploader"] label {
    font-size: 0.78rem;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---- Alerts / messages ---- */
div[data-testid="stSuccess"] {
    background: rgba(16, 185, 129, 0.08) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 8px !important;
    color: #10b981 !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
div[data-testid="stWarning"] {
    background: rgba(245, 158, 11, 0.08) !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
    border-radius: 8px !important;
    color: #f59e0b !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
div[data-testid="stInfo"] {
    background: rgba(59, 130, 246, 0.08) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    border-radius: 8px !important;
    color: #60a5fa !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
div[data-testid="stError"] {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 8px !important;
    color: #ef4444 !important;
}

/* ---- Metrics ---- */
div[data-testid="stMetric"] {
    background: #131722;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
}
div[data-testid="stMetric"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.5rem !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* ---- Latex formula box ---- */
.formula-box {
    background: #131722;
    border: 1px solid #1e2535;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0 1.5rem 0;
}

/* ---- Subheaders & Markdown headers ---- */
h2, h3 {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    color: #e2e8f0 !important;
}
h3 { font-size: 0.95rem !important; }

/* ---- Horizontal rule ---- */
hr {
    border: none;
    border-top: 1px solid #1e2535;
    margin: 2.5rem 0;
}

/* ---- Container (border) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #131722;
    border: 1px solid #1e2535 !important;
    border-radius: 12px;
    padding: 1rem;
}

/* ---- Progress bar ---- */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
    border-radius: 4px;
}

/* ---- Caption text ---- */
div[data-testid="stCaptionContainer"] p {
    color: #475569 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em;
}

/* ---- Markdown bold labels ---- */
strong {
    color: #94a3b8;
    font-weight: 600;
}

/* ---- Matplotlib plots background ---- */
.stPlot > div {
    background: transparent !important;
}

/* ---- Tab-like divider ---- */
.run-header {
    background: linear-gradient(90deg, #131722, #0d0f14);
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.run-header .run-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.run-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: rgba(59,130,246,0.12);
    color: #3b82f6;
    border: 1px solid rgba(59,130,246,0.3);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Benchmark functions
# ======================================

def f1(X): 
    return np.sum(X**2)

def f2(X): 
    return np.sum(np.abs(X)) + np.prod(np.abs(X))

def f5(X):
    xi1, xi = X[1:], X[:-1]
    return np.sum(100*(xi1**2 - xi)**2 + (1 - xi)**2)

def f7(X):
    D = len(X)
    return np.sum(np.arange(1, D+1) * X**4) + np.random.rand()

def f8(X): 
    return np.sum(-X * np.sin(np.sqrt(np.abs(X))))

def f9(X): 
    return np.sum(X**2 - 10*np.cos(2*np.pi*X) + 10)

def f11(X):
    D = len(X)
    indices = np.arange(1, D+1)
    return 1 + np.sum(X**2)/4000 - np.prod(np.cos(X/np.sqrt(indices)))

functions = {
    "F1": f1, 
    "F2": f2, 
    "F5": f5, 
    "F7": f7, 
    "F8": f8, 
    "F9": f9, 
    "F11": f11
}

expected_csv = {
    "F1": "Population_F1-UM.csv",
    "F2": "Population_F2-UM.csv",
    "F5": "Population_F5-UM.csv",
    "F7": "Population_F7-UM.csv",
    "F8": "Population_F8-MM.csv",
    "F9": "Population_F9-MM.csv",
    "F11": "Population_F11-MM.csv"
}

formulas = {
    "F1": r"f(x)=\sum_{i=1}^{D} x_i^2",
    "F2": r"f(x)=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|",
    "F5": r"f(x)=\sum_{i=1}^{D-1}\left[100(x_{i+1}^2-x_i)^2+(1-x_i)^2\right]",
    "F7": r"f(x)=\sum_{i=1}^{D} i\,x_i^4 + \text{rand}(0,1)",
    "F8": r"f(x)=\sum_{i=1}^{D}-x_i\sin(\sqrt{|x_i|})",
    "F9": r"f(x)=\sum_{i=1}^{D}\left[x_i^2-10\cos(2\pi x_i)+10\right]",
    "F11": r"f(x)=1+\frac{1}{4000}\sum_{i=1}^{D}x_i^2-\prod_{i=1}^{D}\cos\left(\frac{x_i}{\sqrt{i}}\right)"
}

# ======================================
# Matplotlib style helper
# ======================================

def apply_dark_style(fig, axes_list=None):
    """Apply consistent dark theme to matplotlib figures."""
    fig.patch.set_facecolor('#0d0f14')
    if axes_list is None:
        axes_list = fig.get_axes()
    for ax in axes_list:
        ax.set_facecolor('#131722')
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2535')
        ax.title.set_fontfamily('monospace')
        ax.title.set_fontsize(9)

def apply_dark_style_3d(fig, ax):
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#131722')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1e2535')
    ax.yaxis.pane.set_edgecolor('#1e2535')
    ax.zaxis.pane.set_edgecolor('#1e2535')
    ax.tick_params(colors='#64748b', labelsize=7)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.zaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    ax.title.set_fontfamily('monospace')

# ======================================
# Header
# ======================================

st.markdown("""
<div class="app-header">
    <div class="accent-bar"></div>
    <h1>Metaheuristics Benchmark</h1>
    <div class="subtitle">Particle Swarm Optimization — Benchmark Functions</div>
</div>
""", unsafe_allow_html=True)

# ======================================
# Configuration sidebar-style top panel
# ======================================

st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 2, 2])

with col_cfg1:
    func_name = st.selectbox("Function", list(functions.keys()))

with col_cfg2:
    D = st.number_input("Dimension (D)", 2, 1000, 30)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        low = st.number_input("Range min", min_value=-500.0, max_value=0.0, value=-100.0)
    with col_r2:
        high = st.number_input("Range max", min_value=0.0, max_value=500.0, value=100.0)

with col_cfg3:
    population_size = st.slider("Population Size", 10, 500, 100, step=10)
    runs = st.slider("Number of Runs", 1, 100, 10)

# ======================================
# Formula display
# ======================================

st.markdown('<div class="section-label">Function Definition</div>', unsafe_allow_html=True)
st.markdown('<div class="formula-box">', unsafe_allow_html=True)
st.latex(formulas[func_name])
st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# Import CSV Population & Evaluation
# ======================================

st.markdown('<div class="section-label">Population Evaluation</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    f"Upload CSV for {func_name} — expected: {expected_csv[func_name]}",
    type="csv"
)

if uploaded_file is not None:

    if uploaded_file.name != expected_csv[func_name]:
        st.error(f"Wrong file! Expected {expected_csv[func_name]}")
        st.stop()

    st.success(f"Correct file: {uploaded_file.name}")

    df = pd.read_csv(uploaded_file)
    pop = df.values[:, :int(D)]

    if st.button("Evaluate Population"):

        all_runs_fitness = []

        for r in range(runs):
            sample_size = min(population_size, len(pop))
            idx = np.random.choice(len(pop), sample_size, replace=False)
            sample = pop[idx]
            fitness_vals = np.array([functions[func_name](ind) for ind in sample])
            all_runs_fitness.append(fitness_vals)

        all_runs_fitness = np.concatenate(all_runs_fitness)

        st.markdown('<div class="section-label">Statistics</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Min (Best)", f"{np.min(all_runs_fitness):.4f}")
        with m2:
            st.metric("Max (Worst)", f"{np.max(all_runs_fitness):.4f}")
        with m3:
            st.metric("Mean / STD", f"{all_runs_fitness.mean():.4f} / {all_runs_fitness.std():.4f}")

        # 2D Contour Plot
        st.markdown('<div class="section-label">2D Contour Plot</div>', unsafe_allow_html=True)

        X = np.linspace(low, high, 100)
        Y = np.linspace(low, high, 100)
        Xg, Yg = np.meshgrid(X, Y)
        Z = np.zeros_like(Xg)

        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                vec = np.zeros(D)
                vec[0] = Xg[i, j]
                vec[1] = Yg[i, j]
                Z[i, j] = functions[func_name](vec)

        fig_contour, ax_contour = plt.subplots(figsize=(8, 5))
        apply_dark_style(fig_contour, [ax_contour])
        contour = ax_contour.contour(Xg, Yg, Z, levels=30, cmap="plasma")
        ax_contour.scatter(pop[:, 0], pop[:, 1], c="#f87171", s=10, alpha=0.7)
        ax_contour.set_title(f"Contour Plot — {func_name}")
        ax_contour.set_xlabel("x1")
        ax_contour.set_ylabel("x2")
        fig_contour.tight_layout()
        st.pyplot(fig_contour, use_container_width=True)

        # 3D Surface Plot
        st.markdown('<div class="section-label">3D Surface Plot</div>', unsafe_allow_html=True)

        fig_surface = plt.figure(figsize=(8, 5))
        ax_surface = fig_surface.add_subplot(111, projection='3d')
        apply_dark_style_3d(fig_surface, ax_surface)
        ax_surface.plot_surface(Xg, Yg, Z, cmap="plasma", alpha=0.85)
        ax_surface.set_title(f"Surface Plot — {func_name}")
        ax_surface.set_xlabel("x1")
        ax_surface.set_ylabel("x2")
        ax_surface.set_zlabel("f(x)")
        fig_surface.tight_layout()
        st.pyplot(fig_surface, use_container_width=True)

# ======================================
# PSO Algorithm (unchanged)
# ======================================

def PSO(func, D, N, low, high, T, w, c1, c2):

    k = 0.2
    vmax = k * (high - low)

    X = np.random.uniform(low, high, (N, D))
    V = np.zeros((N, D))

    pbest = X.copy()
    pbest_fitness = np.array([func(x) for x in X])

    gbest_index = np.argmin(pbest_fitness)
    gbest = pbest[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]

    history_best = []
    history_avg = []
    trajectory = []
    history_positions = []

    first_positions = X.copy()

    stagnation_counter = 0
    last_iter = 0

    for t in range(T):

        last_iter = t
        old_best = gbest_fitness

        for i in range(N):

            r1 = np.random.rand(D)
            r2 = np.random.rand(D)

            V[i] = (
                w * V[i]
                + c1 * r1 * (pbest[i] - X[i])
                + c2 * r2 * (gbest - X[i])
            )

            V[i] = np.clip(V[i], -vmax, vmax)

            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], low, high)

        fitness = np.array([func(x) for x in X])

        for i in range(N):
            if fitness[i] < pbest_fitness[i]:
                pbest[i] = X[i].copy()
                pbest_fitness[i] = fitness[i]

        best_index = np.argmin(pbest_fitness)

        if pbest_fitness[best_index] < gbest_fitness:
            gbest = pbest[best_index].copy()
            gbest_fitness = pbest_fitness[best_index]

        history_best.append(gbest_fitness)
        history_avg.append(np.mean(fitness))

        trajectory.append(X[0, :2].copy())

        history_positions.append(X.copy())

        if gbest_fitness == old_best:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= 30:
            break

    return (
        first_positions,
        X.copy(),
        history_best,
        history_avg,
        np.array(trajectory),
        gbest_fitness,
        last_iter,
        gbest.copy(),
        history_positions
    )

# ======================================
# PSO Interface
# ======================================

st.markdown("---")
st.markdown('<div class="section-label">PSO Hyperparameters</div>', unsafe_allow_html=True)

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    T = st.number_input("Max Iterations (T)", 1, 1000, 200)
with col_p2:
    w = st.number_input("w — Inertia", value=0.5)
with col_p3:
    c1 = st.number_input("c1 — Cognitive", value=2.0)
with col_p4:
    c2 = st.number_input("c2 — Social", value=2.0)

if st.button("Run PSO"):

    func = functions[func_name]

    first_pos, final_pos, best_curve, avg_curve, traj, final_best, last_iter, gbest_pos, _ = PSO(
        func, D, population_size, low, high, T, w, c1, c2
    )

    st.success("Optimization complete")

    X_grid = np.linspace(low, high, 100)
    Y_grid = np.linspace(low, high, 100)
    Xg, Yg = np.meshgrid(X_grid, Y_grid)
    Z = np.zeros_like(Xg)

    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            vec = np.zeros(D)
            vec[0] = Xg[i, j]
            vec[1] = Yg[i, j]
            Z[i, j] = func(vec)

    st.markdown('<div class="section-label">PSO — Search History</div>', unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("**1st Iteration**")
        init_fitness = np.array([func(x) for x in first_pos])
        best_init = first_pos[np.argmin(init_fitness)]

        fig1, ax1 = plt.subplots(figsize=(7, 5))
        apply_dark_style(fig1, [ax1])
        ax1.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.5)
        ax1.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.3)
        ax1.scatter(first_pos[:, 0], first_pos[:, 1], c="#94a3b8", s=10, alpha=0.6)
        ax1.scatter(best_init[0], best_init[1], c="#f59e0b", s=80, zorder=5)
        ax1.set_xlim(low, high)
        ax1.set_ylim(low, high)
        ax1.set_title(f"Search — Initial — {func_name}")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    with col_h2:
        st.markdown("**Final Iteration**")
        final_fitness = np.array([func(x) for x in final_pos])
        best_final = final_pos[np.argmin(final_fitness)]

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        apply_dark_style(fig2, [ax2])
        ax2.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.5)
        ax2.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.3)
        ax2.scatter(final_pos[:, 0], final_pos[:, 1], c="#94a3b8", s=10, alpha=0.6)
        ax2.scatter(best_final[0], best_final[1], c="#f59e0b", s=80, zorder=5)
        ax2.set_xlim(low, high)
        ax2.set_ylim(low, high)
        ax2.set_title(f"Search — Final — {func_name}")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.markdown('<div class="section-label">Statistics</div>', unsafe_allow_html=True)

    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        st.metric("Initial Best", f"{np.min(init_fitness):.2f}")
    with sm2:
        st.metric("Final Best", f"{final_best:.4f}")
    with sm3:
        st.metric("Stagnation at iteration", str(last_iter))

    st.markdown('<div class="section-label">Convergence & Trajectory</div>', unsafe_allow_html=True)

    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        st.markdown("**Convergence Curve**")
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        apply_dark_style(fig3, [ax3])
        ax3.plot(best_curve, color="#3b82f6", linewidth=1.5)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Best Fitness")
        ax3.set_title("Convergence")
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with col_c2:
        st.markdown("**Average Fitness**")
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        apply_dark_style(fig4, [ax4])
        ax4.plot(avg_curve, color="#06b6d4", linewidth=1.5)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Average Fitness")
        ax4.set_title("Population Average")
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)

    with col_c3:
        st.markdown("**Trajectory — 1st Particle**")
        fig5, ax5 = plt.subplots(figsize=(6, 3.5))
        apply_dark_style(fig5, [ax5])
        ax5.plot(traj[:, 0], traj[:, 1], color="#a78bfa", linewidth=1.2)
        ax5.set_xlabel("x1")
        ax5.set_ylabel("x2")
        ax5.set_title("Particle Trajectory")
        fig5.tight_layout()
        st.pyplot(fig5, use_container_width=True)


# ======================================
# Running Multiple PSO Experiments
# ======================================

st.markdown("---")
st.markdown('<div class="section-label">Multi-Run Analysis</div>', unsafe_allow_html=True)

with st.container(border=True):

    st.markdown("**Running PSO with multiple populations**")
    multi_runs = st.slider("Number of runs", min_value=1, max_value=100, value=30, key="multi_runs_slider")

    if st.button("Evaluate", key="btn_multi_run"):

        func = functions[func_name]

        X_grid = np.linspace(low, high, 100)
        Y_grid = np.linspace(low, high, 100)
        Xg, Yg = np.meshgrid(X_grid, Y_grid)
        Z = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                vec = np.zeros(D)
                vec[0] = Xg[i, j]
                vec[1] = Yg[i, j]
                Z[i, j] = func(vec)

        all_best_curves   = []
        all_avg_curves    = []
        all_traj_x1       = []
        all_first_pos     = []
        all_final_pos     = []
        all_gbest_pos     = []
        all_gbest_fitness = []
        all_history_pos   = [] 

        progress = st.progress(0, text="Running PSO experiments...")

        for r in range(multi_runs):

            first_pos, final_pos, best_curve, avg_curve, traj, gbest_fit, _, gbest_pos, hist_pos = PSO(
                func, D, population_size, low, high, T, w, c1, c2
            )

            all_best_curves.append(best_curve)
            all_avg_curves.append(avg_curve)
            all_traj_x1.append(traj[:, 0])
            all_first_pos.append(first_pos)
            all_final_pos.append(final_pos)
            all_gbest_pos.append(gbest_pos[:2])
            all_gbest_fitness.append(gbest_fit)
            all_history_pos.append(hist_pos)  

            progress.progress((r + 1) / multi_runs, text=f"Run {r+1}/{multi_runs}...")

        progress.empty()

        max_len_best = max(len(c) for c in all_best_curves)
        max_len_avg  = max(len(c) for c in all_avg_curves)
        max_len_traj = max(len(t) for t in all_traj_x1)

        def pad(arr_list, length):
            padded = []
            for a in arr_list:
                if len(a) < length:
                    a = np.concatenate([a, np.full(length - len(a), a[-1])])
                padded.append(a)
            return np.array(padded)

        best_mat = pad(all_best_curves, max_len_best)
        avg_mat  = pad(all_avg_curves,  max_len_avg)
        traj_mat = pad(all_traj_x1,     max_len_traj)

        mean_best_curve = best_mat.mean(axis=0)
        mean_avg_curve  = avg_mat.mean(axis=0)
        mean_traj_x1    = traj_mat.mean(axis=0)

        gbest_arr            = np.array(all_gbest_fitness)
        overall_best_fitness = np.min(gbest_arr)
        overall_mean_fitness = np.mean(gbest_arr)
        overall_std_fitness  = np.std(gbest_arr)

        best_run_idx  = np.argmin(all_gbest_fitness)
        best_overall  = all_gbest_pos[best_run_idx]
        gbest_pos_arr = np.array(all_gbest_pos)

        # ---- 3 column layout ----
        col_left, col_mid, col_right = st.columns([1.2, 1.5, 1.5])

        with col_left:
            st.markdown(f"**Function — {func_name}**")
            fig_3d = plt.figure(figsize=(6, 5))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            apply_dark_style_3d(fig_3d, ax_3d)
            ax_3d.plot_surface(Xg, Yg, Z, cmap="plasma", alpha=0.85)
            ax_3d.set_xlabel("x₁", fontsize=8)
            ax_3d.set_ylabel("x₂", fontsize=8)
            ax_3d.tick_params(labelsize=7)
            fig_3d.tight_layout()
            st.pyplot(fig_3d, use_container_width=True)

        with col_mid:
            st.markdown("**Search History — First Iteration**")
            fig_first, ax_first = plt.subplots(figsize=(7, 6))
            apply_dark_style(fig_first, [ax_first])
            ax_first.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.5)
            ax_first.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.3)

            for fp in all_first_pos:
                ax_first.scatter(fp[:, 0], fp[:, 1],
                                 c="#94a3b8", s=6, alpha=0.3, zorder=3)

            for fp in all_first_pos:
                fit_fp = np.array([func(x) for x in fp])
                best_fp = fp[np.argmin(fit_fp)]
                ax_first.scatter(best_fp[0], best_fp[1],
                                 c="#f59e0b", s=40, zorder=4, alpha=0.7)

            ax_first.set_xlim(low, high)
            ax_first.set_ylim(low, high)
            ax_first.set_title(f"Initial Distribution — {func_name}", fontsize=9)
            ax_first.set_xlabel("x₁")
            ax_first.set_ylabel("x₂")

            legend_first = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#94a3b8',
                       markersize=5, label="All particles (init)"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b',
                       markersize=7, label="Best per run (init)"),
            ]
            ax_first.legend(handles=legend_first, fontsize=7, loc="upper right",
                            facecolor='#131722', edgecolor='#1e2535', labelcolor='#94a3b8')
            fig_first.tight_layout()
            st.pyplot(fig_first, use_container_width=True)

        with col_right:
            st.markdown("**Search History — Final Iteration**")
            fig_final, ax_final = plt.subplots(figsize=(7, 6))
            apply_dark_style(fig_final, [ax_final])
            ax_final.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.5)
            ax_final.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.3)

            for hist in all_history_pos:
                for step_pos in hist:
                    ax_final.scatter(step_pos[:, 0], step_pos[:, 1],
                                     c="#94a3b8", s=2, alpha=0.04, zorder=2)

            ax_final.scatter(gbest_pos_arr[:, 0], gbest_pos_arr[:, 1],
                             c="#f59e0b", s=60, zorder=5, alpha=0.85,
                             label="Best solution per run")

            ax_final.scatter(best_overall[0], best_overall[1],
                 c="#f87171", s=80, zorder=6, marker="o",
                 edgecolors="white", linewidths=1,
                 label="Best global solution")

            ax_final.set_xlim(low, high)
            ax_final.set_ylim(low, high)
            ax_final.set_title(f"Final Distribution — {func_name}", fontsize=9)
            ax_final.set_xlabel("x₁")
            ax_final.set_ylabel("x₂")
            ax_final.legend(fontsize=7, loc="upper right",
                            facecolor='#131722', edgecolor='#1e2535', labelcolor='#94a3b8')
            fig_final.tight_layout()
            st.pyplot(fig_final, use_container_width=True)

        # ---- Stats ----
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Best", f"{overall_best_fitness:.4f}")
        with stat_col2:
            st.metric("Mean (avg error)", f"{overall_mean_fitness:.4f}")
        with stat_col3:
            st.metric("STD", f"{overall_std_fitness:.4f}")

        # ---- Bottom row ----
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            st.markdown("**Convergence Curve**")
            st.caption("Mean Best Fitness across all runs")
            fig_cc, ax_cc = plt.subplots(figsize=(6, 4))
            apply_dark_style(fig_cc, [ax_cc])
            ax_cc.plot(mean_best_curve, color="#3b82f6", linewidth=1.5)
            ax_cc.set_xlabel("Iteration")
            ax_cc.set_ylabel("Fitness")
            ax_cc.set_title("Convergence Curve")
            fig_cc.tight_layout()
            st.pyplot(fig_cc, use_container_width=True)

        with col_b2:
            st.markdown("**Trajectory — 1st Particle**")
            st.caption("Mean x₁⁽¹⁾ across all runs")
            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
            apply_dark_style(fig_tr, [ax_tr])
            ax_tr.plot(mean_traj_x1, color="#a78bfa", linewidth=1.5)
            ax_tr.set_xlabel("Iteration")
            ax_tr.set_ylabel("x₁⁽¹⁾")
            ax_tr.set_title("Trajectory of 1st solution")
            fig_tr.tight_layout()
            st.pyplot(fig_tr, use_container_width=True)

        with col_b3:
            st.markdown("**Average Population Fitness**")
            st.caption("Mean population fitness across all runs")
            fig_af, ax_af = plt.subplots(figsize=(6, 4))
            apply_dark_style(fig_af, [ax_af])
            ax_af.plot(mean_avg_curve, color="#06b6d4", linewidth=1.5)
            ax_af.set_xlabel("Iteration")
            ax_af.set_ylabel("Fitness")
            ax_af.set_title("Average Fitness of Population")
            fig_af.tight_layout()
            st.pyplot(fig_af, use_container_width=True)