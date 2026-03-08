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
    h1 { color: #1a73e8; }
    h2, h3 { color: #2c3e50; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px; }
    div[data-testid="metric-container"] {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #1a73e8;
    }
    section[data-testid="stSidebar"] {
        background: #0e1117;
        border-right: 1px solid #2c2c2c;
    }
    section[data-testid="stSidebar"] h2 {
        border-bottom: none;
        font-size: 1rem;
        color: #4d9fff;
        margin-top: 1rem;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }
    hr { border-top: 2px solid #e0e0e0; margin: 1.5rem 0; }
    details summary { font-weight: 600; color: #2c3e50; }
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
    "F1":  r"f(x)=\sum_{i=1}^{D} x_i^2",
    "F2":  r"f(x)=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|",
    "F5":  r"f(x)=\sum_{i=1}^{D-1}\left[100(x_{i+1}^2-x_i)^2+(1-x_i)^2\right]",
    "F7":  r"f(x)=\sum_{i=1}^{D} i\,x_i^4 + \text{rand}(0,1)",
    "F8":  r"f(x)=\sum_{i=1}^{D}-x_i\sin(\sqrt{|x_i|})",
    "F9":  r"f(x)=\sum_{i=1}^{D}\left[x_i^2-10\cos(2\pi x_i)+10\right]",
    "F11": r"f(x)=1+\frac{1}{4000}\sum_{i=1}^{D}x_i^2-\prod_{i=1}^{D}\cos\left(\frac{x_i}{\sqrt{i}}\right)"
}

# ======================================
# Sidebar – Configuration
# ======================================

with st.sidebar:
    st.title("Configuration")

    st.markdown("## Function")
    func_name = st.selectbox("Select function", list(functions.keys()), label_visibility="collapsed")

    st.markdown("## Search Space")
    D = st.number_input("Dimension (D)", 2, 1000, 30)
    col_lo, col_hi = st.columns(2)
    with col_lo:
        low = st.number_input("Min", min_value=-500.0, max_value=0.0, value=-100.0)
    with col_hi:
        high = st.number_input("Max", min_value=0.0, max_value=500.0, value=100.0)

    st.markdown("## Population")
    population_size = st.slider("Population size", 10, 500, 100, step=10)
    runs = st.slider("Number of runs", 1, 100, 10)

    st.markdown("## PSO Parameters")
    T  = st.number_input("Max iterations (T)", 1, 1000, 200)
    w  = st.number_input("w (inertia)",   value=0.3)
    c1 = st.number_input("c1 (cognitive)", value=1.4)
    c2 = st.number_input("c2 (social)",    value=1.4)

    st.markdown("---")
    st.caption("Best defaults: c1=2, c2=2, w=0.5, T=200, N=30")

# ======================================
# Header
# ======================================

st.title("Metaheuristics — Optimization Benchmark")
st.markdown(
    f"**Selected function:** `{func_name}` &nbsp;|&nbsp; "
    f"**Dimension:** `{D}` &nbsp;|&nbsp; "
    f"**Range:** `[{low}, {high}]`"
)
st.latex(formulas[func_name])
st.markdown("---")

# ======================================
# Helper – build 2-D grid for plots
# ======================================

def build_grid(func, D, low, high, n=100):
    X = np.linspace(low, high, n)
    Y = np.linspace(low, high, n)
    Xg, Yg = np.meshgrid(X, Y)
    Z = np.zeros_like(Xg)
    for i in range(n):
        for j in range(n):
            vec = np.zeros(D)
            vec[0] = Xg[i, j]
            vec[1] = Yg[i, j]
            Z[i, j] = func(vec)
    return Xg, Yg, Z

# ======================================
# Section 1 – CSV Population Evaluation
# ======================================

with st.expander("Population Evaluation (CSV Upload)", expanded=True):

    uploaded_file = st.file_uploader(
        f"Upload CSV for **{func_name}** (expected: `{expected_csv[func_name]}`)",
        type="csv",
    )

    if uploaded_file is not None:
        if uploaded_file.name != expected_csv[func_name]:
            st.error(f"Wrong file! Expected `{expected_csv[func_name]}`")
        else:
            st.success(f"Correct file: `{uploaded_file.name}`")

            df  = pd.read_csv(uploaded_file)
            pop = df.values[:, :int(D)]

            if st.button("Evaluate population", type="primary"):

                all_runs_fitness = []
                for r in range(runs):
                    sample_size = min(population_size, len(pop))
                    idx = np.random.choice(len(pop), sample_size, replace=False)
                    sample = pop[idx]
                    fitness_vals = np.array([functions[func_name](ind) for ind in sample])
                    all_runs_fitness.append(fitness_vals)

                all_runs_fitness = np.concatenate(all_runs_fitness)

                m1, m2, m3 = st.columns(3)
                m1.metric("Min (Best)",    f"{np.min(all_runs_fitness):.4f}")
                m2.metric("Max (Worst)",   f"{np.max(all_runs_fitness):.4f}")
                m3.metric("Mean ± STD",    f"{all_runs_fitness.mean():.4f} ± {all_runs_fitness.std():.4f}")

                st.markdown("#### Visualization")
                Xg, Yg, Z = build_grid(functions[func_name], D, low, high)

                vc1, vc2 = st.columns(2)

                with vc1:
                    st.markdown("**2D Contour Plot**")
                    fig_contour, ax_contour = plt.subplots(figsize=(6, 5))
                    ax_contour.contour(Xg, Yg, Z, levels=30, cmap="viridis")
                    ax_contour.scatter(pop[:, 0], pop[:, 1], c="red", s=10)
                    ax_contour.set_title(f"Contour Plot ({func_name})")
                    ax_contour.set_xlabel("x1")
                    ax_contour.set_ylabel("x2")
                    st.pyplot(fig_contour, use_container_width=True)

                with vc2:
                    st.markdown("**3D Surface Plot**")
                    fig_surface = plt.figure(figsize=(6, 5))
                    ax_surface  = fig_surface.add_subplot(111, projection="3d")
                    ax_surface.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.8)
                    ax_surface.set_title(f"Surface Plot ({func_name})")
                    ax_surface.set_xlabel("x1")
                    ax_surface.set_ylabel("x2")
                    ax_surface.set_zlabel("f(x)")
                    st.pyplot(fig_surface, use_container_width=True)

# ======================================
# PSO Algorithm
# ======================================

def PSO(func, D, N, low, high, T, w, c1, c2):
    k    = 0.2
    vmax = k * (high - low)

    X = np.random.uniform(low, high, (N, D))
    V = np.zeros((N, D))

    pbest         = X.copy()
    pbest_fitness = np.array([func(x) for x in X])

    gbest_index   = np.argmin(pbest_fitness)
    gbest         = pbest[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]

    history_best = []
    history_avg  = []
    trajectory   = []

    first_positions    = X.copy()
    stagnation_counter = 0
    last_iter          = 0

    for t in range(T):
        last_iter = t
        old_best  = gbest_fitness

        for i in range(N):
            r1   = np.random.rand(D)
            r2   = np.random.rand(D)
            V[i] = (
                w  * V[i]
                + c1 * r1 * (pbest[i] - X[i])
                + c2 * r2 * (gbest    - X[i])
            )
            V[i] = np.clip(V[i], -vmax, vmax)
            X[i] = np.clip(X[i] + V[i], low, high)

        fitness = np.array([func(x) for x in X])

        for i in range(N):
            if fitness[i] < pbest_fitness[i]:
                pbest[i]         = X[i].copy()
                pbest_fitness[i] = fitness[i]

        best_index = np.argmin(pbest_fitness)
        if pbest_fitness[best_index] < gbest_fitness:
            gbest         = pbest[best_index].copy()
            gbest_fitness = pbest_fitness[best_index]

        history_best.append(gbest_fitness)
        history_avg.append(np.mean(fitness))
        trajectory.append(X[0, :2].copy())

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
        None,
    )

# ======================================
# Section 2 – Single PSO Run
# ======================================

st.markdown("---")
st.header("PSO — Single Run")

if st.button("Run PSO", type="primary", key="btn_single_run"):

    func = functions[func_name]

    first_pos, final_pos, best_curve, avg_curve, traj, final_best, last_iter, gbest_pos, _ = PSO(
        func, D, population_size, low, high, T, w, c1, c2
    )

    st.success("Optimization finished!")

    Xg, Yg, Z = build_grid(func, D, low, high)

    # ── Search history ──
    col_init, col_final = st.columns(2)

    init_fitness    = np.array([func(x) for x in first_pos])
    best_init       = first_pos[np.argmin(init_fitness)]
    final_fitness   = np.array([func(x) for x in final_pos])
    best_final_pt   = final_pos[np.argmin(final_fitness)]

    with col_init:
        st.markdown("**Search History — 1st Iteration**")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
        ax1.contour(Xg, Yg, Z,  levels=30, colors="white", linewidths=0.3, alpha=0.5)
        ax1.scatter(first_pos[:, 0], first_pos[:, 1], c="black", s=10)
        ax1.scatter(best_init[0], best_init[1], c="red", s=80, zorder=5, label="Best")
        ax1.set_xlim(low, high)
        ax1.set_ylim(low, high)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.legend(fontsize=8)
        st.pyplot(fig1, use_container_width=True)

    with col_final:
        st.markdown("**Search History — Final Iteration**")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
        ax2.contour(Xg, Yg, Z,  levels=30, colors="white", linewidths=0.3, alpha=0.5)
        ax2.scatter(final_pos[:, 0], final_pos[:, 1], c="black", s=10)
        ax2.scatter(best_final_pt[0], best_final_pt[1], c="red", s=80, zorder=5, label="Best")
        ax2.set_xlim(low, high)
        ax2.set_ylim(low, high)
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.legend(fontsize=8)
        st.pyplot(fig2, use_container_width=True)

    # ── Statistics ──
    st.markdown("#### Statistics")
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Initial Best",  f"{np.min(init_fitness):.2f}")
    sm2.metric("Initial Worst", f"{np.max(init_fitness):.2f}")
    sm3.metric("Final Best",    f"{final_best:.4f}")
    st.info(f"Stagnation reached at iteration **{last_iter}**")

    # ── Curves ──
    st.markdown("#### Convergence & Trajectory")
    cc1, cc2, cc3 = st.columns(3)

    with cc1:
        st.markdown("**Convergence Curve**")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(best_curve, color="red")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Best Fitness")
        st.pyplot(fig3, use_container_width=True)

    with cc2:
        st.markdown("**Average Fitness**")
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.plot(avg_curve, color="blue")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Avg Fitness")
        st.pyplot(fig4, use_container_width=True)

    with cc3:
        st.markdown("**Trajectory of 1st Particle**")
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        ax5.plot(traj[:, 0], traj[:, 1], color="green")
        ax5.set_xlabel("x1")
        ax5.set_ylabel("x2")
        st.pyplot(fig5, use_container_width=True)

# ======================================
# Section 3 – Multiple PSO Runs
# ======================================

st.markdown("---")
st.header("PSO — Multiple Runs Analysis")

with st.container(border=True):

    multi_runs = st.slider("Number of runs:", min_value=1, max_value=100, value=30, key="multi_runs_slider")

    if st.button("Evaluate", type="primary", key="btn_multi_run"):

        func = functions[func_name]
        Xg, Yg, Z = build_grid(func, D, low, high)

        all_best_curves  = []
        all_avg_curves   = []
        all_traj_x1      = []
        all_first_pos    = []
        all_final_pos    = []
        all_gbest_pos    = []
        all_gbest_fitness = []

        progress = st.progress(0, text="Running PSO experiments...")

        for r in range(multi_runs):
            first_pos, final_pos, best_curve, avg_curve, traj, gbest_fit, _, gbest_pos, _ = PSO(
                func, D, population_size, low, high, T, w, c1, c2
            )
            all_best_curves.append(best_curve)
            all_avg_curves.append(avg_curve)
            all_traj_x1.append(traj[:, 0])
            all_first_pos.append(first_pos)
            all_final_pos.append(final_pos)
            all_gbest_pos.append(gbest_pos[:2])
            all_gbest_fitness.append(gbest_fit)
            progress.progress((r + 1) / multi_runs, text=f"Run {r+1}/{multi_runs}...")

        progress.empty()

        def pad(arr_list, length):
            padded = []
            for a in arr_list:
                if len(a) < length:
                    a = np.concatenate([a, np.full(length - len(a), a[-1])])
                padded.append(a)
            return np.array(padded)

        max_len_best = max(len(c) for c in all_best_curves)
        max_len_avg  = max(len(c) for c in all_avg_curves)
        max_len_traj = max(len(t) for t in all_traj_x1)

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

        best_run_idx   = np.argmin(all_gbest_fitness)
        best_overall   = all_gbest_pos[best_run_idx]
        gbest_pos_arr  = np.array(all_gbest_pos)

        # ── Row 1: 3D | First iter | Final iter ──
        st.markdown("#### Search History")
        col_left, col_mid, col_right = st.columns([1.2, 1.5, 1.5])

        with col_left:
            st.markdown(f"**3D Surface ({func_name})**")
            fig_3d = plt.figure(figsize=(6, 5))
            ax_3d  = fig_3d.add_subplot(111, projection="3d")
            ax_3d.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.9)
            ax_3d.set_xlabel("x₁", fontsize=8)
            ax_3d.set_ylabel("x₂", fontsize=8)
            ax_3d.tick_params(labelsize=7)
            st.pyplot(fig_3d, use_container_width=True)

        with col_mid:
            st.markdown("**First Iteration**")
            fig_first, ax_first = plt.subplots(figsize=(7, 6))
            ax_first.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
            ax_first.contour(Xg, Yg, Z,  levels=30, colors="white", linewidths=0.3, alpha=0.5)
            for fp in all_first_pos:
                ax_first.scatter(fp[:, 0], fp[:, 1], c="black", s=6, alpha=0.3, zorder=3)
            for fp in all_first_pos:
                fit_fp   = np.array([func(x) for x in fp])
                best_fp  = fp[np.argmin(fit_fp)]
                ax_first.scatter(best_fp[0], best_fp[1], c="orange", s=40, zorder=4, alpha=0.6)
            ax_first.set_xlim(low, high)
            ax_first.set_ylim(low, high)
            ax_first.set_title(f"First Iteration ({func_name})", fontsize=9)
            ax_first.set_xlabel("x₁")
            ax_first.set_ylabel("x₂")
            ax_first.legend(
                handles=[
                    Line2D([0],[0], marker="o", color="w", markerfacecolor="black",  markersize=5, label="All particles"),
                    Line2D([0],[0], marker="o", color="w", markerfacecolor="orange", markersize=7, label="Best per run"),
                ],
                fontsize=7, loc="upper right",
            )
            st.pyplot(fig_first, use_container_width=True)

        with col_right:
            st.markdown("**Final Iteration**")
            fig_final, ax_final = plt.subplots(figsize=(7, 6))
            ax_final.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
            ax_final.contour(Xg, Yg, Z,  levels=30, colors="white", linewidths=0.3, alpha=0.5)
            for fp in all_final_pos:
                ax_final.scatter(fp[:, 0], fp[:, 1], c="black", s=6, alpha=0.3, zorder=3)
            ax_final.scatter(
                gbest_pos_arr[:, 0], gbest_pos_arr[:, 1],
                c="orange", s=60, zorder=5, alpha=0.8, label="Best per run",
            )
            ax_final.scatter(
                best_overall[0], best_overall[1],
                c="red", s=120, zorder=6, marker="*", label="Best global",
            )
            ax_final.set_xlim(low, high)
            ax_final.set_ylim(low, high)
            ax_final.set_title(f"Final Iteration ({func_name})", fontsize=9)
            ax_final.set_xlabel("x₁")
            ax_final.set_ylabel("x₂")
            ax_final.legend(fontsize=7, loc="upper right")
            st.pyplot(fig_final, use_container_width=True)

        # ── Statistics ──
        st.markdown("#### Global Statistics")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Best",           f"{overall_best_fitness:.4f}")
        stat_col2.metric("Mean (avg err)", f"{overall_mean_fitness:.4f}")
        stat_col3.metric("STD",            f"{overall_std_fitness:.4f}")

        # ── Bottom row: curves ──
        st.markdown("#### Convergence, Trajectory & Average Fitness")
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            st.markdown("**Convergence Curve**")
            st.caption("Mean Best Fitness of All Runs vs. Iteration")
            fig_cc, ax_cc = plt.subplots(figsize=(6, 4))
            ax_cc.plot(mean_best_curve, color="red")
            ax_cc.set_xlabel("Iteration")
            ax_cc.set_ylabel("Fitness")
            ax_cc.set_title("Convergence Curve")
            st.pyplot(fig_cc, use_container_width=True)

        with col_b2:
            st.markdown("**Trajectory of 1st Particle**")
            st.caption("Mean x₁⁽¹⁾ of all Runs vs. Iteration")
            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
            ax_tr.plot(mean_traj_x1, color="green")
            ax_tr.set_xlabel("Iteration")
            ax_tr.set_ylabel("x₁⁽¹⁾")
            ax_tr.set_title("Trajectory of 1st solution")
            st.pyplot(fig_tr, use_container_width=True)

        with col_b3:
            st.markdown("**Average Population Fitness**")
            st.caption("Mean Population Average Fitness of All Runs vs. Iteration")
            fig_af, ax_af = plt.subplots(figsize=(6, 4))
            ax_af.plot(mean_avg_curve, color="blue")
            ax_af.set_xlabel("Iteration")
            ax_af.set_ylabel("Fitness")
            ax_af.set_title("Average Fitness of population")
            st.pyplot(fig_af, use_container_width=True)