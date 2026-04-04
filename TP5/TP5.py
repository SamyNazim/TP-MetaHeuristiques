import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.datasets import load_digits, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ======================================
# Page config
# ======================================

st.set_page_config(
    page_title="Metaheuristics – Benchmark",
    page_icon=None,
    layout="wide",
)

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

# =========================
# Formulas  
# =========================

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
# Predefined Test Cases
# ======================================

PREDEFINED_CASES = {
    "Case 1 – SF=25, α=0.9": {
        "SF": 25,
        "alpha": 0.9,
        "solution": [
            0.74, 0.56, 0.79, 0.92, 0.28, 0.13, 0.53, 0.80, 0.49, 0.91,
            0.91, 0.88, 0.71, 0.96, 0.31, 0.30, 0.01, 0.14, 0.36, 0.42,
            0.53, 0.99, 0.73, 0.53, 0.84, 0.10, 0.34, 0.63, 0.02, 0.29,
            0.46, 0.30, 0.18, 0.21, 0.23, 0.78, 0.59, 0.50, 0.27, 0.30,
            0.36, 0.99, 0.15, 0.60, 0.03, 0.37, 0.52, 0.12, 0.32, 0.69,
            0.48, 0.91, 0.45, 0.57, 0.46, 0.62, 0.68, 0.48, 0.27, 0.94,
            0.47, 0.70, 0.12, 0.35
        ],
        "indices": [0, 1, 2, 3, 7, 9, 10, 11, 12, 13, 21, 22, 24, 27, 35, 36, 41, 43, 49, 51, 53, 55, 56, 59, 61],
    },
    "Case 2 – SF=10, α=0.9": {
        "SF": 10,
        "alpha": 0.9,
        "solution": [
            0.80, 0.70, 0.89, 0.55, 0.78, 0.63, 0.36, 0.83, 0.18, 0.94,
            0.31, 0.22, 0.53, 0.69, 0.41, 0.52, 0.55, 0.23, 0.74, 0.73,
            0.82, 0.45, 0.35, 0.67, 0.12, 0.62, 0.38, 0.93, 0.04, 0.54,
            0.72, 0.09, 0.23, 0.36, 0.21, 0.56, 0.07, 0.37, 0.60, 0.31,
            0.73, 0.24, 0.71, 0.46, 0.94, 0.17, 0.00, 0.65, 0.48, 0.19,
            0.34, 0.15, 0.42, 0.52, 0.31, 0.29, 0.34, 0.99, 0.59, 0.76,
            0.32, 0.55, 0.16, 0.39
        ],
        "indices": [0, 2, 4, 7, 9, 20, 27, 44, 57, 59],
    },
}

# ======================================
# Interface Streamlit
# ======================================

st.title("PW - Metaheuristics")
st.subheader("Optimization Benchmark Problems")

func_name = st.selectbox("Function", list(functions.keys()))

D = st.number_input("Dimension (D)", 2, 1000, 30)

low = st.number_input("Range min", min_value=-500.0, max_value=0.0, value=-100.0)
high = st.number_input("Range max", min_value=0.0, max_value=500.0, value=100.0)

population_size = st.slider("Population Size", 10, 500, 100, step=10)
runs = st.slider("Number of Runs", 1, 100, 10)

# =========================
# Printing formula
# =========================

st.write("### Function formula")
st.latex(formulas[func_name])

# ======================================
# Import CSV Population & Evaluation
# ======================================

uploaded_file = st.file_uploader(
    f"Upload CSV for {func_name} (expected: {expected_csv[func_name]})",
    type="csv"
)

if uploaded_file is not None:

    if uploaded_file.name != expected_csv[func_name]:
        st.error(f"Wrong file! Expected {expected_csv[func_name]}")
        st.stop()

    st.success(f"Correct file: {uploaded_file.name}")

    df = pd.read_csv(uploaded_file)
    pop = df.values[:, :int(D)]

    # ======================================
    # Evaluate Population
    # ======================================

    if st.button("Evaluate population"):

        all_runs_fitness = []

        for r in range(runs):
            sample_size = min(population_size, len(pop))
            idx = np.random.choice(len(pop), sample_size, replace=False)
            sample = pop[idx]
            fitness_vals = np.array([functions[func_name](ind) for ind in sample])
            all_runs_fitness.append(fitness_vals)

        all_runs_fitness = np.concatenate(all_runs_fitness)

        st.subheader("Statistics")
        st.success(f"Min (Best) = {np.min(all_runs_fitness):.4f}")
        st.warning(f"Max (Worst) = {np.max(all_runs_fitness):.4f}")
        st.info(f"Mean = {all_runs_fitness.mean():.4f} — STD = {all_runs_fitness.std():.4f}")

        # ======================================
        # 2D Contour Plot
        # ======================================

        st.subheader("2D Contour Plot")

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

        fig_contour, ax_contour = plt.subplots()
        contour = ax_contour.contour(Xg, Yg, Z, levels=30, cmap="viridis")
        ax_contour.scatter(pop[:, 0], pop[:, 1], c="red", s=10)
        ax_contour.set_title(f"Contour Plot ({func_name})")
        ax_contour.set_xlabel("x1")
        ax_contour.set_ylabel("x2")
        st.pyplot(fig_contour, use_container_width=True)

        # ======================================
        # 3D Surface Plot
        # ======================================

        st.subheader("3D Surface Plot")

        fig_surface = plt.figure()
        ax_surface = fig_surface.add_subplot(111, projection='3d')
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

st.subheader("PSO Hyperparameters")

T = st.number_input("Max Iteration (T)", 1, 1000, 200)
w = st.number_input("w (inertia)", value=0.5)
c1 = st.number_input("c1 (cognitive)", value=2.0)
c2 = st.number_input("c2 (social)", value=2.0)

if st.button("Run PSO"):

    func = functions[func_name]

    first_pos, final_pos, best_curve, avg_curve, traj, final_best, last_iter, gbest_pos, _ = PSO(
        func, D, population_size, low, high, T, w, c1, c2
    )

    st.success("Optimization Finished")

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

    st.markdown("### Application of PSO")

    # ======================================
    # 1ST ITERATION
    # ======================================

    st.subheader("Search History - 1st Iteration")

    init_fitness = np.array([func(x) for x in first_pos])
    best_init = first_pos[np.argmin(init_fitness)]

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.contour(Xg, Yg, Z, levels=30, cmap="viridis")
    ax1.scatter(first_pos[:, 0], first_pos[:, 1], c="black", s=10)
    ax1.scatter(best_init[0], best_init[1], c="red", s=80)
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    st.pyplot(fig1, use_container_width=True)

    # ======================================
    # FINAL ITERATION
    # ======================================

    st.subheader("Search History - Final Iteration")

    final_fitness = np.array([func(x) for x in final_pos])
    best_final = final_pos[np.argmin(final_fitness)]

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.contour(Xg, Yg, Z, levels=30, cmap="viridis")
    ax2.scatter(final_pos[:, 0], final_pos[:, 1], c="black", s=10)
    ax2.scatter(best_final[0], best_final[1], c="red", s=80)
    ax2.set_xlim(low, high)
    ax2.set_ylim(low, high)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    st.pyplot(fig2, use_container_width=True)

    # ======================================
    # STATISTICS
    # ======================================

    st.subheader("Statistics")

    st.markdown("**Initial population:**")
    st.markdown(
        f"Best — {np.min(init_fitness):.2f}, "
        f"Worst — {np.max(init_fitness):.2f}"
    )

    st.markdown("**Final population:**")
    st.markdown(f"Best — {final_best:.4f}")

    st.markdown(f"**Stagnation — Iteration N°{last_iter}**")

    # ======================================
    # Convergence Curve
    # ======================================

    st.subheader("Convergence Curve")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(best_curve)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Best Fitness")
    st.pyplot(fig3, use_container_width=True)

    # ======================================
    # Average Fitness
    # ======================================

    st.subheader("Average Fitness")

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(avg_curve)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Average Fitness")
    st.pyplot(fig4, use_container_width=True)

    # ======================================
    # Trajectory
    # ======================================

    st.subheader("Trajectory of 1st Particle")

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(traj[:, 0], traj[:, 1])
    ax5.set_xlabel("x1")
    ax5.set_ylabel("x2")
    st.pyplot(fig5, use_container_width=True)


# ======================================
# Running Multiple PSO Experiments
# ======================================

st.markdown("---")
st.header("Running PSO with multiple populations")

with st.container(border=True):

    st.subheader("Running Multiple Populations")

    multi_runs = st.slider("Multiple run:", min_value=1, max_value=100, value=30, key="multi_runs_slider")

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

        col_left, col_mid, col_right = st.columns([1.2, 1.5, 1.5])

        with col_left:
            st.markdown(f"**Function ({func_name})**")
            fig_3d = plt.figure(figsize=(6, 5))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.9)
            ax_3d.set_xlabel("x₁", fontsize=8)
            ax_3d.set_ylabel("x₂", fontsize=8)
            ax_3d.tick_params(labelsize=7)
            st.pyplot(fig_3d, use_container_width=True)

        with col_mid:
            st.markdown("**Search History — First Iteration**")
            fig_first, ax_first = plt.subplots(figsize=(7, 6))
            ax_first.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
            ax_first.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.5)

            for fp in all_first_pos:
                ax_first.scatter(fp[:, 0], fp[:, 1], c="black", s=6, alpha=0.3, zorder=3)

            for fp in all_first_pos:
                fit_fp = np.array([func(x) for x in fp])
                best_fp = fp[np.argmin(fit_fp)]
                ax_first.scatter(best_fp[0], best_fp[1], c="orange", s=40, zorder=4, alpha=0.6)

            ax_first.set_xlim(low, high)
            ax_first.set_ylim(low, high)
            ax_first.set_title(f"Search History ({func_name}), First Iteration", fontsize=9)
            ax_first.set_xlabel("x₁")
            ax_first.set_ylabel("x₂")

            from matplotlib.lines import Line2D
            legend_first = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markersize=5, label="All particles (init)"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                       markersize=7, label="Best particle per run (init)"),
            ]
            ax_first.legend(handles=legend_first, fontsize=7, loc="upper right")
            st.pyplot(fig_first, use_container_width=True)

        with col_right:
            st.markdown("**Search History — Final Iteration**")
            fig_final, ax_final = plt.subplots(figsize=(7, 6))
            ax_final.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
            ax_final.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.5)

            for hist in all_history_pos:
                for step_pos in hist:
                    ax_final.scatter(step_pos[:, 0], step_pos[:, 1],
                                     c="black", s=2, alpha=0.04, zorder=2)

            ax_final.scatter(gbest_pos_arr[:, 0], gbest_pos_arr[:, 1],
                             c="orange", s=60, zorder=5, alpha=0.85,
                             label="Best solution per run")
            ax_final.scatter(best_overall[0], best_overall[1],
                             c="red", s=60, zorder=6, marker="o",
                             edgecolors="white", linewidths=1,
                             label="Best global solution")

            ax_final.set_xlim(low, high)
            ax_final.set_ylim(low, high)
            ax_final.set_title(f"Search History ({func_name}), Final Iteration", fontsize=9)
            ax_final.set_xlabel("x₁")
            ax_final.set_ylabel("x₂")
            ax_final.legend(fontsize=7, loc="upper right")
            st.pyplot(fig_final, use_container_width=True)

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Best", f"{overall_best_fitness:.4f}")
        with stat_col2:
            st.metric("Mean (avg error)", f"{overall_mean_fitness:.4f}")
        with stat_col3:
            st.metric("STD", f"{overall_std_fitness:.4f}")

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
            st.markdown("**Trajectory of the First Solution in the Population**")
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

# ======================================
# Part 3 — Feature Selection with PSO
# ======================================

st.markdown("---")
st.header("Feature Selection with PSO")

# -------------------------------------------------------
# load dataset
# -------------------------------------------------------

@st.cache_data
def load_synthetic():
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=5,
        n_redundant=10,
        random_state=42
    )
    return X, y

@st.cache_data
def load_digits_data():
    digits = load_digits()
    return digits.data, digits.target

# -------------------------------------------------------
# KNN accuracy on selected features
# -------------------------------------------------------

def evaluate_knn(X, y, selected_indices, k=5):
    X_sel = X[:, selected_indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.3, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# -------------------------------------------------------
# Fitness function  f(x) = α·f1(x) + (1-α)·f2(x)
# -------------------------------------------------------

def fitness_fs(solution, X, y, SF, alpha, k=5):
    D_feat = X.shape[1]
    indices = np.argsort(solution)[-SF:]
    accuracy = evaluate_knn(X, y, indices, k)
    f1_val = 1.0 - accuracy         
    f2_val = SF / D_feat           
    return alpha * f1_val + (1 - alpha) * f2_val, accuracy, sorted(indices.tolist())

# -------------------------------------------------------
# PSO for Feature Selection
# -------------------------------------------------------

def PSO_FS(X, y, SF, alpha, N=30, T=100, w=0.5, c1=2.0, c2=2.0, k_knn=5):
    D_feat = X.shape[1]
    low_fs, high_fs = 0.0, 1.0
    vmax = 0.2 * (high_fs - low_fs)

    pos = np.random.uniform(low_fs, high_fs, (N, D_feat))
    vel = np.zeros((N, D_feat))

    pbest = pos.copy()
    pbest_fit = np.array([fitness_fs(pos[i], X, y, SF, alpha, k_knn)[0] for i in range(N)])

    gbest_idx = np.argmin(pbest_fit)
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    stag = 0
    for t in range(T):
        old_best = gbest_fit
        for i in range(N):
            r1, r2 = np.random.rand(D_feat), np.random.rand(D_feat)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gbest - pos[i])
            vel[i] = np.clip(vel[i], -vmax, vmax)
            pos[i] = np.clip(pos[i] + vel[i], low_fs, high_fs)

        fits = np.array([fitness_fs(pos[i], X, y, SF, alpha, k_knn)[0] for i in range(N)])
        for i in range(N):
            if fits[i] < pbest_fit[i]:
                pbest[i] = pos[i].copy()
                pbest_fit[i] = fits[i]

        best_i = np.argmin(pbest_fit)
        if pbest_fit[best_i] < gbest_fit:
            gbest = pbest[best_i].copy()
            gbest_fit = pbest_fit[best_i]

        stag = stag + 1 if gbest_fit == old_best else 0
        if stag >= 20:
            break

    final_fit, final_acc, final_indices = fitness_fs(gbest, X, y, SF, alpha, k_knn)
    return gbest, final_fit, final_acc, final_indices

# -------------------------------------------------------
# UI
# -------------------------------------------------------

fs_col1, fs_col2, fs_col3 = st.columns([1, 1, 2])

with fs_col1:
    st.markdown("**Data**")
    dataset_choice = st.radio("Dataset", ["Synthetic", "Digits"], key="fs_dataset", label_visibility="collapsed")

with fs_col2:
    SF = st.number_input("Selected Features (SF)", min_value=0, max_value=64, value=5, key="fs_sf")
    alpha_fs = st.number_input("α", min_value=0.0, max_value=1.0, value=0.9, step=0.05, key="fs_alpha")

with fs_col3:
    st.markdown("")
    st.markdown("")

    # -------------------------------------------------------
    # Predefined test case selector
    # -------------------------------------------------------
    case_options = ["— None (manual) —"] + list(PREDEFINED_CASES.keys())
    selected_case = st.selectbox("Load predefined test case", case_options, key="fs_case")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        btn_eval = st.button("Model Evaluation", key="btn_fs_eval")
    with btn_col2:
        btn_reeval = st.button("Model Re-evaluation", key="btn_fs_reeval")

# Load dataset
if dataset_choice == "Synthetic":
    X_fs, y_fs = load_synthetic()
else:
    X_fs, y_fs = load_digits_data()

D_fs = X_fs.shape[1]

# -------------------------------------------------------
# Session state for PSO solution
# -------------------------------------------------------

if "fs_solution" not in st.session_state:
    st.session_state.fs_solution    = None
    st.session_state.fs_indices     = None
    st.session_state.fs_fitness     = None
    st.session_state.fs_accuracy    = None
    st.session_state.fs_SF          = None
    st.session_state.fs_alpha_stored = None   # ← clé distincte du widget fs_alpha

# -------------------------------------------------------
# Input validation helper
# -------------------------------------------------------

def validate_inputs(sf_val, alpha_val):
    """Returns (is_valid, error_message)."""
    if sf_val == 0:
        return False, "**Selected Features (SF) cannot be 0.** Please choose at least 1 feature."
    if alpha_val == 0.0:
        return False, "**Alpha (α) cannot be 0.** A value of 0 means the classifier accuracy is completely ignored. Please set α > 0."
    return True, ""

# -------------------------------------------------------
# Model Evaluation — run PSO  OR  load predefined case
# -------------------------------------------------------

if btn_eval:
    # Validation
    is_valid, err_msg = validate_inputs(int(SF), float(alpha_fs))
    if not is_valid:
        st.error(err_msg)
    else:
        # Load predefined case if selected
        if selected_case != "— None (manual) —":
            case_data = PREDEFINED_CASES[selected_case]
            solution_arr = np.array(case_data["solution"])

            # Pad or trim solution to match D_fs if needed
            if len(solution_arr) < D_fs:
                solution_arr = np.concatenate([solution_arr, np.random.uniform(0, 1, D_fs - len(solution_arr))])
            elif len(solution_arr) > D_fs:
                solution_arr = solution_arr[:D_fs]

            sf_to_use    = case_data["SF"]
            alpha_to_use = case_data["alpha"]

            fit_val, acc_val, sel_indices = fitness_fs(
                solution_arr, X_fs, y_fs, SF=sf_to_use, alpha=alpha_to_use
            )

            st.session_state.fs_solution     = solution_arr
            st.session_state.fs_indices      = sel_indices
            st.session_state.fs_fitness      = fit_val
            st.session_state.fs_accuracy     = acc_val
            st.session_state.fs_SF           = sf_to_use
            st.session_state.fs_alpha_stored = alpha_to_use   

            st.info(f"Loaded predefined solution: **{selected_case}**")

        else:
            # Normal PSO run
            with st.spinner("Running PSO for Feature Selection..."):
                solution, fit_val, acc_val, sel_indices = PSO_FS(
                    X_fs, y_fs, SF=int(SF), alpha=float(alpha_fs),
                    N=30, T=100, w=0.5, c1=2.0, c2=2.0
                )
            st.session_state.fs_solution     = solution
            st.session_state.fs_indices      = sel_indices
            st.session_state.fs_fitness      = fit_val
            st.session_state.fs_accuracy     = acc_val
            st.session_state.fs_SF           = int(SF)
            st.session_state.fs_alpha_stored = float(alpha_fs)   

# -------------------------------------------------------
# Model Re-evaluation — re-evaluate stored solution with new SF / alpha
# -------------------------------------------------------

if btn_reeval:
    if st.session_state.fs_solution is None:
        st.warning("Please run Model Evaluation first.")
    else:
        # Validation
        is_valid, err_msg = validate_inputs(int(SF), float(alpha_fs))
        if not is_valid:
            st.error(err_msg)
        else:
            solution = st.session_state.fs_solution
            fit_val, acc_val, sel_indices = fitness_fs(
                solution, X_fs, y_fs, SF=int(SF), alpha=float(alpha_fs)
            )
            st.session_state.fs_indices      = sel_indices
            st.session_state.fs_fitness      = fit_val
            st.session_state.fs_accuracy     = acc_val
            st.session_state.fs_SF           = int(SF)
            st.session_state.fs_alpha_stored = float(alpha_fs)   

# -------------------------------------------------------
# Display results
# -------------------------------------------------------

if st.session_state.fs_solution is not None:

    solution   = st.session_state.fs_solution
    sel_idx    = st.session_state.fs_indices
    fit_val    = st.session_state.fs_fitness
    acc_val    = st.session_state.fs_accuracy
    sf_val     = st.session_state.fs_SF
    alpha_val  = st.session_state.get("fs_alpha_stored", float(alpha_fs))   

    sol_str    = " | ".join([f"{v:.2f}" for v in solution])
    idx_str    = " | ".join([str(i) for i in sel_idx])

    with st.container(border=True):
        st.markdown("**Solution**")
        st.text_area(
            label="Solution details",
            value=f"Solution:\n{sol_str}\n\nIndices of selected features:\n{idx_str}",
            height=160,
            label_visibility="collapsed"
        )

    st.markdown(
        f"**Fitness** — {fit_val:.2f}, "
        f"**Accuracy** — {acc_val:.2f}, "
        f"**Selected Features** — {sf_val}, "
        f"**α** — {alpha_val}"
    )