import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# ======================================
# Fonctions d'évaluation
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

# ======================================
# Interface Streamlit
# ======================================

st.title("PW - Metaheuristics")
st.subheader("Optimization Benchmark Problems")

func_name = st.selectbox("Function", list(functions.keys()))

D = st.number_input("Dimension (D)", 2, 1000, 30)
low = st.number_input("Range min", -100.0)
high = st.number_input("Range max", 100.0)

population_size = st.slider("Population Size", 10, 500, 100, step=10)
runs = st.slider("Number of Runs", 1, 100, 10)

uploaded_file = st.file_uploader(
    f"Upload CSV for {func_name} (expected: {expected_csv[func_name]})",
    type="csv"
)

# ======================================
# Import CSV Population
# ======================================

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

    stagnation_counter = 0
    first_positions = X.copy()
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
        gbest.copy()
    )

# ======================================
# Metaheuristic Interface
# ======================================

st.subheader("Metaheuristic")

T = st.number_input("Max Iteration (T)", 1, 1000, 200)
w = st.number_input("w (inertia)", value=0.3)
c1 = st.number_input("c1 (cognitive)", value=1.4)
c2 = st.number_input("c2 (social)", value=1.4)

if st.button("Run Metaheuristic"):

    func = functions[func_name]

    first_pos, final_pos, best_curve, avg_curve, traj, final_best, last_iter, gbest_pos = PSO(
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

        # Precompute contour grid
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

        # Run PSO multiple times
        all_best_curves   = []   # best fitness per iteration, per run
        all_avg_curves    = []   # avg fitness per iteration, per run
        all_traj_x1       = []   # x1 of 1st particle per iteration, per run
        all_final_pos     = []   # final positions per run
        all_gbest_pos     = []   # global best position of each run
        all_gbest_fitness = []   # global best fitness of each run
        all_iter_best_pos = []   # gbest position at each iteration (last run shown)

        progress = st.progress(0, text="Running PSO experiments...")

        for r in range(multi_runs):

            _, final_pos, best_curve, avg_curve, traj, gbest_fit, _, gbest_pos = PSO(
                func, D, population_size, low, high, T, w, c1, c2
            )

            all_best_curves.append(best_curve)
            all_avg_curves.append(avg_curve)
            all_traj_x1.append(traj[:, 0])
            all_final_pos.append(final_pos)
            all_gbest_pos.append(gbest_pos[:2])
            all_gbest_fitness.append(gbest_fit)

            progress.progress((r + 1) / multi_runs, text=f"Run {r+1}/{multi_runs}...")

        progress.empty()

        # Pad curves to same length (max_iters)
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

        overall_best_fitness = np.min(all_gbest_fitness)
        overall_mean_fitness = np.mean(all_gbest_fitness)
        overall_std_fitness  = np.std(all_gbest_fitness)

        # ======================================
        # Layout: 3D surface | Search History | Stats
        # ======================================

        col_left, col_mid, col_right = st.columns([1.2, 1.5, 1])

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
            # Build best-solution-per-iteration points across all runs
            # For visualisation we collect gbest position updates per run
            fig_hist, ax_hist = plt.subplots(figsize=(7, 6))
            ax_hist.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
            ax_hist.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.5)

            # Best of each run (orange dots)
            gbest_arr = np.array(all_gbest_pos)
            ax_hist.scatter(gbest_arr[:, 0], gbest_arr[:, 1],
                            c="orange", s=60, zorder=4, label="Best solution of each run")

            # Final positions of last run (black dots, representative)
            for fp in all_final_pos:
                ax_hist.scatter(fp[:, 0], fp[:, 1],
                                c="black", s=6, alpha=0.3, zorder=3)

            # Best solution across all runs (red dot)
            best_run_idx = np.argmin(all_gbest_fitness)
            best_overall = all_gbest_pos[best_run_idx]
            ax_hist.scatter(best_overall[0], best_overall[1],
                            c="red", s=120, zorder=5, marker="*",
                            label="Best solution across all runs")

            ax_hist.set_xlim(low, high)
            ax_hist.set_ylim(low, high)
            ax_hist.set_title(f"Search History ({func_name}), Final Iteration", fontsize=9)
            ax_hist.set_xlabel("x₁")
            ax_hist.set_ylabel("x₂")
            ax_hist.legend(fontsize=7, loc="upper right")
            st.pyplot(fig_hist, use_container_width=True)

        with col_right:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.markdown(f"**Best** — {overall_best_fitness:.2f},")
            st.markdown(f"**Mean (average error)** — {overall_mean_fitness:.2f},")
            st.markdown(f"**STD** — {overall_std_fitness:.2f},")

        # ======================================
        # Bottom row: Convergence | Trajectory | Avg Fitness
        # ======================================

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
        # Results summary box
        # ======================================

        st.markdown("---")
        st.markdown("**Results**")

        res_col1, res_col2 = st.columns([1, 1])
        with res_col2:
            with st.container(border=True):
                st.markdown(f"**Best** — {overall_best_fitness:.2f},")
                st.markdown("")
                st.markdown(f"**Mean (average error)** — {overall_mean_fitness:.2f},")
                st.markdown("")
                st.markdown(f"**STD** — {overall_std_fitness:.2f},")