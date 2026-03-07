import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# ======================================
# Fonctions d’évaluation
# ======================================
def f1(X): return np.sum(X**2)
def f2(X): return np.sum(abs(X)) + np.prod(abs(X))
def f5(X):
    xi1, xi = X[1:], X[:-1]
    return np.sum(100*(xi1**2 - xi)**2 + (1 - xi)**2)
def f7(X):
    D = len(X)
    return np.sum(np.arange(1, D+1) * X**4) + np.random.rand()
def f8(X): return np.sum(-X * np.sin(np.sqrt(abs(X))))
def f9(X): return np.sum(X**2 - 10*np.cos(2*np.pi*X) + 10)
def f11(X):
    D = len(X)
    indices = np.arange(1, D+1)
    return 1 + np.sum(X**2)/4000 - np.prod(np.cos(X/np.sqrt(indices)))

functions = {"F1": f1, "F2": f2, "F5": f5, "F7": f7, "F8": f8, "F9": f9, "F11": f11}

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

D = st.number_input("Dimension (D)", 1, 1000, 30)
low = st.number_input("Range min", -100.0)
high = st.number_input("Range max", 100.0)

population_size = st.slider("Population Size", 10, 500, 100, step=10)
runs = st.slider("Number of Runs", 1, 100, 10)

uploaded_file = st.file_uploader(
    f"Upload CSV for {func_name} (expected: {expected_csv[func_name]})",
    type="csv"
)


# ==============================================
# Importation de la population depuis un CSV
# ==============================================

if uploaded_file is not None:

    if uploaded_file.name != expected_csv[func_name]:
        st.error(f"Wrong file! Expected {expected_csv[func_name]}")
        st.stop()

    st.success(f"Correct file: {uploaded_file.name}")

    df = pd.read_csv(uploaded_file)
    pop = df.values[:, :int(D)]

    # ======================================
    # Bouton d’évaluation
    # ======================================
    if st.button("Evaluate population"):

        all_runs_fitness = []

        # ======================================
        # CALCUL DES FITNESS
        # ======================================
        for r in range(runs):
            sample_size = min(population_size, len(pop))
            idx = np.random.choice(len(pop), sample_size, replace=False)
            sample = pop[idx]

            fitness_vals = np.array([functions[func_name](ind) for ind in sample])
            all_runs_fitness.append(fitness_vals)

        all_runs_fitness = np.concatenate(all_runs_fitness)

        # ======================================
        # STATISTIQUES
        # ======================================
        st.write("### Statistics")
        st.success(f"Min (Best) = {np.min(all_runs_fitness):.4f}")
        st.warning(f"Max (Worst) = {np.max(all_runs_fitness):.4f}")
        st.info(f"Mean = {all_runs_fitness.mean():.4f} — STD = {all_runs_fitness.std():.4f}")

        # ======================================
        # 2D CONTOUR PLOT
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

        fig_contour, ax_contour = plt.subplots(figsize=(8, 6))
        contour = ax_contour.contour(Xg, Yg, Z, levels=30, linewidths=0.7, cmap="viridis")
        ax_contour.clabel(contour, inline=True, fontsize=8)

        # Ajouter des points de population
        idx = np.random.choice(len(pop), min(80, len(pop)), replace=False)
        sample2d = pop[idx, :2]
        ax_contour.scatter(sample2d[:, 0], sample2d[:, 1], c="red", s=20)

        ax_contour.set_title(f"Contour Plot ({func_name})")
        ax_contour.set_xlabel("x1")
        ax_contour.set_ylabel("x2")

        st.pyplot(fig_contour)

        # ======================================
        # 3D SURFACE PLOT
        # ======================================
        st.subheader("3D Surface Plot")

        fig_surface = plt.figure(figsize=(8, 6))
        ax_surface = fig_surface.add_subplot(111, projection='3d')

        ax_surface.plot_surface(Xg, Yg, Z, rstride=1, cstride=1, alpha=0.75, cmap="viridis")
        ax_surface.set_title(f"Surface Plot ({func_name})")
        ax_surface.set_xlabel("x1")
        ax_surface.set_ylabel("x2")
        ax_surface.set_zlabel("f(x)")

        st.pyplot(fig_surface)

#-------------------
# Algorithme PSO
#=====================

def PSO(func, D, N, low, high, T, w, c1, c2):

    X = np.random.uniform(low, high, (N, D))
    V = np.zeros((N, D))
    v_max = (high - low) * 0.5  # FIX 1: larger v_max for better exploration

    pbest = X.copy()
    pbest_fitness = np.array([func(x) for x in X])

    gbest = pbest[np.argmin(pbest_fitness)].copy()
    gbest_fitness = np.min(pbest_fitness)

    history_best = []
    history_avg = []
    trajectory = []
    stagnation_counter = 0

    first_positions = X.copy()

    w_max, w_min = 0.9, 0.4  # FIX 2: inertia bounds for linear decay

    for t in range(T):

        old_best = gbest_fitness

        # FIX 2: linearly decay inertia from 0.9 → 0.4
        w_t = w_max - (w_max - w_min) * (t / T)

        for i in range(N):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)

            V[i] = (
                w_t * V[i]  # FIX 2: use decaying w_t
                + c1 * r1 * (pbest[i] - X[i])
                + c2 * r2 * (gbest - X[i])
            )

            V[i] = np.clip(V[i], -v_max, v_max)
            X[i] = np.clip(X[i] + V[i], low, high)

        fitness = np.array([func(x) for x in X])

        for i in range(N):
            if fitness[i] < pbest_fitness[i]:
                pbest[i] = X[i].copy()
                pbest_fitness[i] = fitness[i]

        best_index = np.argmin(pbest_fitness)
        gbest = pbest[best_index].copy()
        gbest_fitness = pbest_fitness[best_index]

        history_best.append(gbest_fitness)
        history_avg.append(np.mean(fitness))
        trajectory.append(X[0, :2].copy())

        # FIX 3: looser stagnation tolerance + more patience
        if abs(gbest_fitness - old_best) < 1e-10:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= 80:  # FIX 3: was 20
            break

    return first_positions, X.copy(), history_best, history_avg, np.array(trajectory), gbest_fitness, t
        

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

    first_pos, final_pos, best_curve, avg_curve, traj, final_best, last_iter = PSO(
        func, D, population_size, low, high, T, w, c1, c2
    )

    st.success("Optimization Finished")

    # ======================================
    # 1ST ITERATION
    # ======================================

    st.subheader("Search History - 1st Iteration")

    init_fitness = np.array([func(x) for x in first_pos])
    best_init = first_pos[np.argmin(init_fitness)]

    # ======================================
    # FINAL ITERATION
    # ======================================

    final_fitness = np.array([func(x) for x in final_pos])
    best_final = final_pos[np.argmin(final_fitness)]

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
    # Compute contour grid
    # ======================================


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

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.contour(Xg, Yg, Z, levels=30, cmap="viridis")
    ax1.scatter(first_pos[:, 0], first_pos[:, 1], c="black", s=10)
    ax1.scatter(best_init[0], best_init[1], c="red", s=80)
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    st.pyplot(fig1)

    # ======================================
    # FINAL ITERATION
    # ======================================

    st.subheader("Search History - Final Iteration")

    final_fitness = np.array([func(x) for x in final_pos])
    best_final = final_pos[np.argmin(final_fitness)]

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.contour(Xg, Yg, Z, levels=30, cmap="viridis")
    ax2.scatter(final_pos[:, 0], final_pos[:, 1], c="black", s=10)
    ax2.scatter(best_final[0], best_final[1], c="red", s=80)
    ax2.set_xlim(low, high)
    ax2.set_ylim(low, high)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    st.pyplot(fig2)

