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

D = st.number_input("Dimension (D)", min_value=1, max_value=1000, value=30)
low = st.number_input("Range min", value=-100.0)
high = st.number_input("Range max", value=100.0)

population_size = st.slider("Population Size", 10, 500, 100, step=10)
runs = st.slider("Number of Runs", 1, 100, 10)

# ==============================================
# Importation de la population depuis un CSV
# ==============================================

uploaded_file = st.file_uploader(
    f"Upload CSV for {func_name} (expected: {expected_csv[func_name]})",
    type="csv"
)

if uploaded_file:

    if uploaded_file.name != expected_csv[func_name]:
        st.error(f"Wrong file! Expected {expected_csv[func_name]}")
        st.stop()

    st.success(f"Correct file: {uploaded_file.name}")

    df = pd.read_csv(uploaded_file)

    # Prenons seulement les D colonnes utiles
    pop = df.values[:, :int(D)]

if st.button("Evaluate population"):

    all_runs_fitness = []

    # ======================================
    # CALCUL DES FITNESS (pour stats + scatter)
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
    st.success(f"Min = {np.min(all_runs_fitness):.4f}")
    st.warning(f"Max = {np.max(all_runs_fitness):.4f}")
    st.info(f"Mean = {all_runs_fitness.mean():.4f} — STD = {all_runs_fitness.std():.4f}")

    # ======================================
    # SCATTER 2D
    # ======================================
    fig2d, ax2d = plt.subplots(figsize=(8,5))

    for r in range(runs):
        sample_size = min(population_size, len(pop))
        idx = np.random.choice(len(pop), sample_size, replace=False)
        sample = pop[idx]

        ax2d.scatter(sample[:,0], sample[:,1], alpha=0.3)

    ax2d.set_title(f"{func_name} - {runs} Runs")
    ax2d.set_xlabel("x1")
    ax2d.set_ylabel("x2")
    ax2d.grid(True)
    st.pyplot(fig2d)
