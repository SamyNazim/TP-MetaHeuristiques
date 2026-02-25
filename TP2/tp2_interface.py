import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D


# ===============================
# Fonctions d'évaluation
# ===============================

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
formulas = {
    "F1": r"f(x)=\sum x_i^2",
    "F2": r"f(x)=\sum |x_i| + \prod |x_i|",
    "F5": r"f(x)=\sum 100(x_{i+1}^2-x_i)^2+(1-x_i)^2",
    "F7": r"f(x)=\sum i x_i^4 + rand()",
    "F8": r"f(x)=-\sum x_i sin(sqrt(|x_i|))",
    "F9": r"f(x)=\sum x_i^2 - 10 cos(2π x_i)+10",
    "F11": r"f(x)=1 + (1/4000)∑x_i^2 - ∏cos(x_i/sqrt(i))"
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

# ===============================
# Interface Streamlit
# ===============================

st.title("PW - Metaheuristics")
st.subheader("Optimization Benchmark Problems")

# Choix de fonction
func_name = st.selectbox("Function", list(functions.keys()))
st.write("### Formula")
st.latex(formulas[func_name])

# Slider population
population_size = st.slider("Population Size", 10, 500, 100, step=10)

# Dimension & bornes
D = st.number_input("Dimension (D)", 1, 1000, 30)
col1, col2 = st.columns(2)
low = col1.number_input("Range min", -100.0)
high = col2.number_input("Range max", 100.0)

# Génération solution candidate
if "solution" not in st.session_state: st.session_state.solution = None
if st.button("Generate solution"):
    st.session_state.solution = np.random.uniform(low, high, D)
if st.session_state.solution is not None:
    st.write("Candidate solution:", st.session_state.solution)

# Évaluation
if st.button("Evaluate solution") and st.session_state.solution is not None:
    fitness = functions[func_name](st.session_state.solution)
    st.success(f"Fitness = {fitness:.4f}")

# ===============================
# Upload CSV
# ===============================
uploaded_file = st.file_uploader(
    f"Upload CSV for {func_name} (expected: {expected_csv[func_name]})", type="csv"
)

if uploaded_file is not None:

    # Vérification strict nom fichier
    if uploaded_file.name != expected_csv[func_name]:
        st.error(f"❌ Wrong file! Expected {expected_csv[func_name]}")
        st.stop()
    st.success(f"Correct file: {uploaded_file.name}")

    df = pd.read_csv(uploaded_file)
    st.subheader("CSV Preview")
    st.dataframe(df.head())

    if len(df.columns) < 2:
        st.error("CSV must have at least 2 columns")
        st.stop()

    # Choix colonnes X/Y
    x_col = st.selectbox("X column", df.columns, index=0)
    y_col = st.selectbox("Y column", df.columns, index=1)
    x, y = df[x_col], df[y_col]

    # Scatter 2D
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x, y, alpha=0.6)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"{x_col} vs {y_col}"); ax.grid(True)
    st.pyplot(fig)

    # Min / Max fitness
    try:
        pop = df.values[:, :D]
        fitness_values = np.array([functions[func_name](ind) for ind in pop[:population_size]])
        st.success(f"Min fitness: {fitness_values.min():.4f}")
        st.warning(f"Max fitness: {fitness_values.max():.4f}")
    except Exception as e:
        st.error(f"Error computing fitness: {e}")