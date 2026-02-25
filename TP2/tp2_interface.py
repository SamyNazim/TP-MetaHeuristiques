import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Fonctions benchmark
# =========================

def f1(X):
    return np.sum(X**2)

def f2(X):
    return np.sum(abs(X)) + np.prod(abs(X))

def f5(X):
    xi1 = X[1:]
    xi = X[:-1]
    return np.sum(100 * (xi1**2 - xi1)**2 + (1 - xi)**2)

def f7(X):
    D = len(X)
    indices = np.arange(1, D+1)
    return np.sum(indices * (X**4)) + np.random.rand()

def f8(X):
    return np.sum(-X * np.sin(np.sqrt(abs(X))))

def f9(X):
    return np.sum(X**2 - 10*np.cos(2*np.pi*X) + 10)

def f11(X):
    D = len(X)
    indices = np.arange(1, D+1)
    sum_part = np.sum(X**2) / 4000
    prod_part = np.prod(np.cos(X / np.sqrt(indices)))
    return 1 + sum_part - prod_part


functions = {
    "F1": f1,
    "F2": f2,
    "F5": f5,
    "F7": f7,
    "F8": f8,
    "F9": f9,
    "F11": f11
}

# =========================
# Formules mathématiques
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

# =========================
# Interface
# =========================

st.title("PW - Metaheuristics")
st.subheader("Optimization Problem Initialization")

st.write("Standard Continuous Optimization Benchmark Problems")

# Paramètres
D = st.number_input("Dimension (D)", min_value=1, value=30)

col1, col2 = st.columns(2)
with col1:
    low = st.number_input("Range min", value=-100.0)
with col2:
    high = st.number_input("Range max", value=100.0)

func_name = st.selectbox("Function", list(functions.keys()))

# =========================
# Affichage formule
# =========================

st.write("### Function formula")
st.latex(formulas[func_name])

# =========================
# Génération solution
# =========================

if "solution" not in st.session_state:
    st.session_state.solution = None

if st.button("Generate solution"):
    st.session_state.solution = np.random.uniform(low, high, D)

st.write("Candidate solution example:")

if st.session_state.solution is not None:
    st.write(st.session_state.solution)

# =========================
# Évaluation
# =========================

if st.button("Evaluate solution") and st.session_state.solution is not None:
    fitness = functions[func_name](st.session_state.solution)
    st.success(f"Fitness = {fitness:.4f}")

uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu du CSV")
    st.dataframe(df.head())


x_col = st.selectbox(df.columns, index= 0)
y_col = st.selectbox(df.columns, index= 1)

x = df[x_col]
y_raw = df[y_col]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, color='green', alpha=0.6)
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_title(f"{x_col} vs {y_col}")
ax.grid(True)