# ==============================================================================
# IMPORTS — on charge toutes les bibliothèques nécessaires
# ==============================================================================

import numpy as np          # NumPy : bibliothèque pour les calculs mathématiques et les tableaux
import pandas as pd         # Pandas : bibliothèque pour lire et manipuler des données tabulaires (CSV, etc.)
import matplotlib.pyplot as plt  # Matplotlib : bibliothèque pour créer des graphiques
import streamlit as st      # Streamlit : bibliothèque pour créer une interface web interactive
from mpl_toolkits.mplot3d import Axes3D  # Outil de Matplotlib pour faire des graphiques en 3D
from collections import Counter          # Counter : permet de compter des éléments facilement (non utilisé activement ici)
from sklearn.datasets import load_digits, make_classification  # Datasets de sklearn : jeux de données prêts à l'emploi
from sklearn.neighbors import KNeighborsClassifier  # KNN : algorithme de classification par plus proches voisins
from sklearn.metrics import accuracy_score           # Calcule la précision d'un classifieur (% de bonnes prédictions)
from sklearn.model_selection import train_test_split # Divise les données en ensemble d'entraînement et de test


# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Metaheuristics – Benchmark",  # Titre affiché dans l'onglet du navigateur
    page_icon=None,                           # Pas d'icône spéciale
    layout="wide",                            # Utilise toute la largeur de l'écran
)


# ==============================================================================
# FONCTIONS DE BENCHMARK
# Ces fonctions sont des problèmes d'optimisation classiques.
# Le but d'un algorithme d'optimisation est de trouver le vecteur X
# qui MINIMISE la valeur retournée par ces fonctions.
# ==============================================================================

def f1(X):
    # F1 — Sphère : somme des carrés de chaque variable
    # Minimum global = 0, atteint quand tous les xi = 0
    return np.sum(X**2)

def f2(X):
    # F2 — Schwefel 2.22 : somme des valeurs absolues + produit des valeurs absolues
    # Minimum global = 0, atteint quand tous les xi = 0
    return np.sum(np.abs(X)) + np.prod(np.abs(X))

def f5(X):
    # F5 — Rosenbrock : fonction en "vallée de banane", très difficile à minimiser
    # xi1 = éléments à partir du 2ème, xi = éléments jusqu'à l'avant-dernier
    xi1, xi = X[1:], X[:-1]
    # Formule : somme de 100*(x_{i+1}^2 - x_i)^2 + (1 - x_i)^2
    return np.sum(100*(xi1**2 - xi)**2 + (1 - xi)**2)

def f7(X):
    # F7 — Quartic avec bruit aléatoire : somme pondérée de x_i^4 + bruit
    D = len(X)  # D = nombre de variables (dimension)
    return np.sum(np.arange(1, D+1) * X**4) + np.random.rand()  # np.random.rand() ajoute un bruit aléatoire entre 0 et 1

def f8(X):
    # F8 — Schwefel : chaque variable multipliée par le sinus de sa racine carrée
    # Minimum global difficile à trouver car beaucoup de minima locaux
    return np.sum(-X * np.sin(np.sqrt(np.abs(X))))

def f9(X):
    # F9 — Rastrigin : beaucoup de minima locaux, minimum global = 0 quand tous xi = 0
    return np.sum(X**2 - 10*np.cos(2*np.pi*X) + 10)

def f11(X):
    # F11 — Griewank : produit de cosinus qui crée de nombreux minima locaux
    D = len(X)                                # Nombre de dimensions
    indices = np.arange(1, D+1)              # Tableau [1, 2, 3, ..., D]
    return 1 + np.sum(X**2)/4000 - np.prod(np.cos(X/np.sqrt(indices)))

# Dictionnaire qui associe le nom de chaque fonction à la fonction elle-même
# Permet d'appeler la bonne fonction selon le choix de l'utilisateur
functions = {
    "F1": f1,
    "F2": f2,
    "F5": f5,
    "F7": f7,
    "F8": f8,
    "F9": f9,
    "F11": f11
}

# Dictionnaire qui associe chaque fonction au nom du fichier CSV attendu
# Permet de vérifier que l'utilisateur uploade le bon fichier
expected_csv = {
    "F1": "Population_F1-UM.csv",
    "F2": "Population_F2-UM.csv",
    "F5": "Population_F5-UM.csv",
    "F7": "Population_F7-UM.csv",
    "F8": "Population_F8-MM.csv",
    "F9": "Population_F9-MM.csv",
    "F11": "Population_F11-MM.csv"
}


# ==============================================================================
# FORMULES EN LATEX
# Ces chaînes de caractères contiennent les formules mathématiques
# au format LaTeX pour les afficher joliment dans l'interface
# ==============================================================================

formulas = {
    "F1":  r"f(x)=\sum_{i=1}^{D} x_i^2",
    "F2":  r"f(x)=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|",
    "F5":  r"f(x)=\sum_{i=1}^{D-1}\left[100(x_{i+1}^2-x_i)^2+(1-x_i)^2\right]",
    "F7":  r"f(x)=\sum_{i=1}^{D} i\,x_i^4 + \text{rand}(0,1)",
    "F8":  r"f(x)=\sum_{i=1}^{D}-x_i\sin(\sqrt{|x_i|})",
    "F9":  r"f(x)=\sum_{i=1}^{D}\left[x_i^2-10\cos(2\pi x_i)+10\right]",
    "F11": r"f(x)=1+\frac{1}{4000}\sum_{i=1}^{D}x_i^2-\prod_{i=1}^{D}\cos\left(\frac{x_i}{\sqrt{i}}\right)"
}


# ==============================================================================
# CAS PRÉDÉFINIS (pour la sélection de features)
# Des solutions déjà calculées sont stockées ici pour pouvoir les recharger
# sans relancer l'algorithme PSO (utile pour comparer ou tester rapidement)
# ==============================================================================

PREDEFINED_CASES = {
    "Case 1 – SF=25, α=0.9": {
        "SF": 25,        # Nombre de features sélectionnées
        "alpha": 0.9,    # Poids de la précision dans la fitness (90% précision, 10% réduction)
        "solution": [    # Vecteur de position de la meilleure particule trouvée
            0.74, 0.56, 0.79, 0.92, 0.28, 0.13, 0.53, 0.80, 0.49, 0.91,
            0.91, 0.88, 0.71, 0.96, 0.31, 0.30, 0.01, 0.14, 0.36, 0.42,
            0.53, 0.99, 0.73, 0.53, 0.84, 0.10, 0.34, 0.63, 0.02, 0.29,
            0.46, 0.30, 0.18, 0.21, 0.23, 0.78, 0.59, 0.50, 0.27, 0.30,
            0.36, 0.99, 0.15, 0.60, 0.03, 0.37, 0.52, 0.12, 0.32, 0.69,
            0.48, 0.91, 0.45, 0.57, 0.46, 0.62, 0.68, 0.48, 0.27, 0.94,
            0.47, 0.70, 0.12, 0.35
        ],
        "indices": [0, 1, 2, 3, 7, 9, 10, 11, 12, 13, 21, 22, 24, 27, 35,
                    36, 41, 43, 49, 51, 53, 55, 56, 59, 61],  # Indices des 25 features retenues
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
        "indices": [0, 2, 4, 7, 9, 20, 27, 44, 57, 59],  # Indices des 10 features retenues
    },
}


# ==============================================================================
# INTERFACE STREAMLIT — Partie principale
# Tout ce qui suit crée les éléments visuels de la page web
# ==============================================================================

st.title("PW - Metaheuristics")                          # Titre principal de la page
st.subheader("Optimization Benchmark Problems")          # Sous-titre

# Menu déroulant pour choisir la fonction à optimiser
func_name = st.selectbox("Function", list(functions.keys()))

# Champ numérique pour choisir la dimension D (nombre de variables du problème)
D = st.number_input("Dimension (D)", 2, 1000, 30)

# Bornes de l'espace de recherche : les particules ne sortiront pas de [low, high]
low  = st.number_input("Range min", min_value=-500.0, max_value=0.0,   value=-100.0)
high = st.number_input("Range max", min_value=0.0,    max_value=500.0, value=100.0)

# Curseur pour choisir la taille de la population (nombre de particules/individus)
population_size = st.slider("Population Size", 10, 500, 100, step=10)

# Curseur pour choisir le nombre de répétitions de l'évaluation
runs = st.slider("Number of Runs", 1, 100, 10)


# ==============================================================================
# AFFICHAGE DE LA FORMULE MATHÉMATIQUE
# ==============================================================================

st.write("### Function formula")
st.latex(formulas[func_name])  # Affiche la formule LaTeX de la fonction sélectionnée


# ==============================================================================
# UPLOAD D'UN FICHIER CSV ET ÉVALUATION DE LA POPULATION
# ==============================================================================

# Bouton d'upload : l'utilisateur peut importer un fichier CSV contenant une population
uploaded_file = st.file_uploader(
    f"Upload CSV for {func_name} (expected: {expected_csv[func_name]})",
    type="csv"  # On accepte uniquement les fichiers .csv
)

if uploaded_file is not None:  # Si un fichier a été uploadé...

    if uploaded_file.name != expected_csv[func_name]:  # Vérifie que c'est le bon fichier
        st.error(f"Wrong file! Expected {expected_csv[func_name]}")  # Affiche une erreur si mauvais fichier
        st.stop()  # Arrête l'exécution du script

    st.success(f"Correct file: {uploaded_file.name}")  # Message de succès si bon fichier

    df  = pd.read_csv(uploaded_file)       # Lit le fichier CSV et le transforme en DataFrame (tableau)
    pop = df.values[:, :int(D)]            # Extrait les D premières colonnes comme matrice numpy

    # Bouton pour lancer l'évaluation de la population importée
    if st.button("Evaluate population"):

        all_runs_fitness = []  # Liste qui va stocker les fitness de tous les runs

        for r in range(runs):  # Répète l'évaluation 'runs' fois
            sample_size = min(population_size, len(pop))  # Taille de l'échantillon = min(population demandée, taille réelle)
            idx    = np.random.choice(len(pop), sample_size, replace=False)  # Tire au hasard des individus (sans remise)
            sample = pop[idx]  # Extrait ces individus
            fitness_vals = np.array([functions[func_name](ind) for ind in sample])  # Évalue chaque individu
            all_runs_fitness.append(fitness_vals)  # Ajoute les résultats à la liste

        all_runs_fitness = np.concatenate(all_runs_fitness)  # Fusionne tous les runs en un seul tableau

        # Affichage des statistiques
        st.subheader("Statistics")
        st.success(f"Min (Best) = {np.min(all_runs_fitness):.4f}")           # Meilleure valeur (la plus basse)
        st.warning(f"Max (Worst) = {np.max(all_runs_fitness):.4f}")          # Pire valeur (la plus haute)
        st.info(f"Mean = {all_runs_fitness.mean():.4f} — STD = {all_runs_fitness.std():.4f}")  # Moyenne et écart-type

        # ---- GRAPHIQUE 2D CONTOUR ----
        st.subheader("2D Contour Plot")

        X = np.linspace(low, high, 100)   # 100 points régulièrement espacés sur l'axe x
        Y = np.linspace(low, high, 100)   # 100 points régulièrement espacés sur l'axe y
        Xg, Yg = np.meshgrid(X, Y)        # Crée une grille 2D à partir de X et Y
        Z = np.zeros_like(Xg)             # Matrice de zéros de même taille que la grille

        for i in range(Xg.shape[0]):       # Parcourt les lignes de la grille
            for j in range(Xg.shape[1]):   # Parcourt les colonnes de la grille
                vec    = np.zeros(D)       # Crée un vecteur de D zéros
                vec[0] = Xg[i, j]         # Assigne la valeur x au 1er élément
                vec[1] = Yg[i, j]         # Assigne la valeur y au 2ème élément
                Z[i, j] = functions[func_name](vec)  # Calcule la valeur de la fonction en ce point

        fig_contour, ax_contour = plt.subplots()  # Crée une figure et des axes matplotlib
        contour = ax_contour.contour(Xg, Yg, Z, levels=30, cmap="viridis")  # Dessine les courbes de niveau
        ax_contour.scatter(pop[:, 0], pop[:, 1], c="red", s=10)             # Superpose les points de la population en rouge
        ax_contour.set_title(f"Contour Plot ({func_name})")                 # Titre du graphique
        ax_contour.set_xlabel("x1")                                         # Étiquette axe X
        ax_contour.set_ylabel("x2")                                         # Étiquette axe Y
        st.pyplot(fig_contour, use_container_width=True)                    # Affiche le graphique dans Streamlit

        # ---- GRAPHIQUE 3D SURFACE ----
        st.subheader("3D Surface Plot")

        fig_surface = plt.figure()                              # Crée une nouvelle figure
        ax_surface  = fig_surface.add_subplot(111, projection='3d')  # Ajoute des axes en mode 3D
        ax_surface.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.8)  # Dessine la surface 3D (alpha=transparence)
        ax_surface.set_title(f"Surface Plot ({func_name})")    # Titre
        ax_surface.set_xlabel("x1")                            # Axe X
        ax_surface.set_ylabel("x2")                            # Axe Y
        ax_surface.set_zlabel("f(x)")                          # Axe Z = valeur de la fonction
        st.pyplot(fig_surface, use_container_width=True)       # Affiche dans Streamlit


# ==============================================================================
# ALGORITHME PSO — Particle Swarm Optimization
#
# Principe : un essaim de N particules explore l'espace de recherche.
# Chaque particule a une position (solution candidate) et une vitesse.
# Elle est attirée vers sa meilleure position passée (pbest)
# et vers la meilleure position jamais trouvée par l'essaim (gbest).
# ==============================================================================

def PSO(func, D, N, low, high, T, w, c1, c2):
    """
    func  : la fonction à minimiser
    D     : nombre de dimensions (variables)
    N     : nombre de particules dans l'essaim
    low   : borne inférieure de l'espace de recherche
    high  : borne supérieure de l'espace de recherche
    T     : nombre maximum d'itérations
    w     : inertie (à quel point la particule garde sa direction actuelle)
    c1    : coefficient cognitif (attraction vers son propre meilleur)
    c2    : coefficient social (attraction vers le meilleur global)
    """

    k    = 0.2                    # Facteur de limitation de la vitesse (20% de la plage)
    vmax = k * (high - low)       # Vitesse maximale autorisée pour chaque particule

    # Initialisation aléatoire des positions : chaque particule démarre à une position aléatoire
    X = np.random.uniform(low, high, (N, D))  # Matrice N×D : N particules, D dimensions
    V = np.zeros((N, D))                      # Matrice N×D de vitesses initialisées à 0

    pbest         = X.copy()  # Chaque particule commence avec sa position initiale comme meilleur personnel
    pbest_fitness = np.array([func(x) for x in X])  # Évalue la fitness de chaque particule

    gbest_index   = np.argmin(pbest_fitness)   # Indice de la particule avec la meilleure fitness
    gbest         = pbest[gbest_index].copy()  # Position du meilleur global (copie pour éviter les modifications)
    gbest_fitness = pbest_fitness[gbest_index] # Valeur de fitness du meilleur global

    history_best  = []   # Historique du meilleur fitness à chaque itération
    history_avg   = []   # Historique de la fitness moyenne à chaque itération
    trajectory    = []   # Historique de la position (x1, x2) de la 1ère particule
    history_positions = [] # Historique complet des positions de toutes les particules

    first_positions = X.copy()  # Sauvegarde des positions initiales (pour comparaison visuelle)

    stagnation_counter = 0  # Compteur : combien d'itérations sans amélioration
    last_iter          = 0  # Mémorise l'itération où l'algorithme s'est arrêté

    for t in range(T):  # Boucle principale : jusqu'à T itérations maximum

        last_iter  = t           # Mémorise l'itération courante
        old_best   = gbest_fitness  # Sauvegarde le meilleur avant mise à jour (pour détecter la stagnation)

        for i in range(N):  # Mise à jour de chaque particule

            r1 = np.random.rand(D)  # Vecteur aléatoire entre 0 et 1 (pour la composante cognitive)
            r2 = np.random.rand(D)  # Vecteur aléatoire entre 0 et 1 (pour la composante sociale)

            # Mise à jour de la vitesse selon la formule PSO :
            # V = inertie + attraction_personnelle + attraction_sociale
            V[i] = (
                w  * V[i]                      # Inertie : conserve une partie de la vitesse actuelle
                + c1 * r1 * (pbest[i] - X[i]) # Vers son propre meilleur (nostalgie)
                + c2 * r2 * (gbest   - X[i])  # Vers le meilleur global (conformisme)
            )

            V[i] = np.clip(V[i], -vmax, vmax)  # Limite la vitesse entre -vmax et +vmax

            X[i] = X[i] + V[i]                 # Déplace la particule selon sa nouvelle vitesse
            X[i] = np.clip(X[i], low, high)    # Garde la particule dans les bornes autorisées

        fitness = np.array([func(x) for x in X])  # Évalue la fitness de toutes les particules

        for i in range(N):  # Met à jour le meilleur personnel de chaque particule
            if fitness[i] < pbest_fitness[i]:  # Si la nouvelle position est meilleure...
                pbest[i]         = X[i].copy()  # ...mémorise cette position comme nouveau meilleur personnel
                pbest_fitness[i] = fitness[i]   # ...mémorise cette fitness

        best_index = np.argmin(pbest_fitness)  # Trouve la particule avec le meilleur score global

        if pbest_fitness[best_index] < gbest_fitness:  # Si ce score bat le meilleur global...
            gbest         = pbest[best_index].copy()   # ...met à jour la position du meilleur global
            gbest_fitness = pbest_fitness[best_index]  # ...met à jour la fitness du meilleur global

        history_best.append(gbest_fitness)         # Enregistre le meilleur fitness de cette itération
        history_avg.append(np.mean(fitness))       # Enregistre la fitness moyenne de cette itération
        trajectory.append(X[0, :2].copy())         # Enregistre la position (x1, x2) de la 1ère particule
        history_positions.append(X.copy())         # Enregistre les positions de TOUTES les particules

        # Détection de la stagnation
        if gbest_fitness == old_best:   # Si le meilleur n'a pas changé...
            stagnation_counter += 1     # ...incrémente le compteur de stagnation
        else:
            stagnation_counter = 0      # Sinon, remet le compteur à zéro

        if stagnation_counter >= 30:    # Si pas d'amélioration depuis 30 itérations → on arrête
            break

    # Retourne tous les résultats utiles
    return (
        first_positions,    # Positions initiales de toutes les particules
        X.copy(),           # Positions finales de toutes les particules
        history_best,       # Courbe du meilleur fitness par itération
        history_avg,        # Courbe de la fitness moyenne par itération
        np.array(trajectory),  # Trajectoire de la 1ère particule
        gbest_fitness,      # Meilleure valeur de fitness trouvée
        last_iter,          # Dernière itération atteinte
        gbest.copy(),       # Position de la meilleure solution globale
        history_positions   # Historique complet des positions
    )


# ==============================================================================
# INTERFACE PSO — Hyperparamètres
# ==============================================================================

st.subheader("PSO Hyperparameters")

T  = st.number_input("Max Iteration (T)", 1, 1000, 200)  # Nombre max d'itérations
w  = st.number_input("w (inertia)",  value=0.5)           # Coefficient d'inertie
c1 = st.number_input("c1 (cognitive)", value=2.0)         # Coefficient cognitif
c2 = st.number_input("c2 (social)",    value=2.0)         # Coefficient social

if st.button("Run PSO"):  # Quand l'utilisateur clique sur "Run PSO"

    func = functions[func_name]  # Récupère la fonction choisie dans le dictionnaire

    # Lance le PSO et récupère tous les résultats
    first_pos, final_pos, best_curve, avg_curve, traj, final_best, last_iter, gbest_pos, _ = PSO(
        func, D, population_size, low, high, T, w, c1, c2
    )

    st.success("Optimization Finished")  # Message de succès

    # Prépare la grille pour les graphiques de contour
    X_grid = np.linspace(low, high, 100)   # 100 valeurs sur l'axe x
    Y_grid = np.linspace(low, high, 100)   # 100 valeurs sur l'axe y
    Xg, Yg = np.meshgrid(X_grid, Y_grid)  # Crée la grille 2D
    Z = np.zeros_like(Xg)                 # Initialise la matrice des valeurs de la fonction

    for i in range(Xg.shape[0]):   # Pour chaque ligne de la grille
        for j in range(Xg.shape[1]):  # Pour chaque colonne
            vec    = np.zeros(D)      # Vecteur de D zéros
            vec[0] = Xg[i, j]        # Coordonnée x
            vec[1] = Yg[i, j]        # Coordonnée y
            Z[i, j] = func(vec)      # Valeur de la fonction en ce point

    st.markdown("### Application of PSO")

    # ---- GRAPHIQUE 1 : PREMIÈRE ITÉRATION ----
    st.subheader("Search History - 1st Iteration")

    init_fitness = np.array([func(x) for x in first_pos])  # Évalue les positions initiales
    best_init    = first_pos[np.argmin(init_fitness)]       # Trouve la meilleure position initiale

    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.contour(Xg, Yg, Z, levels=30, cmap="viridis")                    # Dessine les courbes de niveau
    ax1.scatter(first_pos[:, 0], first_pos[:, 1], c="black", s=10)       # Toutes les particules en noir
    ax1.scatter(best_init[0], best_init[1], c="red", s=80)               # Meilleure particule en rouge
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    st.pyplot(fig1, use_container_width=True)

    # ---- GRAPHIQUE 2 : DERNIÈRE ITÉRATION ----
    st.subheader("Search History - Final Iteration")

    final_fitness = np.array([func(x) for x in final_pos])  # Évalue les positions finales
    best_final    = final_pos[np.argmin(final_fitness)]      # Trouve la meilleure position finale

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.contour(Xg, Yg, Z, levels=30, cmap="viridis")
    ax2.scatter(final_pos[:, 0], final_pos[:, 1], c="black", s=10)  # Positions finales en noir
    ax2.scatter(best_final[0], best_final[1], c="red", s=80)        # Meilleure finale en rouge
    ax2.set_xlim(low, high)
    ax2.set_ylim(low, high)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    st.pyplot(fig2, use_container_width=True)

    # ---- STATISTIQUES ----
    st.subheader("Statistics")
    st.markdown("**Initial population:**")
    st.markdown(f"Best — {np.min(init_fitness):.2f}, Worst — {np.max(init_fitness):.2f}")

    st.markdown("**Final population:**")
    st.markdown(f"Best — {final_best:.4f}")  # Meilleur score final trouvé par PSO

    st.markdown(f"**Stagnation — Iteration N°{last_iter}**")  # Itération où PSO s'est arrêté

    # ---- COURBE DE CONVERGENCE ----
    st.subheader("Convergence Curve")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(best_curve)        # Trace le meilleur fitness par itération
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Best Fitness")
    st.pyplot(fig3, use_container_width=True)

    # ---- FITNESS MOYENNE ----
    st.subheader("Average Fitness")

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(avg_curve)         # Trace la fitness moyenne de la population par itération
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Average Fitness")
    st.pyplot(fig4, use_container_width=True)

    # ---- TRAJECTOIRE DE LA 1ÈRE PARTICULE ----
    st.subheader("Trajectory of 1st Particle")

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(traj[:, 0], traj[:, 1])  # Trace le chemin (x1, x2) parcouru par la 1ère particule
    ax5.set_xlabel("x1")
    ax5.set_ylabel("x2")
    st.pyplot(fig5, use_container_width=True)


# ==============================================================================
# PARTIE MULTI-RUNS — PSO lancé plusieurs fois
# Le PSO est aléatoire → les résultats varient à chaque run.
# On lance plusieurs fois et on moyenne pour des stats plus fiables.
# ==============================================================================

st.markdown("---")  # Ligne de séparation horizontale
st.header("Running PSO with multiple populations")

with st.container(border=True):  # Encadré visuel

    st.subheader("Running Multiple Populations")

    # Curseur pour choisir combien de fois on relance le PSO
    multi_runs = st.slider("Multiple run:", min_value=1, max_value=100, value=30, key="multi_runs_slider")

    if st.button("Evaluate", key="btn_multi_run"):  # Bouton de lancement

        func = functions[func_name]  # Récupère la fonction choisie

        # Prépare la grille de visualisation (même logique que précédemment)
        X_grid = np.linspace(low, high, 100)
        Y_grid = np.linspace(low, high, 100)
        Xg, Yg = np.meshgrid(X_grid, Y_grid)
        Z = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                vec    = np.zeros(D)
                vec[0] = Xg[i, j]
                vec[1] = Yg[i, j]
                Z[i, j] = func(vec)

        # Listes pour stocker les résultats de chaque run
        all_best_curves   = []  # Courbes de convergence de chaque run
        all_avg_curves    = []  # Courbes de fitness moyenne de chaque run
        all_traj_x1       = []  # Trajectoires en x1 de chaque run
        all_first_pos     = []  # Positions initiales de chaque run
        all_final_pos     = []  # Positions finales de chaque run
        all_gbest_pos     = []  # Meilleure position globale de chaque run
        all_gbest_fitness = []  # Meilleure fitness globale de chaque run
        all_history_pos   = []  # Historique complet des positions de chaque run

        progress = st.progress(0, text="Running PSO experiments...")  # Barre de progression

        for r in range(multi_runs):  # Répète le PSO multi_runs fois

            # Lance un run de PSO et récupère tous les résultats
            first_pos, final_pos, best_curve, avg_curve, traj, gbest_fit, _, gbest_pos, hist_pos = PSO(
                func, D, population_size, low, high, T, w, c1, c2
            )

            all_best_curves.append(best_curve)      # Sauvegarde la courbe de convergence
            all_avg_curves.append(avg_curve)         # Sauvegarde la courbe moyenne
            all_traj_x1.append(traj[:, 0])           # Sauvegarde la trajectoire en x1
            all_first_pos.append(first_pos)           # Sauvegarde les positions initiales
            all_final_pos.append(final_pos)           # Sauvegarde les positions finales
            all_gbest_pos.append(gbest_pos[:2])       # Sauvegarde les 2 premières coordonnées du meilleur
            all_gbest_fitness.append(gbest_fit)       # Sauvegarde la meilleure fitness
            all_history_pos.append(hist_pos)          # Sauvegarde l'historique complet

            progress.progress((r + 1) / multi_runs, text=f"Run {r+1}/{multi_runs}...")  # Met à jour la barre

        progress.empty()  # Efface la barre de progression une fois terminé

        # Trouve la longueur maximale des courbes (elles peuvent varier à cause de la stagnation)
        max_len_best = max(len(c) for c in all_best_curves)
        max_len_avg  = max(len(c) for c in all_avg_curves)
        max_len_traj = max(len(t) for t in all_traj_x1)

        def pad(arr_list, length):
            """
            Complète chaque tableau avec sa dernière valeur pour qu'ils aient tous la même longueur.
            Cela permet de les empiler et de calculer des moyennes.
            """
            padded = []
            for a in arr_list:
                if len(a) < length:  # Si le tableau est trop court...
                    a = np.concatenate([a, np.full(length - len(a), a[-1])])  # ...complète avec la dernière valeur
                padded.append(a)
            return np.array(padded)  # Retourne un tableau 2D

        # Complète et empile toutes les courbes
        best_mat = pad(all_best_curves, max_len_best)  # Matrice (multi_runs × max_len_best)
        avg_mat  = pad(all_avg_curves,  max_len_avg)   # Matrice (multi_runs × max_len_avg)
        traj_mat = pad(all_traj_x1,     max_len_traj)  # Matrice (multi_runs × max_len_traj)

        # Calcule les courbes moyennes sur tous les runs
        mean_best_curve = best_mat.mean(axis=0)  # Moyenne colonne par colonne (par itération)
        mean_avg_curve  = avg_mat.mean(axis=0)
        mean_traj_x1    = traj_mat.mean(axis=0)

        # Statistiques globales sur les meilleures fitness de chaque run
        gbest_arr            = np.array(all_gbest_fitness)  # Tableau des meilleures fitness
        overall_best_fitness = np.min(gbest_arr)            # Meilleure parmi tous les runs
        overall_mean_fitness = np.mean(gbest_arr)           # Moyenne des meilleures par run
        overall_std_fitness  = np.std(gbest_arr)            # Écart-type (variabilité)

        best_run_idx  = np.argmin(all_gbest_fitness)   # Indice du meilleur run
        best_overall  = all_gbest_pos[best_run_idx]    # Position de la meilleure solution globale
        gbest_pos_arr = np.array(all_gbest_pos)        # Tableau de toutes les meilleures positions

        # Disposition en 3 colonnes
        col_left, col_mid, col_right = st.columns([1.2, 1.5, 1.5])

        with col_left:
            # ---- GRAPHIQUE 3D ----
            st.markdown(f"**Function ({func_name})**")
            fig_3d = plt.figure(figsize=(6, 5))
            ax_3d  = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.9)  # Surface 3D de la fonction
            ax_3d.set_xlabel("x₁", fontsize=8)
            ax_3d.set_ylabel("x₂", fontsize=8)
            ax_3d.tick_params(labelsize=7)
            st.pyplot(fig_3d, use_container_width=True)

        with col_mid:
            # ---- PREMIÈRE ITÉRATION (tous les runs) ----
            st.markdown("**Search History — First Iteration**")
            fig_first, ax_first = plt.subplots(figsize=(7, 6))
            ax_first.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)  # Fond coloré
            ax_first.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.5)  # Lignes de contour

            for fp in all_first_pos:  # Pour chaque run
                ax_first.scatter(fp[:, 0], fp[:, 1], c="black", s=6, alpha=0.3, zorder=3)  # Toutes les particules

            for fp in all_first_pos:  # Pour chaque run
                fit_fp   = np.array([func(x) for x in fp])  # Évalue les positions initiales
                best_fp  = fp[np.argmin(fit_fp)]              # Meilleure particule du run
                ax_first.scatter(best_fp[0], best_fp[1], c="orange", s=40, zorder=4, alpha=0.6)  # En orange

            ax_first.set_xlim(low, high)
            ax_first.set_ylim(low, high)
            ax_first.set_title(f"Search History ({func_name}), First Iteration", fontsize=9)
            ax_first.set_xlabel("x₁")
            ax_first.set_ylabel("x₂")

            from matplotlib.lines import Line2D  # Pour créer une légende personnalisée
            legend_first = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markersize=5, label="All particles (init)"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                       markersize=7, label="Best particle per run (init)"),
            ]
            ax_first.legend(handles=legend_first, fontsize=7, loc="upper right")
            st.pyplot(fig_first, use_container_width=True)

        with col_right:
            # ---- DERNIÈRE ITÉRATION (tous les runs) ----
            st.markdown("**Search History — Final Iteration**")
            fig_final, ax_final = plt.subplots(figsize=(7, 6))
            ax_final.contourf(Xg, Yg, Z, levels=30, cmap="Blues_r", alpha=0.7)
            ax_final.contour(Xg, Yg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.5)

            for hist in all_history_pos:      # Pour chaque run
                for step_pos in hist:         # Pour chaque itération du run
                    ax_final.scatter(step_pos[:, 0], step_pos[:, 1],
                                     c="black", s=2, alpha=0.04, zorder=2)  # Trace toutes les positions (très transparent)

            ax_final.scatter(gbest_pos_arr[:, 0], gbest_pos_arr[:, 1],
                             c="orange", s=60, zorder=5, alpha=0.85,
                             label="Best solution per run")  # Meilleure solution de chaque run
            ax_final.scatter(best_overall[0], best_overall[1],
                             c="red", s=60, zorder=6, marker="o",
                             edgecolors="white", linewidths=1,
                             label="Best global solution")  # Meilleure solution parmi tous les runs

            ax_final.set_xlim(low, high)
            ax_final.set_ylim(low, high)
            ax_final.set_title(f"Search History ({func_name}), Final Iteration", fontsize=9)
            ax_final.set_xlabel("x₁")
            ax_final.set_ylabel("x₂")
            ax_final.legend(fontsize=7, loc="upper right")
            st.pyplot(fig_final, use_container_width=True)

        # ---- MÉTRIQUES ----
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Best", f"{overall_best_fitness:.4f}")       # Meilleure fitness sur tous les runs
        with stat_col2:
            st.metric("Mean (avg error)", f"{overall_mean_fitness:.4f}")  # Fitness moyenne
        with stat_col3:
            st.metric("STD", f"{overall_std_fitness:.4f}")          # Écart-type

        # ---- 3 GRAPHIQUES EN BAS ----
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            st.markdown("**Convergence Curve**")
            st.caption("Mean Best Fitness of All Runs vs. Iteration")
            fig_cc, ax_cc = plt.subplots(figsize=(6, 4))
            ax_cc.plot(mean_best_curve, color="red")  # Courbe de convergence moyenne
            ax_cc.set_xlabel("Iteration")
            ax_cc.set_ylabel("Fitness")
            ax_cc.set_title("Convergence Curve")
            st.pyplot(fig_cc, use_container_width=True)

        with col_b2:
            st.markdown("**Trajectory of the First Solution in the Population**")
            st.caption("Mean x₁⁽¹⁾ of all Runs vs. Iteration")
            fig_tr, ax_tr = plt.subplots(figsize=(6, 4))
            ax_tr.plot(mean_traj_x1, color="green")  # Trajectoire moyenne de x1 de la 1ère particule
            ax_tr.set_xlabel("Iteration")
            ax_tr.set_ylabel("x₁⁽¹⁾")
            ax_tr.set_title("Trajectory of 1st solution")
            st.pyplot(fig_tr, use_container_width=True)

        with col_b3:
            st.markdown("**Average Population Fitness**")
            st.caption("Mean Population Average Fitness of All Runs vs. Iteration")
            fig_af, ax_af = plt.subplots(figsize=(6, 4))
            ax_af.plot(mean_avg_curve, color="blue")  # Fitness moyenne de la population (moyennée sur les runs)
            ax_af.set_xlabel("Iteration")
            ax_af.set_ylabel("Fitness")
            ax_af.set_title("Average Fitness of population")
            st.pyplot(fig_af, use_container_width=True)


# ==============================================================================
# PARTIE 3 — SÉLECTION DE FEATURES AVEC PSO
#
# Objectif : trouver automatiquement les features (colonnes) les plus utiles
# dans un dataset pour entraîner un classifieur KNN avec la meilleure précision,
# tout en utilisant le moins de features possible.
# ==============================================================================

st.markdown("---")
st.header("Feature Selection with PSO")


# ==============================================================================
# CHARGEMENT DES DATASETS
# @st.cache_data = le résultat est mis en cache → le dataset n'est chargé qu'une seule fois
# ==============================================================================

@st.cache_data
def load_synthetic():
    # Crée un dataset artificiel avec 1000 exemples et 50 features
    # Dont seulement 5 sont vraiment informatives
    X, y = make_classification(
        n_samples=1000,     # 1000 exemples
        n_features=50,      # 50 features au total
        n_informative=5,    # Seulement 5 features réellement utiles
        n_redundant=10,     # 10 features redondantes (combinaisons des informatives)
        random_state=42     # Graine aléatoire pour reproductibilité
    )
    return X, y

@st.cache_data
def load_digits_data():
    # Charge le dataset "Digits" de sklearn :
    # 1797 images de chiffres manuscrits (0 à 9), chacune décrite par 64 pixels
    digits = load_digits()
    return digits.data, digits.target


# ==============================================================================
# FONCTION D'ÉVALUATION KNN
# Évalue la précision d'un classifieur KNN sur un sous-ensemble de features
# ==============================================================================

def evaluate_knn(X, y, selected_indices, k=5):
    """
    X               : toutes les données (exemples × features)
    y               : étiquettes (classes)
    selected_indices: indices des features à utiliser
    k               : nombre de voisins pour le KNN
    """
    X_sel = X[:, selected_indices]  # Ne garde que les features sélectionnées
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.3, random_state=42  # 70% entraînement, 30% test
    )
    knn = KNeighborsClassifier(n_neighbors=k)  # Crée le classifieur KNN
    knn.fit(X_train, y_train)                  # Entraîne sur les données d'entraînement
    y_pred = knn.predict(X_test)               # Prédit sur les données de test
    return accuracy_score(y_test, y_pred)      # Retourne le % de bonnes prédictions


# ==============================================================================
# FONCTION DE FITNESS POUR LA SÉLECTION DE FEATURES
# f(x) = α × (1 - accuracy) + (1 - α) × (SF / D_total)
#
# Le but est de MINIMISER cette valeur, donc :
# - Minimiser (1 - accuracy) = maximiser la précision
# - Minimiser (SF / D_total) = utiliser le moins de features possible
# - α contrôle l'équilibre entre ces deux objectifs
# ==============================================================================

def fitness_fs(solution, X, y, SF, alpha, k=5):
    """
    solution : vecteur de la particule (valeurs entre 0 et 1 pour chaque feature)
    X        : données
    y        : étiquettes
    SF       : nombre de features à sélectionner
    alpha    : poids de la précision dans le calcul de la fitness
    k        : nombre de voisins KNN
    """
    D_feat  = X.shape[1]                         # Nombre total de features dans le dataset
    indices = np.argsort(solution)[-SF:]          # Sélectionne les SF features avec les plus grandes valeurs dans la solution
    accuracy = evaluate_knn(X, y, indices, k)    # Évalue la précision KNN avec ces features
    f1_val = 1.0 - accuracy                      # Erreur de classification (à minimiser)
    f2_val = SF / D_feat                         # Proportion de features utilisées (à minimiser)
    return alpha * f1_val + (1 - alpha) * f2_val, accuracy, sorted(indices.tolist())
    # Retourne : valeur de fitness, précision, et liste triée des indices sélectionnés


# ==============================================================================
# PSO POUR LA SÉLECTION DE FEATURES
# Même logique que le PSO classique, mais adapté à la sélection de features.
# Chaque particule est un vecteur de valeurs entre 0 et 1 (une valeur par feature).
# Les SF features avec les valeurs les plus élevées sont sélectionnées.
# ==============================================================================

def PSO_FS(X, y, SF, alpha, N=30, T=100, w=0.5, c1=2.0, c2=2.0, k_knn=5):
    """
    X      : données
    y      : étiquettes
    SF     : nombre de features à sélectionner
    alpha  : poids de la précision dans la fitness
    N      : nombre de particules
    T      : nombre max d'itérations
    w, c1, c2 : hyperparamètres PSO
    k_knn  : nombre de voisins KNN
    """
    D_feat      = X.shape[1]          # Nombre de features (dimensions de la solution)
    low_fs      = 0.0                 # Borne inférieure (chaque dimension entre 0 et 1)
    high_fs     = 1.0                 # Borne supérieure
    vmax        = 0.2 * (high_fs - low_fs)  # Vitesse max = 20% de la plage

    pos = np.random.uniform(low_fs, high_fs, (N, D_feat))  # Positions initiales aléatoires
    vel = np.zeros((N, D_feat))                             # Vitesses initiales à 0

    pbest     = pos.copy()  # Meilleurs personnels = positions initiales
    pbest_fit = np.array([fitness_fs(pos[i], X, y, SF, alpha, k_knn)[0] for i in range(N)])  # Fitness initiales

    gbest_idx = np.argmin(pbest_fit)    # Indice du meilleur global initial
    gbest     = pbest[gbest_idx].copy() # Position du meilleur global
    gbest_fit = pbest_fit[gbest_idx]    # Fitness du meilleur global

    stag = 0  # Compteur de stagnation
    for t in range(T):  # Boucle PSO
        old_best = gbest_fit
        for i in range(N):  # Pour chaque particule
            r1, r2 = np.random.rand(D_feat), np.random.rand(D_feat)  # Nombres aléatoires
            # Mise à jour de la vitesse (formule PSO standard)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gbest - pos[i])
            vel[i] = np.clip(vel[i], -vmax, vmax)             # Limite la vitesse
            pos[i] = np.clip(pos[i] + vel[i], low_fs, high_fs)  # Déplace et garde dans [0, 1]

        # Évalue toutes les particules
        fits = np.array([fitness_fs(pos[i], X, y, SF, alpha, k_knn)[0] for i in range(N)])

        for i in range(N):  # Met à jour les meilleurs personnels
            if fits[i] < pbest_fit[i]:
                pbest[i]     = pos[i].copy()
                pbest_fit[i] = fits[i]

        best_i = np.argmin(pbest_fit)  # Meilleur de l'essaim
        if pbest_fit[best_i] < gbest_fit:  # Si meilleur que le meilleur global
            gbest     = pbest[best_i].copy()
            gbest_fit = pbest_fit[best_i]

        stag = stag + 1 if gbest_fit == old_best else 0  # Stagnation ?
        if stag >= 20:  # Arrêt si 20 itérations sans amélioration
            break

    # Évalue la solution finale pour obtenir précision et indices
    final_fit, final_acc, final_indices = fitness_fs(gbest, X, y, SF, alpha, k_knn)
    return gbest, final_fit, final_acc, final_indices


# ==============================================================================
# INTERFACE — Sélection de Features
# ==============================================================================

fs_col1, fs_col2, fs_col3 = st.columns([1, 1, 2])  # Disposition en 3 colonnes

with fs_col1:
    st.markdown("**Data**")
    # Boutons radio pour choisir le dataset
    dataset_choice = st.radio("Dataset", ["Synthetic", "Digits"], key="fs_dataset", label_visibility="collapsed")

with fs_col2:
    # Nombre de features à sélectionner
    SF       = st.number_input("Selected Features (SF)", min_value=0, max_value=64, value=5, key="fs_sf")
    # Paramètre alpha : équilibre précision vs. nombre de features
    alpha_fs = st.number_input("α", min_value=0.0, max_value=1.0, value=0.9, step=0.05, key="fs_alpha")

with fs_col3:
    st.markdown("")
    st.markdown("")

    # Menu déroulant pour charger un cas prédéfini
    case_options  = ["— None (manual) —"] + list(PREDEFINED_CASES.keys())
    selected_case = st.selectbox("Load predefined test case", case_options, key="fs_case")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        btn_eval   = st.button("Model Evaluation",    key="btn_fs_eval")    # Lance PSO ou charge le cas prédéfini
    with btn_col2:
        btn_reeval = st.button("Model Re-evaluation", key="btn_fs_reeval")  # Réévalue avec de nouveaux paramètres

# Chargement du dataset choisi
if dataset_choice == "Synthetic":
    X_fs, y_fs = load_synthetic()   # Dataset artificiel
else:
    X_fs, y_fs = load_digits_data() # Dataset de chiffres manuscrits

D_fs = X_fs.shape[1]  # Nombre total de features du dataset


# ==============================================================================
# SESSION STATE — Stockage persistant entre les clics
# Streamlit recharge tout le script à chaque interaction.
# st.session_state permet de garder des valeurs entre les clics.
# ==============================================================================

if "fs_solution" not in st.session_state:  # Si pas encore initialisé...
    st.session_state.fs_solution = None    # Vecteur solution de la particule gagnante
    st.session_state.fs_indices  = None    # Indices des features sélectionnées
    st.session_state.fs_fitness  = None    # Valeur de fitness
    st.session_state.fs_accuracy = None    # Précision KNN
    st.session_state.fs_SF       = None    # Nombre de features sélectionnées


# ==============================================================================
# VALIDATION DES ENTRÉES
# Vérifie que les paramètres sont valides avant de lancer le calcul
# ==============================================================================

def validate_inputs(sf_val, alpha_val):
    """Retourne (est_valide, message_d_erreur)."""
    if sf_val == 0:       # SF = 0 n'a pas de sens (aucune feature sélectionnée)
        return False, "❌ **Selected Features (SF) cannot be 0.** Please choose at least 1 feature."
    if alpha_val == 0.0:  # Alpha = 0 ignorerait complètement la précision
        return False, "❌ **Alpha (α) cannot be 0.** A value of 0 means the classifier accuracy is completely ignored. Please set α > 0."
    return True, ""  # Tout est valide


# ==============================================================================
# ACTION : BOUTON "MODEL EVALUATION"
# Lance PSO pour trouver les meilleures features, OU charge un cas prédéfini
# ==============================================================================

if btn_eval:
    is_valid, err_msg = validate_inputs(int(SF), float(alpha_fs))  # Vérifie les paramètres
    if not is_valid:
        st.error(err_msg)  # Affiche l'erreur et ne continue pas
    else:
        if selected_case != "— None (manual) —":  # Si un cas prédéfini est sélectionné
            case_data    = PREDEFINED_CASES[selected_case]     # Récupère les données du cas
            solution_arr = np.array(case_data["solution"])     # Convertit la solution en tableau NumPy

            # Ajuste la taille de la solution si nécessaire
            if len(solution_arr) < D_fs:   # Si la solution est plus courte que le nombre de features...
                solution_arr = np.concatenate([solution_arr, np.random.uniform(0, 1, D_fs - len(solution_arr))])
            elif len(solution_arr) > D_fs: # Si elle est plus longue...
                solution_arr = solution_arr[:D_fs]  # ...on tronque

            sf_to_use    = case_data["SF"]     # Nombre de features du cas prédéfini
            alpha_to_use = case_data["alpha"]  # Alpha du cas prédéfini

            # Évalue la solution prédéfinie
            fit_val, acc_val, sel_indices = fitness_fs(
                solution_arr, X_fs, y_fs, SF=sf_to_use, alpha=alpha_to_use
            )

            # Stocke les résultats dans la session
            st.session_state.fs_solution = solution_arr
            st.session_state.fs_indices  = sel_indices
            st.session_state.fs_fitness  = fit_val
            st.session_state.fs_accuracy = acc_val
            st.session_state.fs_SF       = sf_to_use
            st.session_state.fs_alpha    = alpha_to_use

            st.info(f"📂 Loaded predefined solution: **{selected_case}**")

        else:  # Sinon, lance le PSO normalement
            with st.spinner("Running PSO for Feature Selection..."):  # Indicateur de chargement
                solution, fit_val, acc_val, sel_indices = PSO_FS(
                    X_fs, y_fs, SF=int(SF), alpha=float(alpha_fs),
                    N=30, T=100, w=0.5, c1=2.0, c2=2.0
                )
            # Stocke les résultats dans la session
            st.session_state.fs_solution  = solution
            st.session_state.fs_indices   = sel_indices
            st.session_state.fs_fitness   = fit_val
            st.session_state.fs_accuracy  = acc_val
            st.session_state.fs_SF        = int(SF)
            st.session_state.fs_alpha     = float(alpha_fs)


# ==============================================================================
# ACTION : BOUTON "MODEL RE-EVALUATION"
# Réévalue la solution déjà trouvée avec de nouveaux paramètres SF / alpha
# (sans relancer le PSO → plus rapide)
# ==============================================================================

if btn_reeval:
    if st.session_state.fs_solution is None:  # Si aucune solution n'a encore été calculée
        st.warning("Please run Model Evaluation first.")
    else:
        is_valid, err_msg = validate_inputs(int(SF), float(alpha_fs))
        if not is_valid:
            st.error(err_msg)
        else:
            solution = st.session_state.fs_solution  # Récupère la solution déjà calculée
            fit_val, acc_val, sel_indices = fitness_fs(
                solution, X_fs, y_fs, SF=int(SF), alpha=float(alpha_fs)  # Réévalue avec nouveaux paramètres
            )
            # Met à jour les résultats dans la session
            st.session_state.fs_indices   = sel_indices
            st.session_state.fs_fitness   = fit_val
            st.session_state.fs_accuracy  = acc_val
            st.session_state.fs_SF        = int(SF)
            st.session_state.fs_alpha     = float(alpha_fs)


# ==============================================================================
# AFFICHAGE DES RÉSULTATS DE LA SÉLECTION DE FEATURES
# ==============================================================================

if st.session_state.fs_solution is not None:  # Si une solution existe dans la session

    solution  = st.session_state.fs_solution  # Vecteur solution
    sel_idx   = st.session_state.fs_indices   # Indices des features sélectionnées
    fit_val   = st.session_state.fs_fitness   # Valeur de fitness
    acc_val   = st.session_state.fs_accuracy  # Précision KNN
    sf_val    = st.session_state.fs_SF        # Nombre de features
    alpha_val = st.session_state.get("fs_alpha", float(alpha_fs))  # Alpha utilisé

    # Formate le vecteur solution en chaîne lisible (ex: "0.74 | 0.56 | ...")
    sol_str = " | ".join([f"{v:.2f}" for v in solution])
    # Formate les indices sélectionnés (ex: "0 | 2 | 7 | ...")
    idx_str = " | ".join([str(i) for i in sel_idx])

    with st.container(border=True):
        st.markdown("**Solution**")
        st.text_area(
            label="Solution details",
            value=f"Solution:\n{sol_str}\n\nIndices of selected features:\n{idx_str}",
            height=160,
            label_visibility="collapsed"  # Cache le label (déjà affiché au-dessus)
        )

    # Résumé en une ligne : fitness, précision, nombre de features, alpha
    st.markdown(
        f"**Fitness** — {fit_val:.2f}, "
        f"**Accuracy** — {acc_val:.2f}, "
        f"**Selected Features** — {sf_val}, "
        f"**α** — {alpha_val}"
    )
