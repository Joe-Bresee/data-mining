# SENG 474 — Data Mining

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Supported-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Supported-150458.svg)](https://pandas.pydata.org/)

Repository containing coursework, laboratory implementations, and comprehensive machine learning pipeline projects for **SENG 474: Data Mining** (Summer 2026) by **Joseph Bresee**.

This repository showcases foundational data mining methodologies, mathematical derivations, from-scratch algorithmic implementations (such as a Multi-Layer Perceptron and Logistic Regression with backpropagation), robust feature preprocessing pipelines, and practical model evaluations across diverse benchmark datasets.

---

## Table of Contents

- [Repository Overview](#repository-overview)
- [Repository Structure](#repository-structure)
- [Assignment 1 — End-to-End Pipeline & Custom Models](#assignment-1--end-to-end-pipeline--custom-models)
  - [Key Findings & Performance Comparison](#key-findings--performance-comparison)
- [Laboratory Modules](#laboratory-modules)
  - [Lab 01: Exploratory Data Analysis & Visualization](#lab-01-exploratory-data-analysis--visualization)
  - [Lab 02: Feature Engineering & Preprocessing Pipelines](#lab-02-feature-engineering--preprocessing-pipelines)
  - [Lab 03: Linear, Polynomial, & Regularized Regression](#lab-03-linear-polynomial--regularized-regression)
  - [Lab 04: Unsupervised Clustering & K-Means](#lab-04-unsupervised-clustering--k-means)
  - [Lab 05: Classification & Evaluation Metrics](#lab-05-classification--evaluation-metrics)
  - [Lab 06: Logistic Regression & Decision Boundaries](#lab-06-logistic-regression--decision-boundaries)
  - [Lab 07: Support Vector Machines (SVM) & RBF Kernels](#lab-07-support-vector-machines-svm--rbf-kernels)
  - [Lab 10: Word Embeddings & Vector Semantics](#lab-10-word-embeddings--vector-semantics)
- [Setup & Installation](#setup--installation)
- [Tech Stack & Dependencies](#tech-stack--dependencies)

---

## Repository Overview

The repository covers both theoretical foundations and hands-on empirical evaluations across key domains in data mining and applied machine learning:

1. **Data Preprocessing & Cleaning:** Missing data imputation (median/mean), extreme outlier detection using IQR and K-Means distance metrics, logarithmic transformations (`log1p`), and robust Min-Max and Standard scaling.
2. **From-Scratch Algorithm Implementation:** Object-oriented estimators adhering to Scikit-Learn API standards (`BaseEstimator`, `ClassifierMixin`), implementing forward propagation, gradient descent, and backpropagation without high-level automated differentiation engines.
3. **Model Evaluation & Generalization:** Stratified $K$-fold cross-validation, learning curves, bias-variance trade-off diagnosis, ROC/AUC metrics, precision-recall trade-offs, and confusion matrix analysis.
4. **Natural Language Processing:** Vector space modeling, GloVe word embeddings, dimensionality reduction (PCA), cosine similarity metrics, and semantic vector arithmetic.

---

## Repository Structure

```text
data-mining/
├── a1/                                           # Major Project: Comprehensive Pipeline & From-Scratch Models
│   ├── Data_Mining_Assignment_1_post.ipynb       # Full reproducible Jupyter notebook
│   ├── Data_Mining_Assignment_1_post.html        # Exported HTML report with interactive outputs & charts
│   ├── Data_Mining_Assignment_1_post.pdf         # Formatted PDF report
│   ├── data_mining_a1.pdf                        # Project specifications & rubric
│   └── readme.md                                 # Technical summary report and experimental results
│
├── labs/                                         # Lab Notebooks with Empirical Experiments & Exercises
│   ├── lab_01_data_analysis (1).ipynb            # EDA, stratified sampling, distribution analysis
│   ├── lab_02_joseph_bresee.ipynb                # Feature engineering, imputation, custom pipeline construction
│   ├── lab_03_joseph_bresee.ipynb                # Normal equation, polynomial regression, learning curves, Ridge
│   ├── lab_04_joseph_bresee.ipynb                # Lloyd's K-Means algorithm, non-convex clustering analysis
│   ├── lab_05_joseph_bresee.ipynb                # Binary classification, SGD, ROC/AUC, confusion matrices (MNIST)
│   ├── lab_06_logistic_regression_post1_*.ipynb  # Logistic regression, sigmoid math, 2D decision boundary equations
│   ├── lab_07_svm_post1_joseph_bresee.ipynb      # Support Vector Machines, soft margins (C), RBF kernel (gamma)
│   └── lab_10_word_embeddings_post1.ipynb        # GloVe embeddings, PCA 3D projection, vector arithmetic
│
└── README.md                                     # Main repository documentation
```

---

## Assignment 1 — End-to-End Pipeline & Custom Models

Located in `a1/`, this project implements an end-to-end data mining and machine learning workflow on the **Wine Quality dataset** (6,821 samples, 13 features) covering regression, binary classification, and clustering tasks.

### Pipeline Highlights
- **Exploratory Data Analysis (EDA):** Discovered extreme synthetic anomalies by computing sample distances to K-Means cluster centroids (cutoff distance $\approx 8$ on logarithmic histograms). Identified $18.87\%$ missingness in the `alcohol` attribute.
- **Preprocessing Pipeline:**
  - *Median Imputation:* Handled missing values without skewing distributions.
  - *Feature Pruning:* Dropped `volatile acidity` due to negative anomalies and near-zero correlation; dropped `free sulfur dioxide` due to high collinearity with `total sulfur dioxide`.
  - *Distribution Normalization:* Applied `log1p` transformations to right-skewed features (`sulphates`, `residual sugar`).
  - *Outlier Filtering:* Filtered extreme non-target records using Interquartile Range ($Q_1 - 3 \times \text{IQR}$ to $Q_3 + 3 \times \text{IQR}$).
  - *Normalization:* Min-Max scaled features to $[0, 1]$ for stable gradient descent.
- **Custom Model Implementations:**
  - **`JoesMLP` (Multi-Layer Perceptron):** A multi-layer neural network built from scratch with NumPy. Features configurable hidden dimensions, ReLU activations, Sigmoid classification output, and analytical backpropagation gradient updates.
  - **`JoesLogisticRegression`:** Custom binary logistic regressor implemented from scratch using batch gradient descent and sigmoid log-loss optimization.
  - **Ridge Regression (Scikit-Learn):** L2-regularized linear model tuned over regularization parameter $\alpha \in [10^{-4}, 10^3]$ to mitigate collinearity.
  - **K-Means Clustering (Scikit-Learn):** Evaluated elbow plot inertia curves identifying optimal cluster count ($k=3$) and centroid convergence behaviors.

### Key Findings & Performance Comparison

Hyperparameters were tuned via 5-Fold Stratified Cross-Validation on an 80/20 train/test split:

| Model | Implementation Type | Best Hyperparameters | Train Error / MSE | Test Error / MSE | Notes |
|---|---|---|---|---|---|
| **Multi-Layer Perceptron (`JoesMLP`)** | Custom (from scratch) | Architecture: `(128,)`, $\eta = 1.0$, 1000 iter | **0.0066** | **0.0072 (0.72%)** | Successfully captured non-linear feature interactions |
| **Logistic Regression (`JoesLogisticRegression`)** | Custom (from scratch) | $\eta = 0.5$, 1000 iter | 0.0158 | 0.0128 (1.28%) | Fast, reliable linear decision boundary |
| **Ridge Regression** | Scikit-Learn | Regularization $\alpha = 1.0$ | 0.0163 MSE | 0.0159 MSE | Stable generalization on continuous quality target |
| **K-Means Clustering** | Scikit-Learn | $k = 3$, init: `k-means++` | N/A (Inertia plateau) | N/A | Centroids reliably converged in $<300$ iterations |

---

## Laboratory Modules

Each lab in `labs/` investigates foundational theoretical concepts paired with practical coding solutions.

### Lab 01: Exploratory Data Analysis & Visualization
- **File:** `lab_01_data_analysis (1).ipynb`
- **Core Topics:** Exploring dataset schemas, descriptive statistics, feature distributions, identifying skewness, handling target variables, and stratified test set generation.
- **Datasets:** California Housing dataset and Scikit-Learn Diabetes dataset.
- **Key Practice:** Implemented `pd.cut()` categorization to create discrete bins (e.g., income and BMI categories) to maintain distribution parity via stratified train-test splits.

### Lab 02: Feature Engineering & Preprocessing Pipelines
- **File:** `lab_02_joseph_bresee.ipynb`
- **Core Topics:** Feature engineering via attribute ratios (e.g., rooms-per-household, bedrooms-per-room), automated imputation strategies with `SimpleImputer`, standard scaling with `StandardScaler`, and Scikit-Learn `Pipeline` integration.
- **Datasets:** California Housing and Scikit-Learn Diabetes.
- **Key Practice:** Built full multi-stage preprocessing pipelines ensuring zero data leakage between training and evaluation phases.

### Lab 03: Linear, Polynomial, & Regularized Regression
- **File:** `lab_03_joseph_bresee.ipynb`
- **Core Topics:** Analytical Normal Equation $\theta = (X^T X)^{-1} X^T Y$, Scikit-Learn `LinearRegression`, non-linear modeling via `PolynomialFeatures`, learning curve analysis (evaluating underfitting vs. overfitting), the bias-variance trade-off, Stochastic Gradient Descent (`SGDRegressor`), and L2 Ridge regularization.
- **Key Practice:** Generated noisy quadratic synthetic datasets ($y = -2x^2 + x + 42 + \epsilon$), constructed polynomial transformation pipelines, extracted parameter coefficients, and evaluated model RMSE.

### Lab 04: Unsupervised Clustering & K-Means
- **File:** `lab_04_joseph_bresee.ipynb`
- **Core Topics:** Mathematical formulation of Lloyd's algorithm, iterative centroid assignment and update steps, local minima sensitivity, and K-Means++ centroid initialization.
- **Datasets:** 3D Iris dataset and non-convex `make_moons` dataset.
- **Key Practice:** Compared step-by-step Lloyd's convergence with Scikit-Learn's optimized `KMeans` across multiple cluster counts $k \in \{2, 4, 8, 16\}$, evaluating algorithm behavior on non-spherical geometries.

### Lab 05: Classification & Evaluation Metrics
- **File:** `lab_05_joseph_bresee.ipynb`
- **Core Topics:** Linear classification using `SGDClassifier`, Stratified $K$-fold cross-validation (`cross_val_score`, `cross_val_predict`), confusion matrix interpretation, precision, recall, and $F_1$-score balancing, decision threshold manipulation, and ROC curves / AUC scores.
- **Dataset:** MNIST handwritten digit classification.
- **Key Practice:** Designed and evaluated a binary "4-detector" against standard baseline classifiers, plotting ROC curves and analyzing trade-offs between false positives and false negatives.

### Lab 06: Logistic Regression & Decision Boundaries
- **File:** `lab_06_logistic_regression_post1_joseph_bresee.ipynb`
- **Core Topics:** Sigmoid activation function derivation, binary cross-entropy (log-loss) objective function, probability estimation (`predict_proba`), and explicit geometric derivation of the 2D linear decision boundary:
  $$x_2 = -\frac{\theta_1 x_1 + \theta_0}{\theta_2}$$
- **Dataset:** Fisher's Iris dataset (Iris Virginica vs. non-Virginica).
- **Key Practice:** Trained logistic regression models on sepal dimensions, plotted the exact 2D decision boundary line against data points, and validated generalization via ROC/AUC analysis and confusion matrices.

### Lab 07: Support Vector Machines (SVM) & RBF Kernels
- **File:** `lab_07_svm_post1_joseph_bresee.ipynb`
- **Core Topics:** Maximal margin hyperplanes, support vectors, soft-margin formulation with slack variables, the regularization parameter $C$ (tolerance for margin violations), and the non-linear Kernel Trick using the Radial Basis Function (RBF) kernel:
  $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
- **Dataset:** Scikit-Learn `make_moons` dataset.
- **Key Practice:** Conducted grid search experiments balancing $C$ and $\gamma$ to avoid both underfitting and overfitting, verifying model robustness against newly generated noisy test instances.

### Lab 10: Word Embeddings & Vector Semantics
- **File:** `lab_10_word_embeddings_post1.ipynb`
- **Core Topics:** Distributed word representations in NLP, pre-trained GloVe embeddings (`glove.6B.50d`), Tokenizer vocabulary mapping, Principal Component Analysis (PCA) for 3D visualization, and vector distance metrics (Euclidean distance vs. Cosine similarity).
- **Key Practice:** Demonstrated semantic concept encoding through vector arithmetic:
  $$\vec{v}_{\text{aunt}} - \vec{v}_{\text{uncle}} \approx \vec{v}_{\text{woman}} - \vec{v}_{\text{man}}$$
  Confirmed gender-direction encoding across high-dimensional semantic spaces using cosine similarity.

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone git@github.com:Joe-Bresee/data-mining.git
cd data-mining
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn tensorflow jupyter
```

### 4. Launch Jupyter Notebook / Lab
```bash
jupyter notebook
# or
jupyter lab
```

> **Note on Lab 10 Word Embeddings:** Lab 10 utilizes the GloVe 50-dimensional pre-trained vectors (`glove.6B.50d.txt`). The raw embedding file is omitted from the git tree due to file size constraints and can be downloaded from [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/).

---

## Tech Stack & Dependencies

- **Language:** Python 3.x
- **Scientific Computing & Data Manipulation:** `NumPy`, `Pandas`, `SciPy`
- **Machine Learning & Modeling:** `Scikit-Learn`
- **Deep Learning / NLP Utilities:** `TensorFlow` / `Keras` (tokenization & sequences)
- **Data Visualization:** `Matplotlib`, `Seaborn`
- **Interactive Computing:** `Jupyter Notebook` / `IPython`

