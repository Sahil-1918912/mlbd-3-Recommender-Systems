# MLBD-3: Recommender Systems 

This project implements multiple recommender system approaches in a single notebook: `recommender.ipynb`, using the MovieLens dataset in `data/movielens/`.

## Project structure

```
.
├── recommender.ipynb
├── data/
│   └── movielens/
│       ├── movies.csv
│       ├── ratings.csv
│       ├── links.csv
│       ├── tags.csv
│       └── README.txt
└── images/
```

## What is implemented in the notebook

### Part 1: Content-Based Filtering
- Task 1: TF-IDF based movie recommendation (genres + cosine similarity)
- Task 2: User-profile based content recommender

### Part 2: Collaborative Filtering
- Task 3: User-based collaborative filtering
- Task 4: Item-based collaborative filtering
- Includes RMSE, Precision@K, Recall@K evaluation pipelines

### Part 3: Matrix Factorization
- Task 5: SVD using `scipy.sparse.linalg.svds`
- Task 6: Matrix factorization with `surprise` library (`scikit-surprise`)

### Part 4: Hybrid Recommenders
- Task 7: Hybrid recommendation model
	- Weighted hybrid (content + collaborative)
	- Meta-learning hybrid using `RandomForestRegressor`

### Part 5: Advanced Methods
- Task 8: Content-based recommendation with a neural network (`tensorflow/keras`)
- Task 9: Reinforcement-learning inspired recommenders
	- Epsilon-greedy multi-armed bandit
	- UCB
	- Q-learning

### Part 6: Explainability
- Task 10: Feature-based explanations with SHAP
- Task 11: Neighborhood-based explanations for CF
- Task 12: Model-agnostic explanations with LIME
- Task 13: Explainability discussion and bias analysis

## Python version

Recommended: **Python 3.10 or 3.11**

(`scikit-surprise` and TensorFlow are typically less reliable on Python 3.12 in many setups.)

## Dependency installation

You can install everything at once, or install in stages as needed.

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2) Base dependencies (required for most tasks)

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter ipykernel
```

### 3) Install dependencies when needed by notebook tasks

- For **Task 6 (Surprise SVD)**:

```bash
pip install scikit-surprise
```

- For **Task 8 (Neural Network recommender)**:

```bash
pip install tensorflow
```

- For **Task 10 (SHAP explainability)**:

```bash
pip install shap
```

- For **Task 12 (LIME explainability)**:

```bash
pip install lime
```

### 4) Install all optional dependencies in one shot

```bash
pip install scikit-surprise tensorflow shap lime
```

## Linux troubleshooting (important)

If `pip install scikit-surprise` fails on Linux, install build tools first:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev
pip install cython numpy
pip install scikit-surprise
```

If TensorFlow install fails due to compatibility, verify Python version:

```bash
python --version
```

Then use Python 3.10/3.11 in your virtual environment.

## Running the notebook

```bash
jupyter notebook recommender.ipynb
```

or in VS Code:
1. Open `recommender.ipynb`
2. Select your `.venv` Python interpreter/kernel
3. Run cells in order from top to bottom

## Notes

- The notebook contains direct install cells for some packages:
	- `!pip install scikit-surprise`
	- `!pip install shap`
	- `!pip install lime`
- Prefer installing in the environment before running to avoid kernel/package mismatch.


## Author
Sahil Narkhede - B23CS1060