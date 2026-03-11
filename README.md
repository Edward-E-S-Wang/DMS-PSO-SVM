## What is DMS-PSO-SVM?

### Basic idea

DMS-PSO-SVM is a hybrid optimization and classification framework that combines **Dynamic Multi-Swarm Particle Swarm Optimization (DMS-PSO)** with a **Support Vector Machine (SVM)**.

In this framework:

- **SVM** is used as the final classifier
- **DMS-PSO** is used to automatically optimize the key hyperparameters of SVM, typically:
  - `C`: penalty parameter
  - `gamma`: kernel parameter of the RBF kernel

Because the performance of SVM is highly sensitive to hyperparameter selection, a strong optimization strategy is often necessary. Instead of relying on manual tuning or exhaustive grid search, DMS-PSO-SVM uses an intelligent swarm-based search strategy to identify better parameter combinations efficiently.

---

### Why SVM needs optimization

SVM is a powerful supervised learning algorithm, especially suitable for:

- high-dimensional data
- small-sample datasets
- nonlinear classification tasks

However, its classification performance depends strongly on the choice of hyperparameters.

For an RBF-kernel SVM:

- `C` controls the trade-off between maximizing the margin and minimizing classification error
- `gamma` determines how far the influence of a single sample reaches in feature space

If these parameters are not properly selected:

- the model may overfit
- the decision boundary may become unstable
- generalization performance may decrease significantly

Therefore, hyperparameter optimization is a key step in building a reliable SVM classifier.

---

### What is PSO-SVM?

PSO-SVM is a common hybrid approach that uses **Particle Swarm Optimization (PSO)** to search for the optimal SVM hyperparameters.

In standard PSO:

- each particle represents a candidate solution, such as a pair of values `(C, gamma)`
- particles move through the search space by updating their velocity and position
- each particle learns from:
  - its own historical best position (**personal best**, `pbest`)
  - the best position found by the entire swarm (**global best**, `gbest`)

The algorithm iteratively updates the swarm until convergence, and the best parameter combination is then used to train the final SVM model.

This PSO-SVM framework is widely used because it is more flexible and often more efficient than manual search or grid search.

---

### Limitations of traditional PSO-SVM

Although PSO-SVM is effective, standard PSO still has several well-known limitations, especially in complex or high-dimensional optimization tasks:

1. **Premature convergence**  
   The swarm may converge too early to a local optimum, especially when particles quickly cluster around a suboptimal region.

2. **Loss of population diversity**  
   As iterations proceed, particles tend to become increasingly similar, which reduces the exploration ability of the algorithm.

3. **Weak local-global balance**  
   Standard PSO may struggle to maintain a good balance between:
   - global exploration of the search space
   - local exploitation around promising solutions

4. **Overdependence on a single global best**  
   Since all particles are influenced by the same `gbest`, the swarm may become overly centralized, limiting search diversity.

Because of these limitations, standard PSO-SVM may fail to find the best SVM parameters, particularly when the optimization landscape is complex.

---

## Improvements of DMS-PSO-SVM over PSO-SVM

DMS-PSO-SVM improves traditional PSO-SVM by replacing standard PSO with **Dynamic Multi-Swarm Particle Swarm Optimization (DMS-PSO)**.

The key idea of DMS-PSO is to divide the entire population into several smaller **sub-swarms**, and then periodically reorganize these sub-swarms during the optimization process. This dynamic mechanism enhances diversity and improves the balance between exploration and exploitation.

### 1. From single-swarm to multi-swarm optimization

In traditional PSO-SVM:

- all particles belong to one swarm
- all particles are guided by the same global best solution

In DMS-PSO-SVM:

- particles are divided into multiple **sub-swarms**
- each sub-swarm searches semi-independently
- each sub-swarm maintains its own **local best** (`lbest`)

This design allows different groups of particles to explore different regions of the parameter space simultaneously.

**Advantage:**  
It reduces the risk that the entire population becomes trapped in the same local optimum.

---

### 2. Introduction of local best learning instead of purely global learning

Traditional PSO typically emphasizes the influence of the global best particle. In contrast, DMS-PSO introduces **local best guidance** within each sub-swarm.

This means that each particle updates itself based on:

- its own best historical position (`pbest`)
- the best solution found within its current sub-swarm (`lbest`)

rather than relying only on one global leader.

**Advantage:**  
This mechanism preserves local search diversity and avoids excessive dependence on a single global optimum candidate.

---

### 3. Dynamic regrouping mechanism

One of the most important improvements in DMS-PSO is the **dynamic regrouping strategy**.

After a fixed number of iterations, particles are randomly reassigned into new sub-swarms. In your implementation, this is controlled by:

```python
if t % regroup_period == 0:
    rng.shuffle(indices)
    subswarm_idxs = np.array_split(indices, n_swarms)
````

This means that particles periodically leave their current sub-swarm and join a new one, allowing information exchange across different search groups.

**Advantage:**

* prevents long-term isolation of sub-swarms
* allows promising search patterns to spread across the population
* improves communication between particles
* helps the algorithm escape local optima

This dynamic regrouping is the core feature that distinguishes DMS-PSO from ordinary multi-swarm or standard PSO methods.

---

### 4. Better balance between exploration and exploitation

A good optimization algorithm must achieve two goals:

* **Exploration**: searching broadly across the parameter space
* **Exploitation**: refining promising candidate solutions

Standard PSO sometimes shifts too quickly from exploration to exploitation, which can lead to premature convergence.

DMS-PSO-SVM improves this balance by:

* using multiple sub-swarms for parallel exploration
* using local best guidance for focused refinement
* using regrouping to renew diversity and expand search coverage

**Advantage:**
The algorithm is generally more robust in complex optimization landscapes and more likely to find better hyperparameter combinations.

---

### 5. Improved robustness and global search capability

Because DMS-PSO maintains higher particle diversity and avoids over-centralization, it usually has better global search performance than standard PSO.

For SVM hyperparameter tuning, this means:

* more stable optimization behavior
* reduced chance of poor local solutions
* potentially better classification performance on unseen data

This is especially useful for biomedical datasets such as radiomics, where:

* feature space is often high-dimensional
* sample size is limited
* data distribution may be complex and nonlinear

---

## Compared with traditional PSO-SVM, DMS-PSO-SVM introduces several important improvements:

* it divides the swarm into multiple sub-swarms instead of using a single swarm
* it uses local best learning instead of relying only on a global best
* it periodically reshuffles particles through a dynamic regrouping strategy
* it better preserves population diversity during optimization
* it reduces premature convergence and improves global search capability
* it provides a more effective and robust way to optimize SVM hyperparameters

In short, **DMS-PSO-SVM can be regarded as an enhanced PSO-SVM framework** that is more suitable for difficult parameter optimization problems, especially in small-sample, high-dimensional, and nonlinear classification tasks.

---

## Conceptual comparison between PSO-SVM and DMS-PSO-SVM

| Aspect                                | PSO-SVM                      | DMS-PSO-SVM                              |
| ------------------------------------- | ---------------------------- | ---------------------------------------- |
| Swarm structure                       | Single swarm                 | Multiple dynamic sub-swarms              |
| Learning strategy                     | Personal best + global best  | Personal best + local best               |
| Search behavior                       | Easier to become centralized | More distributed and flexible            |
| Information exchange                  | Mainly through global best   | Through regrouping and local competition |
| Suitability for complex search spaces | Limited                      | More suitable                            |

---

## Interpretation in this project

In this project, DMS-PSO-SVM is used to optimize the two most important hyperparameters of the RBF-SVM:

* `C`
* `gamma`

Each particle represents one candidate pair of parameter values in log-space. During iterative optimization:

1. particles explore the parameter space
2. each sub-swarm updates according to its own local best solution
3. sub-swarms are periodically regrouped
4. the best discovered parameter combination is used to train the final SVM classifier

The quality of each particle is evaluated by **cross-validated ROC-AUC**, making the optimization process directly aligned with classification performance.

This design makes the framework particularly suitable for structured biomedical classification tasks where robust hyperparameter tuning is critical.

---

# DMS-PSO-SVM for Binary Classification

A Python implementation of a **DMS-PSO-SVM** framework for binary classification, integrating **Dynamic Multi-Swarm Particle Swarm Optimization (DMS-PSO)** with an **RBF-kernel Support Vector Machine (SVM)** to automatically optimize hyperparameters and build a robust predictive model.

This project is especially suitable for **small-sample, high-dimensional structured datasets**, such as radiomics, pathomics, or other biomedical feature-based classification tasks.

---

## Overview

Support Vector Machine (SVM) is a widely used machine learning algorithm for binary classification, particularly effective in high-dimensional problems. However, the performance of SVM strongly depends on the selection of hyperparameters, especially:

- `C`: penalty parameter controlling the trade-off between margin maximization and classification error
- `gamma`: kernel coefficient in the RBF kernel, controlling the influence range of individual samples

Traditional hyperparameter tuning methods such as grid search are often computationally expensive and may not efficiently explore the search space. To address this issue, this project adopts **Dynamic Multi-Swarm Particle Swarm Optimization (DMS-PSO)** to optimize SVM hyperparameters automatically.

The workflow includes:

1. Loading and preprocessing raw tabular data
2. Handling complex-like or non-standard numeric entries
3. Splitting data into training and validation sets
4. Using DMS-PSO to search for the optimal `C` and `gamma`
5. Training the final SVM model with the best parameters
6. Evaluating model performance on an independent validation set

---

## Features

- Implements **DMS-PSO** for SVM hyperparameter optimization
- Uses **cross-validated ROC-AUC** as the fitness criterion
- Supports **binary classification**
- Handles **complex-like string values** by converting them to real values
- Reports multiple evaluation metrics:
  - AUC
  - Accuracy
  - Sensitivity
  - Specificity
- Easy to adapt to radiomics and other structured biomedical datasets

---

## Requirements

Install the required Python packages before running the project:

```bash
pip install pandas numpy scikit-learn
```

---

## Input Data Format

The input file should be a CSV file, for example:

```csv
label,feature1,feature2,feature3,...
0,0.123,1.456,2.789,...
1,0.234,1.567,2.890,...
...
```

### Data format requirements

* The **first column** must be the binary label
* The **remaining columns** are feature values
* All values should be numeric or convertible to numeric
* If a value is in a **complex-like string form**, the script will attempt to parse it and keep its **real part**

Example:

* `"1.23"` → `1.23`
* `"(2.5+0j)"` → `2.5`

---

## Workflow

### 1. Data Loading

The script reads the CSV dataset using `pandas`:

```python
df = pd.read_csv('SData.csv')
```

---

### 2. Data Conversion

A helper function `to_real()` is used to convert each value to a real float:

* First tries direct `float(x)`
* If that fails, tries `complex(x)` and extracts the real part
* Raises an error if conversion is impossible

This design is helpful when exported feature tables contain complex-like strings.

---

### 3. Feature and Label Separation

* `y`: first column as class label
* `X`: remaining columns as feature matrix

```python
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values
```

---

### 4. Train/Validation Split

The dataset is split into:

* **80% training set**
* **20% validation set**

with **stratified sampling** to preserve class distribution:

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

---

### 5. DMS-PSO Hyperparameter Optimization

The core of this project is the `dms_pso_svm()` function.

#### What is DMS-PSO?

Dynamic Multi-Swarm Particle Swarm Optimization (DMS-PSO) is an enhanced version of standard PSO. Instead of using a single swarm, particles are divided into multiple sub-swarms, and these sub-swarms are periodically regrouped to maintain diversity and improve global search capability.

#### Optimized parameters

This implementation optimizes:

* `C` in the range `[1e-3, 1e3]`
* `gamma` in the range `[1e-4, 1e1]`

The search is performed in **log-space**:

* `log10(C)`
* `log10(gamma)`

#### Fitness function

The fitness is defined as:

```python
1 - mean cross-validated ROC-AUC
```

That is, higher AUC corresponds to lower fitness, and the optimizer attempts to minimize the fitness value.

#### Cross-validation strategy

The script uses:

* `StratifiedKFold`
* `5-fold cross-validation`
* scoring metric: `roc_auc`

This makes the optimization process more stable and suitable for imbalanced biomedical datasets.

---

## Main Parameters

The default parameters in `dms_pso_svm()` are:

```python
n_particles=30
n_swarms=3
n_iter=50
regroup_period=10
C_bounds=(1e-3, 1e3)
gamma_bounds=(1e-4, 1e1)
inertia_weight=0.729
c1=1.49445
c2=1.49445
cv_folds=5
random_state=42
```

### Parameter description

* `n_particles`: number of particles in the population
* `n_swarms`: number of sub-swarms
* `n_iter`: total optimization iterations
* `regroup_period`: how often particles are regrouped into new sub-swarms
* `C_bounds`: search range of SVM penalty parameter `C`
* `gamma_bounds`: search range of RBF kernel parameter `gamma`
* `inertia_weight`: inertia term controlling exploration and exploitation
* `c1`: cognitive learning factor
* `c2`: social learning factor
* `cv_folds`: number of folds in cross-validation
* `random_state`: seed for reproducibility

> **Note:** The parameter settings provided here are for demonstration and reference only. They may need adjustment depending on the dataset size, feature dimensionality, and task characteristics.

---

## Model Training

After optimization, the best `C` and `gamma` are used to train a final SVM model on the training set:

```python
final_svm = SVC(C=best_C, kernel='rbf', gamma=best_gamma,
                probability=True, random_state=random_state)
final_svm.fit(X, y)
```

---

## Validation Metrics

The trained model is evaluated on the validation set using:

* **AUC**
* **Accuracy**
* **Sensitivity**
* **Specificity**

### Metric definitions

* **AUC**: Area under the ROC curve, measuring ranking performance
* **Accuracy**: Overall classification correctness
* **Sensitivity**: Ability to correctly identify positive samples
* **Specificity**: Ability to correctly identify negative samples

The code computes these metrics as follows:

```python
auc = roc_auc_score(y_val, y_prob)
acc = accuracy_score(y_val, y_pred)
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
```

---

## Decision Threshold

In this implementation, the classification threshold is manually set to:

```python
y_pred = (y_prob >= 0.47).astype(int)
```

Instead of using the default threshold of `0.5`, this allows threshold adjustment according to specific task requirements, such as:

* maximizing sensitivity
* balancing sensitivity and specificity
* improving clinical utility in biomedical applications

---

## How to Run

Place your dataset file `SData.csv` in the project directory, then run:

```bash
python main.py
```

Example output:

```bash
Optimized C: 12.34567, gamma: 0.05678
Validation AUC: 0.842
Validation Accuracy: 0.800
Validation Sensitivity: 0.778
Validation Specificity: 0.821
```

---

## Example Use Cases

This framework can be adapted to a wide range of structured binary classification problems, including:

* radiomics-based disease diagnosis
* pathomics-based risk stratification
* clinical outcome prediction
* biomarker-based classification
* other small-sample, high-dimensional biomedical learning tasks

---

## Citation

If you use this code in your research, please cite this original methodological references:
- Li Y, Zhang D, Wang Y, et al. *Construction of an oligometastatic prediction model for nasopharyngeal carcinoma patients based on pathomics features and dynamic multi-swarm particle swarm optimization support vector machine*. Frontiers in Oncology, 2025. DOI: 10.3389/fonc.2025.1589919


