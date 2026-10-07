[README (3).md](https://github.com/user-attachments/files/33177571/README.3.md)
# Trees, Bagging, Boosting and PCA / Feature Selection

A complete machine-learning assignment using the **Dry Bean Dataset**.  
The notebook covers Decision Trees, Bagging, Random Forest, Gradient Boosting, XGBoost, PCA, and feature selection, with a consistent train/test protocol and reproducible random seed.

## Dataset

**Dry Bean Dataset**

- **13,611 samples**
- **16 numerical features**
- **7 bean varieties**
- Target column: `Class`

Classes:

`SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `HOROZ`, `SIRA`, `DERMASON`

### Input features

- `Area`
- `Perimeter`
- `MajorAxisLength`
- `MinorAxisLength`
- `AspectRation`
- `Eccentricity`
- `ConvexArea`
- `EquivDiameter`
- `Extent`
- `Solidity`
- `roundness`
- `Compactness`
- `ShapeFactor1`
- `ShapeFactor2`
- `ShapeFactor3`
- `ShapeFactor4`

## Experimental protocol

The notebook uses the same reproducible setup throughout the assignment:

- **80% training / 20% test**
- `random_state = 42`
- Stratified train/test split
- Duplicate rows are removed **before** splitting
- Random components use `random_state = 42`

The data loader accepts either `.csv` or `.xlsx` files.

## Assignment workflow

### 1. Decision Tree

The notebook starts with exploratory analysis and data preparation, then trains a default Decision Tree.

The exploratory analysis shows that the strongest class-separating variables are mainly size-related features such as `Area`, `ConvexArea`, `EquivDiameter`, `Perimeter`, `MinorAxisLength`, and `MajorAxisLength`.

It also identifies substantial feature redundancy, especially among size and shape measurements.

The Decision Tree analysis then compares:

- Gini vs. Entropy
- Different `max_depth` values
- Cost-complexity pruning with `ccp_alpha`
- Cross-validation based pruning

The best pruned tree reaches about **91.18% test accuracy**, while the cross-validation-selected pruning configuration reaches about **90.70%**.

### 2. Bagging

Bagging is evaluated with:

- RBF SVM
- Decision Tree

The Decision Tree benefits much more from bagging than the SVM, which is consistent with the higher variance of an unpruned tree.

Observed test accuracy:

| Model | Test Accuracy |
|---|---:|
| SVM (RBF) | 92.10% |
| Bagged SVM | **92.25%** |
| Decision Tree | 89.66% |
| Bagged Decision Tree | 91.36% |

### 3. Random Forest

Random Forest is evaluated by changing:

- Number of trees
- `max_features`

The results show that performance largely plateaus after roughly 50–100 trees.

The final Random Forest configuration achieves about **91.99% test accuracy**.

### 4. Boosting

Two boosting approaches are studied:

#### Gradient Boosting

The number of estimators and learning rate are compared.

The selected configuration uses:

- `n_estimators = 100`
- `learning_rate = 0.1`

Test accuracy: **91.77%**

#### XGBoost

XGBoost is evaluated with different `subsample` values.

The best configuration in the executed notebook is selected automatically from:

- `subsample = 0.6`
- `subsample = 0.8`
- `subsample = 1.0`

The best observed XGBoost configuration reaches **92.25% test accuracy**.

## Final model comparison

The main test-set accuracy comparison is:

| Model | Test Accuracy |
|---|---:|
| Decision Tree | 89.66% |
| Bagged Decision Tree | 91.36% |
| Gradient Boosting | 91.77% |
| Random Forest | 91.99% |
| Bagged SVM | **92.25%** |
| XGBoost (best subsample) | **92.25%** |

The executed notebook identifies **Bagged SVM** as the best model in the weighted precision/recall/accuracy/F1 comparison.

For the six-model comparison, the single Decision Tree is clearly the weakest model, while the top ensemble models are very close to one another.

## PCA and Feature Selection

The best-performing model is then used to study dimensionality reduction.

### PCA

Standardized features are transformed using PCA.

Only **4 principal components** are required to retain at least **95% of the variance**:

- 4 components
- Cumulative explained variance: **95.04%**

However, the Bagged SVM accuracy drops from **92.25%** to **89.48%** at 4 components.

The PCA sweep shows that performance recovers as more components are retained:

| PCA Components | Cumulative Variance | Test Accuracy |
|---:|---:|---:|
| 2 | 81.90% | 87.01% |
| 3 | 89.91% | 88.45% |
| 4 | 95.04% | 89.48% |
| 5 | 97.76% | 91.69% |
| 6 | 98.90% | 92.10% |
| 8 | 99.93% | 92.21% |
| 10 | 99.99% | 92.21% |
| 16 | 100.00% | 92.17% |

### Feature Selection

The top six features selected from Random Forest feature importance are:

1. `Perimeter`
2. `ShapeFactor3`
3. `Compactness`
4. `ShapeFactor1`
5. `MinorAxisLength`
6. `ConvexArea`

Using only these six original features with the Bagged SVM gives:

**90.73% test accuracy**

Compared with the original 16-feature Bagged SVM:

- Original: **92.25%**
- Feature-selected: **90.73%**
- Difference: about **−1.51 percentage points**

## Key findings

- A single Decision Tree has noticeably higher variance and the lowest test accuracy.
- Bagging provides a larger improvement for the Decision Tree than for the SVM.
- Random Forest and boosting methods achieve performance close to Bagged SVM.
- **Bagged SVM gives the best overall result in the final weighted-metric comparison.**
- PCA provides strong dimensionality compression, but keeping only 95% variance is not enough to preserve the best classification performance.
- Feature selection reduces the number of measured inputs while keeping their physical meaning, but it also reduces accuracy.
- For this dataset, the notebook recommends keeping the **original 16 features** with the best-performing model rather than aggressively reducing dimensionality.

## Training-time comparison

The notebook also compares training time. Because the models use different learning strategies, their computational costs differ substantially:

- Single Decision Tree: very fast
- Bagged models: more expensive but parallelizable
- Random Forest: moderate training cost
- XGBoost: relatively fast
- Scikit-learn Gradient Boosting: the slowest among the compared ensemble models in this experiment

Exact times depend on the machine and environment.

## Project structure

```text
.
├── Trees_Assignment_Solution.ipynb
├── Dry_Bean_Dataset.csv        # or .xlsx
└── README.md
```

## Requirements

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost openpyxl jupyter
```

`openpyxl` is only needed when using the Excel version of the dataset.

## How to run

1. Put the Dry Bean dataset in the same directory as the notebook.
2. Open `Trees_Assignment_Solution.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
3. Run the cells from top to bottom.
4. The notebook automatically searches for the Dry Bean dataset in common locations and loads either CSV or Excel format.

## Reproducibility

All random components use:

```python
RANDOM_STATE = 42
```

The fixed random seed and stratified 80/20 split make the experiment reproducible under the same software and dataset versions.

## Conclusion

This notebook demonstrates the complete progression from a single Decision Tree to ensemble learning and dimensionality reduction.

The main practical conclusion is that **Bagged SVM provides the strongest overall performance on the given test split, reaching approximately 92.25% accuracy**, while Random Forest and XGBoost offer very competitive alternatives.

PCA and feature selection are useful for studying dimensionality and interpretability, but on this 16-feature dataset their reduction in input dimensionality does not provide enough practical benefit to justify the observed loss in predictive performance.
