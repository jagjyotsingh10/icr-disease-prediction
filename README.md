# ICR Disease Prediction

Binary classifier that predicts whether a patient has any of three age-related medical conditions from 56
anonymised health measurements. Built for the Kaggle competition
[ICR – Identifying Age-Related Conditions](https://www.kaggle.com/competitions/icr-identify-age-related-conditions).

**Best result:** Random Forest and XGBoost both reached **90.3% accuracy** on a held-out validation set,
with **82% recall** on the minority (has-condition) class.

**Stack:** Python · pandas · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · CatBoost · matplotlib · Plotly

## The problem

- 617 labelled patients and 56 features, plus a supplementary "greeks" metadata table.
- Imbalanced: only about 1 in 6 patients is in the positive class, so plain accuracy is misleading and
  recall on that class matters most.

## Approach

1. **Cleaning.** Imputed missing values with KNN (k = 5), converted the date column in the metadata into a
   numeric "days since" feature, label-encoded the categorical column `EJ`, and checked for duplicates.
2. **Exploration.** Box plots and histograms for every feature, plus the class distribution.
3. **Validation split.** 70/30 stratified train/validation split, so both sets keep the same class ratio.
4. **Scaling and balancing.** Standardised the features, then oversampled the training set with SMOTE
   (431 → 712 rows). The validation set was never resampled.
5. **Feature selection.** Kept the 25 features chosen by both Pearson correlation (|r| ≥ 0.1) and
   SelectKBest (ANOVA F-test).
6. **Modelling.** Tuned eight classifiers with GridSearchCV and 5-fold cross-validation.

## Results (validation set, 186 patients)

| Model | Accuracy | Recall (has condition) | F1 (has condition) |
|---|---|---|---|
| **Random Forest** | **90.3%** | **81.8%** | **0.75** |
| **XGBoost** | **90.3%** | **81.8%** | **0.75** |
| CatBoost | 88.7% | 78.8% | 0.71 |
| Naive Bayes | 87.6% | 57.6% | 0.62 |
| SVM (RBF) | 86.6% | 51.5% | 0.58 |
| Logistic Regression (L1) | 86.0% | 78.8% | 0.67 |
| Decision Tree | 86.0% | 72.7% | 0.65 |
| K-Nearest Neighbours | 84.9% | 87.9% | 0.67 |

The final submission used the tuned Random Forest to output class probabilities for the test set.

## What I learned / would improve

- On imbalanced medical data, recall on the positive class matters more than overall accuracy. KNN had the
  lowest accuracy but caught the most positive cases.
- The competition scores balanced log loss, so the next step would be to tune models on probability
  calibration rather than accuracy.
- Cross-validated scores across the whole dataset would be more reliable than a single 186-row validation split.

## Run it

1. Download `train.csv`, `test.csv` and `greeks.csv` from the
   [competition data page](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data)
   into this folder. The data is not included here because of the competition's rules.
2. Install the dependencies and open the notebook:

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost catboost missingno seaborn matplotlib plotly colorama
jupyter notebook ICR_Final_Submission.ipynb
```
