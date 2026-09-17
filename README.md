# TaskAware
"""An objective-aware and explainable assistant that helps non-experts define the right ML task before automating the modelling workflow"""
2.""" TaskAware should not always find a target variable. It should first determine whether the user has a meaningful prediction target. If a suitable target exists, it follows a supervised-learning path: classification or regression. If no suitable target exists, it offers unsupervised analysis, such as clustering"""
3.""" “TaskAware supports two situations. When the user has a meaningful target column, the system validates that target and routes the dataset to classification or regression. However, not every dataset has a target. If the user does not provide a target, or if the system finds that the available columns are mainly identifiers, descriptive attributes, or unsuitable outputs, TaskAware does not force a classification or regression problem. Instead, it offers clustering to discover natural groups in the data. Clustering uses only the input features and does not require a target label"""


## Intelligent Dataset Analysis and Target Variable Recommendation System

TaskAware is a machine learning assistance application designed to help users analyze a dataset and identify a suitable target variable.

The system allows users to upload a dataset, performs basic dataset validation and initial exploratory data analysis (EDA), analyzes the available columns, and recommends the most suitable target variable.

The user can then either accept the recommended target variable or manually select a different column.

This project is currently under development. The present implementation represents **Phase 1**, which focuses on dataset analysis and target variable recommendation.

---

# Project Objective

In machine learning, selecting the correct target variable is an important step.

However, users may upload a dataset without clearly knowing:

- What problem the dataset can solve
- Which column should be used as the target variable
- Whether the target is suitable for machine learning
- Which columns are useful for prediction

TaskAware aims to assist the user by automatically analyzing the dataset and recommending possible target variables.

---

# Current Phase: Phase 1 to phase 10

TaskAware — Complete 10-Phase Development Plan
" Phase 1 " — Dataset Upload, Validation & Initial Analysis

Goal: Accept a dataset and understand its basic structure before making any ML decision.

Modules
modules/
├── data_loader.py
├── data_validator.py
└── initial_eda.py
Functions

1. Dataset upload

CSV
XLSX

2. Dataset validation

File format
Empty dataset
Number of rows/columns
Missing values
Duplicate rows
Data types
Constant columns
Possible identifier columns
Invalid/inconsistent values

3. Initial EDA

Dataset shape
Column names
Data types
Missing-value summary
Unique values
Numerical/categorical column counts
Basic statistics
Output
Dataset
    ↓
Validation
    ↓
Initial EDA
    ↓
Dataset Overview
Status

✅ Already partially implemented in your project.

" Phase 2 " — Intelligent Objective & Target Analysis

This is one of the most important phases of TaskAware.

Goal: Understand what the user wants to predict/discover and determine whether a target is appropriate.

Module
modules/objective_target.py
Three situations
Case 1 — User provides target

Example:

Objective:
Predict final academic performance

Target:
final_grade

TaskAware should validate:

Target: final_grade

Data type: Categorical
Unique values: 5
Missing values: 0
Identifier likelihood: Low

Problem type:
Classification

Conclusion:
Highly suitable

And explain why.

Case 2 — User gives objective but no target

Example:

Objective:
Predict student academic performance

TaskAware analyzes the columns.

It should produce:

Recommended Target
------------------
final_grade
Reason:
• Represents final academic outcome
• Categorical target
• 5 classes
• No missing values
• Strong alignment with objective

Other possible target:
final_exam_score

Not recommended:
student_id

Do not show every column as equally suitable.

Case 3 — User selects a different target

Suppose TaskAware recommends:

final_grade

but user selects:

final_exam_score

TaskAware compares:

Property	Recommended	User Selected
Column	final_grade	final_exam_score
Type	Categorical	Numerical
Task	Classification	Regression
Missing	0	dataset-dependent
Meaning	Final grade	Exam score
Suitability	High	High/Medium

Then explain:

final_exam_score is a valid target, but selecting it changes the ML problem from classification to regression. final_grade was originally preferred because it directly represents the categorical academic outcome described by the objective.

Important

Target recommendation should use:

Objective alignment
+
Data type
+
Cardinality
+
Missingness
+
ID detection
+
Variability
+
Outcome characteristics
+
Column name as a supporting signal

Not simply:

if "final" in column:
    recommend()
Output
Objective
   ↓
Target validation/recommendation
   ↓
Candidate comparison
   ↓
User confirmation

" Phase 3 me" — Detailed EDA

Once the target is confirmed, TaskAware performs target-specific analysis.

Module
modules/detailed_eda.py
Numerical analysis
Mean
Median
Standard deviation
Min/max
Quartiles
Distribution
Outliers
Skewness
Categorical analysis
Frequency
Percentage
Number of categories
Rare categories
Target analysis

For classification:

Class distribution
Class imbalance
Most common class
Least common class

For regression:

Target distribution
Mean
Median
Range
Variance
Outliers
Relationship analysis
Feature ↔ Target

Examples:

Correlation
Grouped means
Box plots
Target-wise feature distributions
Output
Confirmed Target
       ↓
Detailed EDA
       ↓
Charts + Statistics
       ↓
Data Understanding
Phase 4 — Data Preprocessing & Feature Preparation

Goal: Convert raw data into ML-ready data.

Modules
modules/
├── preprocessing.py
└── feature_selection.py
Preprocessing
Missing values

Numerical:

Median / Mean

Categorical:

Mode / Constant category
Categorical encoding

For example:

gender
Male
Female

→ One-hot encoding.

Numerical scaling

For algorithms that benefit from scaling:

StandardScaler
Remove unsuitable columns

For example:

student_id

should normally be removed from features.

Important: prevent data leakage

Use:

Pipeline
+
ColumnTransformer

The transformations must be fitted only using training data/folds.

Feature selection

Possible methods:

Correlation
Mutual information
Variance filtering
Model-based importance

Do not automatically delete features just because correlation is low.

Output
Raw Dataset
    ↓
Missing-value handling
    ↓
Encoding
    ↓
Scaling
    ↓
ID/irrelevant column removal
    ↓
Feature preparation
    ↓
ML-ready dataset
Phase 5 — Problem Type Detection & Decision Engine

This phase connects the user's objective with the ML task.

Modules
modules/
├── decision_engine.py
└── problem_type_detector.py

TaskAware should decide between:

                Confirmed requirement
                         ↓
               ┌─────────┴─────────┐
               ↓                   ↓
          Target exists       No target
               ↓                   ↓
          Supervised          Ask about
               ↓              clustering
        ┌──────┴──────┐            ↓
        ↓             ↓       Unsupervised
 Classification    Regression    Clustering
Classification

Examples:

Pass / Fail
Spam / Not Spam
Grade A/B/C/D
Disease / No Disease
Regression

Examples:

House price
Salary
Exam score
Temperature
Clustering

Examples:

Customer segmentation
Student groups
Product groups
Behavior patterns
Important principle

A dataset is not automatically supervised or unsupervised.

The task depends on:

User objective
+
Target availability
+
Target suitability
Phase 6 — Classification & Regression

This is the main supervised-learning engine.

6A — Classification
Module
modules/model_trainer.py

Candidate models:

Logistic Regression
Decision Tree
Random Forest

Optional later:

SVM
XGBoost
Gradient Boosting
Workflow
Preprocessed data
      ↓
Train/Test Split
      ↓
Cross Validation
      ↓
Train models
      ↓
Evaluate models
      ↓
Compare models
Metrics

Use multiple appropriate metrics:

Accuracy
Precision
Recall
F1-score
ROC-AUC where applicable
Confusion Matrix

Do not blindly select the model with highest accuracy.

6B — Regression

Candidate models:

Linear Regression
Ridge Regression
Decision Tree Regressor
Random Forest Regressor

Optional:

Gradient Boosting
XGBoost
Metrics
MAE
MSE
RMSE
R²
Output

Example:

Model              RMSE      R²
--------------------------------
Linear Regression  8.42     0.71
Decision Tree      7.31     0.79
Random Forest      5.84     0.88

Then TaskAware can recommend:

Recommended model:
Random Forest

Reason:
Highest validation performance according to the selected
regression evaluation criteria.
Phase 7 — Clustering / Unsupervised Learning

This is the phase that directly answers your guild's clustering question.

Module
modules/clustering.py
Step 1 — Feature preparation

Remove:

IDs
constant columns
obviously irrelevant columns

Then:

Missing-value handling
       ↓
Categorical encoding
       ↓
Scaling
Step 2 — Determine candidate K

For K-Means:

K = 2
K = 3
K = 4
K = 5
...
Step 3 — Evaluate clusters

Use:

Silhouette Score

Higher is generally better.

Davies-Bouldin Index

Lower is generally better.

Calinski-Harabasz Index

Higher is generally better.

Step 4 — Cluster visualization

Use PCA:

High-dimensional features
        ↓
       PCA
        ↓
      2D space
        ↓
Cluster visualization
Step 5 — Cluster profiles

This is extremely important.

Do not stop at:

"Here is a colored scatter plot."

Instead:

Cluster 0
-----------
Average study time: 5.8 hrs
Average attendance: 91%
Average previous grade: 82

Cluster 1
-----------
Average study time: 2.9 hrs
Average attendance: 72%
Average previous grade: 61

Then the user can understand what makes the groups different.

Step 6 — Stability

Run clustering with different seeds/sample variations and check whether the structure is reasonably stable.

Output
Clustering
    ↓
Preprocessing
    ↓
K candidates
    ↓
Metrics
    ↓
Recommended configuration
    ↓
Visualization
    ↓
Cluster profiles
    ↓
User interpretation

And TaskAware should clearly warn:

These clusters represent patterns discovered from the dataset and are not ground-truth labels.

Phase 8 — Model Selection & Explainability

This phase makes the project more than an ordinary ML pipeline.

Modules
modules/
├── model_selector.py
└── explainability.py
Best model selection

TaskAware should use the correct metric depending on the problem.

Classification

Potential priority:

F1 / Recall / Precision / ROC-AUC / Accuracy

depending on the objective and class distribution.

Regression

Potential priority:

RMSE / MAE / R²

depending on the objective.

The system should record why the selected model won.

SHAP

For supervised models:

Best Model
    ↓
SHAP
    ↓
Feature contribution
    ↓
Explanation

Example:

Feature                 Importance
----------------------------------
attendance_percent       High
previous_grade           High
study_time_hours         Medium
sleep_hours              Low

SHAP can also explain individual predictions.

Example:

The predicted score was influenced positively by high attendance and previous grade, while low study time contributed negatively.

For clustering

Do not force SHAP into clustering.

Instead use:

Cluster profiles
+
Feature differences
+
Feature distributions
Phase 9 — AI Explanation + OKF Knowledge Reuse

This is where your LLM and OKF components come in.

Modules
modules/
├── ai_explainer.py
└── okf_manager.py
Correct architecture
              Python ML Engine
                    ↓
          ┌──────────────────┐
          │ Validated Results│
          └────────┬─────────┘
                   ↓
          ┌──────────────────┐
          │   OKF Knowledge  │
          └────────┬─────────┘
                   ↓
          Existing explanation?
             ↓            ↓
            Yes           No
             ↓            ↓
          Reuse        LLM
                         ↓
                    Explanation
                         ↓
                    Save to OKF
                         ↓
                    Return result
What the LLM receives

Not the raw dataset.

Instead:

{
  "problem_type": "classification",
  "target": "final_grade",
  "model": "Random Forest",
  "accuracy": 0.91,
  "f1": 0.89,
  "important_features": [
    "attendance_percent",
    "previous_grade"
  ]
}

Then the LLM explains those results.

This prevents hallucination

The LLM should not calculate:

Accuracy
RMSE
SHAP values
cluster metrics
feature importance

Python calculates them.

LLM explains them.

Phase 10 — Reports, Database, Authentication, UI & Deployment

This is the final integration phase.

Database

Use SQLite initially.

Tables:

Users
Projects
Datasets
AnalysisRuns
ModelResults
Reports

Example:

Users
-----
user_id
name
email
password_hash
created_at
AnalysisRuns
------------
analysis_id
dataset_id
objective
target
problem_type
status
created_at
Final Web Application

Your pages can become:

Landing Page
     ↓
Register / Login
     ↓
Dashboard
     ↓
Create Project
     ↓
Upload Dataset
     ↓
Validation
     ↓
Initial EDA
     ↓
Objective + Target
     ↓
Target Recommendation
     ↓
User Confirmation
     ↓
Detailed EDA
     ↓
Preprocessing
     ↓
Problem Type
     ↓
 ┌───────────┬────────────┐
 ↓           ↓            ↓
Classification Regression Clustering
 ↓           ↓            ↓
Model       Model        Clusters
 ↓           ↓            ↓
Evaluation  Evaluation   Evaluation
 └───────────┼────────────┘
             ↓
        Best Result
             ↓
       Explainability
             ↓
       AI Explanation
             ↓
            OKF
             ↓
       Final Dashboard
             ↓
        Generate Report
Complete 10-Phase Architecture
Phase	Main Work	Main Modules
1	Upload, validation, initial EDA	data_loader, data_validator, initial_eda
2	Objective & target intelligence	objective_target
3	Detailed target-based EDA	detailed_eda
4	Preprocessing & feature preparation	preprocessing, feature_selection
5	Decision & problem-type detection	decision_engine, problem_type_detector
6	Classification & regression	model_trainer, model_evaluator
7	Clustering	clustering
8	Best model & explainability	model_selector, explainability
9	AI explanation & OKF	ai_explainer, okf_manager
10	Database, authentication, reports, UI, deployment	Flask routes, DB, report_generator
The final project flow in one sentence

TaskAware takes a raw dataset and the user's objective, validates and understands the data, helps identify or validate an appropriate target when necessary, determines whether prediction or group discovery is appropriate, executes a leakage-safe ML pipeline, evaluates and explains the results, reuses knowledge through OKF, and produces an understandable final report.

What makes your project different

The strongest part is not the number of algorithms.

Your contribution is the decision layer:

                "What should I do with this dataset?"
                              ↓
                       TaskAware
                              ↓
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓
       Known Target     Choose Target    Discover Groups
             ↓                ↓                ↓
       Classification/   Target          Clustering
        Regression      Comparison
             └────────────────┼────────────────┘
                              ↓
                         Valid ML
                              ↓
                     Explainable Results
                              ↓
                       AI + OKF