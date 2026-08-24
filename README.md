# TaskAware

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

# Current Phase: Phase 1

The current version of TaskAware performs the following tasks:

```text
Dataset Upload
      ↓
Dataset Loading
      ↓
Dataset Validation
      ↓
Initial Exploratory Data Analysis
      ↓
Column Analysis
      ↓
Target Variable Scoring
      ↓
Target Variable Recommendation
      ↓
User Confirmation / Manual Selection