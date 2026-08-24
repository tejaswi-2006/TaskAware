import pandas as pd


def perform_initial_eda(df):
    """
    Performs initial exploratory data analysis
    on the uploaded dataset.
    """

    eda_results = {}

    # Numerical columns
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Unique values for every column
    unique_values = {}

    for column in df.columns:
        unique_values[column] = int(df[column].nunique())

    # Statistical summary for numerical columns
    if numerical_columns:
        statistical_summary = (
            df[numerical_columns]
            .describe()
            .round(2)
            .to_dict()
        )
    else:
        statistical_summary = {}

    # Store results
    eda_results["numerical_columns"] = numerical_columns
    eda_results["categorical_columns"] = categorical_columns
    eda_results["unique_values"] = unique_values
    eda_results["statistical_summary"] = statistical_summary

    return eda_results