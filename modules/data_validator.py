def validate_dataset(df):
    """
    Performs basic validation on the uploaded dataset.
    """

    validation_results = {}

    # 1. Check if dataset is empty
    validation_results["is_empty"] = df.empty

    # 2. Count missing values
    validation_results["missing_values"] = (
        df.isnull().sum().to_dict()
    )

    # 3. Count duplicate rows
    validation_results["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    # 4. Find constant columns
    constant_columns = []

    for column in df.columns:
        if df[column].nunique() <= 1:
            constant_columns.append(column)

    validation_results["constant_columns"] = constant_columns

    # 5. Find possible ID columns
    possible_id_columns = []

    for column in df.columns:

        # If almost every value is unique,
        # it may be an ID column
        unique_ratio = df[column].nunique() / len(df)

        if unique_ratio > 0.95:
            possible_id_columns.append(column)

    validation_results["possible_id_columns"] = possible_id_columns

    return validation_results