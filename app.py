from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
from werkzeug.utils import secure_filename

# ============================================================
# PROJECT MODULES
# ============================================================

from modules.data_validator import validate_dataset
from modules.initial_eda import perform_initial_eda
from modules.objective_target import analyze_target_candidates
from modules.problem_type_detector import detect_problem_type
from modules.detailed_eda import perform_detailed_eda
from modules.preprocessing import prepare_preprocessing
from modules.feature_selection import perform_feature_selection
from modules.model_trainer import train_models
from modules.model_selector import select_best_model
from modules.clustering import perform_clustering


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

CHARTS_FOLDER = "results/charts"

ALLOWED_EXTENSIONS = {
    "csv",
    "xlsx"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["CHARTS_FOLDER"] = CHARTS_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    CHARTS_FOLDER,
    exist_ok=True
)


# ============================================================
# FILE TYPE CHECK
# ============================================================

def allowed_file(filename):
    """
    Check whether the uploaded file is CSV or XLSX.
    """

    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(filepath):
    """
    Load CSV or XLSX dataset.
    """

    extension = filepath.rsplit(
        ".",
        1
    )[1].lower()

    if extension == "csv":

        return pd.read_csv(filepath)

    elif extension == "xlsx":

        return pd.read_excel(filepath)

    raise ValueError(
        "Unsupported file type."
    )


# ============================================================
# FIND TARGET INFORMATION
# ============================================================

def get_target_info(
    candidates,
    target_column
):
    """
    Find information about a particular target column.
    """

    for candidate in candidates:

        if candidate["column"] == target_column:

            return candidate

    return None


# ============================================================
# GET USABLE TARGETS
# ============================================================

def get_usable_candidates(candidates):
    """
    Return target candidates that are reasonably usable.

    Identifier columns and clearly unsuitable columns
    are excluded.
    """

    return [
        candidate

        for candidate in candidates

        if not candidate["is_id"]

        and candidate["suitability"] in [
            "Highly Suitable",
            "Suitable",
            "Possible"
        ]
    ]


# ============================================================
# GET RECOMMENDED TARGET
# ============================================================

def get_recommended_candidate(candidates):
    """
    Get the highest-ranked usable target.
    """

    usable_candidates = get_usable_candidates(
        candidates
    )

    if not usable_candidates:

        return None

    return usable_candidates[0]


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload_dataset():

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "upload.html"
        )

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "dataset" not in request.files:

        return "No dataset was selected."

    file = request.files["dataset"]

    if not file.filename:

        return "Please select a dataset."

    # --------------------------------------------------------
    # CHECK EXTENSION
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        return (
            "Invalid file type. "
            "Please upload a CSV or XLSX file."
        )

    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    if not filename:

        return "Invalid file name."

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    try:

        file.save(filepath)

    except Exception as e:

        return (
            "Unable to save dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # EMPTY DATASET
    # --------------------------------------------------------

    if df.empty:

        return (
            "The uploaded dataset is empty."
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    try:

        validation_results = validate_dataset(df)

    except Exception as e:

        validation_results = {
            "success": False,
            "message": (
                f"Validation failed: {str(e)}"
            )
        }

    # --------------------------------------------------------
    # INITIAL EDA
    # --------------------------------------------------------

    try:

        eda_results = perform_initial_eda(df)

    except Exception as e:

        eda_results = {
            "success": False,
            "message": (
                f"Initial EDA failed: {str(e)}"
            )
        }

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    rows, columns = df.shape

    column_names = df.columns.tolist()

    data_types = (
        df.dtypes
        .astype(str)
        .to_dict()
    )

    missing_values = (
        df.isnull()
        .sum()
        .to_dict()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    preview = df.head().to_html(
        classes="table",
        index=False
    )

    # --------------------------------------------------------
    # SHOW OVERVIEW
    # --------------------------------------------------------

    return render_template(

        "dataset_overview.html",

        filename=filename,

        rows=rows,

        columns=columns,

        column_names=column_names,

        data_types=data_types,

        missing_values=missing_values,

        duplicate_rows=duplicate_rows,

        preview=preview,

        validation_results=validation_results,

        eda_results=eda_results
    )


# ============================================================
# OBJECTIVE + TARGET ANALYSIS
# ============================================================

@app.route(
    "/analyze-objective",
    methods=["POST"]
)
def analyze_objective():

    # --------------------------------------------------------
    # GET USER INPUT
    # --------------------------------------------------------

    objective = request.form.get(
        "objective",
        ""
    ).strip()

    selected_target = request.form.get(
        "target",
        ""
    ).strip()

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    # --------------------------------------------------------
    # OBJECTIVE IS OPTIONAL
    # --------------------------------------------------------

    # Do NOT reject an empty objective.
    #
    # TaskAware must also work when the user does not know
    # what objective to provide.

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not filename:

        return (
            "Dataset information is missing."
        )

    filename = secure_filename(
        filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        return (
            "Dataset file not found. "
            "Please upload the dataset again."
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # ANALYZE TARGETS
    # --------------------------------------------------------

    try:

        candidates = analyze_target_candidates(

            df,

            objective=(
                objective
                if objective
                else None
            )
        )

    except Exception as e:

        return (
            "Unable to analyze target candidates: "
            f"{str(e)}"
        )

    # ========================================================
    # CASE 1: USER PROVIDED TARGET
    # ========================================================

    if selected_target:

        # ----------------------------------------------------
        # TARGET EXISTS?
        # ----------------------------------------------------

        if selected_target not in df.columns:

            return (
                f"Target '{selected_target}' "
                "does not exist in the dataset."
            )

        selected_info = get_target_info(
            candidates,
            selected_target
        )

        if selected_info is None:

            return (
                "Unable to analyze the selected target."
            )

        # ----------------------------------------------------
        # SHOW VALIDATION
        # ----------------------------------------------------

        return render_template(

            "objective_analysis.html",

            filename=filename,

            objective=objective,

            selected_target=selected_target,

            selected_info=selected_info,

            candidates=candidates
        )

    # ========================================================
    # CASE 2: NO TARGET PROVIDED
    # ========================================================

    usable_candidates = get_usable_candidates(
        candidates
    )

    # --------------------------------------------------------
    # NO SUITABLE TARGET
    # --------------------------------------------------------

    if not usable_candidates:

        return render_template(

            "objective_analysis.html",

            filename=filename,

            objective=objective,

            selected_target=None,

            selected_info=None,

            candidates=[],

            no_target=True
        )

    # --------------------------------------------------------
    # SHOW RECOMMENDATION
    # --------------------------------------------------------

    return render_template(

        "objective_analysis.html",

        filename=filename,

        objective=objective,

        selected_target=None,

        selected_info=None,

        candidates=usable_candidates,

        no_target=False
    )


# ============================================================
# COMPARE TARGET
# ============================================================

@app.route(
    "/compare-target",
    methods=["POST"]
)
def compare_target():

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    objective = request.form.get(
        "objective",
        ""
    ).strip()

    selected_target = request.form.get(
        "target",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not filename:

        return (
            "Dataset information is missing."
        )

    if not selected_target:

        return (
            "Please select a target."
        )

    filename = secure_filename(
        filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        return (
            "Dataset file not found."
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    try:

        candidates = analyze_target_candidates(

            df,

            objective=(
                objective
                if objective
                else None
            )
        )

    except Exception as e:

        return (
            "Unable to analyze target candidates: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # SELECTED TARGET
    # --------------------------------------------------------

    selected_info = get_target_info(
        candidates,
        selected_target
    )

    if selected_info is None:

        return (
            f"Unable to analyze target "
            f"'{selected_target}'."
        )

    # --------------------------------------------------------
    # BLOCK IDENTIFIER
    # --------------------------------------------------------

    if selected_info["is_id"]:

        return render_template(

            "objective_analysis.html",

            filename=filename,

            objective=objective,

            selected_target=selected_target,

            selected_info=selected_info,

            candidates=get_usable_candidates(
                candidates
            )
        )

    # --------------------------------------------------------
    # ORIGINAL RECOMMENDATION
    # --------------------------------------------------------

    recommended_info = get_recommended_candidate(
        candidates
    )

    if recommended_info is None:

        return render_template(

            "objective_analysis.html",

            filename=filename,

            objective=objective,

            selected_target=None,

            selected_info=None,

            candidates=[],

            no_target=True
        )

    # --------------------------------------------------------
    # COMPARISON PAGE
    # --------------------------------------------------------

    return render_template(

        "target_comparison.html",

        filename=filename,

        objective=objective,

        selected_info=selected_info,

        recommended_info=recommended_info
    )


# ============================================================
# CONFIRM TARGET
# ============================================================

@app.route(
    "/confirm-target",
    methods=["POST"]
)
def confirm_target():

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    selected_target = request.form.get(
        "target",
        ""
    ).strip()

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    objective = request.form.get(
        "objective",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not selected_target:

        return (
            "No target variable selected."
        )

    if not filename:

        return (
            "Dataset information is missing."
        )

    filename = secure_filename(
        filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        return (
            "Dataset file not found."
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # TARGET EXISTS
    # --------------------------------------------------------

    if selected_target not in df.columns:

        return (
            f"Target '{selected_target}' "
            "does not exist in the dataset."
        )

    # --------------------------------------------------------
    # RE-ANALYZE TARGET
    #
    # This is important:
    # the backend validates the target again before accepting it.
    # --------------------------------------------------------

    try:

        candidates = analyze_target_candidates(

            df,

            objective=(
                objective
                if objective
                else None
            )
        )

    except Exception as e:

        return (
            "Unable to validate target: "
            f"{str(e)}"
        )

    selected_info = get_target_info(

        candidates,

        selected_target
    )

    if selected_info is None:

        return (
            "Unable to validate the selected target."
        )

    # --------------------------------------------------------
    # BLOCK ID TARGET
    # --------------------------------------------------------

    if selected_info["is_id"]:

        return (
            f"'{selected_target}' cannot be used as "
            "a target because it appears to be an "
            "identifier column."
        )

    # --------------------------------------------------------
    # BLOCK CONSTANT TARGET
    # --------------------------------------------------------

    if selected_info["unique_values"] <= 1:

        return (
            f"'{selected_target}' cannot be used as "
            "a target because it contains insufficient "
            "variation."
        )

    # --------------------------------------------------------
    # PROBLEM TYPE
    # --------------------------------------------------------

    problem_type_result = detect_problem_type(

        df,

        selected_target,

        objective=objective
    )

    # --------------------------------------------------------
    # CONFIRMED TARGET PAGE
    # --------------------------------------------------------

    return render_template(

        "target_confirmed.html",

        selected_target=selected_target,

        problem_type_result=problem_type_result,

        filename=filename,

        objective=objective,

        selected_info=selected_info
    )


# ============================================================
# DETAILED EDA
# ============================================================

@app.route(
    "/detailed-eda",
    methods=["POST"]
)
def detailed_eda():

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    target_column = request.form.get(
        "target",
        ""
    ).strip()

    problem_type = request.form.get(
        "problem_type",
        ""
    ).strip()

    objective = request.form.get(
        "objective",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not filename:

        return (
            "Dataset information is missing."
        )

    if not problem_type:

        return (
            "Problem type information is missing."
        )

    # --------------------------------------------------------
    # CLUSTERING: TARGET IS NOT REQUIRED
    # --------------------------------------------------------

    if not target_column and problem_type != "Clustering":

        return (
            "No target variable was provided."
        )

    filename = secure_filename(
        filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        return (
            "Dataset file not found."
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # CHART OUTPUT PATH
    # --------------------------------------------------------

    chart_base = os.path.splitext(
        filename
    )[0]

    if target_column:

        chart_base = (
            f"{chart_base}_{target_column}"
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    else:

        chart_base = (
            f"{chart_base}_clustering"
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

    # --------------------------------------------------------
    # PERFORM DETAILED EDA
    # --------------------------------------------------------

    try:

        eda_results = perform_detailed_eda(

            df,

            target_column=target_column,

            problem_type=problem_type,

            charts_dir=app.config[
                "CHARTS_FOLDER"
            ],

            chart_id=chart_base
        )

    except Exception as e:

        return (
            "Unable to perform detailed EDA: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # RESULTS PAGE
    # --------------------------------------------------------

    return render_template(

        "detailed_eda.html",

        filename=filename,

        target_column=target_column,

        problem_type=problem_type,

        objective=objective,

        eda=eda_results,

        charts_folder=app.config[
            "CHARTS_FOLDER"
        ]
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route(
    "/preprocessing",
    methods=["POST"]
)
def preprocessing():

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    target_column = request.form.get(
        "target",
        ""
    ).strip()

    problem_type = request.form.get(
        "problem_type",
        ""
    ).strip()

    objective = request.form.get(
        "objective",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not filename:

        return (
            "Dataset information is missing."
        )

    if not problem_type:

        return (
            "Problem type information is missing."
        )

    if not target_column and problem_type != "Clustering":

        return (
            "No target variable was provided."
        )

    filename = secure_filename(
        filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        return (
            "Dataset file not found."
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # PREPROCESSING PLAN
    # --------------------------------------------------------

    try:

        prep = prepare_preprocessing(
            df,
            target_column,
            problem_type
        )

    except Exception as e:

        return (
            "Unable to prepare preprocessing: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # FEATURE SELECTION REPORT
    # --------------------------------------------------------

    try:

        selection = perform_feature_selection(
            df,
            prep["roles"]["feature_columns"],
            target_column=(
                prep["target_column"]
            )
        )

    except Exception as e:

        selection = {
            "feature_count": 0,
            "high_correlation": [],
            "high_corr_count": 0,
            "low_variance": [],
            "low_variance_count": 0,
            "leakage_candidates": [],
            "leakage_count": 0
        }

    # --------------------------------------------------------
    # RESULTS PAGE
    # --------------------------------------------------------

    return render_template(

        "preprocessing.html",

        filename=filename,

        target_column=prep["target_column"],

        problem_type=problem_type,

        objective=objective,

        prep=prep,

        selection=selection
    )


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route(
    "/model-training",
    methods=["POST"]
)
def model_training():

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    target_column = request.form.get(
        "target",
        ""
    ).strip()

    problem_type = request.form.get(
        "problem_type",
        ""
    ).strip()

    objective = request.form.get(
        "objective",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not filename:

        return (
            "Dataset information is missing."
        )

    if not problem_type:

        return (
            "Problem type information is missing."
        )

    if not target_column and problem_type != "Clustering":

        return (
            "No target variable was provided."
        )

    if problem_type == "Clustering":

        return (
            "Clustering is not part of the supervised "
            "model training stage."
        )

    filename = secure_filename(
        filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        return (
            "Dataset file not found."
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    try:

        training = train_models(
            df,
            target_column,
            problem_type
        )

    except Exception as e:

        return (
            "Unable to train models: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    best = select_best_model(
        training["results"],
        problem_type
    )

    # --------------------------------------------------------
    # RESULTS PAGE
    # --------------------------------------------------------

    return render_template(

        "model_results.html",

        filename=filename,

        target_column=target_column,

        problem_type=problem_type,

        objective=objective,

        training=training,

        best=best
    )


# ============================================================
# CLUSTERING
# ============================================================

@app.route(
    "/clustering",
    methods=["POST"]
)
def clustering():

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    filename = request.form.get(
        "filename",
        ""
    ).strip()

    objective = request.form.get(
        "objective",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not filename:

        return (
            "Dataset information is missing."
        )

    filename = secure_filename(
        filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.exists(filepath):

        return (
            "Dataset file not found."
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        df = load_dataset(filepath)

    except Exception as e:

        return (
            "Unable to read dataset: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # CHART OUTPUT PATH
    # --------------------------------------------------------

    chart_base = (
        os.path.splitext(
            filename
        )[0]
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    # --------------------------------------------------------
    # PERFORM CLUSTERING
    # --------------------------------------------------------

    try:

        clustering_results = perform_clustering(
            df,
            objective=(
                objective
                if objective
                else None
            ),
            charts_dir=app.config[
                "CHARTS_FOLDER"
            ],
            chart_id=chart_base
        )

    except Exception as e:

        return (
            "Unable to perform clustering: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # RESULTS PAGE
    # --------------------------------------------------------

    return render_template(

        "clustering_results.html",

        filename=filename,

        objective=objective,

        result=clustering_results
    )


# ============================================================
# SERVE CHART IMAGES
# ============================================================

@app.route("/charts/<path:chart_filename>")
def serve_chart(chart_filename):

    return send_from_directory(
        app.config["CHARTS_FOLDER"],
        chart_filename
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )