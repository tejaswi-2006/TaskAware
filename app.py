from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, session
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ============================================================
# PROJECT MODULES
# ============================================================

from modules.data_loader import load_dataset, allowed_file
from modules.data_validator import validate_dataset
from modules.initial_eda import perform_initial_eda
from modules.objective_target import analyze_target_candidates
from modules.problem_type_detector import detect_problem_type
from modules.decision_engine import decide_route
from modules.detailed_eda import perform_detailed_eda
from modules.preprocessing import prepare_preprocessing
from modules.feature_selection import perform_feature_selection
from modules.model_trainer import train_models, train_named_model, TaskAwareTrainingError
from modules.model_selector import select_best_model
from modules.clustering import perform_clustering
from modules.explainability import perform_explainability
from modules.ai_explainer import (
    build_presentation,
    generate_explanation,
    get_api_key,
)
import modules.okf_manager as okf_manager
from modules.okf_manager import okf_summary
from modules.report_generator import collect_report_data, save_report
import modules.database as database
from modules.database import init_db


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

CHARTS_FOLDER = "results/charts"

REPORTS_FOLDER = "results/reports"

ALLOWED_EXTENSIONS = {
    "csv",
    "xlsx"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["CHARTS_FOLDER"] = CHARTS_FOLDER

app.config["REPORTS_FOLDER"] = REPORTS_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    CHARTS_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORTS_FOLDER,
    exist_ok=True
)

# ============================================================
# SESSION + DATABASE INIT
# ============================================================

app.secret_key = os.environ.get(
    "TASKAWARE_SECRET_KEY",
    "taskaware-dev-secret-key"
)

init_db()


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
# LOGIN REQUIRED DECORATOR
# ============================================================

def login_required(view):

    @wraps(view)
    def wrapped(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please log in to continue.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return view(*args, **kwargs)

    return wrapped


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "GET":

        return render_template(
            "auth.html",
            mode="register"
        )

    username = request.form.get(
        "username",
        ""
    ).strip()

    email = request.form.get(
        "email",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    confirm = request.form.get(
        "confirm_password",
        ""
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not username or not email or not password:

        flash(
            "All fields are required.",
            "error"
        )

        return render_template(
            "auth.html",
            mode="register"
        )

    if password != confirm:

        flash(
            "Passwords do not match.",
            "error"
        )

        return render_template(
            "auth.html",
            mode="register"
        )

    if len(password) < 6:

        flash(
            "Password must be at least 6 characters.",
            "error"
        )

        return render_template(
            "auth.html",
            mode="register"
        )

    # --------------------------------------------------------
    # CREATE USER
    # --------------------------------------------------------

    password_hash = generate_password_hash(
        password
    )

    user_id, error = database.create_user(
        username,
        email,
        password_hash
    )

    if error:

        flash(
            "Username or email already exists.",
            "error"
        )

        return render_template(
            "auth.html",
            mode="register"
        )

    session["user_id"] = user_id

    session["username"] = username

    flash(
        "Registration successful. Welcome!",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "GET":

        return render_template(
            "auth.html",
            mode="login"
        )

    username = request.form.get(
        "username",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    user = database.get_user_by_username(
        username
    )

    if not user or not check_password_hash(
        user["password_hash"],
        password
    ):

        flash(
            "Invalid username or password.",
            "error"
        )

        return render_template(
            "auth.html",
            mode="login"
        )

    session["user_id"] = user["id"]

    session["username"] = user["username"]

    flash(
        "Logged in successfully.",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    projects = database.list_projects(
        session["user_id"]
    )

    return render_template(
        "dashboard.html",
        username=session.get("username"),
        projects=projects
    )


# ============================================================
# CREATE PROJECT
# ============================================================

@app.route(
    "/project/new",
    methods=["POST"]
)
@login_required
def new_project():

    name = request.form.get(
        "name",
        ""
    ).strip()

    description = request.form.get(
        "description",
        ""
    ).strip()

    if not name:

        flash(
            "Project name is required.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    database.create_project(
        session["user_id"],
        name,
        description or None
    )

    flash(
        "Project created.",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# PROJECT DETAIL
# ============================================================

@app.route("/project/<int:project_id>")
@login_required
def project_detail(project_id):

    project = database.get_project(
        project_id,
        session["user_id"]
    )

    if not project:

        flash(
            "Project not found.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    runs = database.list_analysis_runs(
        project_id
    )

    return render_template(
        "project_detail.html",
        project=project,
        runs=runs
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

        project_id = request.args.get(
            "project_id",
            type=int
        )

        if project_id and "user_id" in session:

            project = database.get_project(
                project_id,
                session["user_id"]
            )

            if project:
                session["active_project_id"] = project_id
            else:
                session.pop(
                    "active_project_id",
                    None
                )

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
# EXPLAINABILITY
# ============================================================

@app.route(
    "/explainability",
    methods=["POST"]
)
def explainability():

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

    selected_model = request.form.get(
        "selected_model",
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

    # ========================================================
    # CLUSTERING EXPLAINABILITY (PROFILES ALREADY COMPUTED)
    # ========================================================

    if problem_type == "Clustering":

        return render_template(
            "explainability.html",
            filename=filename,
            target_column=None,
            problem_type=problem_type,
            objective=objective,
            is_clustering=True,
            shap_result=None
        )

    # ========================================================
    # SUPERVISED EXPLAINABILITY (SHAP)
    # ========================================================

    if not target_column:

        return (
            "No target variable was provided."
        )

    if not selected_model:

        # Retrain all models and pick the best one
        try:

            training = train_models(
                df,
                target_column,
                problem_type
            )

            best = select_best_model(
                training["results"],
                problem_type
            )

            if best is None:
                return (
                    "Unable to select a model for "
                    "explanation."
                )

            selected_model = best["name"]

        except Exception as e:

            return (
                "Unable to select a model: "
                f"{str(e)}"
            )

    # --------------------------------------------------------
    # RETRAIN THE SELECTED MODEL
    # --------------------------------------------------------

    try:

        fitted = train_named_model(
            df,
            target_column,
            problem_type,
            selected_model
        )

    except Exception as e:

        return (
            "Unable to retrain the model: "
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
    # SHAP EXPLANATION
    # --------------------------------------------------------

    try:

        shap_result = perform_explainability(
            df,
            target_column,
            problem_type,
            pipeline=fitted["pipeline"],
            X=fitted["X_test"],
            y=fitted["y_test"],
            charts_dir=app.config[
                "CHARTS_FOLDER"
            ],
            chart_id=f"{chart_base}_{selected_model}"
        )

    except Exception as e:

        return (
            "Unable to explain the model: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # RESULTS PAGE
    # --------------------------------------------------------

    return render_template(
        "explainability.html",
        filename=filename,
        target_column=target_column,
        problem_type=problem_type,
        objective=objective,
        selected_model=selected_model,
        is_clustering=False,
        shap_result=shap_result
    )


# ============================================================
# AI EXPLANATION
# ============================================================

@app.route(
    "/ai-explanation",
    methods=["POST"]
)
def ai_explanation():

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

    selected_model = request.form.get(
        "selected_model",
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

    # ========================================================
    # BUILD STRUCTURED CONTEXT
    # ========================================================

    context = {
        "problem_type": problem_type,
        "target": target_column or None,
        "model": selected_model or None,
        "objective": objective or None,
        "data_quality": None,
        "metrics": None,
        "top_features": [],
        "feature_importance": [],
        "cluster_profiles": None,
        "evaluation_notes": None,
        "n_clusters": None,
        "method": None,
        "n_records": int(len(df)),
        "n_test": None,
        "target_classes": None,
        "selection_reason": None,
        "source_columns": {},
    }

    shap_result = None

    # ------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------

    if problem_type == "Clustering":

        try:

            cluster_result = perform_clustering(
                df,
                objective=(
                    objective
                    if objective
                    else None
                )
            )

            if cluster_result.get("success"):

                context["model"] = "K-Means"

                context["method"] = "Cluster Profiles"

                context["n_clusters"] = (
                    cluster_result["n_clusters"]
                )

                context["top_features"] = [
                    item["column"]
                    for item in cluster_result[
                        "distinguishing_features"
                    ][:6]
                ]

                context["cluster_profiles"] = [
                    {
                        "cluster": p["cluster"],
                        "size": p["size"],
                        "features": [
                            {
                                "column": f["column"],
                                "value": (
                                    f["mean"]
                                    if f["type"] == "numerical"
                                    else f["dominant"]
                                )
                            }
                            for f in p["features"][:6]
                        ]
                    }
                    for p in cluster_result[
                        "profiles"
                    ]
                ]

                context["evaluation_notes"] = (
                    f"Clusters were selected using the "
                    f"silhouette score (K="
                    f"{cluster_result['n_clusters']}). "
                    "Clusters are discovered patterns, "
                    "not ground-truth labels."
                )

        except Exception:
            pass

    # ------------------------------------------------
    # SUPERVISED
    # ------------------------------------------------

    else:

        if not target_column:
            return (
                "No target variable was provided."
            )

        if not selected_model:

            try:

                training = train_models(
                    df,
                    target_column,
                    problem_type
                )

                best = select_best_model(
                    training["results"],
                    problem_type
                )

                if best is not None:
                    selected_model = best["name"]

                    context["selection_reason"] = (
                        best.get("reason")
                    )

            except Exception:
                pass

        if selected_model:

            context["model"] = selected_model

            try:

                fitted = train_named_model(
                    df,
                    target_column,
                    problem_type,
                    selected_model
                )

                # ----- SHAP / importance -----
                shap_result = perform_explainability(
                    df,
                    target_column,
                    problem_type,
                    pipeline=fitted["pipeline"],
                    X=fitted["X_test"],
                    y=fitted["y_test"]
                )

                context["top_features"] = (
                    shap_result.get("top_features", [])
                )

                context["feature_importance"] = (
                    shap_result.get(
                        "feature_importance",
                        []
                    )[:6]
                )

                context["method"] = shap_result.get("method")

                context["source_columns"] = (
                    shap_result.get("source_columns", {})
                )

                context["n_test"] = int(
                    len(fitted["X_test"])
                )

                target_info = fitted.get("target_info") or {}

                context["target_classes"] = (
                    target_info.get("classes")
                )

                # ----- Metrics -----
                from sklearn.metrics import (
                    accuracy_score,
                    f1_score,
                    precision_score,
                    recall_score,
                    r2_score,
                    mean_absolute_error,
                    mean_squared_error
                )

                y_pred = fitted["pipeline"].predict(
                    fitted["X_test"]
                )

                y_true = fitted["y_test"]

                if problem_type == "Classification":

                    context["metrics"] = {
                        "accuracy": round(
                            float(
                                accuracy_score(
                                    y_true,
                                    y_pred
                                )
                            ),
                            3
                        ),
                        "f1_score": round(
                            float(
                                f1_score(
                                    y_true,
                                    y_pred,
                                    average="macro",
                                    zero_division=0
                                )
                            ),
                            3
                        ),
                        "precision": round(
                            float(
                                precision_score(
                                    y_true,
                                    y_pred,
                                    average="macro",
                                    zero_division=0
                                )
                            ),
                            3
                        ),
                        "recall": round(
                            float(
                                recall_score(
                                    y_true,
                                    y_pred,
                                    average="macro",
                                    zero_division=0
                                )
                            ),
                            3
                        )
                    }

                else:

                    context["metrics"] = {
                        "r2": round(
                            float(
                                r2_score(
                                    y_true,
                                    y_pred
                                )
                            ),
                            3
                        ),
                        "mae": round(
                            float(
                                mean_absolute_error(
                                    y_true,
                                    y_pred
                                )
                            ),
                            3
                        ),
                        "rmse": round(
                            float(
                                np.sqrt(
                                    mean_squared_error(
                                        y_true,
                                        y_pred
                                    )
                                )
                            ),
                            3
                        )
                    }

            except Exception:
                pass

    # ========================================================
    # GENERATE EXPLANATION (OKF-aware)
    # ========================================================

    explanation_result = generate_explanation(
        context,
        use_okf=True,
        okf_manager=okf_manager
    )

    presentation = build_presentation(
        context,
        explanation_result
    )

    # --------------------------------------------------------
    # RESULTS PAGE
    # --------------------------------------------------------

    return render_template(
        "ai_explanation.html",
        filename=filename,
        target_column=target_column or None,
        problem_type=problem_type,
        objective=objective,
        selected_model=selected_model or None,
        context=context,
        explanation_result=explanation_result,
        presentation=presentation,
        shap_result=shap_result,
        api_configured=bool(get_api_key())
    )


# ============================================================
# OKF STATUS
# ============================================================

@app.route("/okf")
def okf_status():

    return render_template(
        "okf_status.html",
        summary=okf_summary()
    )


# ============================================================
# FINAL REPORT
# ============================================================

@app.route(
    "/report",
    methods=["POST"]
)
def report():

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

    selected_model = request.form.get(
        "selected_model",
        ""
    ).strip()

    explanation = request.form.get(
        "explanation",
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
    # COLLECT REPORT DATA
    # --------------------------------------------------------

    try:

        report_data = collect_report_data(
            df,
            filename,
            objective,
            target_column,
            problem_type,
            explanation,
            selected_model=(
                selected_model
                if selected_model
                else None
            )
        )

    except Exception as e:

        return (
            "Unable to generate report: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # SAVE + RENDER
    # --------------------------------------------------------

    try:

        html_path, html_content = save_report(
            report_data,
            app.config["REPORTS_FOLDER"]
        )

    except Exception as e:

        return (
            "Unable to save report: "
            f"{str(e)}"
        )

    # --------------------------------------------------------
    # SAVE ANALYSIS RUN (when logged in)
    # --------------------------------------------------------

    run_id = None

    project_id = session.get(
        "active_project_id"
    )

    if project_id and "user_id" in session:

        project = database.get_project(
            project_id,
            session["user_id"]
        )

        if project:

            run_id = database.create_analysis_run(
                project_id,
                filename,
                problem_type,
                objective=(
                    objective
                    if objective
                    else None
                ),
                target_column=(
                    target_column
                    if target_column
                    else None
                ),
                selected_model=(
                    selected_model
                    if selected_model
                    else None
                ),
                explanation=(
                    explanation
                    if explanation
                    else None
                ),
                status="completed"
            )

    return render_template(
        "report.html",
        report=report_data,
        report_saved=True,
        run_id=run_id,
        project_id=project_id
    )


# ============================================================
# SERVE SAVED REPORTS
# ============================================================

@app.route("/reports/<path:report_filename>")
def serve_report(report_filename):

    return send_from_directory(
        app.config["REPORTS_FOLDER"],
        report_filename
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