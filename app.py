from flask import Flask, render_template, request
import os
import pandas as pd
from werkzeug.utils import secure_filename

# Import project modules
from modules.data_validator import validate_dataset
from modules.initial_eda import perform_initial_eda
from modules.objective_target import analyze_target_candidates


app = Flask(__name__)

# Upload configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------------
# CHECK ALLOWED FILE TYPE
# -----------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# -----------------------------------
# HOME PAGE
# -----------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------
# UPLOAD DATASET
# -----------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():

    if request.method == "POST":

        # Check if dataset exists
        if "dataset" not in request.files:
            return "No file selected"

        file = request.files["dataset"]

        # Check if filename is empty
        if file.filename == "":
            return "Please select a file"

        # Check file type
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            # Save uploaded file
            file.save(filepath)

            # Read dataset
            if filename.lower().endswith(".csv"):

                df = pd.read_csv(filepath)

            else:

                df = pd.read_excel(filepath)


            # -----------------------------------
            # DATASET VALIDATION
            # -----------------------------------

            validation_results = validate_dataset(df)


            # -----------------------------------
            # INITIAL EDA
            # -----------------------------------

            eda_results = perform_initial_eda(df)


            # -----------------------------------
            # TARGET VARIABLE ANALYSIS
            # -----------------------------------

            target_candidates = analyze_target_candidates(df)


            # -----------------------------------
            # DATASET OVERVIEW INFORMATION
            # -----------------------------------

            rows, columns = df.shape

            column_names = df.columns.tolist()

            data_types = (
                df.dtypes.astype(str).to_dict()
            )

            missing_values = (
                df.isnull().sum().to_dict()
            )

            duplicate_rows = int(
                df.duplicated().sum()
            )

            preview = df.head().to_html(
                classes="table",
                index=False
            )


            # -----------------------------------
            # SHOW DATASET OVERVIEW
            # -----------------------------------

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

                eda_results=eda_results,

                target_candidates=target_candidates
            )


        return (
            "Invalid file type. "
            "Please upload CSV or XLSX."
        )


    return render_template("upload.html")


# -----------------------------------
# TARGET SELECTION PAGE
# -----------------------------------

@app.route("/target-selection", methods=["POST"])
def target_selection():

    # Get recommended target from form
    recommended_target = request.form.get(
        "recommended_target"
    )

    # Get uploaded filename
    filename = request.form.get("filename")

    # We need all column names.
    # For now, find the latest uploaded file.

    uploaded_files = os.listdir(
        app.config["UPLOAD_FOLDER"]
    )

    if not uploaded_files:

        return "No uploaded dataset found."

    # Get latest file
    latest_file = max(
        uploaded_files,
        key=lambda x: os.path.getctime(
            os.path.join(
                app.config["UPLOAD_FOLDER"],
                x
            )
        )
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        latest_file
    )

    # Read dataset again
    if latest_file.lower().endswith(".csv"):

        df = pd.read_csv(filepath)

    else:

        df = pd.read_excel(filepath)


    # Get column names
    column_names = df.columns.tolist()


    # Show target selection page
    return render_template(
        "objective_target.html",

        recommended_target=recommended_target,

        column_names=column_names
    )


# -----------------------------------
# CONFIRM TARGET VARIABLE
# -----------------------------------

@app.route("/confirm-target", methods=["POST"])
def confirm_target():

    # Get selected target
    selected_target = request.form.get("target")

    return render_template(
        "target_confirmed.html",

        selected_target=selected_target
    )


# -----------------------------------
# RUN APPLICATION
# -----------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )