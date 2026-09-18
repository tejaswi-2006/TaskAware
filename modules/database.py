import os
import sqlite3
from datetime import datetime


# ============================================================
# DATABASE LAYER
#
# Tables:
#   users       - registered users
#   projects    - user-created projects
#   analysis_runs - each completed analysis run
# ============================================================

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "instance",
    "taskaware.db"
)


def get_db_path(db_path=None):
    return db_path or DEFAULT_DB_PATH


def connect(db_path=None):
    """
    Open a database connection.
    """

    path = get_db_path(db_path)

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    conn = sqlite3.connect(path)

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# SCHEMA
# ============================================================

def init_db(db_path=None):
    """
    Create tables if they do not exist.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            dataset_filename TEXT NOT NULL,
            objective TEXT,
            target_column TEXT,
            problem_type TEXT NOT NULL,
            selected_model TEXT,
            status TEXT NOT NULL,
            explanation TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id)
        );
        """
    )

    conn.commit()

    conn.close()


# ============================================================
# USERS
# ============================================================

def create_user(
    username,
    email,
    password_hash,
    db_path=None
):
    """
    Insert a new user. Returns (user_id, error).
    """

    conn = connect(db_path)

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users
                (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                email,
                password_hash,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        conn.commit()

        user_id = cursor.lastrowid

        conn.close()

        return user_id, None

    except sqlite3.IntegrityError as e:

        conn.close()

        return None, str(e)


def get_user_by_username(username, db_path=None):
    """
    Look up a user by username.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM users
        WHERE username = ?
        """,
        (username,)
    )

    row = cursor.fetchone()

    conn.close()

    return dict(row) if row else None


def get_user_by_id(user_id, db_path=None):
    """
    Look up a user by id.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM users
        WHERE id = ?
        """,
        (user_id,)
    )

    row = cursor.fetchone()

    conn.close()

    return dict(row) if row else None


# ============================================================
# PROJECTS
# ============================================================

def create_project(
    user_id,
    name,
    description=None,
    db_path=None
):
    """
    Create a new project.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO projects
            (user_id, name, description, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            name,
            description,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()

    project_id = cursor.lastrowid

    conn.close()

    return project_id


def list_projects(user_id, db_path=None):
    """
    List all projects for a user.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT p.*,
               COUNT(r.id) AS run_count
        FROM projects p
        LEFT JOIN analysis_runs r
               ON r.project_id = p.id
        WHERE p.user_id = ?
        GROUP BY p.id
        ORDER BY p.created_at DESC
        """,
        (user_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [dict(r) for r in rows]


def get_project(project_id, user_id, db_path=None):
    """
    Get a single project owned by a user.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM projects
        WHERE id = ? AND user_id = ?
        """,
        (project_id, user_id)
    )

    row = cursor.fetchone()

    conn.close()

    return dict(row) if row else None


# ============================================================
# ANALYSIS RUNS
# ============================================================

def create_analysis_run(
    project_id,
    dataset_filename,
    problem_type,
    objective=None,
    target_column=None,
    selected_model=None,
    explanation=None,
    status="completed",
    db_path=None
):
    """
    Record a completed analysis run.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO analysis_runs
            (project_id, dataset_filename, objective,
             target_column, problem_type, selected_model,
             status, explanation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            dataset_filename,
            objective,
            target_column,
            problem_type,
            selected_model,
            status,
            explanation,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()

    run_id = cursor.lastrowid

    conn.close()

    return run_id


def list_analysis_runs(project_id, db_path=None):
    """
    List analysis runs for a project.
    """

    conn = connect(db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM analysis_runs
        WHERE project_id = ?
        ORDER BY created_at DESC
        """,
        (project_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [dict(r) for r in rows]