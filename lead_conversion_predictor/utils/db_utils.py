"""
db_utils.py — SQLite helper for storing and retrieving predictions.

Author: [Your Name]
College Project
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../app/predictions.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the predictions table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
            age             INTEGER,
            income          INTEGER,
            lead_source     TEXT,
            website_visits  INTEGER,
            time_spent      REAL,
            pages_viewed    INTEGER,
            email_opened    INTEGER,
            prev_interaction INTEGER,
            lead_score      INTEGER,
            industry        TEXT,
            follow_up_calls INTEGER,
            model_used      TEXT,
            prediction      INTEGER,
            probability     REAL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(input_data: dict, model_name: str, prediction: int, probability: float):
    """Insert a single prediction record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            age, income, lead_source, website_visits, time_spent,
            pages_viewed, email_opened, prev_interaction, lead_score,
            industry, follow_up_calls, model_used, prediction, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        input_data["age"],
        input_data["income"],
        input_data["lead_source"],
        input_data["website_visits"],
        input_data["time_spent_on_site"],
        input_data["pages_viewed"],
        input_data["email_opened"],
        input_data["previous_interaction"],
        input_data["lead_score"],
        input_data["industry"],
        input_data["follow_up_calls"],
        model_name,
        prediction,
        round(probability, 4),
    ))
    conn.commit()
    conn.close()


def fetch_all_predictions() -> pd.DataFrame:
    """Return all saved predictions as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY timestamp DESC", conn
    )
    conn.close()
    return df


def fetch_summary_stats() -> dict:
    """Return quick summary stats from the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    converted = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(probability) FROM predictions")
    avg_prob = cursor.fetchone()[0] or 0.0

    conn.close()
    return {
        "total_predictions": total,
        "predicted_conversions": converted,
        "avg_probability": round(avg_prob, 4),
    }
