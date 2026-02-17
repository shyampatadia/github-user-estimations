"""Database operations for GitHub User Estimation project."""

import sqlite3
from pathlib import Path
from typing import Optional

from . import config


def get_connection(db_path: str = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = config.DATA_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            github_id INTEGER NOT NULL,
            is_valid BOOLEAN NOT NULL,
            login TEXT,
            type TEXT,
            public_repos INTEGER,
            public_gists INTEGER,
            followers INTEGER,
            following INTEGER,
            created_at TEXT,
            updated_at TEXT,
            sampled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            run_id TEXT NOT NULL,
            stratum_id TEXT,
            token_id INTEGER,
            response_time_ms INTEGER,
            http_status INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_samples_run_id ON samples(run_id);
        CREATE INDEX IF NOT EXISTS idx_samples_github_id ON samples(github_id);
        CREATE INDEX IF NOT EXISTS idx_samples_stratum ON samples(stratum_id);

        CREATE TABLE IF NOT EXISTS ground_truth (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            github_id INTEGER UNIQUE NOT NULL,
            is_valid BOOLEAN NOT NULL,
            login TEXT,
            type TEXT,
            public_repos INTEGER,
            public_gists INTEGER,
            followers INTEGER,
            following INTEGER,
            created_at TEXT,
            updated_at TEXT,
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stratum_id TEXT NOT NULL,
            token_id INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_ground_truth_stratum ON ground_truth(stratum_id);

        CREATE TABLE IF NOT EXISTS estimation_runs (
            id TEXT PRIMARY KEY,
            run_type TEXT NOT NULL,
            stratum_id TEXT,
            sample_size INTEGER NOT NULL,
            sampling_rate REAL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            valid_count INTEGER,
            total_sampled INTEGER,
            estimated_total REAL,
            estimated_rate REAL,
            ci_lower REAL,
            ci_upper REAL,
            bootstrap_samples INTEGER,
            ground_truth_total INTEGER,
            relative_error REAL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            ci_lower REAL,
            ci_upper REAL,
            FOREIGN KEY (run_id) REFERENCES estimation_runs(id),
            UNIQUE(run_id, metric_name)
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);

        CREATE TABLE IF NOT EXISTS frontier_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            max_id_found INTEGER NOT NULL,
            max_id_created_at TEXT,
            max_id_login TEXT
        );
    """)
    conn.commit()

    # Deduplicate metrics table and add unique index (migration)
    try:
        conn.execute("""
            DELETE FROM metrics WHERE id NOT IN (
                SELECT MIN(id) FROM metrics GROUP BY run_id, metric_name
            )
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_unique ON metrics(run_id, metric_name)")
        conn.commit()
    except Exception:
        pass  # Index already exists or table is empty


def insert_ground_truth_batch(conn: sqlite3.Connection, records: list[dict]):
    conn.executemany("""
        INSERT OR IGNORE INTO ground_truth
            (github_id, is_valid, login, type, public_repos, public_gists,
             followers, following, created_at, updated_at, stratum_id, token_id)
        VALUES
            (:github_id, :is_valid, :login, :type, :public_repos, :public_gists,
             :followers, :following, :created_at, :updated_at, :stratum_id, :token_id)
    """, records)
    conn.commit()


def insert_samples_batch(conn: sqlite3.Connection, records: list[dict]):
    conn.executemany("""
        INSERT INTO samples
            (github_id, is_valid, login, type, public_repos, public_gists,
             followers, following, created_at, updated_at, run_id, stratum_id,
             token_id, response_time_ms, http_status)
        VALUES
            (:github_id, :is_valid, :login, :type, :public_repos, :public_gists,
             :followers, :following, :created_at, :updated_at, :run_id, :stratum_id,
             :token_id, :response_time_ms, :http_status)
    """, records)
    conn.commit()


def insert_estimation_run(conn: sqlite3.Connection, run: dict):
    conn.execute("""
        INSERT OR REPLACE INTO estimation_runs
            (id, run_type, stratum_id, sample_size, sampling_rate, started_at,
             completed_at, valid_count, total_sampled, estimated_total, estimated_rate,
             ci_lower, ci_upper, bootstrap_samples, ground_truth_total, relative_error, notes)
        VALUES
            (:id, :run_type, :stratum_id, :sample_size, :sampling_rate, :started_at,
             :completed_at, :valid_count, :total_sampled, :estimated_total, :estimated_rate,
             :ci_lower, :ci_upper, :bootstrap_samples, :ground_truth_total, :relative_error, :notes)
    """, run)
    conn.commit()


def insert_metrics_batch(conn: sqlite3.Connection, metrics: list[dict]):
    conn.executemany("""
        INSERT OR REPLACE INTO metrics (run_id, metric_name, metric_value, ci_lower, ci_upper)
        VALUES (:run_id, :metric_name, :metric_value, :ci_lower, :ci_upper)
    """, metrics)
    conn.commit()


def insert_frontier(conn: sqlite3.Connection, max_id: int, created_at: str = None, login: str = None):
    conn.execute("""
        INSERT INTO frontier_tracking (max_id_found, max_id_created_at, max_id_login)
        VALUES (?, ?, ?)
    """, (max_id, created_at, login))
    conn.commit()


def get_ground_truth_by_stratum(conn: sqlite3.Connection, stratum_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM ground_truth WHERE stratum_id = ?", (stratum_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_ground_truth_count(conn: sqlite3.Connection, stratum_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM ground_truth WHERE stratum_id = ?", (stratum_id,)
    ).fetchone()
    return row["cnt"]


def get_ground_truth_valid_count(conn: sqlite3.Connection, stratum_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM ground_truth WHERE stratum_id = ? AND is_valid = 1",
        (stratum_id,),
    ).fetchone()
    return row["cnt"]


def get_all_ground_truth(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM ground_truth").fetchall()
    return [dict(r) for r in rows]


def get_samples_by_run(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM samples WHERE run_id = ?", (run_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_estimation_runs(conn: sqlite3.Connection, run_type: str = None) -> list[dict]:
    if run_type:
        rows = conn.execute(
            "SELECT * FROM estimation_runs WHERE run_type = ?", (run_type,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM estimation_runs").fetchall()
    return [dict(r) for r in rows]


def get_tokens(db_path: str = None) -> list[dict]:
    if db_path is None:
        db_path = config.TOKENS_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tokens").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_collected_ids_for_stratum(conn: sqlite3.Connection, stratum_id: str) -> set:
    rows = conn.execute(
        "SELECT github_id FROM ground_truth WHERE stratum_id = ?", (stratum_id,)
    ).fetchall()
    return {r["github_id"] for r in rows}


def get_collected_sample_ids_for_run(conn: sqlite3.Connection, run_id: str) -> set:
    rows = conn.execute(
        "SELECT github_id FROM samples WHERE run_id = ?", (run_id,)
    ).fetchall()
    return {r["github_id"] for r in rows}


def get_sample_count_for_run(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM samples WHERE run_id = ?", (run_id,)
    ).fetchone()
    return row["cnt"]
