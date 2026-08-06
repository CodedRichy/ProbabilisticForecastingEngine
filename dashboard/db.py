"""
Apollo Dashboard — DB query layer.

Thin wrappers over ExperimentTracker SQLite DB that return DataFrames
ready for Streamlit display. All paths resolved relative to project root.
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "discovery" / "tracking" / "apollo.db"


def db_exists() -> bool:
    return DB_PATH.exists()


def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def overview_stats() -> dict:
    if not db_exists():
        return {"n_hypotheses": 0, "n_tests": 0, "n_signals": 0, "n_batches": 0}
    conn = _connect()
    n_hyp = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
    n_tests = conn.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0]
    n_signals = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE status='validated'"
    ).fetchone()[0]
    n_batches = conn.execute(
        "SELECT COUNT(DISTINCT generation_batch) FROM hypotheses "
        "WHERE generation_batch IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return {"n_hypotheses": n_hyp, "n_tests": n_tests,
            "n_signals": n_signals, "n_batches": n_batches}


def gate_funnel_df() -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame(
            {"gate": ["Gate 1", "Gate 2", "Gate 3", "Gate 4"],
             "passed": [0, 0, 0, 0],
             "total": [0, 0, 0, 0]}
        )
    conn = _connect()
    rows = []
    for gate in [1, 2, 3, 4]:
        passed = conn.execute(
            "SELECT COUNT(DISTINCT hypothesis_id) FROM test_runs "
            "WHERE gate=? AND passed=1", (gate,)
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(DISTINCT hypothesis_id) FROM test_runs WHERE gate=?", (gate,)
        ).fetchone()[0]
        rows.append({"gate": f"Gate {gate}", "passed": passed, "total": total})
    conn.close()
    return pd.DataFrame(rows)


def signals_df() -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT signal_id, description, operator, status, "
        "brier_skill_avg, roi_avg, effect_size_avg, "
        "gate1_p, gate2_p, gate4_p, validated_at "
        "FROM signals ORDER BY brier_skill_avg DESC",
        conn,
    )
    conn.close()
    return df


def hypotheses_df() -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()
    conn = _connect()
    df = pd.read_sql_query(
        """
        SELECT h.hypothesis_id, h.description, h.level, h.target, h.operator,
               h.generation_batch, h.created_at,
               MAX(CASE WHEN t.passed=1 THEN t.gate ELSE 0 END) AS best_gate_passed,
               MAX(t.gate)                                        AS highest_gate_tested,
               MIN(t.p_value_raw)                                 AS best_p_raw
        FROM hypotheses h
        LEFT JOIN test_runs t ON h.hypothesis_id = t.hypothesis_id
        GROUP BY h.hypothesis_id
        ORDER BY h.created_at DESC
        """,
        conn,
    )
    conn.close()
    return df


def hypothesis_runs_df(hypothesis_id: str) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT run_id, gate, league, passed, p_value_raw, p_value_by, "
        "brier_model, brier_baseline, brier_skill, roi_3pct, n_observations, "
        "rejection_reason, completed_at "
        "FROM test_runs WHERE hypothesis_id=? ORDER BY gate, completed_at",
        conn,
        params=(hypothesis_id,),
    )
    conn.close()
    return df


def tests_timeline_df() -> pd.DataFrame:
    """Daily test counts with running cumulative total."""
    if not db_exists():
        return pd.DataFrame()
    conn = _connect()
    df = pd.read_sql_query(
        """
        SELECT DATE(completed_at) AS date, COUNT(*) AS n_tests
        FROM test_runs
        WHERE completed_at IS NOT NULL
        GROUP BY DATE(completed_at)
        ORDER BY date
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["cumulative"] = df["n_tests"].cumsum()
    return df


def pvalue_dist_df() -> pd.DataFrame:
    """All Gate 1 raw p-values for distribution analysis."""
    if not db_exists():
        return pd.DataFrame()
    conn = _connect()
    df = pd.read_sql_query(
        "SELECT p_value_raw FROM test_runs "
        "WHERE gate=1 AND p_value_raw IS NOT NULL",
        conn,
    )
    conn.close()
    return df


def pass_rate_by_level_df() -> pd.DataFrame:
    """Gate 1 pass rate grouped by hypothesis level × target."""
    if not db_exists():
        return pd.DataFrame()
    conn = _connect()
    df = pd.read_sql_query(
        """
        SELECT h.level, h.target,
               COUNT(*)                                            AS total,
               SUM(CASE WHEN t.passed=1 THEN 1 ELSE 0 END)       AS passed
        FROM hypotheses h
        JOIN test_runs t ON h.hypothesis_id = t.hypothesis_id
        WHERE t.gate = 1
        GROUP BY h.level, h.target
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["pass_rate"] = df["passed"] / df["total"].replace(0, 1)
    return df


def recent_batches_df(n: int = 10) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()
    conn = _connect()
    df = pd.read_sql_query(
        """
        SELECT generation_batch AS batch,
               COUNT(*)         AS hypotheses,
               MIN(created_at)  AS created_at
        FROM hypotheses
        WHERE generation_batch IS NOT NULL
        GROUP BY generation_batch
        ORDER BY created_at DESC
        LIMIT ?
        """,
        conn,
        params=(n,),
    )
    conn.close()
    return df
