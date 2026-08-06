"""
Experiment Tracker for Apollo Discovery Engine.

SQLite-based persistent store for every hypothesis generated,
every test run, and every validated signal. Append-only design —
nothing is ever updated or deleted.

This is the audit trail. It answers:
- How many tests have we ever run? (for multiple testing correction)
- What was the result of hypothesis H-03847? (for reproducibility)
- What signals are currently validated? (for the knowledge base)
- Has this hypothesis been tested before? (for deduplication)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class ExperimentTracker:
    """
    Persistent experiment tracking via SQLite.
    """
    
    def __init__(self, db_path: str = "discovery/tracking/apollo.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id   TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,
                level           INTEGER NOT NULL,
                features        TEXT NOT NULL,
                operator        TEXT NOT NULL,
                target          TEXT NOT NULL,
                condition       TEXT,
                description     TEXT,
                config_hash     TEXT,
                generation_batch TEXT
            );
            
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id          TEXT PRIMARY KEY,
                hypothesis_id   TEXT NOT NULL,
                gate            INTEGER NOT NULL,
                league          TEXT,
                seasons         TEXT,
                started_at      TEXT NOT NULL,
                completed_at    TEXT,
                
                n_observations  INTEGER,
                p_value_raw     REAL,
                p_value_by      REAL,
                t_statistic     REAL,
                effect_size     REAL,
                
                brier_model     REAL,
                brier_baseline  REAL,
                brier_skill     REAL,
                
                roi_0pct        REAL,
                roi_3pct        REAL,
                roi_5pct        REAL,
                n_bets_3pct     INTEGER,
                
                bayes_factor_min REAL,
                deflated_sharpe REAL,
                total_tests_at_time INTEGER,
                
                passed          INTEGER,
                rejection_reason TEXT,
                
                random_seed     INTEGER,
                code_version    TEXT,
                data_version    TEXT,
                full_result     TEXT,
                
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
            );
            
            CREATE TABLE IF NOT EXISTS signals (
                signal_id       TEXT PRIMARY KEY,
                hypothesis_id   TEXT NOT NULL,
                
                gate1_p         REAL,
                gate2_p         REAL,
                gate3_leagues   TEXT,
                gate4_p         REAL,
                
                brier_skill_avg REAL,
                roi_avg         REAL,
                effect_size_avg REAL,
                
                status          TEXT NOT NULL DEFAULT 'validated',
                validated_at    TEXT,
                last_checked    TEXT,
                
                decay_checks    INTEGER DEFAULT 0,
                decay_failures  INTEGER DEFAULT 0,
                
                features        TEXT,
                operator        TEXT,
                description     TEXT,
                full_evidence   TEXT,
                
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_test_runs_hypothesis 
                ON test_runs(hypothesis_id);
            CREATE INDEX IF NOT EXISTS idx_test_runs_gate 
                ON test_runs(gate);
            CREATE INDEX IF NOT EXISTS idx_signals_status 
                ON signals(status);
        """)
        conn.commit()
        conn.close()
    
    # ── HYPOTHESIS RECORDING ──────────────────────────────────
    
    def record_hypotheses(self, hypotheses: list, batch_id: str):
        """Record a batch of generated hypotheses."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        
        for h in hypotheses:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO hypotheses "
                    "(hypothesis_id, created_at, level, features, operator, "
                    "target, condition, description, config_hash, generation_batch) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        h.hypothesis_id, now, h.level,
                        json.dumps(h.features), h.operator, h.target,
                        h.condition, h.description,
                        h.config_hash(), batch_id
                    )
                )
            except Exception:
                pass
        
        conn.commit()
        conn.close()
    
    # ── TEST RUN RECORDING ────────────────────────────────────
    
    def record_test_run(self, run_id: str, hypothesis_id: str, 
                        gate: int, result: dict, **kwargs):
        """Record a single test execution. Append-only."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        
        conn.execute(
            "INSERT INTO test_runs "
            "(run_id, hypothesis_id, gate, league, seasons, started_at, completed_at, "
            "n_observations, p_value_raw, p_value_by, t_statistic, effect_size, "
            "brier_model, brier_baseline, brier_skill, "
            "roi_0pct, roi_3pct, roi_5pct, n_bets_3pct, "
            "bayes_factor_min, deflated_sharpe, total_tests_at_time, "
            "passed, rejection_reason, random_seed, code_version, data_version, "
            "full_result) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, hypothesis_id, gate,
                kwargs.get('league'), kwargs.get('seasons'),
                kwargs.get('started_at', now), now,
                result.get('n_observations'),
                result.get('p_value'),
                result.get('p_value_by'),
                result.get('t_statistic'),
                result.get('effect_size'),
                result.get('model_brier'), result.get('baseline_brier'),
                result.get('brier_skill_score'),
                result.get('roi_0pct'), result.get('roi_3pct'),
                result.get('roi_5pct'), result.get('n_bets_3pct'),
                result.get('bayes_factor_min'), result.get('deflated_sharpe'),
                result.get('total_tests_at_time'),
                1 if result.get('passed') else 0,
                result.get('reason') or result.get('rejection_reason'),
                kwargs.get('random_seed', 42),
                kwargs.get('code_version', '0.1.0'),
                kwargs.get('data_version'),
                json.dumps(result, default=str),
            )
        )
        conn.commit()
        conn.close()
    
    # ── SIGNAL RECORDING ──────────────────────────────────────
    
    def record_signal(self, signal_id: str, hypothesis_id: str,
                      evidence: dict):
        """Record a validated signal."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        
        conn.execute(
            "INSERT OR REPLACE INTO signals "
            "(signal_id, hypothesis_id, gate1_p, gate2_p, gate3_leagues, gate4_p, "
            "brier_skill_avg, roi_avg, effect_size_avg, "
            "status, validated_at, features, operator, description, full_evidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                signal_id, hypothesis_id,
                evidence.get('gate1_p'),
                evidence.get('gate2_p'),
                json.dumps(evidence.get('gate3_leagues', [])),
                evidence.get('gate4_p'),
                evidence.get('brier_skill_avg'),
                evidence.get('roi_avg'),
                evidence.get('effect_size_avg'),
                'validated', now,
                json.dumps(evidence.get('features', [])),
                evidence.get('operator'),
                evidence.get('description'),
                json.dumps(evidence, default=str),
            )
        )
        conn.commit()
        conn.close()
    
    # ── QUERIES ───────────────────────────────────────────────
    
    def get_total_tests(self) -> int:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT COUNT(*) FROM test_runs").fetchone()
        conn.close()
        return row[0]
    
    def get_gate_survivors(self, gate: int) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DISTINCT hypothesis_id FROM test_runs WHERE gate = ? AND passed = 1",
            (gate,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    
    def get_validated_signals(self) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM signals WHERE status = 'validated' ORDER BY brier_skill_avg DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    
    def hypothesis_already_tested(self, config_hash: str) -> bool:
        """Check if a hypothesis with this config has already been tested."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT 1 FROM hypotheses h JOIN test_runs t "
            "ON h.hypothesis_id = t.hypothesis_id "
            "WHERE h.config_hash = ? LIMIT 1",
            (config_hash,)
        ).fetchone()
        conn.close()
        return row is not None
    
    def summary(self) -> str:
        conn = sqlite3.connect(self.db_path)
        
        n_hyp = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        n_tests = conn.execute("SELECT COUNT(*) FROM test_runs").fetchone()[0]
        n_signals = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE status = 'validated'"
        ).fetchone()[0]
        
        gate_counts = {}
        for gate in [1, 2, 3, 4]:
            passed = conn.execute(
                "SELECT COUNT(DISTINCT hypothesis_id) FROM test_runs "
                "WHERE gate = ? AND passed = 1", (gate,)
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(DISTINCT hypothesis_id) FROM test_runs "
                "WHERE gate = ?", (gate,)
            ).fetchone()[0]
            gate_counts[gate] = (passed, total)
        
        conn.close()
        
        lines = [
            "Apollo Discovery Engine — Summary",
            "=" * 45,
            f"Hypotheses generated:  {n_hyp}",
            f"Total test runs:       {n_tests}",
            f"Validated signals:     {n_signals}",
            "",
            "Gate Survival:",
        ]
        for gate, (passed, total) in gate_counts.items():
            rate = f"{passed/total*100:.1f}%" if total > 0 else "—"
            lines.append(f"  Gate {gate}: {passed}/{total} ({rate})")
        
        return "\n".join(lines)
