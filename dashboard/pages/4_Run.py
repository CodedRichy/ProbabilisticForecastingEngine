import sys
import subprocess
import threading
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from dashboard.style import inject
from dashboard.components import sidebar
from dashboard.db import recent_batches_df

ROOT   = Path(__file__).parent.parent.parent
LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]

st.set_page_config(page_title="Apollo · Run", page_icon="🚀", layout="wide")
inject()
sidebar()

st.title("Discovery Run Launcher")


def _stream(process, key: str):
    for line in iter(process.stdout.readline, ""):
        st.session_state[key] = st.session_state.get(key, "") + line
    process.wait()
    st.session_state["_status"] = "done" if process.returncode == 0 else "error"


# ── Run history ───────────────────────────────────────────────
with st.expander("📜  Run history (recent batches)", expanded=False):
    batches = recent_batches_df(15)
    if batches.empty:
        st.caption("No batches yet.")
    else:
        st.dataframe(
            batches,
            use_container_width=True,
            hide_index=True,
            column_config={
                "batch":      st.column_config.TextColumn("Batch ID"),
                "hypotheses": st.column_config.NumberColumn("Hypotheses", format="%d"),
                "created_at": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD HH:mm"),
            },
        )

st.divider()

# ── Configuration ─────────────────────────────────────────────
st.subheader("Configure Batch")

col1, col2, col3 = st.columns(3)
with col1:
    league = st.selectbox("League", LEAGUES)
    batch_name = st.text_input(
        "Batch name",
        placeholder="Auto-generated if empty",
    )
with col2:
    max_pairwise = st.slider(
        "Max pairwise features",
        min_value=10, max_value=200, value=60, step=10,
        help="Higher = more hypotheses tested, longer runtime",
    )
with col3:
    gate1_only = st.checkbox(
        "Gate 1 only",
        value=False,
        help="Stop after Gate 1 — useful for initial exploration",
    )
    parallel = st.checkbox(
        "Parallel workers",
        value=True,
        help="Use multiprocessing — disable if debugging",
    )

# Build command preview
_cmd_parts = [
    f"python -m discovery.run",
    f"--league {league}",
    f"--max-pairwise {max_pairwise}",
]
if batch_name.strip():
    _cmd_parts.append(f"--batch {batch_name.strip()}")
if gate1_only:
    _cmd_parts.append("--gate1-only")
if not parallel:
    _cmd_parts.append("--no-parallel")

st.code(" ".join(_cmd_parts), language="bash")

# ── Launch / Stop ─────────────────────────────────────────────
status = st.session_state.get("_status", "idle")

c_launch, c_stop, _ = st.columns([1, 1, 5])

with c_launch:
    if st.button("🚀  Launch", type="primary", disabled=(status == "running"), use_container_width=True):
        cmd = [
            sys.executable, "-m", "discovery.run",
            "--league", league,
            "--max-pairwise", str(max_pairwise),
        ]
        if batch_name.strip():
            cmd += ["--batch", batch_name.strip()]
        if gate1_only:
            cmd.append("--gate1-only")
        if not parallel:
            cmd.append("--no-parallel")

        st.session_state["_output"] = ""
        st.session_state["_status"] = "running"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        st.session_state["_proc"] = proc

        threading.Thread(
            target=_stream, args=(proc, "_output"), daemon=True
        ).start()

with c_stop:
    if st.button("⏹  Stop", disabled=(status != "running"), use_container_width=True):
        proc = st.session_state.get("_proc")
        if proc:
            proc.terminate()
        st.session_state["_status"] = "stopped"

# ── Status banner ─────────────────────────────────────────────
st.divider()

if status == "running":
    st.info("⏳ Running… page auto-refreshes every second.")
elif status == "done":
    st.success("✅ Batch completed successfully.")
elif status == "error":
    st.error("❌ Run exited with errors — review the logs below.")
elif status == "stopped":
    st.warning("⏹ Stopped by user.")

# ── Live log ──────────────────────────────────────────────────
output = st.session_state.get("_output", "")
if output:
    line_count = output.count("\n")
    st.caption(f"{line_count} lines of output")

    # Colour-code key lines
    highlighted = []
    for line in output.splitlines():
        if "ERROR" in line or "FAILED" in line:
            highlighted.append(f"# ❌  {line}")
        elif "survivor" in line.lower() or "passed" in line.lower():
            highlighted.append(f"# ✅  {line}")
        elif line.startswith("  [") or line.startswith("["):
            highlighted.append(f"# ──  {line}")
        else:
            highlighted.append(line)

    st.code("\n".join(highlighted), language="python")

# Auto-refresh while running
if status == "running":
    time.sleep(1)
    st.rerun()
