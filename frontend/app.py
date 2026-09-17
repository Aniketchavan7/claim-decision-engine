"""Streamlit Web Interface for Reviewing and Analyzing Insurance Claims."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

# Setup page layout
st.set_page_config(
    page_title="Claim Decision Engine | Multi-Agent RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Custom CSS for styling
st.markdown(
    """
<style>
    .badge-admissible { background-color: #28a745; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-limits { background-color: #17a2b8; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-partial { background-color: #fd7e14; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-not-admissible { background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-review { background-color: #6c757d; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .metric-card { 
        background-color: #1e293b; 
        color: #f8fafc; 
        border: 1px solid #334155; 
        border-left: 4px solid #38bdf8; 
        border-radius: 8px; 
        padding: 14px; 
        margin-bottom: 12px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-card b { color: #38bdf8; }
    .metric-card code { background-color: #0f172a; color: #a5f3fc; padding: 2px 6px; border-radius: 4px; }
    .citation-card { 
        background-color: #1e293b; 
        color: #f8fafc; 
        border-left: 4px solid #818cf8; 
        border-top: 1px solid #334155; 
        border-right: 1px solid #334155; 
        border-bottom: 1px solid #334155; 
        padding: 12px 16px; 
        margin-bottom: 14px; 
        border-radius: 0 8px 8px 0; 
    }
    .citation-card b { color: #a5b4fc; }
    .citation-card code { background-color: #0f172a; color: #c7d2fe; padding: 2px 6px; border-radius: 4px; }
    .citation-card pre { background-color: #0f172a; color: #e2e8f0; padding: 10px; border-radius: 6px; border: 1px solid #334155; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_public_cases() -> List[Dict[str, Any]]:
    path = Path("candidate_data/public_test_cases.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_custom_cases() -> List[Dict[str, Any]]:
    path = Path("evaluation/custom_test_cases.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


public_cases = load_public_cases()
custom_cases = load_custom_cases()
all_preloaded = {c["case_id"]: c for c in (public_cases + custom_cases)}

# Sidebar: Controls and Input Selection
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Claim Adjudication")
    st.caption("Policy-Aware Multi-Agent RAG Engine")

    st.divider()

    # Health indicator
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            st.success("API Backend: Connected", icon="🟢")
        else:
            st.warning("API Backend: Unhealthy", icon="🟡")
    except Exception:
        st.error("API Backend: Offline", icon="🔴")

    st.subheader("Select Claim Case")
    source_type = st.radio(
        "Source",
        ["Preloaded Test Cases", "Upload JSON Case", "Manual Paste"],
        label_visibility="collapsed",
    )

    selected_case = None

    if source_type == "Preloaded Test Cases":
        case_options = list(all_preloaded.keys())
        if case_options:
            sel_id = st.selectbox("Choose Case ID", case_options)
            selected_case = all_preloaded[sel_id]
            st.info(f"Task: {selected_case.get('task', 'Adjudicate claim')}")

    elif source_type == "Upload JSON Case":
        uploaded_file = st.file_uploader("Upload JSON", type=["json"])
        if uploaded_file:
            try:
                selected_case = json.load(uploaded_file)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    elif source_type == "Manual Paste":
        pasted_text = st.text_area("Paste Claim JSON", height=250)
        if pasted_text:
            try:
                selected_case = json.loads(pasted_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    run_btn = st.button("🚀 Analyze Claim Case", type="primary", use_container_width=True)


# Main layout
st.header("Policy-Aware Multi-Agent Claim Decision Engine")
st.caption(
    "Universal Sompo General Insurance Co. Ltd. (Policy UNIHLIP18004V011718) | IRDAI Evidence-Grounded Multi-Agent RAG"
)

if not selected_case:
    st.info("👈 Select or upload a claim case from the sidebar to inspect and analyze.")
    st.stop()

# Display Case overview
with st.expander(f"📋 Case Context: {selected_case.get('case_id', 'Unknown')}", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Patient Age:** {selected_case.get('patient', {}).get('age', 'N/A')}")
        st.markdown(f"**Diagnosis:** `{selected_case.get('treatment', {}).get('diagnosis', 'N/A')}`")
    with col2:
        st.markdown(f"**Procedure:** {selected_case.get('treatment', {}).get('procedure', 'N/A')}")
        st.markdown(f"**Type:** {selected_case.get('treatment', {}).get('type', 'N/A')}")
    with col3:
        st.markdown(f"**Sum Insured:** ₹{selected_case.get('sum_insured_inr', 0):,}")
        st.markdown(f"**Continuous Months:** {selected_case.get('continuous_coverage_months', 0)}")
    with col4:
        exp = selected_case.get("expenses_inr", {})
        total_exp = sum(exp.values()) if isinstance(exp, dict) else 0
        st.markdown(f"**Total Claimed:** ₹{total_exp:,}")
        st.markdown(f"**Hospital:** {selected_case.get('hospital', {}).get('name', 'N/A')}")

# Execution handler
if run_btn:
    with st.spinner("Multi-agent workflow in progress... (Case Analysis ➔ Evidence Retrieval ➔ Coverage & Exclusion ➔ Decision ➔ Validation)"):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{API_BASE_URL}/analyze",
                json=selected_case,
                timeout=240,
            )
            elapsed = time.time() - t0

            if resp.status_code != 200:
                st.error(f"Error {resp.status_code}: {resp.text}")
                st.stop()

            decision_data = resp.json()
            st.session_state["latest_decision"] = decision_data
            st.session_state["elapsed"] = elapsed
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

# Render Results
if "latest_decision" in st.session_state:
    data = st.session_state["latest_decision"]
    elapsed = st.session_state.get("elapsed", 0.0)

    st.divider()

    # Decision Banner
    dec_val = data.get("decision", "NEEDS_REVIEW")
    confidence = data.get("confidence", 0.0)

    badge_class = {
        "ADMISSIBLE": "badge-admissible",
        "ADMISSIBLE_WITH_LIMITS": "badge-limits",
        "PARTIALLY_ADMISSIBLE": "badge-partial",
        "NOT_ADMISSIBLE": "badge-not-admissible",
        "NEEDS_REVIEW": "badge-review",
    }.get(dec_val, "badge-review")

    top_col1, top_col2, top_col3, top_col4 = st.columns([2, 1, 1, 1])

    with top_col1:
        st.markdown(
            f"### Decision: <span class='{badge_class}'>{dec_val}</span>",
            unsafe_allow_html=True,
        )
    with top_col2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    with top_col3:
        val_status = data.get("validation", {}).get("status", "PASS")
        val_icon = "✅" if val_status == "PASS" else "⚠️"
        st.metric("Validation Gate", f"{val_icon} {val_status}")
    with top_col4:
        st.metric("Analysis Time", f"{elapsed:.2f}s")

    # If abstained / needs review
    if dec_val == "NEEDS_REVIEW":
        st.warning(
            "⚠️ **System Abstained**: The engine has withheld an automated approval or rejection because critical required policy evidence or hospital qualification is missing or inconclusive."
        )

    tabs = st.tabs(["📝 Key Findings & Limits", "🔍 Grounded Citations", "⚙️ Execution Trace", "📊 Confidence Breakdown"])

    with tabs[0]:
        col_findings, col_limits = st.columns([1, 1])

        with col_findings:
            st.subheader("Key Material Findings")
            findings = data.get("key_findings", [])
            if findings:
                for f in findings:
                    st.markdown(f"- {f}")
            else:
                st.write("No findings reported.")

            missing = data.get("missing_evidence", [])
            if missing:
                st.subheader("Missing Evidence Identified")
                for m in missing:
                    st.markdown(f"🔸 *{m}*")

        with col_limits:
            st.subheader("Applicable Limits & Deductions")
            limits = data.get("applicable_limits", [])
            if limits:
                for lim in limits:
                    claimed_val = lim.get('claimed_amount')
                    payable_val = lim.get('payable_amount')
                    claimed_str = f"₹{int(claimed_val):,}" if claimed_val is not None else "N/A"
                    payable_str = f"₹{int(payable_val):,}" if payable_val is not None else "N/A"
                    st.markdown(
                        f"""
                    <div class="metric-card">
                        <b>{str(lim.get('limit_type', 'Limit')).upper()}</b>: {lim.get('description', '')}<br/>
                        <b>Claimed:</b> {claimed_str} | 
                        <b>Payable:</b> {payable_str} | 
                        <b>Reference:</b> <code>{lim.get('policy_reference', 'N/A')}</code>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No policy caps, sub-limits, or deductions deducted from payable amount.")

    with tabs[1]:
        st.subheader("Inspectable Policy Citations (IRDAI Source)")
        citations = data.get("citations", [])
        if citations:
            for i, cit in enumerate(citations, 1):
                st.markdown(
                    f"""
                <div class="citation-card">
                    <b>Citation #{i}:</b> <i>"{cit.get('claim', '')}"</i><br/>
                    <b>Policy Source:</b> {cit.get('source', 'policy.pdf')} | 
                    <b>Page:</b> {cit.get('page', 'N/A')} | 
                    <b>Section:</b> {cit.get('section', 'N/A')} | 
                    <b>Chunk ID:</b> <code>{cit.get('chunk_id', 'N/A')}</code>
                    <br/><br/>
                    <details>
                        <summary>View Quoted Policy Clause</summary>
                        <pre style="white-space: pre-wrap; font-size: 12px; margin-top: 8px;">{cit.get('chunk_text', 'No excerpt text provided.')}</pre>
                    </details>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No explicit policy citations attached to this decision.")

    with tabs[2]:
        st.subheader("Multi-Agent Execution Trace (Auditable & Concise)")
        trace = data.get("trace", [])
        if trace:
            for step in trace:
                agent = step.get("agent_name", "Agent")
                action = step.get("action", "")
                dur = step.get("duration_ms", 0)
                status_step = step.get("status", "completed")
                retrievals = step.get("retrieval_count", 0)

                with st.expander(f"🤖 {agent} — {action} ({dur}ms)", expanded=False):
                    st.write(f"**Action Description:** {action}")
                    st.write(f"**Status:** `{status_step}`")
                    if retrievals > 0:
                        st.write(f"**Retrieved Passages:** {retrievals}")
                    details = step.get("details", {})
                    if details:
                        st.json(details)
        else:
            st.write("No execution trace recorded.")

    with tabs[3]:
        st.subheader("Evidence-Based Confidence Decomposition")
        st.caption("Derived strictly from observable evidence signals — not a black-box probability.")

        cb = data.get("confidence_breakdown", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Evidence Support", f"{cb.get('evidence_support', 0.0) * 100:.0f}%")
        with c2:
            st.metric("Citation Coverage", f"{cb.get('citation_coverage', 0.0) * 100:.0f}%")
        with c3:
            st.metric("Retrieval Quality", f"{cb.get('retrieval_quality', 0.0) * 100:.0f}%")
        with c4:
            st.metric("Critical Evidence Intact", "YES" if cb.get("missing_critical", 1.0) == 1.0 else "NO (Abstain)")
