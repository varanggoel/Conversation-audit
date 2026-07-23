import json

import pandas as pd
import streamlit as st

from analyzer import analyze, compare_chats
from parser import parse_chat

st.set_page_config(page_title="Edoofa Auditor", layout="wide")

st.title("🎓 Edoofa Conversation Audit Tool")

st.markdown(
    """
    Upload one or more exported WhatsApp chats and receive a structured audit report.
    The analysis uses an Edoofa-specific framework covering professionalism, empathy,
    responsiveness, trust, and other quality signals across the full conversation.
    """
)

st.info(
    "Audit Framework: Professionalism, Responsiveness, Information Accuracy, Broken Promises, Transparency, Empathy, Sales Pressure, Personalization, Trust Building, Concern Resolution, Follow-up Quality, Tone Changes, Ethical Counseling, Compliance Risks, and Conversation Flow."
)

files = st.file_uploader("Upload Chats", type=["txt"], accept_multiple_files=True)

if files:
    if st.button("Run Audit"):
        reports = []

        for file in files:
            text = file.read().decode("utf-8")
            messages = parse_chat(text)
            report = analyze(messages)
            reports.append(report)

        if len(reports) > 1:
            aggregate = compare_chats(reports)
            st.subheader("Cross-Conversation Patterns")
            st.json(aggregate)

        for index, report in enumerate(reports, start=1):
            st.divider()
            st.header(f"Conversation {index} Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Score", f"{report['overall_score']}/100")
            c2.metric("Risk", report["risk"])
            c3.metric("Findings", len(report["findings"]))
            critical = len([f for f in report["findings"] if f["severity"] == "Critical"])
            c4.metric("Critical", critical)

            st.subheader("Audit Framework")
            framework_df = pd.DataFrame(
                [{"Category": item["name"], "Measure": item["measure"], "Why it matters": item["why"]} for item in report["framework"]]
            )
            st.dataframe(framework_df, use_container_width=True, hide_index=True)

            st.subheader("Category Scores")
            scores = pd.DataFrame(report["category_scores"].items(), columns=["Category", "Score"])
            st.bar_chart(scores.set_index("Category"))

            st.subheader("Findings")
            for finding in report["findings"]:
                with st.expander(f"{finding['severity']} • {finding['category']}"):
                    st.write("**Summary**")
                    st.write(finding["summary"])

                    st.write("**Evidence**")
                    st.write(finding["evidence"])

                    st.write("**Why it Matters**")
                    st.write(finding["why"])

                    st.write("**Recommendation**")
                    st.success(finding["recommendation"])

            st.subheader("Counselor Strengths")
            for strength in report["strengths"]:
                st.success(strength)

            st.subheader("Coaching Priorities")
            for priority in report["coaching_priorities"]:
                st.warning(priority)
