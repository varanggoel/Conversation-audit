import streamlit as st
import pandas as pd
import json

from parser import parse_chat
from analyzer import analyze

st.set_page_config(
    page_title="Edoofa Auditor",
    layout="wide"
)

st.title("🎓 Edoofa Conversation Audit Tool")

st.markdown(
"""
Upload one or more exported WhatsApp chats.
The system analyzes communication quality using the Edoofa Audit Framework.
"""
)

files = st.file_uploader(
    "Upload Chats",
    type=["txt"],
    accept_multiple_files=True
)

if files:

    if st.button("Run Audit"):

        reports = []

        for file in files:

            text = file.read().decode("utf-8")

            messages = parse_chat(text)

            report = analyze(messages)

            reports.append(report)

        for report in reports:

            data = json.loads(report)

            st.divider()

            st.header("Conversation Summary")

            c1,c2,c3,c4 = st.columns(4)

            c1.metric(
                "Overall Score",
                f"{data['overall_score']}/100"
            )

            c2.metric(
                "Risk",
                data["risk"]
            )

            c3.metric(
                "Findings",
                len(data["findings"])
            )

            critical = len([
                f for f in data["findings"]
                if f["severity"]=="Critical"
            ])

            c4.metric(
                "Critical",
                critical
            )

            st.subheader("Category Scores")

            scores = pd.DataFrame(
                data["category_scores"].items(),
                columns=["Category","Score"]
            )

            st.bar_chart(
                scores.set_index("Category")
            )

            st.subheader("Findings")

            for finding in data["findings"]:

                with st.expander(
                    f"{finding['severity']} • {finding['category']}"
                ):

                    st.write("**Summary**")
                    st.write(finding["summary"])

                    st.write("**Evidence**")
                    st.write(finding["evidence"])

                    st.write("**Why it Matters**")
                    st.write(finding["why"])

                    st.write("**Recommendation**")
                    st.success(finding["recommendation"])

            st.subheader("Counselor Strengths")

            for s in data["strengths"]:
                st.success(s)

            st.subheader("Coaching Priorities")

            for s in data["coaching_priorities"]:
                st.warning(s)