import streamlit as st
import json
import pickle
import os
from dotenv import load_dotenv
from BM_25 import load_students, load_jsonl_file, preprocess_jobs, build_bm25_model, match_students_to_jobs
from chatbot_together import analyze_matches

# ────────────── Load Environment Variables ────────────── #
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ────────────── Streamlit App ────────────── #
st.set_page_config(page_title="Profeshare Job Matcher", layout="wide")
st.title("🔍 Profeshare Job Matcher")

uploaded_file = st.file_uploader("📁 Upload student profile JSON file", type=["json"])
interest_input = st.text_input("💡 Enter interests (separated by '+')", placeholder="e.g. frontend+developer+>10LPA+hybrid")

if uploaded_file and interest_input:
    try:
        # ───── Parse and Validate JSON ───── #
        student_data = json.load(uploaded_file)
        if not isinstance(student_data, list):
            student_data = [student_data]

        # ───── Update Interests ───── #
        interest_list = [i.strip() for i in interest_input.split("+") if i.strip()]
        for student in student_data:
            student.setdefault("job_preferences", {})["interests"] = interest_list

        # ───── Save Temp Student File ───── #
        with open("students.json", "w") as f:
            json.dump(student_data, f, indent=2)
        st.success("✅ Interests updated and student profile processed!")

        # ───── Load & Preprocess Job Data ───── #
        jobs = []
        for part_file in ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]:
            data = load_jsonl_file(part_file)
            jobs.extend(data)

        job_texts, job_index = preprocess_jobs(jobs)
        bm25 = build_bm25_model(job_texts)
        
        # ───── Match Students to Jobs ───── #
        matches = match_students_to_jobs(student_data, jobs, bm25, job_index, top_n=10)
        with open("student_job_matches.pkl", "wb") as f:
            pickle.dump(matches, f)
        st.success("🎯 Top job matches generated using BM25!")
        st.write(matches)

        # ───── Run LLM Reasoning ───── #
        final_response = analyze_matches("student_job_matches.pkl", student_data)
        st.markdown("## 🤖 LLM Career Analysis")
        st.write(final_response)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
