import streamlit as st
import pandas as pd
from pre_processing import preprocess_and_cluster_jobs  # your clustering function

# Streamlit setup
st.set_page_config(page_title="Job Match Alert", layout="wide")
st.title("💼 Job Match Alert App")
st.write("Matches jobs to your profile based on skill clustering.")

# Sidebar input
st.sidebar.header("Your Skills")
user_skills_input = st.sidebar.text_input("Enter your skills (comma-separated)", value="python, machine learning, data analysis")

# Clean skills input
user_skills = [skill.strip().lower() for skill in user_skills_input.split(",") if skill.strip()]

# Load and cluster job data
try:
    with st.spinner("🔧 Loading and Clustering Jobs..."):
        clustered_df = preprocess_and_cluster_jobs("scraped_jobs.csv", n_clusters=5)
except FileNotFoundError:
    st.error("❌ 'scraped_jobs.csv' not found. Please provide a valid job listings CSV.")
    st.stop()

# Display clustered data
st.subheader("📊 Clustered Job Listings")
st.dataframe(clustered_df[['Title', 'Company', 'Location', 'Skills', 'Cluster']])

# Match jobs by user skills
st.subheader("🎯 Matching Jobs Based on Your Skills")
clustered_df['Skills'] = clustered_df['Skills'].fillna('').str.lower()
matching_jobs = clustered_df[clustered_df['Skills'].apply(
    lambda s: any(skill in s for skill in user_skills)
)]

if not matching_jobs.empty:
    st.success(f"🎉 Found {len(matching_jobs)} matching jobs!")
    st.dataframe(matching_jobs[['Title', 'Company', 'Location', 'Skills']])
else:
    st.warning("No matching jobs found for your skills.")
