import streamlit as st
import pandas as pd
from webscrapping import scrape_karkidi_jobs  # your scrape function
from pre_processing import preprocess_and_cluster_jobs  # your clustering function
import joblib

# Streamlit setup
st.set_page_config(page_title="Job Match Alert", layout="wide")
st.title("💼 Job Match Alert App")
st.write("Scrapes Karkidi jobs, clusters them by skills, and matches jobs to your profile.")

# Sidebar input
st.sidebar.header("Search Settings")
keyword = st.sidebar.text_input("Job Keyword", value="data science")
pages = st.sidebar.slider("Number of Pages to Scrape", 1, 5, 2)
user_skills_input = st.sidebar.text_input("Your Skills (comma-separated)", value="python, machine learning, data analysis")

# Clean skills input
user_skills = [skill.strip().lower() for skill in user_skills_input.split(",") if skill.strip()]

# Button to run the workflow
if st.button("🔍 Scrape and Match Jobs"):
    with st.spinner("🔄 Scraping job listings..."):
        scraped_df = scrape_karkidi_jobs(keyword=keyword, pages=pages)

    if scraped_df.empty:
        st.warning("No jobs found. Try a different keyword or increase page count.")
    else:
        # Save scraped jobs to CSV for clustering
        scraped_df.to_csv("scraped_jobs.csv", index=False)
        st.success(f"✅ Scraped {len(scraped_df)} job postings.")

        with st.spinner("🔧 Preprocessing and Clustering..."):
            clustered_df = preprocess_and_cluster_jobs("scraped_jobs.csv", n_clusters=5)

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