import streamlit as st
import pandas as pd
import joblib
from webscrapping import scrape_karkidi_jobs

# Load models
vectorizer = joblib.load("/home/user/Desktop/Notifications/vectorizer.joblib")
kmeans = joblib.load("/home/user/Desktop/Notifications/kmeans_model.joblib")

# Title
st.set_page_config(page_title="Job Alert App", layout="wide")
st.title("🚀 AI-Powered Job Recommender")

# Sidebar for user inputs
st.sidebar.header("Your Preferences")
user_skills = st.sidebar.text_input("Enter your skills (comma-separated)", "data science, machine learning, ai")
trigger_scrape = st.sidebar.button("🔄 Scrape New Jobs")

# Load or scrape data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/home/user/Desktop/Notifications/clustered_jobs.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Title", "Company", "Location", "Date", "Skills", "Cluster"])
    return df

df_jobs = load_data()

if trigger_scrape:
    st.info("Scraping latest jobs from Karkidi...")
    scraped = scrape_karkidi_jobs("data science", pages=1)
    if scraped.empty:
        st.warning("No new jobs found.")
    else:
        scraped['Skills'] = scraped['Skills'].fillna('').str.lower().str.strip()
        X_new = vectorizer.transform(scraped['Skills'])
        scraped['Cluster'] = kmeans.predict(X_new)

        df_jobs = pd.concat([df_jobs, scraped], ignore_index=True)
        df_jobs.drop_duplicates(subset=["Title", "Company", "Location"], inplace=True)
        df_jobs.to_csv("new_jobs_clustered.csv", index=False)
        st.success("Jobs updated!")

# Skill filtering
if user_skills:
    prefs = [s.strip().lower() for s in user_skills.split(',')]
    matched = df_jobs[df_jobs['Skills'].apply(lambda x: any(p in x for p in prefs))]
    st.subheader(f"🎯 Matched Jobs for: `{', '.join(prefs)}`")
    st.write(f"Found {len(matched)} matching jobs.")

    for _, job in matched.iterrows():
        with st.expander(f"🔹 {job['Title']} at {job['Company']} ({job['Location']})"):
            st.write(f"📅 Posted on: {job['Date']}")
            st.write(f"🛠 Skills: `{job['Skills']}`")
            st.write(f"🔗 [View Posting](https://www.karkidi.com/)")  # Placeholder

else:
    st.info("Enter your skills to get personalized job recommendations.")

# Display all jobs in a table
st.divider()
st.subheader("📄 All Available Jobs (Clustered)")
st.dataframe(df_jobs[['Title', 'Company', 'Location', 'Date', 'Skills', 'Cluster']], use_container_width=True)
