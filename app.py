import streamlit as st
import pandas as pd
import joblib
import sys
from webscrapping import scrape_karkidi_jobs  # Make sure this file and function exist

# --- 1. Tokenizer used during vectorizer training ---
def skill_tokenizer(text):
    return text.lower().split(", ")

# Ensure tokenizer is discoverable when loading vectorizer
sys.modules['__main__'].skill_tokenizer = skill_tokenizer

# --- 2. Load vectorizer and clustering model ---
try:
    vectorizer = joblib.load("vectorizer.joblib")
    kmeans = joblib.load("kmeans_model.joblib")
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")
    st.stop() #https://hj9cbiz3dk7pqrq7mkirwo.streamlit.app

# --- 3. Streamlit UI config ---
st.set_page_config(page_title="Job Alert App", layout="wide")
st.title("🚀 AI-Powered Job Recommender")

# --- 4. Sidebar for user input ---
st.sidebar.header("Your Preferences")
user_skills = st.sidebar.text_input("Enter your skills (comma-separated)", "data science, machine learning, ai")
trigger_scrape = st.sidebar.button("🔄 Scrape New Jobs")

# --- 5. Load job data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/home/user/Desktop/Notifications/clustered_jobs.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Title", "Company", "Location", "Date", "Skills", "Cluster"])
    return df

df_jobs = load_data()

# Ensure 'Date' column exists
if 'Date' not in df_jobs.columns:
    df_jobs['Date'] = ''

# --- 6. Scrape new jobs ---
if trigger_scrape:
    st.info("Scraping latest jobs from Karkidi...")
    try:
        scraped = scrape_karkidi_jobs("data science", pages=1)
    except Exception as e:
        st.error(f"❌ Scraping failed: {e}")
        scraped = pd.DataFrame()

    if scraped.empty:
        st.warning("No new jobs found.")
    else:
        scraped['Skills'] = scraped['Skills'].fillna('').str.lower().str.strip()

        try:
            X_new = vectorizer.transform(scraped['Skills'])
            scraped['Cluster'] = kmeans.predict(X_new)
        except Exception as e:
            st.error(f"❌ Vectorization or clustering failed: {e}")
            st.stop()

        df_jobs = pd.concat([df_jobs, scraped], ignore_index=True)
        df_jobs.drop_duplicates(subset=["Title", "Company", "Location"], inplace=True)

        # Save to CSV
        df_jobs.to_csv("/home/user/Desktop/Notifications/clustered_jobs.csv", index=False)
        st.success("✅ Jobs updated!")

# --- 7. Match user skills to jobs ---
if user_skills:
    prefs = [s.strip().lower() for s in user_skills.split(',')]
    matched = df_jobs[df_jobs['Skills'].apply(lambda x: any(p in x for p in prefs))]

    st.subheader(f"🎯 Matched Jobs for: `{', '.join(prefs)}`")
    st.write(f"Found **{len(matched)}** matching jobs.")

    for _, job in matched.iterrows():
        with st.expander(f"🔹 {job['Title']} at {job['Company']} ({job['Location']})"):
            posted_date = job['Date'] if pd.notna(job['Date']) and job['Date'] != '' else 'N/A'
            st.write(f"📅 Posted on: {posted_date}")
            st.write(f"🛠 Skills: `{job['Skills']}`")
            st.write(f"🔗 [View Posting](https://www.karkidi.com/)")  # Optional: Update with actual job URL
else:
    st.info("Enter your skills to get personalized job recommendations.")

# --- 8. Show all jobs ---
st.divider()
st.subheader("📄 All Available Jobs (Clustered)")
st.dataframe(df_jobs[['Title', 'Company', 'Location', 'Date', 'Skills', 'Cluster']], use_container_width=True)
