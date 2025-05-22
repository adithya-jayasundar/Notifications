import streamlit as st
import pandas as pd
import joblib
import sys
from webscrapping import scrape_karkidi_jobs  # Your scraper function module

# --- 1. Define the tokenizer exactly as used during vectorizer training ---
def skill_tokenizer(text):
    # Tokenize by splitting on commas and stripping spaces, lowercase
    return text.lower().split(", ")

# --- 2. Inject tokenizer into main module so joblib can find it when loading ---
sys.modules['__main__'].skill_tokenizer = skill_tokenizer

# --- 3. Load vectorizer and clustering model ---
vectorizer = joblib.load("vectorizer.joblib")
kmeans = joblib.load("kmeans_model.joblib")

# --- 4. Streamlit page config ---
st.set_page_config(page_title="Job Alert App", layout="wide")
st.title("🚀 AI-Powered Job Recommender")

# --- 5. Sidebar: User inputs ---
st.sidebar.header("Your Preferences")
user_skills = st.sidebar.text_input("Enter your skills (comma-separated)", "data science, machine learning, ai")
trigger_scrape = st.sidebar.button("🔄 Scrape New Jobs")

# --- 6. Load existing jobs data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/home/user/Desktop/Notifications/clustered_jobs.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Title", "Company", "Location", "Date", "Skills", "Cluster"])
    return df

df_jobs = load_data()
if 'Date' not in df_jobs.columns:
    df_jobs['Date'] = ''

# --- 7. Scrape new jobs and update dataset ---
if trigger_scrape:
    st.info("Scraping latest jobs from Karkidi...")
    try:
        scraped = scrape_karkidi_jobs("data science", pages=1)  # Adjust parameters if needed
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        scraped = pd.DataFrame()

    if scraped.empty:
        st.warning("No new jobs found.")
    else:
        # Normalize skills text
        scraped['Skills'] = scraped['Skills'].fillna('').str.lower().str.strip()

        # Vectorize skills for clustering
        X_new = vectorizer.transform(scraped['Skills'])

        # Predict clusters
        scraped['Cluster'] = kmeans.predict(X_new)

        # Append and deduplicate
        df_jobs = pd.concat([df_jobs, scraped], ignore_index=True)
        df_jobs.drop_duplicates(subset=["Title", "Company", "Location"], inplace=True)

        # Save updated CSV
        df_jobs.to_csv("/home/user/Desktop/Notifications/clustered_jobs.csv", index=False)
        st.success("Jobs updated!")

# --- 8. Filter and display matched jobs ---
if user_skills:
    prefs = [s.strip().lower() for s in user_skills.split(',')]
    matched = df_jobs[df_jobs['Skills'].apply(lambda x: any(p in x for p in prefs))]

    st.subheader(f"🎯 Matched Jobs for: `{', '.join(prefs)}`")
    st.write(f"Found {len(matched)} matching jobs.")

    for _, job in matched.iterrows():
        with st.expander(f"🔹 {job['Title']} at {job['Company']} ({job['Location']})"):
            date_posted = job['Date'] if pd.notna(job['Date']) and job['Date'] != '' else 'N/A'
            st.write(f"📅 Posted on: {date_posted}")
            st.write(f"🛠 Skills: `{job['Skills']}`")
            st.write(f"🔗 [View Posting](https://www.karkidi.com/)")  # Replace with actual job URL if available
else:
    st.info("Enter your skills to get personalized job recommendations.")

# --- 9. Show all jobs in a table ---
st.divider()
st.subheader("📄 All Available Jobs (Clustered)")
st.dataframe(df_jobs[['Title', 'Company', 'Location', 'Date', 'Skills', 'Cluster']], use_container_width=True)
