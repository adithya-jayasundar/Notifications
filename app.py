import streamlit as st
import pandas as pd
import joblib
import sys
from webscrapping import scrape_karkidi_jobs  # Your scraper function module

# --- 1. Define the custom tokenizer function exactly as used in vectorizer training ---
def skill_tokenizer(text):
    # Example: tokenizes by splitting on commas and stripping spaces, all lowercase
    return text.lower().split(", ")

# --- 2. Inject this tokenizer into the main module namespace so joblib can find it on load ---
sys.modules['__main__'].skill_tokenizer = skill_tokenizer

# --- 3. Load your vectorizer and clustering model ---
# Make sure to use the absolute or relative paths where you saved these files
vectorizer = joblib.load("/home/user/Desktop/Notifications/vectorizer.joblib")
kmeans = joblib.load("/home/user/Desktop/Notifications/kmeans_model.joblib")

# --- 4. Streamlit page config and title ---
st.set_page_config(page_title="Job Alert App", layout="wide")
st.title("🚀 AI-Powered Job Recommender")

# --- 5. Sidebar input for user skills and scraping trigger button ---
st.sidebar.header("Your Preferences")
user_skills = st.sidebar.text_input("Enter your skills (comma-separated)", "data science, machine learning, ai")
trigger_scrape = st.sidebar.button("🔄 Scrape New Jobs")

# --- 6. Load existing job data from CSV ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/home/user/Desktop/Notifications/clustered_jobs.csv")
    except FileNotFoundError:
        # Empty DataFrame with expected columns if CSV does not exist
        df = pd.DataFrame(columns=["Title", "Company", "Location", "Date", "Skills", "Cluster"])
    return df

df_jobs = load_data()

# --- 7. Scrape new jobs and update dataset ---
if trigger_scrape:
    st.info("Scraping latest jobs from Karkidi...")
    scraped = scrape_karkidi_jobs("data science", pages=1)  # You can parameterize the keyword or pages
    
    if scraped.empty:
        st.warning("No new jobs found.")
    else:
        # Clean and prepare skills text
        scraped['Skills'] = scraped['Skills'].fillna('').str.lower().str.strip()

        # Vectorize the skills column using your loaded vectorizer
        X_new = vectorizer.transform(scraped['Skills'])

        # Predict clusters for the scraped jobs
        scraped['Cluster'] = kmeans.predict(X_new)

        # Append new jobs to existing DataFrame and drop duplicates
        df_jobs = pd.concat([df_jobs, scraped], ignore_index=True)
        df_jobs.drop_duplicates(subset=["Title", "Company", "Location"], inplace=True)

        # Save the updated CSV back to disk
        df_jobs.to_csv("/home/user/Desktop/Notifications/clustered_jobs.csv", index=False)
        st.success("Jobs updated!")

# --- 8. Filter and display jobs based on user skills input ---
if user_skills:
    prefs = [s.strip().lower() for s in user_skills.split(',')]
    
    # Filter jobs where any of the user skills appear in the job's skills string
    matched = df_jobs[df_jobs['Skills'].apply(lambda x: any(p in x for p in prefs))]
    
    st.subheader(f"🎯 Matched Jobs for: `{', '.join(prefs)}`")
    st.write(f"Found {len(matched)} matching jobs.")
    
    for _, job in matched.iterrows():
        with st.expander(f"🔹 {job['Title']} at {job['Company']} ({job['Location']})"):
            st.write(f"📅 Posted on: {job['Date']}")
            st.write(f"🛠 Skills: `{job['Skills']}`")
            # You can update the link to the actual job posting URL if you have it
            st.write(f"🔗 [View Posting](https://www.karkidi.com/)")  
else:
    st.info("Enter your skills to get personalized job recommendations.")

# --- 9. Display all jobs in a table at the bottom ---
st.divider()
st.subheader("📄 All Available Jobs (Clustered)")
st.dataframe(df_jobs[['Title', 'Company', 'Location', 'Date', 'Skills', 'Cluster']], use_container_width=True)
