import schedule
import time
import joblib
import pandas as pd

from webscrapping import scrape_karkidi_jobs  # your scraping function

# Define user preferences for demo; replace with your actual user data source
user_preferences = {
    "user_1": ["data science", "machine learning", "ai"],
    "user_2": ["marketing", "sales"],
    # Add more users and their preferred clusters or skill keywords here
}

def skill_tokenizer(text):
    return [skill.strip() for skill in text.split(',')]

def daily_scrape_and_predict():
    print("Starting daily scrape and predict...")

    # Scrape new jobs
    new_jobs = scrape_karkidi_jobs(keyword="data science", pages=1)
    if new_jobs.empty:
        print("No new jobs found.")
        return

    new_jobs['Skills'] = new_jobs['Skills'].fillna('').str.lower().str.strip()

    # Load models
    vectorizer = joblib.load('vectorizer.joblib')
    kmeans = joblib.load('kmeans_model.joblib')

    # Vectorize and cluster new jobs
    X_new = vectorizer.transform(new_jobs['Skills'])
    new_jobs['Cluster'] = kmeans.predict(X_new)

    # Load existing clustered jobs if exist, else create empty DataFrame
    try:
        existing_jobs = pd.read_csv("new_jobs_clustered.csv")
    except FileNotFoundError:
        existing_jobs = pd.DataFrame(columns=new_jobs.columns)

    # Combine old and new jobs, avoid duplicates based on job Title + Company + Location (adjust as needed)
    combined_jobs = pd.concat([existing_jobs, new_jobs], ignore_index=True)
    combined_jobs.drop_duplicates(subset=['Title', 'Company', 'Location'], inplace=True)

    # Save combined data
    combined_jobs.to_csv("new_jobs_clustered.csv", index=False)
    print(f"Saved total {len(combined_jobs)} unique jobs to 'new_jobs_clustered.csv'.")

    # Alert users based on preferred categories (using skills keywords)
    for user, prefs in user_preferences.items():
        # Find jobs that have any preferred keyword in the Skills or Titles
        mask = combined_jobs.apply(
            lambda row: any(pref in row['Skills'] or pref in row['Title'].lower() for pref in prefs),
            axis=1
        )
        user_jobs = combined_jobs[mask]

        if not user_jobs.empty:
            print(f"\nHi {user},")
            print(f"Found {len(user_jobs)} new jobs in your preferred categories:")
            for _, job in user_jobs.iterrows():
                print(f"- {job['Title']} at {job['Company']} in {job['Location']}")
            print("\nRegards,\nJob Alert System\n")
        else:
            print(f"No new jobs for {user} in preferred categories.")

# Schedule to run daily at 12:32 (change as needed)
schedule.every().day.at("18:49").do(daily_scrape_and_predict)

print("Scheduler running, waiting for next job...")
while True:
    schedule.run_pending()
    time.sleep(60)
