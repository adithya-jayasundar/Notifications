import warnings
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import joblib

# Suppress the specific sklearn UserWarning about token_pattern being ignored
warnings.filterwarnings(
    "ignore",
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None"
)

def skill_tokenizer(text):
    # Custom tokenizer to split skills by comma and strip spaces
    return [skill.strip() for skill in text.split(',')]

def preprocess_and_cluster_jobs(csv_path="scraped_jobs.csv", n_clusters=5):
    # Step 1: Load the scraped data from CSV
    df_jobs = pd.read_csv(csv_path)
    print(f"Loaded {len(df_jobs)} jobs")

    # Step 2: Clean the Skills column
    df_jobs['Skills'] = df_jobs['Skills'].fillna('').str.lower().str.strip()
    print("Cleaned Skills column (filled missing, lowercased, stripped)")

    # Step 3: Convert Skills text to numeric vectors using CountVectorizer with custom tokenizer
    vectorizer = CountVectorizer(tokenizer=skill_tokenizer)

    # Fit and transform
    X = vectorizer.fit_transform(df_jobs['Skills'])
    print(f"Vectorized skills into matrix of shape: {X.shape}")

    # Step 4: Train KMeans clustering model on the skill vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # explicitly set n_init
    kmeans.fit(X)
    print(f"Trained KMeans with {n_clusters} clusters")

    # Step 5: Assign cluster labels
    df_jobs['Cluster'] = kmeans.labels_
    print("Assigned cluster labels to jobs")

    # Step 6: Save the vectorizer and clustering model to disk
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(kmeans, 'kmeans_model.joblib')
    print("Saved vectorizer and clustering model to disk")

    # Optional: save clustered jobs CSV
    df_jobs.to_csv("clustered_jobs.csv", index=False)
    print("Saved clustered jobs to 'clustered_jobs.csv'")

    return df_jobs

if __name__ == "__main__":
    clustered_jobs_df = preprocess_and_cluster_jobs(csv_path="scraped_jobs.csv", n_clusters=5)
    print(clustered_jobs_df.head())
