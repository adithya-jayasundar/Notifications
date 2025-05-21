import pandas as pd
import joblib

def skill_tokenizer(text):
    return [skill.strip() for skill in text.split(',')]

vectorizer = joblib.load('vectorizer.joblib')
kmeans = joblib.load('kmeans_model.joblib')

new_jobs = pd.read_csv('new_jobs.csv')
new_jobs['Skills'] = new_jobs['Skills'].fillna('').str.lower().str.strip()

# The vectorizer uses the skill_tokenizer defined above internally
X_new = vectorizer.transform(new_jobs['Skills'])
new_jobs['Cluster'] = kmeans.predict(X_new)

print(new_jobs.head())
