# 📬 Job Notifications System

A Python-based system to **scrape job listings**, **cluster jobs based on required skills**, and **send alerts to users** based on their skill preferences. Designed for daily automation using `schedule`.

---

## 📌 Features

- 🕸️ **Web Scraping**: Scrapes job listings from [Karkidi.com](https://www.karkidi.com) using `requests` and `BeautifulSoup`.
- 🔍 **Clustering**: Groups jobs based on required skills using KMeans clustering.
- 🤖 **Skill Matching**: Alerts users if newly scraped jobs match their preferred skill keywords.
- 📅 **Daily Scheduler**: Automates scraping and notifications using the `schedule` module.
- 💾 **Model Persistence**: Saves and reuses vectorizer and clustering models with `joblib`.

---


