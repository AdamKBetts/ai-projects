# Real-time AI News Summarization Dashboard

This project demonstrates a full-stack AI application that processes and summarizes news articles in real-time. It showcases a data pipeline that fetches live data from an external API, performs **Natural Language Processing (NLP)** tasks, and displays the results in an interactive web dashboard.

---

### Features ✨

- **Real-time Data:** Fetches top news headlines using a live API.
- **AI Summarization:** Utilizes a pre-trained **Hugging Face** model to generate concise summaries of news articles.
- **Interactive Dashboard:** Built with **Streamlit** to provide a simple, elegant user interface for filtering and viewing news.
- **End-to-End Pipeline:** Combines data ingestion, AI processing, and web deployment into a single application.

---

### Core Components 🧩

- **Streamlit:** A Python library for creating interactive web applications for data science and machine learning.
- **`newsapi-python`:** The official Python client for the News API, used for data ingestion.
- **Hugging Face Transformers:** Provides the `distilbart-cnn-12-6` model for text summarization.

---

### Installation 🛠️

1.  **Get an API Key:** Sign up for a free API key at [NewsAPI.org](https://newsapi.org/).
2.  **Navigate to the project directory:**
    ```bash
    cd real-time-news-dashboard
    ```
3.  **Install the required libraries:**
    ```bash
    pip install streamlit newsapi-python transformers
    ```

### Usage ▶️

1.  **Add your API Key:** Open `app.py` and replace `'YOUR_NEWS_API_KEY'` with the key you obtained from NewsAPI.
2.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    This will launch a local server and open the dashboard in your web browser.

---
