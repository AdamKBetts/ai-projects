import streamlit as st
import os
from newsapi import NewsApiClient
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    st.error("API key not found. Please create a .env file with your NEWS_API_KEY.")
    st.stop()

# Initialize the News API client
try:
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
except Exception as e:
    st.error(f"Error initializing News API: {e}")
    st.stop()

# Initialize the summarization model
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = get_summarizer()

st.title("AI News Summarizer Dashboard")

# Create a sidebar for user input
st.sidebar.header("Filter News")

# Add popular topics
popular_topics = [
    'Technology', 'Business', 'Sports', 'Politics', 'Health', 'Science', 'Entertainment'
]

# Use a selectbox for popular topics and a text input for custom topics
topic_choice = st.sidebar.selectbox("Select a popular topic", [''] + popular_topics)
custom_topic = st.sidebar.text_input("Or enter your own topic")

# Choose the topic to search for
if custom_topic:
    topic = custom_topic
else:
    topic = topic_choice

country = st.sidebar.selectbox("Select a country", ['us', 'gb', 'de', 'fr', 'jp', 'cn'])

# Fetch and display the news articles
if st.button("Get Latest News"):
    try:
        with st.spinner('Fetching and summarizing news...'):
            top_headlines = newsapi.get_top_headlines(
                q=topic,
                language='en',
                country=country
            )

            if top_headlines['articles']:
                st.subheader(f"Top Headlines for '{topic.capitalize()}'")

                for article in top_headlines['articles'][:5]:
                    with st.expander(f"**{article['title']}**"):
                        st.write(f"Source: {article['source']['name']}")
                        st.write(f"URL: {article['url']}")
                        st.write(f"Description: {article['description']}")

                        # Generate and display summary
                        if article['content']:
                            summary = summarizer(article['content'], max_length=150, min_length=40, do_sample=False)
                            st.markdown(f"**AI Summary:** {summary[0]['summary_text']}")
                        else:
                            st.markdown("**AI Summary:** No content available to summarise.")
            else:
                st.info("No articles found for that topic. Please try another.")
    except Exception as e:
        st.error(f"An error occurred: {e}")