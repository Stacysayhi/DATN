import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import json
import yt_dlp
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
API_KEY = 'AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M'
# Load Model
@st.cache_resource
def load_model():
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model

def analyze_sentiment(text):
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
    return predictions.numpy()[0]

def preprocess_text(text, video_title=""):
    regex_pattern = r"(http|www).*(\/|\/\/)|[-()+*&^%$#!@\";<>\/\.\?]{3,}|\n|#.*|\w*:"
    clean_text = re.sub(r"\s{2,}", " ", re.sub(regex_pattern, " ", text)).replace(video_title, "").strip()
    return word_tokenize(clean_text, format="text")

def get_video_details(video_url, api_key):
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id_match:
        return {"error": "Invalid YouTube URL."}
    video_id = video_id_match.group(1)
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    if not response["items"]:
        return {"error": "Video not found."}
    snippet = response["items"][0]["snippet"]
    stats = response["items"][0]["statistics"]
    return {
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "views": stats.get("viewCount", "N/A"),
        "likes": stats.get("likeCount", "N/A"),
        "comments": stats.get("commentCount", "N/A"),
        "thumbnail": snippet["thumbnails"]["high"]["url"]
    }

def plot_sentiment_pie_chart(positive, negative, total):
    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [positive, negative, total - (positive + negative)]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig

st.set_page_config(page_title="🎥 YouTube Video Sentiment Analysis")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment Analysis 🎯</h1>", unsafe_allow_html=True)

youtube_link = st.text_input("🔗 Enter YouTube Video Link Below:")

if st.button("🔍 Analyze Video"):
    with st.spinner('Fetching video data...'):
        video_info = get_video_details(youtube_link, API_KEY)
        if "error" in video_info:
            st.error(video_info["error"])
        else:
            st.image(video_info["thumbnail"], use_column_width=True)
            st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📹 {video_info['title']}</h2>", unsafe_allow_html=True)
            st.write(f"**Channel:** {video_info['channel']}")
            st.write(f"👁️ Views: {video_info['views']}")
            st.write(f"👍 Likes: {video_info['likes']}")
            st.write(f"💬 Comments: {video_info['comments']}")
            
            # Sentiment Analysis Placeholder
            positive_count, negative_count, total_comments = 10, 5, 30  # Mock values
            fig = plot_sentiment_pie_chart(positive_count, negative_count, total_comments)
            st.pyplot(fig)
