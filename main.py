from googleapiclient.discovery import build
import json
import yt_dlp
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import underthesea
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import os
from dotenv import load_dotenv
import isodate
from datetime import datetime
import time
from googleapiclient.errors import HttpError
from dateutil import parser
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# 🔥 Thay YOUR_YOUTUBE_API_KEY và YOUR_GENAI_API_KEY bằng API key của bạn
YOUTUBE_API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"
GENAI_API_KEY = "AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"

genai.configure(api_key=GENAI_API_KEY)

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)


prompt = """
You are a YouTube video summarizer. Summarize the transcript in 300 words or less with key points.
"""

def get_gemini_response(transcript_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(transcript_text + prompt)
    return response.text

def extract_transcript(video_id, languages=['vi']):
    for lang in languages:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            return " ".join([entry['text'] for entry in transcript_list])
        except NoTranscriptFound:
            continue
    raise Exception("No transcript found in the provided languages.")

def extract_video_id(url):
    patterns = [r'v=([^&]+)', r'youtu\.be/([^?]+)']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_details(video_id):
    try:
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()
        video_data = response['items'][0]
        duration = isodate.parse_duration(video_data['contentDetails']['duration'])
        return {
            'title': video_data['snippet']['title'],
            'channel': video_data['snippet']['channelTitle'],
            'views': video_data['statistics'].get('viewCount', 'N/A'),
            'upload_date': parser.parse(video_data['snippet']['publishedAt']).strftime("%Y-%m-%d"),
            'duration': f"{int(duration.total_seconds() // 60)}:{int(duration.total_seconds() % 60):02}",
            'likes': video_data['statistics'].get('likeCount', 'N/A')
        }
    except HttpError as e:
        raise Exception(f"YouTube API error: {str(e)}")

def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
    return tokenizer, model

def analyze_sentiment(text):
    tokenizer, model = load_sentiment_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
    return predictions.numpy()[0]

def preprocess_text(text):
    return re.sub(r'http\S+|www\S+|[^\w\s]', '', text).strip()

def get_desc_chat(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")
    try:
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        description = response['items'][0]['snippet']['description']
        return preprocess_text(description), []
    except Exception as e:
        raise Exception(f"Error fetching video details: {str(e)}")

def plot_sentiment_pie_chart(positive, negative, total):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive, negative, total - (positive + negative)]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig

st.set_page_config(page_title="YouTube Video Analysis")
st.markdown("## 🎥 YouTube Video Sentiment and Summary 🎯", unsafe_allow_html=True)

youtube_link = st.text_input("Enter YouTube Video Link:")
if st.button("Analyze Video"):
    if not youtube_link:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Fetching video details..."):
            video_id = extract_video_id(youtube_link)
            if video_id:
                try:
                    video_details = get_video_details(video_id)
                    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                    for key, value in video_details.items():
                        st.markdown(f"**{key.capitalize()}:** {value}")
                except Exception as e:
                    st.error(str(e))
            else:
                st.error("Invalid YouTube URL")

if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        try:
            transcript = extract_transcript(video_id, ['vi', 'en'])
            summary = get_gemini_response(transcript)
            st.markdown("### Summary:")
            st.write(summary)
        except Exception as e:
            st.error(str(e))
