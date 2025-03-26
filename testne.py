# Importing Libraries that will be used in this project
import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import os
# from dotenv import load_dotenv
import re
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import isodate
from datetime import datetime
import time
from googleapiclient.errors import HttpError
from dateutil import parser
import matplotlib.pyplot as plt
import io

nltk.download('vader_lexicon')

# Load environment variables
# load_dotenv()

# Configure Google API key
genai.configure(api_key=os.getenv("AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"))
youtube_api_key = os.getenv("AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M")

# Setup YouTube Data API client with explicit endpoint
youtube = build('youtube', 'v3', developerKey=youtube_api_key, discoveryServiceUrl='https://www.googleapis.com/discovery/v1/apis/youtube/v3/rest')

import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re
import time
from dateutil import parser
import isodate

# Initialize YouTube Data API
API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def extract_video_id(youtube_url):
    patterns = [
        r'v=([^&]+)',
        r'youtu\.be/([^?]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def extract_transcript(video_id, languages=['en']):
    for language in languages:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            return " ".join([entry['text'] for entry in transcript_list])
        except NoTranscriptFound:
            continue
    return None

def get_video_details(video_id):
    request = youtube.videos().list(part="snippet,statistics,contentDetails", id=video_id)
    response = request.execute()
    video_data = response['items'][0]
    return {
        'title': video_data['snippet']['title'],
        'channel': video_data['snippet']['channelTitle'],
        'views': video_data['statistics']['viewCount'],
        'likes': video_data['statistics'].get('likeCount', 'N/A'),
        'upload_date': parser.parse(video_data['snippet']['publishedAt']).strftime("%Y-%m-%d"),
        'duration': isodate.parse_duration(video_data['contentDetails']['duration']).total_seconds()
    }

def get_video_comments(video_id, max_comments=100):
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_comments)
    response = request.execute()
    for item in response.get('items', []):
        comments.append(item['snippet']['topLevelComment']['snippet']['textOriginal'])
    return comments

def analyze_sentiment(comments):
    sid = SentimentIntensityAnalyzer()
    positive = [c for c in comments if sid.polarity_scores(c)['compound'] > 0.05]
    negative = [c for c in comments if sid.polarity_scores(c)['compound'] < -0.05]
    return len(positive), len(negative), len(comments)

def generate_summary(transcript):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = """
    Summarize the given YouTube transcript in bullet points within 300 words:
    """
    response = model.generate_content(transcript + prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="🎥 YouTube Analysis")
st.markdown("""<h1 style='text-align: center; color: #FF5733;'>YouTube Video Analysis 🎯</h1>""", unsafe_allow_html=True)

youtube_link = st.text_input("🔗 Enter YouTube Video Link Below:")
if st.button("🔍 Analyze Video"):
    if not youtube_link.strip():
        st.warning("Please enter a valid YouTube link.")
    else:
        with st.spinner('Fetching details...'):
            video_id = extract_video_id(youtube_link)
            if video_id:
                details = get_video_details(video_id)
                comments = get_video_comments(video_id)
                pos, neg, total = analyze_sentiment(comments)
                transcript = extract_transcript(video_id)
                summary = generate_summary(transcript) if transcript else "No transcript available."
                
                st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                st.subheader(f"📹 {details['title']}")
                st.write(f"📺 Channel: {details['channel']}")
                st.write(f"👁️ Views: {details['views']}")
                st.write(f"👍 Likes: {details['likes']}")
                st.write(f"📅 Upload Date: {details['upload_date']}")
                
                fig, ax = plt.subplots()
                ax.pie([pos, neg, total - (pos + neg)], labels=['😊 Positive', '😠 Negative', '😐 Neutral'], autopct='%1.1f%%', startangle=140)
                st.pyplot(fig)
                
                st.subheader("📜 Summary:")
                st.write(summary)
            else:
                st.error("Invalid YouTube URL")
