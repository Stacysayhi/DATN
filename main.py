# from glob import glob
from googleapiclient.discovery import build
import json
import yt_dlp
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import underthesea
import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import os
from dotenv import load_dotenv
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

#Load environment variables
load_dotenv()

# Configure Google API key
genai.configure(api_key=os.getenv("AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"))
youtube_api_key = os.getenv("AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M")

# Setup YouTube Data API client with explicit endpoint
youtube = build('youtube', 'v3', developerKey=youtube_api_key, discoveryServiceUrl='https://www.googleapis.com/discovery/v1/apis/youtube/v3/rest')

# Define the prompt for the gemini model
prompt = """
You are Youtube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 300 words. Please provide the summary of the text given here:  
"""

# Define the function to get the gemini response
def get_gemini_response(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(transcript_text + prompt)
    return response.text

# Define the function to extract the transcript details
def extract_transcript_details(video_id, languages=['vi']):
    for language in languages:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            transcript = " ".join([entry['text'] for entry in transcript_list])
            return transcript
        except NoTranscriptFound:
            continue
    raise Exception("No transcripts found in the provided languages.")

# Function to extract video ID from URL
def extract_video_id(youtube_url):
    patterns = [
        r'v=([^&]+)',  # Standard YouTube URL
        r'youtu\.be/([^?]+)',  # Shortened YouTube URL
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

# Function to get video details from YouTube Data API
def get_video_details(video_id, retries=3):
    for attempt in range(retries):
        try:
            request = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            )
            response = request.execute()
            video_data = response['items'][0]
            
            duration_iso = video_data['contentDetails']['duration']
            duration = isodate.parse_duration(duration_iso)
            formatted_duration = f"{int(duration.total_seconds() // 60)}:{int(duration.total_seconds() % 60):02}"
            
            upload_date_iso = video_data['snippet']['publishedAt']
            upload_date = parser.parse(upload_date_iso).strftime("%Y-%m-%d")
            
            title = video_data['snippet']['title']
            channel_title = video_data['snippet']['channelTitle']
            view_count = video_data['statistics']['viewCount']
            like_count = video_data['statistics'].get('likeCount', 'N/A')
            dislike_count = video_data['statistics'].get('dislikeCount', 'N/A')
            
            return {
                'title': title,
                'channel_title': channel_title,
                'view_count': view_count,
                'upload_date': upload_date,
                'duration': formatted_duration,
                'like_count': like_count,
                'dislike_count': dislike_count
            }
        except HttpError as e:
            if e.resp.status in [500, 503]:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    raise Exception("Failed to retrieve video details after multiple attempts")

def load_model():
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    # model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    return tokenizer, model


def analyze_sentiment(text):
    tokenizer, model = load_model()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
    return predictions.numpy()[0]


def preprocess_model_input_str(text, video_title=""):
    regex_pattern = (
        r"(http|www).*(\/|\/\/)\s?|[-()+*&^%$#!@\";<>\/\.\?]{3,}|\n|#.*|\w*:"
    )
    clean_str = (
        re.sub(r"\s{2,}", " ", re.sub(regex_pattern, " ", text))
        .replace(video_title, "")
        .strip()
    )
    clean_str = underthesea.word_tokenize(clean_str, format="text")
    return clean_str

def get_video_details_with_chat(video_url: str, api_key: str) -> dict:
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id_match:
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    video_id = video_id_match.group(1)

    # Fetch video description using YouTube API
    youtube = build("youtube", "v3", developerKey=api_key)
    try:
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response["items"]:
            return {"error": "Video not found. Check the URL or video ID."}

        description = response["items"][0]["snippet"]["description"]
    except Exception as e:
        return {"error": f"An error occurred while fetching video details: {str(e)}"}

    # Download live chat subtitles using yt_dlp
    ydl_opts = {
        'writesubtitles': True,    # Download subtitles
        'skip_download': True,    # Skip video download
        'subtitleslangs': ['live_chat'],  # Specify live chat subtitles
        # Set output filename to {video_id}.json
        'outtmpl': f'{video_id}',
    }
    live_chat_messages = []

    def parse_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            subtitle_file = f"{video_id}.live_chat.json"
            try:
                data = parse_jsonl(subtitle_file)
                for lc in data:
                    try:
                        lc_actions = lc.get(
                            'replayChatItemAction', {}).get('actions', [])
                        for act in lc_actions:
                            live_chat = act.get('addChatItemAction', {}).get(
                                'item', {}).get('liveChatTextMessageRenderer', None)
                            if live_chat:
                                runs = live_chat['message']['runs']
                                for run in runs:
                                    live_chat_messages.append(run['text'])
                    except:
                        continue
            except FileNotFoundError:
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Live chat file not found: {subtitle_file}"
                }
            except Exception as e:
                print(e)
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Error parsing live chat: {str(e)}"
                }
    except Exception as e:
        print(e)
        return {
            "video_title": info_dict['title'],
            "description": description,
            "live_chat": [],
            "error": f"An error occurred while downloading live chat: {str(e)}"
        }

    return {
        "video_title": info_dict['title'],
        "description": description,
        "live_chat": live_chat_messages
    }

def get_desc_chat(video_url):
    video_info = get_video_details_with_chat(video_url, API_KEY)
    video_description = video_info['description']
    video_title = video_info['video_title']
    video_live_chat = video_info['live_chat']
    clean_description = preprocess_model_input_str(
        video_description, video_title)
    clean_live_chat = [
        preprocess_model_input_str(live_chat) for live_chat in video_live_chat
    ]

    return clean_description, clean_live_chat

def plot_sentiment_pie_chart(positive_count, negative_count, total_comments):
    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [positive_count, negative_count, total_comments - (positive_count + negative_count)]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)  # explode 1st slice (Positive)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig
#########################################################################################################################
# Setup Streamlit app
st.set_page_config(page_title="🎥 YouTube Video Sentiment and Summarization")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment and Summarization 🎯</h1>", unsafe_allow_html=True)

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Unique key for text input
youtube_link = st.text_input("🔗 Enter YouTube Video Link Below:", key="youtube_link_input")

# Add Submit URL button below the URL input field
if st.button("🔍 Analyze Video"):
    if youtube_link.strip() == "":  # Check if the input is empty
        st.session_state.responses = []  # Clear responses
        st.write("The video link has been removed. All previous responses have been cleared.")
    else:
        with st.spinner('Collecting video information...'):
            video_id = extract_video_id(youtube_link)
            if video_id:
                thumbnail_url = f"http://img.youtube.com/vi/{video_id}/0.jpg"
                try:
                    video_details = get_video_details(video_id)
                    
                    # 🔹 Thay comments bằng live chat từ get_desc_chat()
                    _, comments = get_desc_chat(video_id)  

                    # 🔹 Phân tích cảm xúc trên live chat
                    positive_count, negative_count, total_comments = analyze_sentiment(comments)

                    # 🔹 Lấy top 3 positive và negative live chat
                    sid = SentimentIntensityAnalyzer()
                    positive_comments, negative_comments = get_top_comments(comments, sid)

                    response = {
                        'thumbnail_url': thumbnail_url,
                        'video_details': video_details,
                        'comments': {
                            'total_comments': total_comments,
                            'positive_comments': positive_count,
                            'negative_comments': negative_count,
                            'positive_comments_list': positive_comments,
                            'negative_comments_list': negative_comments
                        }
                    }
                    st.session_state.responses.append(response)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Invalid YouTube URL")

# Display stored responses
for idx, response in enumerate(st.session_state.responses):
    video_details = response.get('video_details')
    comments = response.get('comments')

    # Display video details
    if video_details:
        if 'thumbnail_url' in response:
            st.image(response['thumbnail_url'], use_column_width=True)

        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📹 Video Title:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['title']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📺 Channel Name:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['channel_title']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>👁️ Views:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['view_count']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📅 Upload Date:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['upload_date']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>⏱️ Duration:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['duration']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>👍 Likes:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['like_count']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>👎 Dislikes:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['dislike_count']}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>💬 Total Comments:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{comments['total_comments']}</p>", unsafe_allow_html=True)

        # Plot and display pie chart for comments sentiment
        fig = plot_sentiment_pie_chart(comments['positive_comments'], comments['negative_comments'], comments['total_comments'])
        st.pyplot(fig)
        
        st.markdown(f"<h2 style='text-align: center; color: #32CD32;'>👍 Positive Comments:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{comments['positive_comments']} ({(comments['positive_comments']/comments['total_comments'])*100:.2f}%)</p>", unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>👎 Negative Comments:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{comments['negative_comments']} ({(comments['negative_comments']/comments['total_comments'])*100:.2f}%)</p>", unsafe_allow_html=True)
        
        # Add a toggle button to show/hide the top comments
        show_comments = st.checkbox("Show Top Comments", key=f"toggle_comments_{idx}")
        if show_comments:
            st.markdown(f"<h2 style='text-align: center; color: #32CD32;'>👍 Top 3 Positive Comments:</h2>", unsafe_allow_html=True)
            for comment in comments['positive_comments_list']:
                st.markdown(f"<div style='background-color: #DFF0D8; padding: 10px; border-radius: 5px;'>{comment}</div>", unsafe_allow_html=True)

            st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>👎Top 3 Negative Comments:</h2>", unsafe_allow_html=True)
            for comment in comments['negative_comments_list']:
                st.markdown(f"<div style='background-color: #F2DEDE; padding: 10px; border-radius: 5px;'>{comment}</div>", unsafe_allow_html=True)
    
    # Check if detailed notes have not been generated yet
    if 'gemini_response' not in response:
        if st.button("📑 Generate Detailed Summary", key=f"btn_{idx}"):
            with st.spinner('Generating detailed notes...'):
                video_id = extract_video_id(youtube_link)
                if video_id:
                    try:
                        transcript = extract_transcript_details(video_id, languages=['vi', 'en', 'es', 'fr', 'de', 'zh-Hans'])
                        gemini_response = get_gemini_response(transcript, prompt)
                        response['gemini_response'] = gemini_response
                        st.session_state.responses[idx] = response
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Invalid YouTube URL")

    # Display generated summary if available
    if 'gemini_response' in response:
        st.markdown("<h2 style='color: #1E90FF;'>📜 Summary:</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: #F0F8FF; padding: 10px; border-radius: 5px;'>{response['gemini_response']}</div>", unsafe_allow_html=True)
