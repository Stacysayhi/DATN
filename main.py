import re
import json
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yt_dlp
from googleapiclient.discovery import build
import logging
import matplotlib.pyplot as plt
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from dotenv import load_dotenv
from Bard import ChatBard

load_dotenv()

# Your API Key - should be stored securely, not hardcoded
API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"  # Replace with your actual API key
BARD_API_KEY = os.getenv("AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I")

# Cấu hình logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@st.cache_resource
def load_model():
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model


def analyze_sentiment(text):
    tokenizer, model = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer.padding_side = "left"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]  # Move to CPU and convert to NumPy array

    sentiment_labels = ["Negative", "Neutral", "Positive"]
    predicted_class = np.argmax(predictions)  # Get index of max value
    sentiment_label = sentiment_labels[predicted_class]

    return sentiment_label, predictions


def preprocess_model_input_str(text, video_title=""):
    if not text:
        return ""

    regex_pattern = r"(http|www).*(\/|\/\/)\s?|[-()+*&^%$#!@\";<>\/\.\?]{3,}|\n|#.*|\w*:"
    clean_str = re.sub(r"\s{2,}", " ", re.sub(regex_pattern, " ", text)).replace(video_title, "").strip()
    return clean_str


def get_video_details_with_chat(video_url: str, api_key: str) -> dict:
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id_match:
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    video_id = video_id_match.group(1)

    # Fetch video description using YouTube API
    youtube = build("youtube", "v3", developerKey=api_key)
    description = ""
    try:
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response["items"]:
            return {"error": "Video not found. Check the URL or video ID."}

        description = response["items"][0]["snippet"]["description"]
    except Exception as e:
        logging.error(f"Error fetching video details from YouTube API: {str(e)}")
        return {"error": f"An error occurred while fetching video details: {str(e)}"}

    # Download live chat subtitles using yt_dlp
    ydl_opts = {
        'writesubtitles': True,
        'skip_download': True,
        'subtitleslangs': ['live_chat'],
        'outtmpl': f'{video_id}',
    }
    live_chat_messages = []
    subtitle_file = f"{video_id}.live_chat.json"

    def parse_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            try:
                data = parse_jsonl(subtitle_file)
                for lc in data:
                    try:
                        lc_actions = lc.get('replayChatItemAction', {}).get('actions', [])
                        for act in lc_actions:
                            live_chat = act.get('addChatItemAction', {}).get('item', {}).get('liveChatTextMessageRenderer', None)
                            if live_chat:
                                runs = live_chat['message']['runs']
                                for run in runs:
                                    live_chat_messages.append(run['text'])
                    except Exception as e:
                        logging.warning(f"Error processing a live chat message: {str(e)}")
                        continue
            except FileNotFoundError:
                logging.error(f"Live chat file not found: {subtitle_file}")
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Live chat file not found: {subtitle_file}"
                }
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON in live chat file: {str(e)}")
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Error parsing live chat JSON: {str(e)}"
                }

    except Exception as e:
        logging.error(f"An error occurred while downloading live chat: {str(e)}")
        return {
            "video_title": "",
            "description": description,
            "live_chat": [],
            "error": f"An error occurred while downloading live chat: {str(e)}"
        }
    finally:
        try:
            os.remove(subtitle_file)
            logging.info(f"Deleted temporary file: {subtitle_file}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")

    try:
        video_title = info_dict['title']
    except Exception as e:
        video_title = ""
        logging.error(f"Error getting video title: {str(e)}")
        return {
            "video_title": video_title,
            "description": description,
            "live_chat": [],
            "error": f"An error occurred while getting video title: {str(e)}"
        }

    return {
        "video_title": video_title,
        "description": description,
        "live_chat": live_chat_messages
    }

def get_desc_chat(video_url, API_KEY):
    st.write(f"Analyzing video: {video_url}")
    video_info = get_video_details_with_chat(video_url, API_KEY)

    if "error" in video_info:
        st.error(f"Error: {video_info['error']}")
        return "", [], [], {}

    video_description = video_info['description']
    video_title = video_info['video_title']
    video_live_chat = video_info['live_chat']

    clean_description = preprocess_model_input_str(video_description, video_title)
    clean_live_chat = [preprocess_model_input_str(live_chat) for live_chat in video_live_chat]

    return clean_description, clean_live_chat, video_info['video_title'], video_info['live_chat']

def get_top_comments(live_chat, top_n=3):
    comment_sentiments = []

    for comment in live_chat:
        sentiment_label, sentiment_scores = analyze_sentiment(comment)
        if sentiment_label == "Positive":
            comment_sentiments.append((comment, sentiment_scores[2]))  # Positive score
        elif sentiment_label == "Negative":
            comment_sentiments.append((comment, sentiment_scores[0]))  # Negative score

    # Sort by sentiment score and get top comments
    positive_comments = sorted([cs for cs in comment_sentiments if analyze_sentiment(cs[0])[0] == "Positive"], key=lambda x: x[1], reverse=True)[:top_n]
    negative_comments = sorted([cs for cs in comment_sentiments if analyze_sentiment(cs[0])[0] == "Negative"], key=lambda x: x[1], reverse=True)[:top_n]

    return [comment for comment, score in positive_comments], [comment for comment, score in negative_comments]

def plot_sentiment_pie_chart(positive_count, negative_count, total_comments):
    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [positive_count, negative_count, total_comments - (positive_count + negative_count)]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig

def extract_video_id(url):
    pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})")
    match = pattern.search(url)
    if match:
        return match.group(1)
    return None

def get_sub(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["vi"])
        data = []
        for segment in transcript:
            text = segment['text']
            start = segment['start']
            duration = segment['duration']
            data.append([video_id, start, start + duration, text])

        df = pd.DataFrame(data, columns=['video_id', 'start_time', 'end_time', 'text'])
        concatenated_text = ' '.join(df['text'].astype(str))
        return concatenated_text
    except Exception as e:
        logging.error(f"Error getting subtitles: {e}")
        return None

def summarize_transcript(transcript, BARD_API_KEY):
    if not transcript:
        return "No transcript available or error retrieving transcript."
    
    # Initialize ChatBard
    bard = ChatBard(token=BARD_API_KEY)
    
    # Prompt for summarization
    prompt = f"Summarize the following Vietnamese transcript in a concise and informative way: {transcript}"
    
    # Get the summary
    summary = bard.ask(prompt)
    
    return summary

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
    if youtube_link.strip() == "":
        st.session_state.responses = []
        st.write("The video link has been removed. All previous responses have been cleared.")
    else:
        with st.spinner('Collecting video information...'):
            video_id = extract_video_id(youtube_link)
            if video_id:
                try:
                    clean_description, clean_live_chat, video_title, live_chat_messages = get_desc_chat(youtube_link, API_KEY)

                    positive_count = sum(1 for comment in clean_live_chat if analyze_sentiment(comment)[0] == "Positive")
                    negative_count = sum(1 for comment in clean_live_chat if analyze_sentiment(comment)[0] == "Negative")
                    total_comments = len(clean_live_chat)

                    # Get top 3 positive and negative comments
                    positive_comments, negative_comments = get_top_comments(live_chat_messages)

                    response = {
                        'thumbnail_url': f"http://img.youtube.com/vi/{video_id}/0.jpg",
                        'video_details': {
                            'title': video_title,
                            'channel_title': None,
                            'view_count': None,
                            'upload_date': None,
                            'duration': None,
                            'like_count': None,
                            'dislike_count': None
                        },
                        'comments': {
                            'total_comments': total_comments,
                            'positive_comments': positive_count,
                            'negative_comments': negative_count,
                            'positive_comments_list': positive_comments,
                            'negative_comments_list': negative_comments
                        },
                        "description": clean_description,
                        "video_id": video_id # Store video ID
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

        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📝 Description:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{response['description']}</p>", unsafe_allow_html=True)

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

    # Button to generate summary
    if 'transcript_summary' not in response:
        if st.button("📜 Generate Summary", key=f"summarize_{idx}"):
            with st.spinner("Generating summary..."):
                video_id = response["video_id"]  # Get video ID from the response
                if video_id:
                    transcript = get_sub(video_id)
                    if transcript:
                        summary = summarize_transcript(transcript, BARD_API_KEY)
                        if summary:
                            response['transcript_summary'] = summary
                            st.session_state.responses[idx] = response
                        else:
                            st.error("Failed to generate summary.")
                    else:
                        st.error("Failed to retrieve transcript.")
                else:
                    st.error("Video ID not found.")

    # Display generated summary
    if 'transcript_summary' in response:
        st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>📜 Summary:</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: #F0F8FF; padding: 10px; border-radius: 5px;'>{response['transcript_summary']}</div>", unsafe_allow_html=True)
