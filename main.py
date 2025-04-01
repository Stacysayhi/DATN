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
import google.generativeai as genai  # Import the Gemini library

# Your API Key - should be stored securely, not hardcoded
API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"  # Replace with your actual YouTube Data API key
GOOGLE_API_KEY = "AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"  # Replace with your Gemini API key

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_PATH = ""  # Set this to the directory if you have a folder ofr the weights, other wise it would be ""
MODEL_FILE = "sentiment_classifier (1).pth"


@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_PATH, MODEL_FILE)  # Full path to the .pth file
    # tokenizer_path = MODEL_PATH  # Tokenizer usually saved in the same directory as the model
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # **Important:** Replace with the correct model class if needed
        model = AutoModelForSequenceClassification.from_pretrained(model_id) #Or RobertaForSequenceClassification

        # Load the state dictionary from the saved .pth file
        # Try strict=False *only* if you understand the implications
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded successfully from {model_path} and moved to {device}")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        logging.error(f"Error loading model from {model_path}: {e}")
        return None, None  # Return None values to indicate loading failure


def analyze_sentiment(text):
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.error("Model loading failed. Sentiment analysis is unavailable.")
        return "Error", [0, 0, 0]  # Return a default sentiment and scores

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


def extract_video_id(url):
    pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})")
    match = pattern.search(url)
    if match:
        return match.group(1)
    return None

# Helper function to fetch video description from YouTube API
def fetch_video_description(video_id, api_key):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response["items"]:
            return None  # Video not found
        return response["items"][0]["snippet"]["description"]
    except Exception as e:
        logging.error(f"Error fetching video description: {e}")
        return None


def download_live_chat(video_url, video_id):
    ydl_opts = {
        'writesubtitles': True,
        'skip_download': True,
        'subtitleslangs': ['live_chat'],
        'outtmpl': f'{video_id}',
    }
    subtitle_file = f"{video_id}.live_chat.json"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)  # Download the live chat
        return subtitle_file  # Return the filename to be parsed
    except Exception as e:
        logging.error(f"Error downloading live chat: {e}")
        return None

def parse_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        logging.error(f"Live chat file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON in live chat file: {e}")
        return None
    except Exception as e:
        logging.error(f"Error opening/reading file: {e}")
        return None

def extract_live_chat_messages(subtitle_file):
    messages = []
    if not subtitle_file or not os.path.exists(subtitle_file):
        return messages  # Return empty list if no file or file doesn't exist

    data = parse_jsonl(subtitle_file)  # Use the parsing function
    if not data:
        return messages  # Return empty list if parsing failed

    for lc in data:
        try:
            lc_actions = lc.get('replayChatItemAction', {}).get('actions', [])
            for act in lc_actions:
                live_chat = act.get('addChatItemAction', {}).get('item', {}).get('liveChatTextMessageRenderer', None)
                if live_chat:
                    runs = live_chat['message']['runs']
                    for run in runs:
                        messages.append(run['text'])
        except Exception as e:
            logging.warning(f"Error processing a live chat message: {str(e)}")
            continue
    return messages


def get_video_details_with_chat(video_url: str, api_key: str) -> dict:
    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    # 1. Fetch Video Description
    description = fetch_video_description(video_id, api_key)
    if description is None:
        description = ""  # Handle cases where description retrieval fails

    # 2. Download and Parse Live Chat
    subtitle_file = download_live_chat(video_url, video_id)
    live_chat_messages = extract_live_chat_messages(subtitle_file)

    # 3. Clean up the temp file
    if subtitle_file and os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
            logging.info(f"Deleted temporary file: {subtitle_file}")
        except Exception as e:
            logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")

    # Return the data
    return {
        "video_id": video_id,  # Include the video_id here
        "description": description,
        "live_chat": live_chat_messages
    }


def get_desc_chat(video_url, API_KEY):
    st.write(f"Analyzing video: {video_url}")
    video_info = get_video_details_with_chat(video_url, API_KEY)

    if "error" in video_info:
        st.error(f"Error: {video_info['error']}")
        return "", [], [], {}

    # Extract the video_id from video_info
    video_id = video_info.get("video_id")

    # Use the video_id to construct the URL to fetch the title
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)  # Ensure API_KEY is correctly passed
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        video_title = response['items'][0]['snippet']['title']
    except Exception as e:
        logging.error(f"Error fetching video title: {e}")
        video_title = "Video Title Unavailable"  # Fallback title

    video_description = video_info['description']
    video_live_chat = video_info['live_chat']

    clean_description = preprocess_model_input_str(video_description, video_title)
    clean_live_chat = [preprocess_model_input_str(live_chat) for live_chat in video_live_chat]

    return clean_description, clean_live_chat, video_title, video_info['live_chat']


def get_top_comments(live_chat, sentiment_labels, top_n=3):
    """
    Selects the top N positive and negative comments based on sentiment scores.
    Sentiment labels are passed so we don't need to analyze them multiple times.
    """
    positive_comments = []
    negative_comments = []

    for i, comment in enumerate(live_chat):
        if sentiment_labels[i] == "Positive":
            positive_comments.append(comment)
        elif sentiment_labels[i] == "Negative":
            negative_comments.append(comment)

    return positive_comments[:top_n], negative_comments[:top_n]



def plot_sentiment_pie_chart(positive_count, negative_count, total_comments):
    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [positive_count, negative_count, total_comments - (positive_count + negative_count)]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig



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
        logging.error(f"Error getting subtitles for video ID {video_id}: {e}")
        return None

# Define the prompt for the Gemini model
prompt = """
Bạn là người tóm tắt video trên Youtube. Bạn sẽ lấy văn bản ghi chép
và tóm tắt toàn bộ video và cung cấp bản tóm tắt quan trọng theo các điểm
trong vòng 300 từ. Vui lòng cung cấp bản tóm tắt của văn bản được đưa ra ở đây:
"""

# Define the function to get the Gemini response
def get_gemini_response(transcript_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the model
        response = model.generate_content(transcript_text + prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        return None

# Function to create and display the sentiment analysis visualization
def display_sentiment_visualization(video_description, video_live_chat):
    sentiment_labels = ["Negative", "Neutral", "Positive"]

    # Analyze comments
    comments_results = []
    for comment in video_live_chat:
        sentiment_label, scores = analyze_sentiment(comment)
        comments_results.append(
            {
                "Text": comment,
                "Sentiment": sentiment_label,
                **{
                    label: scores[i] * 100
                    for i, label in enumerate(sentiment_labels)
                },
            }
        )

    # Analyze description
    sentiment_label, description_scores = analyze_sentiment(video_description)
    description_scores = description_scores * 100

    # Create visualization
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Description Analysis", "Comments Analysis")
    )

    # Description visualization
    fig.add_trace(
        go.Bar(
            name="Description Sentiment", x=sentiment_labels, y=description_scores
        ),
        row=1,
        col=1,
    )

    # Comments visualization
    for i, label in enumerate(sentiment_labels):
        scores = [result[label] for result in comments_results]
        fig.add_trace(
            go.Bar(name=label, x=list(range(1, len(scores) + 1)), y=scores),
            row=2,
            col=1,
        )

    fig.update_layout(height=700, barmode="group")
    st.plotly_chart(fig)

    # Display results
    st.subheader("Description Analysis")
    st.write(
        f"**Overall Sentiment:** {sentiment_labels[np.argmax(description_scores)]}"
    )
    st.write(
        f"**Scores:** {', '.join([f'{label}: {description_scores[i]:.2f}%' for i, label in enumerate(sentiment_labels)])}"
    )
    st.write(f"**Text:** {video_description}")

    st.subheader("Comments Analysis")
    comments_df = pd.DataFrame(comments_results)
    st.dataframe(comments_df)


# Setup Streamlit app
st.set_page_config(page_title="🎥 YouTube Video Sentiment and Summarization", layout="wide") # Use wide layout
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment and Summarization 🎯</h1>", unsafe_allow_html=True)

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'last_youtube_link' not in st.session_state:
     st.session_state.last_youtube_link = ""

# Unique key for text input
youtube_link = st.text_input("🔗 Enter YouTube Video Link Below:", key="youtube_link_input", value=st.session_state.get("last_youtube_link", ""))

# Placeholder for dynamic content updates
results_placeholder = st.container()

# --- Logic moved below input/button ---

# Add Submit URL button below the URL input field
if st.button("🔍 Analyze Video"):
    if youtube_link.strip() == "":
        st.session_state.responses = []
        st.session_state.last_youtube_link = ""
        st.warning("Input cleared. All previous results removed.")
        # Clear the placeholder explicitly if needed
        results_placeholder.empty()
    elif youtube_link != st.session_state.last_youtube_link:
        # New link entered, clear previous results and analyze
        st.session_state.responses = []
        st.session_state.last_youtube_link = youtube_link # Store the new link

        with st.spinner('Collecting video information...'):
            video_id = extract_video_id(youtube_link)
            if video_id:
                try:
                    # Replace with your actual API_KEY
                    API_KEY = "YOUR_API_KEY" # Define or get your API key
                    clean_description, clean_live_chat, video_title, live_chat_messages = get_desc_chat(youtube_link, API_KEY)

                    # Analyze sentiment for all live chat messages (batched)
                    sentiment_data = []
                    for chat in clean_live_chat:
                        sentiment, _ = analyze_sentiment(chat)
                        sentiment_data.append(sentiment)

                    positive_count = sum(1 for s in sentiment_data if s == "Positive")
                    negative_count = sum(1 for s in sentiment_data if s == "Negative")
                    total_comments = len(sentiment_data) if sentiment_data else 0 # Handle empty chat

                    # Get top comments directly, passing in the sentiment labels we already calculated
                    positive_comments, negative_comments = get_top_comments(live_chat_messages, sentiment_data)

                    # Analyze description sentiment
                    description_sentiment, description_scores = analyze_sentiment(clean_description)

                    response = {
                        'thumbnail_url': f"http://img.youtube.com/vi/{video_id}/0.jpg",
                        'video_details': {
                            'title': video_title,
                            # Add other details if you fetch them
                        },
                        'comments': {
                            'total_comments': total_comments,
                            'positive_comments': positive_count,
                            'negative_comments': negative_count,
                            'positive_comments_list': positive_comments,
                            'negative_comments_list': negative_comments
                        },
                        "description": clean_description or "No description provided.", # Handle empty description
                        "video_id": video_id,
                        "sentiment_data": sentiment_data,
                        "live_chat_messages": live_chat_messages,
                        "description_sentiment": description_sentiment,
                    }
                    st.session_state.responses = [response] # Replace responses with the new one

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.session_state.responses = [] # Clear results on error
            else:
                st.error("Invalid YouTube URL")
                st.session_state.responses = [] # Clear results on invalid URL
    # If the button is clicked but the URL is the same, do nothing, just redisplay.

# Display stored responses using tabs (inside the placeholder)
with results_placeholder:
    if not st.session_state.responses:
        st.info("Enter a YouTube URL and click 'Analyze Video' to see results.")
    else:
        # Displaying only the latest response (index 0, as we clear and add one)
        response = st.session_state.responses[0]
        idx = 0 # Since we only have one response now

        video_details = response.get('video_details')
        comments = response.get('comments')
        live_chat_messages = response.get('live_chat_messages')
        sentiment_data = response.get('sentiment_data')

        # No need for st.header(f"Analysis of Video #{idx+1}") if only showing one

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Video Info", "💬 Live Chat Analysis", "📝 Summary"])

        # --- Tab 1: Video Info (Landscape Layout) ---
        with tab1:
            col1, col2 = st.columns([1, 2]) # Ratio: 1 part for image, 2 parts for text

            with col1:
                st.image(response['thumbnail_url'], use_column_width='always')

            with col2:
                st.markdown(f"<h3 style='color: #FF4500;'>📹 Video Title:</h3>", unsafe_allow_html=True)
                st.markdown(f"{video_details['title']}") # No need for <p> tag, st.markdown handles it

                st.markdown("---") # Visual separator

                st.markdown(f"<h3 style='color: #FF4500;'>📝 Description:</h3>", unsafe_allow_html=True)
                # Use an expander for potentially long descriptions
                with st.expander("View Description"):
                    st.markdown(f"{response['description']}")

                st.markdown("---") # Visual separator

                st.markdown(f"<h3 style='color: #FF4500;'>📊 Description Sentiment:</h3>", unsafe_allow_html=True)
                st.markdown(f"{response['description_sentiment']}")

        # --- Tab 2: Live Chat Analysis ---
        with tab2:
            st.markdown("<h2 style='text-align: center; color: #FF4500;'>💬 Live Chat Sentiment Analysis</h2>", unsafe_allow_html=True)

            # Layout for Pie Chart and Stats
            col_chart, col_stats = st.columns([1, 1])

            with col_chart:
                if comments and comments['total_comments'] > 0:
                     st.markdown("<h3 style='text-align: center;'>Sentiment Distribution</h3>", unsafe_allow_html=True)
                     fig = plot_sentiment_pie_chart(comments['positive_comments'], comments['negative_comments'], comments['total_comments'])
                     st.pyplot(fig, use_container_width=True)
                elif comments and comments['total_comments'] == 0:
                    st.info("No comments found for sentiment analysis.")
                else:
                    st.warning("Comment data not available.")

            with col_stats:
                 st.markdown("<h3 style='text-align: center;'>Statistics</h3>", unsafe_allow_html=True)
                 if comments and comments['total_comments'] > 0:
                    st.metric(label="Total Comments Analyzed", value=f"{comments['total_comments']}")
                    st.metric(label="👍 Positive Comments", value=f"{comments['positive_comments']} ({(comments['positive_comments']/comments['total_comments'])*100:.1f}%)")
                    st.metric(label="👎 Negative Comments", value=f"{comments['negative_comments']} ({(comments['negative_comments']/comments['total_comments'])*100:.1f}%)")
                    neutral_comments = comments['total_comments'] - comments['positive_comments'] - comments['negative_comments']
                    st.metric(label="😐 Neutral Comments", value=f"{neutral_comments} ({(neutral_comments/comments['total_comments'])*100:.1f}%)")
                 else:
                     st.markdown("<p style='text-align: center;'>No comment statistics available.</p>", unsafe_allow_html=True)


            st.markdown("---") # Visual separator

            # Live Chat Table and Top Comments Section
            st.markdown("<h3 style='color: #FF4500;'>📜 Live Chat Messages & Sentiment</h3>", unsafe_allow_html=True)
            if live_chat_messages is not None and sentiment_data is not None and len(live_chat_messages) > 0:
                df = pd.DataFrame({'Live Chat Message': live_chat_messages, 'Sentiment': sentiment_data})
                st.dataframe(df, use_container_width=True) # Use container width
            else:
                st.info("No live chat messages found or available.")

            if comments and comments['total_comments'] > 0:
                # Use st.expander to optionally show top comments
                 with st.expander("View Top 3 Positive/Negative Comments"):
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown(f"<h4 style='color: #32CD32;'>👍 Top Positive Comments:</h4>", unsafe_allow_html=True)
                        if comments['positive_comments_list']:
                            for comment in comments['positive_comments_list']:
                                st.markdown(f"<div style='background-color: #E8F5E9; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black;'>{comment}</div>", unsafe_allow_html=True)
                        else:
                            st.write("No positive comments found.")

                    with col_neg:
                        st.markdown(f"<h4 style='color: #FF6347;'>👎 Top Negative Comments:</h4>", unsafe_allow_html=True)
                        if comments['negative_comments_list']:
                            for comment in comments['negative_comments_list']:
                                st.markdown(f"<div style='background-color: #FFEBEE; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black;'>{comment}</div>", unsafe_allow_html=True)
                        else:
                            st.write("No negative comments found.")
            # No need for the old checkbox logic


        # --- Tab 3: Summary ---
        with tab3:
            st.markdown("<h2 style='text-align: center; color: #1E90FF;'>📜 AI Generated Summary</h2>", unsafe_allow_html=True)

            # Button to generate summary
            if 'transcript_summary' not in response:
                 # Use a unique key based on video_id to avoid conflicts if multiple analyses were possible
                if st.button("✨ Generate Summary", key=f"summarize_{response['video_id']}"):
                    with st.spinner("Generating summary... This may take a moment."):
                        try:
                            video_id = response["video_id"]
                            print(f"Attempting to retrieve transcript for video ID: {video_id}") # Debugging line
                            transcript = get_sub(video_id) # Replace with your actual function call
                            if transcript:
                                summary = get_gemini_response(transcript) # Replace with your actual function call
                                if summary:
                                    # Update the specific response in session state
                                    st.session_state.responses[idx]['transcript_summary'] = summary
                                    # Force rerun to display the summary immediately
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to generate summary from the model.")
                            else:
                                st.warning("⚠️ Could not retrieve transcript. Summary cannot be generated.")
                        except Exception as e:
                            st.error(f"❌ An error occurred during summary generation: {e}")
                            print(f"Summary generation error: {e}") # Log error for debugging

            # Display generated summary if available
            if 'transcript_summary' in response:
                st.markdown(f"<div style='background-color: #E3F2FD; padding: 15px; border-radius: 8px; border: 1px solid #BBDEFB; color: black;'>{response['transcript_summary']}</div>", unsafe_allow_html=True)
            elif 'transcript_summary' not in response and st.session_state.get(f"summarize_{response['video_id']}_clicked", False):
                 # Optional: Indicate that summary generation was attempted but failed or no transcript
                 pass # Error/Warning messages handled above
            else:
                 st.info("Click the 'Generate Summary' button above to create a summary from the video transcript.")
