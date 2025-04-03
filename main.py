import re
import json
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yt_dlp
from googleapiclient.discovery import build
import logging
import matplotlib.pyplot as plt # Keep for potential future use, but not for the main pie chart anymore
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go # <-- Import Plotly
from plotly.subplots import make_subplots
import time

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
    # Using a spinner here is good for the initial load if it's slow
    with st.spinner("Loading Sentiment Analysis Model... This may take a moment."):
        model_path = os.path.join(MODEL_PATH, MODEL_FILE)
        model_id = "wonrax/phobert-base-vietnamese-sentiment"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            print(f"Model loaded successfully from {model_path} and moved to {device}")
            return tokenizer, model
        except Exception as e:
            st.error(f"Fatal Error: Could not load sentiment analysis model. Please check model path and file integrity. Error: {e}", icon="🚨")
            logging.error(f"Error loading model from {model_path}: {e}")
            st.stop() # Stop execution if model fails to load
            return None, None


def analyze_sentiment(text):
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        # Error already shown in load_model, maybe a simpler message here
        st.warning("Sentiment analysis model not available.", icon="⚠️")
        return "Error", [0, 0, 0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer.padding_side = "left"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    sentiment_labels = ["Negative", "Neutral", "Positive"]
    predicted_class = np.argmax(predictions)
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


def fetch_video_description(video_id, api_key):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response["items"]:
            return None
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
        'quiet': True, # Make yt-dlp less verbose in console
        'no_warnings': True,
    }
    subtitle_file = f"{video_id}.live_chat.json"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)
        return subtitle_file
    except yt_dlp.utils.DownloadError as e:
        # Handle specific yt-dlp errors, like no live chat found
        if "live chat" in str(e).lower():
             st.warning("Could not find live chat replay for this video. Analysis will proceed without chat data.", icon="💬")
        else:
            st.error(f"Error downloading video data: {e}", icon="🚨")
        logging.error(f"Error downloading live chat: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during video data download: {e}", icon="🚨")
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
        return messages

    data = parse_jsonl(subtitle_file)
    if not data:
        return messages

    for lc in data:
        try:
            lc_actions = lc.get('replayChatItemAction', {}).get('actions', [])
            for act in lc_actions:
                live_chat = act.get('addChatItemAction', {}).get('item', {}).get('liveChatTextMessageRenderer', None)
                if live_chat:
                    runs = live_chat.get('message', {}).get('runs', []) # Safer access
                    # Combine runs into a single message string
                    full_message = ''.join(run.get('text', '') for run in runs)
                    if full_message: # Ensure we don't add empty messages
                        messages.append(full_message)
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
        description = "" # Default to empty string

    # 2. Download and Parse Live Chat
    subtitle_file = download_live_chat(video_url, video_id) # This now handles errors better
    live_chat_messages = [] # Initialize as empty list
    if subtitle_file: # Only try to parse if download succeeded
        live_chat_messages = extract_live_chat_messages(subtitle_file)

    # 3. Clean up the temp file
    if subtitle_file and os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
            logging.info(f"Deleted temporary file: {subtitle_file}")
        except Exception as e:
            logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")

    return {
        "video_id": video_id,
        "description": description,
        "live_chat": live_chat_messages
    }


def get_desc_chat(video_url, API_KEY):
    # Note: Spinner is now outside this function call
    # st.write(f"Analyzing video: {video_url}") # Less verbose, spinner indicates action
    video_info = get_video_details_with_chat(video_url, API_KEY)

    if "error" in video_info:
        st.error(f"Error: {video_info['error']}", icon="🚨")
        return None, [], "", [] # Return consistent types on error

    video_id = video_info.get("video_id")
    video_title = "Video Title Unavailable" # Default title
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        if response.get('items'):
            video_title = response['items'][0]['snippet']['title']
    except Exception as e:
        st.warning(f"Could not fetch video title: {e}", icon="⚠️")
        logging.error(f"Error fetching video title for {video_id}: {e}")

    video_description = video_info['description']
    video_live_chat_raw = video_info['live_chat'] # Keep raw messages for display

    clean_description = preprocess_model_input_str(video_description, video_title)
    clean_live_chat = [preprocess_model_input_str(chat) for chat in video_live_chat_raw if chat.strip()] # Preprocess non-empty chats

    return clean_description, clean_live_chat, video_title, video_live_chat_raw


def get_top_comments(live_chat_raw, sentiment_labels, top_n=3):
    """Selects top N comments based on calculated sentiment labels."""
    positive_comments = []
    negative_comments = []

    # Ensure raw chat and labels have the same length for safe iteration
    min_len = min(len(live_chat_raw), len(sentiment_labels))

    for i in range(min_len):
        comment = live_chat_raw[i]
        sentiment = sentiment_labels[i]
        if sentiment == "Positive":
            positive_comments.append(comment)
        elif sentiment == "Negative":
            negative_comments.append(comment)

    # Return only top N, even if fewer are found
    return positive_comments[:top_n], negative_comments[:top_n]

# --- NEW Plotly Pie Chart Function ---
def plot_sentiment_pie_chart_plotly(positive_count, negative_count, total_comments):
    """Generates an interactive Plotly pie chart."""
    if total_comments == 0:
        # Return an empty figure or a message if no comments
        fig = go.Figure()
        fig.update_layout(title_text='No comments to analyze', title_x=0.5)
        return fig

    neutral_count = total_comments - (positive_count + negative_count)
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_count, negative_count, neutral_count]
    # More distinct colors
    colors = ['#28a745', '#dc3545', '#6c757d'] # Green, Red, Gray

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,
                                marker_colors=colors,
                                pull=[0.05, 0.05, 0], # Slightly pull positive and negative
                                hole=0.3, # Donut chart style
                                textinfo='percent+value', # Show percentage and count
                                insidetextorientation='radial',
                                hoverinfo='label+percent+value')]) # Tooltip info
    fig.update_layout(
        #title_text='Live Chat Sentiment Distribution',
        #title_x=0.5,
        margin=dict(l=10, r=10, t=30, b=10), # Minimal margins
        legend_title_text='Sentiments',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5), # Horizontal legend below
        # Transparent background can blend better with Streamlit themes
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # Uncomment if using dark theme
        # font_color="white"
    )
    return fig

def get_sub(video_id):
    try:
        # Attempt to get Vietnamese first, fallback to English
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_generated_transcript(['vi']).fetch()
        except: # Fallback to English if Vietnamese not found
             try:
                 transcript = transcript_list.find_generated_transcript(['en']).fetch()
                 st.info("Vietnamese transcript not found, using English for summary.", icon="ℹ️")
             except:
                 st.error(f"No suitable transcript (Vietnamese or English) found for video ID {video_id}.", icon="🚨")
                 return None

        concatenated_text = ' '.join([segment['text'] for segment in transcript])
        return concatenated_text
    except Exception as e:
        st.error(f"Error retrieving transcript for video ID {video_id}: {e}", icon="🚨")
        logging.error(f"Error getting subtitles for video ID {video_id}: {e}")
        return None


# Define the prompt for the Gemini model
prompt = """
Bạn là người tóm tắt video trên Youtube. Bạn sẽ lấy văn bản ghi chép
và tóm tắt toàn bộ video và cung cấp bản tóm tắt quan trọng theo các điểm
trong vòng 300 từ. Vui lòng cung cấp bản tóm tắt của văn bản được đưa ra ở đây:
"""

# Define the function to get the Gemini response, with retry logic
def get_gemini_response_with_retry(transcript_text, max_attempts=3):
    if not transcript_text:
        return "Error: Cannot generate summary from empty transcript."

    # Add context to the prompt (optional but can improve results)
    full_prompt = f"{prompt}\n\nTranscript:\n{transcript_text}"

    for attempt in range(max_attempts):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            # Send only the combined prompt and transcript
            response = model.generate_content(full_prompt)
            # Check for safety ratings or blocks if applicable/needed
            # if response.prompt_feedback.block_reason:
            #     logging.error(f"Gemini response blocked: {response.prompt_feedback.block_reason}")
            #     return f"Error: Content generation blocked due to safety settings ({response.prompt_feedback.block_reason})."
            return response.text
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to generate Gemini response: {e}")
            if attempt < max_attempts - 1:
                st.warning(f"Summary generation attempt {attempt + 1} failed. Retrying...", icon="⏳")
                time.sleep(2 ** attempt) # Exponential backoff
            else:
                st.error(f"Failed to generate summary from Gemini after {max_attempts} attempts. Error: {e}", icon="🚨")
                return None # Indicate final failure
    return None # Should not be reached, but for safety

# --- Streamlit App UI ---
st.set_page_config(page_title="🎥 YouTube Video Analysis", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)
st.markdown("---") # Add a visual separator

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'last_youtube_link' not in st.session_state:
    st.session_state.last_youtube_link = ""

# --- Input Area ---
st.subheader("Enter YouTube Video Link")
st.markdown("e.g., https://www.youtube.com/watch?v=ISrGxpJgLXM&t=3606s")
youtube_link = st.text_input("🔗 Paste the YouTube video URL here:", key="youtube_link_input", label_visibility="collapsed")

# --- Analyze Button and Processing Logic ---
if st.button("🔍 Analyze Video", type="primary"): # Use primary button type
    if not youtube_link or not youtube_link.strip():
        st.warning("Please enter a YouTube video link.", icon="⚠️")
        # Optionally clear previous results if the button is clicked with no link
        # st.session_state.responses = []
        # st.session_state.last_youtube_link = ""
    elif youtube_link == st.session_state.last_youtube_link and st.session_state.responses:
         st.info("Analysis for this video is already displayed below.", icon="ℹ️")
    else:
        # --- NEW: Main spinner for the whole analysis process ---
        with st.spinner('🚀 Analyzing video... Fetching data, processing chat, and evaluating sentiment. Please wait.'):
            st.session_state.responses = [] # Clear previous results for a new analysis
            st.session_state.last_youtube_link = youtube_link # Store the link being analyzed
            video_id = extract_video_id(youtube_link)

            if video_id:
                try:
                    # 1. Get Description and Chat Data
                    desc_clean, chat_clean, title, chat_raw = get_desc_chat(youtube_link, API_KEY)

                    if desc_clean is None: # Handle error from get_desc_chat
                       raise ValueError("Failed to retrieve video details.")

                    # 2. Analyze Live Chat Sentiment (if chat exists)
                    sentiment_data = []
                    positive_count = 0
                    negative_count = 0
                    if chat_clean:
                        # Consider batching sentiment analysis if performance is an issue
                        with st.spinner('Analyzing live chat sentiment...'): # Nested spinner for specific step
                            for chat in chat_clean:
                                sentiment, _ = analyze_sentiment(chat)
                                sentiment_data.append(sentiment)
                            positive_count = sum(1 for s in sentiment_data if s == "Positive")
                            negative_count = sum(1 for s in sentiment_data if s == "Negative")
                    total_comments = len(sentiment_data)

                    # 3. Get Top Comments (use raw chat messages for display)
                    # Pass the *raw* chat messages corresponding to the cleaned ones analyzed
                    # Ensure raw chat list matches the sentiment list length if preprocessing removed items
                    # For simplicity here, assuming chat_raw and sentiment_data correspond correctly
                    # A more robust approach might map indices if filtering happens during cleaning.
                    raw_chat_for_top = chat_raw[:len(sentiment_data)] # Use raw chats corresponding to analyzed ones
                    positive_comments, negative_comments = get_top_comments(raw_chat_for_top, sentiment_data)


                    # 4. Analyze Description Sentiment (if description exists)
                    description_sentiment = "N/A" # Default
                    if desc_clean:
                         with st.spinner('Analyzing description sentiment...'): # Nested spinner
                             description_sentiment, _ = analyze_sentiment(desc_clean)

                    # 5. Store results
                    response_data = {
                        'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", # Medium quality thumb
                        'video_details': {'title': title}, # Simplified details for now
                        'comments': {
                            'total_comments': total_comments,
                            'positive_comments': positive_count,
                            'negative_comments': negative_count,
                            'positive_comments_list': positive_comments,
                            'negative_comments_list': negative_comments
                        },
                        "description": desc_clean if desc_clean else "No description available.",
                        "video_id": video_id,
                        "sentiment_data": sentiment_data,
                        "live_chat_messages_raw": chat_raw, # Store raw for display
                        "description_sentiment": description_sentiment,
                    }
                    st.session_state.responses.append(response_data)
                    st.success("Analysis complete!", icon="✅") # Indicate success after spinner

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}", icon="🚨")
                    logging.error(f"Analysis error for {youtube_link}: {e}", exc_info=True) # Log full traceback
            else:
                st.error("Invalid YouTube URL provided. Please check the link and try again.", icon="🔗")


# --- Display Results Area ---
if not st.session_state.responses:
    st.info("Enter a YouTube video link above and click 'Analyze Video' to see the results.")
else:
    # Only display the latest analysis if needed, or loop as before
    # For now, loop through all stored (usually just one after the logic change)
    for idx, response in enumerate(st.session_state.responses):
        video_details = response.get('video_details', {})
        comments = response.get('comments', {})
        live_chat_messages = response.get('live_chat_messages_raw', []) # Use raw for display
        sentiment_data = response.get('sentiment_data', [])
        video_id = response.get('video_id')

        st.markdown("---")
        st.header(f"📊 Analysis Results for: {video_details.get('title', 'Video')}")

        tab1, tab2, tab3 = st.tabs(["📝 Video Info", "💬 Live Chat Analysis", "📜 Summary"])

        # --- Tab 1: Video Info ---
        with tab1:
            col1, col2 = st.columns([0.6, 0.4]) # Give text slightly more space

            with col1:
                st.subheader("📄 Description")
                # Use an expander for potentially long descriptions
                with st.expander("Click to view description", expanded=False):
                     st.markdown(f"> {response.get('description', 'N/A')}", unsafe_allow_html=True) # Blockquote style

                st.subheader("📈 Description Sentiment")
                sentiment_emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐", "N/A": "❓", "Error": "⚠️"}
                desc_sentiment = response.get('description_sentiment', 'N/A')
                st.markdown(f"**Overall Sentiment:** {desc_sentiment} {sentiment_emoji.get(desc_sentiment, '')}")


            with col2:
                st.subheader("🖼️ Video Thumbnail")
                thumb_url = response.get('thumbnail_url')
                if thumb_url:
                    st.image(thumb_url, use_column_width=True, caption=video_details.get('title', 'Video Thumbnail'))
                else:
                    st.info("Thumbnail not available.")

        # --- Tab 2: Live Chat Analysis ---
        with tab2:
            if not live_chat_messages and comments.get('total_comments', 0) == 0:
                 st.info("No live chat messages were found or could be analyzed for this video.")
            else:
                col1, col2 = st.columns([0.6, 0.4]) # Adjusted column widths

                with col1:
                    st.subheader("🗨️ Live Chat Messages & Sentiment")
                    if live_chat_messages and sentiment_data:
                        # Ensure lists have the same length for DataFrame creation
                        min_len = min(len(live_chat_messages), len(sentiment_data))
                        df_data = {
                            'Live Chat Message': live_chat_messages[:min_len],
                            'Sentiment': sentiment_data[:min_len]
                         }
                        df = pd.DataFrame(df_data)
                        # Use st.dataframe for better display control
                        st.dataframe(df, height=450, use_container_width=True) # Slightly taller, use container width
                    elif comments.get('total_comments', 0) > 0:
                        st.warning("Sentiment data might be missing for some chat messages.", icon="⚠️")
                    else:
                         st.info("No live chat messages to display.")


                with col2:
                    st.subheader("📊 Sentiment Breakdown")
                    if comments and 'total_comments' in comments:
                        total = comments['total_comments']
                        positive = comments['positive_comments']
                        negative = comments['negative_comments']
                        neutral = total - positive - negative

                        if total > 0:
                             # Display Pie Chart using Plotly
                             fig = plot_sentiment_pie_chart_plotly(positive, negative, total)
                             st.plotly_chart(fig, use_container_width=True) # Key: use container width

                             # # Display Metrics using st.metric for a cleaner look
                             # st.metric(label="Total Comments Analyzed", value=f"{total}")

                             # pos_perc = (positive / total) * 100 if total > 0 else 0
                             # neg_perc = (negative / total) * 100 if total > 0 else 0
                             # neu_perc = (neutral / total) * 100 if total > 0 else 0 # Calculate neutral percentage

                             # st.metric(label="😊 Positive", value=f"{positive}", delta=f"{pos_perc:.1f}%")
                             # st.metric(label="😠 Negative", value=f"{negative}", delta=f"{neg_perc:.1f}%")
                             # st.metric(label="😐 Neutral", value=f"{neutral}", delta=f"{neu_perc:.1f}%")

                        else:
                            st.info("No comments were analyzed.")
                    else:
                         st.warning("Comment statistics not available.")

                # --- Top Comments Display (Moved below columns but within Tab 2) ---
                st.markdown("---") # Separator
                st.subheader("⭐ Top Comments")
                if comments and comments['total_comments'] > 0:
                    # Using st.expander instead of checkbox for a cleaner look
                    with st.expander("Show Top Positive & Negative Comments", expanded=False):
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.markdown(f"<h5 style='color: #28a745;'>👍 Top Positive Comments:</h5>", unsafe_allow_html=True)
                            if comments['positive_comments_list']:
                                for comment in comments['positive_comments_list']:
                                    st.markdown(f"<div style='background-color: #121f12; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #28a745;'>{comment}</div>", unsafe_allow_html=True)
                            else:
                                st.caption("No positive comments found.")

                        with col_neg:
                            st.markdown(f"<h5 style='color: #dc3545;'>👎 Top Negative Comments:</h5>", unsafe_allow_html=True)
                            if comments['negative_comments_list']:
                                for comment in comments['negative_comments_list']:
                                    st.markdown(f"<div style='background-color: #121f12; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #dc3545;'>{comment}</div>", unsafe_allow_html=True)
                            else:
                                 st.caption("No negative comments found.")
                else:
                    st.caption("No comments available to display top examples.")


        # --- Tab 3: Summary ---
        with tab3:
            st.subheader("✍️ Video Summary (via Gemini AI)")
            st.markdown("Please try again for 2-3 times until the summary works")
            # Button to trigger summary generation
            summary_key = f"summary_{video_id}_{idx}" # Unique key per video/response
            summary_button_label = "📜 Generate Summary"
            if 'transcript_summary' in response:
                summary_button_label = "🔄 Regenerate Summary" # Change label if summary exists

            if st.button(summary_button_label, key=summary_key):
                with st.spinner("🔄 Generating summary with Gemini AI... This might take a few seconds."):
                    try:
                        transcript = get_sub(video_id) # Fetch transcript
                        if transcript:
                            summary = get_gemini_response_with_retry(transcript) # Call Gemini
                            if summary:
                                # Update the specific response in session state
                                st.session_state.responses[idx]['transcript_summary'] = summary
                                st.rerun() # Rerun to display the new summary immediately
                            else:
                                # Error handled within get_gemini_response_with_retry
                                pass
                        else:
                            # Error handled within get_sub
                            st.warning("Could not generate summary because the transcript is unavailable.", icon="⚠️")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during summary generation: {e}", icon="🚨")
                        logging.error(f"Summary generation error for {video_id}: {e}", exc_info=True)

            # Display generated summary if it exists in the response
            if 'transcript_summary' in response:
                st.markdown(f"<div style='background-color: #121f12; padding: 15px; border-radius: 8px; border-left: 5px solid #0d6efd;'>{response['transcript_summary']}</div>", unsafe_allow_html=True)
            else:
                st.info("Click 'Generate Summary' to create a summary of the video transcript using AI.")

# Optional: Add a footer
st.markdown("---")
st.caption("YOUTUBE LIVECHAT SENTIMENT | TRAN THU HIEN | Summary by Google Gemini")
