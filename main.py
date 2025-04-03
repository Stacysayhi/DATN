import re
import json
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yt_dlp
from googleapiclient.discovery import build
import logging
# import matplotlib.pyplot as plt # Not used for main chart anymore
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go # Import Plotly
from plotly.subplots import make_subplots
import time

# --- Configuration ---
# Store API keys securely (e.g., using Streamlit secrets or environment variables)
# For demonstration purposes, replace placeholders if running locally:
# API_KEY = st.secrets.get("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY_HERE") # Example using secrets
# GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # Example using secrets
API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"  # Replace with your actual YouTube Data API key if needed
GOOGLE_API_KEY = "AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"  # Replace with your Gemini API key if needed

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API (ensure key is valid)
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please check your API key. Error: {e}", icon="🔑")
    logging.error(f"Gemini API configuration error: {e}")
    # Optionally stop execution if Gemini is critical
    # st.stop()


# Model path configuration (adjust if necessary)
MODEL_PATH = ""  # Set to directory if model files are separate
MODEL_FILE = "sentiment_classifier (1).pth" # Ensure this file exists

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the sentiment analysis model and tokenizer."""
    with st.spinner("⏳ Loading Sentiment Analysis Model..."):
        model_path = os.path.join(MODEL_PATH, MODEL_FILE)
        model_id = "wonrax/phobert-base-vietnamese-sentiment"
        try:
            # Ensure the model file exists before attempting to load
            if MODEL_FILE and not os.path.exists(model_path):
                 st.error(f"Model file not found at: {model_path}. Cannot load custom weights.", icon="🚨")
                 # Attempt to load the base model without custom weights as a fallback
                 st.warning(f"Attempting to load base model '{model_id}' without custom weights.", icon="⚠️")
                 tokenizer = AutoTokenizer.from_pretrained(model_id)
                 model = AutoModelForSequenceClassification.from_pretrained(model_id)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                if MODEL_FILE: # Only load state_dict if file is specified and exists
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
                    print(f"Custom weights loaded from {model_path}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            print(f"Model '{model_id}' loaded successfully and moved to {device}")
            return tokenizer, model
        except Exception as e:
            st.error(f"Fatal Error: Could not load sentiment model '{model_id}'. Check path/integrity. Error: {e}", icon="🚨")
            logging.error(f"Error loading model from {model_path} or {model_id}: {e}", exc_info=True)
            # st.stop() # Consider stopping if the model is essential
            return None, None

# --- Sentiment Analysis Function ---
def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.warning("Sentiment analysis model not available.", icon="⚠️")
        return "Error", [0, 0, 0] # Return error state

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure model is on the correct device (might be moved by another process)
    model.to(device)
    tokenizer.padding_side = "left" # Recommended for PhoBERT

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_class = np.argmax(predictions)
        sentiment_label = sentiment_labels[predicted_class]

        return sentiment_label, predictions
    except Exception as e:
        logging.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}", exc_info=True)
        st.warning(f"Could not analyze sentiment for some text due to error: {e}", icon="⚠️")
        return "Error", [0, 0, 0]

# --- Text Preprocessing ---
def preprocess_model_input_str(text, video_title=""):
    """Cleans text for sentiment analysis."""
    if not text:
        return ""
    # Slightly refined regex
    regex_pattern = r"(http|www)\S+|\S*@\S*\s?|[-()+*&^%$#!@\";<>\/\.\?]{3,}|\n|#\w+|\w*:"
    # Remove title first to avoid parts being removed by regex
    text_no_title = text.replace(video_title, "").strip()
    # Apply regex and clean whitespace
    clean_str = re.sub(r"\s{2,}", " ", re.sub(regex_pattern, " ", text_no_title)).strip()
    return clean_str

# --- YouTube Data Extraction ---
def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})")
    match = pattern.search(url)
    return match.group(1) if match else None

def fetch_video_description(video_id, api_key):
    """Fetches video description using YouTube Data API."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        if not response.get("items"):
            logging.warning(f"No video items found for ID: {video_id}")
            return None
        return response["items"][0]["snippet"].get("description") # Safer access
    except Exception as e:
        logging.error(f"Error fetching video description for {video_id}: {e}")
        return None

def download_live_chat(video_url, video_id):
    """Downloads live chat replay using yt-dlp."""
    ydl_opts = {
        'writesubtitles': True,
        'skip_download': True,
        'subtitleslangs': ['live_chat'], # Explicitly request live chat
        'outtmpl': f'{video_id}', # Output filename base
        'quiet': True,
        'no_warnings': True,
        'encoding': 'utf-8', # Ensure UTF-8
    }
    subtitle_file = f"{video_id}.live_chat.json" # Expected output file

    # Clean up potential leftover file from previous run
    if os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
        except OSError as e:
             logging.warning(f"Could not remove pre-existing subtitle file {subtitle_file}: {e}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Check info first if possible, though download=True is needed for subs
            ydl.extract_info(video_url, download=True)

        # Verify file was created
        if os.path.exists(subtitle_file):
            return subtitle_file
        else:
            # This case happens if yt-dlp runs but finds no live chat subtitle track
            logging.warning(f"yt-dlp ran but did not create live chat file for {video_id}. No replay available?")
            st.warning("Could not find live chat replay for this video. Analysis will proceed without chat data.", icon="💬")
            return None

    except yt_dlp.utils.DownloadError as e:
        err_str = str(e).lower()
        if "requested format not available" in err_str or "no closed captions found" in err_str or "live chat" in err_str:
            st.warning("Could not find live chat replay for this video. Analysis will proceed without chat data.", icon="💬")
        else:
            st.error(f"Error downloading video data (yt-dlp): {e}", icon="🚨")
        logging.error(f"yt-dlp DownloadError for {video_url}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during video data download: {e}", icon="🚨")
        logging.error(f"Unexpected error downloading chat for {video_url}: {e}", exc_info=True)
        return None

def parse_jsonl(file_path):
    """Parses a JSON Lines file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as json_err:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {json_err} - Line: {line.strip()}")
                    continue # Skip corrupted lines
        return data
    except FileNotFoundError:
        logging.error(f"Live chat file not found during parsing: {file_path}")
        # Error might have been shown already, avoid redundancy
        # st.error(f"Internal Error: Chat file disappeared before parsing.", icon="🔥")
        return None
    except Exception as e:
        logging.error(f"Error opening/reading chat file {file_path}: {e}")
        st.error(f"Could not read downloaded chat file. Error: {e}", icon="📄")
        return None

def extract_live_chat_messages(subtitle_file):
    """Extracts message text from parsed live chat data."""
    messages = []
    if not subtitle_file or not os.path.exists(subtitle_file):
        return messages # Return empty list if no file

    data = parse_jsonl(subtitle_file)
    if data is None: # Check if parsing failed
        return messages # Return empty list

    processed_count = 0
    error_count = 0
    for lc in data:
        try:
            # Navigate the JSON structure carefully, using .get() for safety
            replay_action = lc.get('replayChatItemAction', {})
            if not replay_action: continue

            actions = replay_action.get('actions', [])
            if not actions: continue

            for act in actions:
                add_chat_item = act.get('addChatItemAction', {})
                if not add_chat_item: continue

                item = add_chat_item.get('item', {})
                if not item: continue

                live_chat_renderer = item.get('liveChatTextMessageRenderer')
                if live_chat_renderer:
                    message_data = live_chat_renderer.get('message', {})
                    runs = message_data.get('runs', [])
                    full_message = ''.join(run.get('text', '') for run in runs if 'text' in run).strip()
                    if full_message: # Ensure we don't add empty messages
                        messages.append(full_message)
                        processed_count += 1
                # Add elif blocks here to handle other types like superchats, memberships if needed
                # elif item.get('liveChatPaidMessageRenderer'): ...
                # elif item.get('liveChatMembershipItemRenderer'): ...

        except Exception as e:
            error_count += 1
            logging.warning(f"Error processing one live chat entry: {e} - Data: {str(lc)[:100]}...")
            continue # Skip to the next entry on error

    if error_count > 0:
        logging.warning(f"Finished processing chat file {subtitle_file}. Processed: {processed_count}, Errors: {error_count}")

    return messages

# --- Combined Data Fetching and Processing ---
def get_video_details_with_chat(video_url: str, api_key: str) -> dict:
    """Fetches description and live chat, handles errors."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    # 1. Fetch Video Description
    description = fetch_video_description(video_id, api_key)
    description = description if description is not None else "" # Ensure string type

    # 2. Download and Parse Live Chat
    subtitle_file = download_live_chat(video_url, video_id) # Handles download errors/warnings
    live_chat_messages = [] # Initialize
    if subtitle_file: # Only try to parse if download likely succeeded
        live_chat_messages = extract_live_chat_messages(subtitle_file)
        # 3. Clean up the temp file
        if os.path.exists(subtitle_file):
            try:
                os.remove(subtitle_file)
                logging.info(f"Deleted temporary chat file: {subtitle_file}")
            except Exception as e:
                logging.warning(f"Error deleting temporary file {subtitle_file}: {e}")
    # else: live_chat_messages remains []

    return {
        "video_id": video_id,
        "description": description,
        "live_chat": live_chat_messages # This is the raw list extracted
    }

# --- *** MODIFIED get_desc_chat Function *** ---
def get_desc_chat(video_url, API_KEY):
    """Gets description, cleans it, gets raw chat, cleans chat, and returns aligned lists."""
    video_info = get_video_details_with_chat(video_url, API_KEY)

    if "error" in video_info:
        st.error(f"Error fetching video data: {video_info['error']}", icon="🚨")
        # Return consistent types on error: desc, clean_chat, title, raw_chat_filtered
        return None, [], "", []

    video_id = video_info.get("video_id")
    video_title = "Video Title Unavailable"
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        if response.get('items'):
            video_title = response['items'][0]['snippet'].get('title', "Video Title Unavailable") # Safer access
    except Exception as e:
        st.warning(f"Could not fetch video title: {e}", icon="⚠️")
        logging.error(f"Error fetching video title for {video_id}: {e}")

    video_description = video_info.get('description', "")
    video_live_chat_raw_original = video_info.get('live_chat', []) # Original raw messages

    # Clean description
    clean_description = preprocess_model_input_str(video_description, video_title)

    # --- Aligned Cleaning of Live Chat ---
    clean_live_chat = []
    filtered_live_chat_raw = [] # Stores raw chats corresponding to clean chats
    if video_live_chat_raw_original:
        for raw_chat in video_live_chat_raw_original:
            if raw_chat and isinstance(raw_chat, str) and raw_chat.strip(): # Basic validation
                clean_chat = preprocess_model_input_str(raw_chat) # Clean it
                if clean_chat: # Only keep if cleaning didn't result in empty string
                    clean_live_chat.append(clean_chat)
                    filtered_live_chat_raw.append(raw_chat) # Keep the corresponding original raw chat
            # else: skip invalid/empty raw chat entries

    # Log counts for debugging potential discrepancies
    # print(f"Original raw chat count: {len(video_live_chat_raw_original)}")
    # print(f"Filtered raw chat count: {len(filtered_live_chat_raw)}")
    # print(f"Cleaned chat count: {len(clean_live_chat)}")

    # Return: cleaned desc, list of cleaned chats, title, list of raw chats *corresponding* to cleaned ones
    return clean_description, clean_live_chat, video_title, filtered_live_chat_raw
# --- *** END MODIFICATION *** ---

# --- Top Comments Selection ---
def get_top_comments(live_chat_raw_filtered, sentiment_labels, top_n=3):
    """Selects top N comments based on sentiment labels. Assumes lists are aligned."""
    positive_comments = []
    negative_comments = []

    # No need to check min_len if the lists are guaranteed to be aligned by get_desc_chat
    # min_len = min(len(live_chat_raw_filtered), len(sentiment_labels)) # Safety check removed for confidence

    # Check if lists are non-empty before iterating
    if not live_chat_raw_filtered or not sentiment_labels:
         return [], []

    # Check if lengths match - this *should* always be true now, but useful assertion
    assert len(live_chat_raw_filtered) == len(sentiment_labels), \
           f"Mismatch! Raw filtered: {len(live_chat_raw_filtered)}, Sentiments: {len(sentiment_labels)}"

    for i in range(len(sentiment_labels)): # Iterate based on sentiment_labels length
        # Access using index 'i' assumes perfect alignment
        comment = live_chat_raw_filtered[i]
        sentiment = sentiment_labels[i]
        if sentiment == "Positive":
            # Append only if we need more positive comments
            if len(positive_comments) < top_n:
                positive_comments.append(comment)
        elif sentiment == "Negative":
            # Append only if we need more negative comments
            if len(negative_comments) < top_n:
                negative_comments.append(comment)

        # Optimization: Stop early if we have found enough of both
        if len(positive_comments) >= top_n and len(negative_comments) >= top_n:
            break

    return positive_comments, negative_comments


# --- Plotly Pie Chart Function ---
def plot_sentiment_pie_chart_plotly(positive_count, negative_count, total_comments):
    """Generates an interactive Plotly pie chart for sentiment distribution."""
    if total_comments == 0:
        fig = go.Figure()
        fig.update_layout(title_text='No comments to analyze', title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    neutral_count = total_comments - (positive_count + negative_count)
    # Ensure neutral count isn't negative due to potential analysis errors
    neutral_count = max(0, neutral_count)

    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_count, negative_count, neutral_count]
    colors = ['#28a745', '#dc3545', '#6c757d'] # Green, Red, Gray

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,
                                marker_colors=colors,
                                pull=[0.05, 0.05, 0], # Slightly pull positive/negative
                                hole=0.3,
                                textinfo='percent+value',
                                insidetextorientation='radial',
                                hoverinfo='label+percent+value')])
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend_title_text='Sentiments',
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        # font_color="white" # Uncomment if needed for dark themes
    )
    return fig

# --- Transcript and Summarization ---
def get_sub(video_id):
    """Retrieves transcript text (Vietnamese preferred, fallback English)."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        try:
            # Try Vietnamese first
            transcript = transcript_list.find_generated_transcript(['vi']).fetch()
            print(f"Found Vietnamese transcript for {video_id}")
        except Exception:
            logging.info(f"Vietnamese transcript not found for {video_id}. Trying English.")
            try:
                 # Fallback to English
                 transcript = transcript_list.find_generated_transcript(['en']).fetch()
                 st.info("Vietnamese transcript not found, using English for summary.", icon="ℹ️")
                 print(f"Found English transcript for {video_id}")
            except Exception as en_e:
                 # No English either
                 st.error(f"No suitable transcript (Vietnamese or English) found for video ID {video_id}.", icon="🚨")
                 logging.error(f"No VI or EN transcript for {video_id}: {en_e}")
                 return None

        # Concatenate transcript segments
        concatenated_text = ' '.join(segment.get('text', '') for segment in transcript)
        return concatenated_text

    except Exception as e:
        # Handle broader errors like TranscriptsDisabled, NoTranscriptFound etc.
        st.error(f"Could not retrieve transcript for video ID {video_id}. Transcripts might be disabled. Error: {e}", icon="🚨")
        logging.error(f"Error getting subtitles for {video_id}: {e}", exc_info=True)
        return None

# Gemini prompt
prompt = """
Bạn là một trợ lý AI chuyên tóm tắt video YouTube. Dựa vào bản ghi chép (transcript) được cung cấp,
hãy tạo một bản tóm tắt súc tích, nêu bật những điểm chính của video. Giới hạn bản tóm tắt
trong khoảng 150-300 từ.

Bản ghi chép:
"""

def get_gemini_response_with_retry(transcript_text, max_attempts=3):
    """Generates summary using Gemini, with retry logic."""
    if not transcript_text or not transcript_text.strip():
        logging.warning("Attempted to summarize empty transcript.")
        return "Error: Cannot generate summary from empty transcript."

    full_prompt = f"{prompt}\n{transcript_text}"

    # Ensure Gemini API is configured
    if not GOOGLE_API_KEY or not genai._config.api_key:
         st.error("Gemini API key not configured. Cannot generate summary.", icon="🔑")
         return None

    for attempt in range(max_attempts):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash") # Use appropriate model
            response = model.generate_content(full_prompt)

            # Basic check if response has text (more robust checks might be needed)
            if hasattr(response, 'text') and response.text:
                # Optional: Check for safety flags if necessary
                # if response.prompt_feedback.block_reason: ...
                return response.text
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 st.error(f"Summary generation blocked by safety settings: {block_reason}", icon="🛡️")
                 logging.error(f"Gemini response blocked: {block_reason}")
                 return f"Error: Content generation blocked ({block_reason})."
            else:
                # Handle cases where response is empty but not blocked
                logging.warning(f"Gemini returned empty response (Attempt {attempt + 1}). Response: {response}")
                if attempt < max_attempts - 1:
                     st.warning(f"Summary generation attempt {attempt + 1} returned empty. Retrying...", icon="⏳")
                     time.sleep(1.5 ** attempt) # Adjusted backoff
                else:
                     st.error(f"Failed to generate summary from Gemini after {max_attempts} attempts (empty response).", icon="🚨")
                     return None # Final failure


        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to generate Gemini response: {e}", exc_info=True)
            if attempt < max_attempts - 1:
                st.warning(f"Summary generation attempt {attempt + 1} failed. Retrying...", icon="⏳")
                time.sleep(1.5 ** attempt) # Adjusted backoff
            else:
                st.error(f"Failed to generate summary from Gemini after {max_attempts} attempts. Error: {e}", icon="🚨")
                return None # Final failure

    return None # Should not be reached, but return None if loop finishes unexpectedly


# --- Streamlit App UI ---
st.set_page_config(page_title="🎥 YouTube Video Analysis", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'last_youtube_link' not in st.session_state:
    st.session_state.last_youtube_link = ""

# --- Input Area ---
st.subheader("Enter YouTube Video Link")
# Example link hint
st.caption("e.g., `https://www.youtube.com/watch?v=YluWWSCdXD4` or `https://www.youtube.com/watch?v=ISrGxpJgLXM`")
youtube_link = st.text_input("🔗 Paste URL here:", key="youtube_link_input", label_visibility="collapsed")

# --- Analyze Button and Processing Logic ---
if st.button("🔍 Analyze Video", type="primary"):
    st.session_state.summary_generated = False # Reset summary flag on new analysis

    if not youtube_link or not youtube_link.strip():
        st.warning("Please enter a YouTube video link.", icon="⚠️")
    elif youtube_link == st.session_state.get('last_youtube_link') and st.session_state.get('responses'):
         st.info("Analysis for this video is already displayed below.", icon="ℹ️")
    else:
        # Clear previous results for a new analysis
        st.session_state.responses = []
        st.session_state.last_youtube_link = youtube_link # Store the link being analyzed

        video_id = extract_video_id(youtube_link)

        if not video_id:
            st.error("Invalid YouTube URL provided. Please check the link format.", icon="🔗")
        else:
            # Main analysis spinner
            with st.spinner('🚀 Analyzing video... This may take a minute for long chats.'):
                analysis_successful = False # Flag to track success
                try:
                    # --- *** MODIFIED Analysis Steps *** ---

                    # 1. Get Description and Aligned Chat Data
                    #    `chat_raw_filtered` now contains raw chats corresponding to `chat_clean`
                    desc_clean, chat_clean, title, chat_raw_filtered = get_desc_chat(youtube_link, API_KEY)

                    # Check if core data retrieval failed critically
                    if desc_clean is None and not chat_clean:
                        # Error was likely shown in get_desc_chat, but raise to stop
                        raise ValueError("Failed to retrieve essential video details (description/chat).")

                    # 2. Analyze Live Chat Sentiment (if `chat_clean` exists)
                    sentiment_data = []
                    positive_count = 0
                    negative_count = 0
                    tokenizer_loaded, model_loaded = load_model() # Check model status once

                    if chat_clean:
                        if tokenizer_loaded is None:
                             st.warning("Sentiment model failed to load. Skipping chat sentiment analysis.", icon="⚠️")
                        else:
                            with st.spinner('💬 Analyzing live chat sentiment...'):
                                for i, chat in enumerate(chat_clean):
                                    # Optional: Add progress update for long chats
                                    # if i % 50 == 0 and i > 0:
                                    #    print(f"Analyzing chat {i}/{len(chat_clean)}...")
                                    sentiment, _ = analyze_sentiment(chat)
                                    sentiment_data.append(sentiment)

                                positive_count = sum(1 for s in sentiment_data if s == "Positive")
                                negative_count = sum(1 for s in sentiment_data if s == "Negative")
                    # `total_comments` is the count of successfully cleaned and analyzed comments
                    total_comments = len(sentiment_data)

                    # 3. Get Top Comments (use the FILTERED raw chat)
                    #    `chat_raw_filtered` and `sentiment_data` are now aligned.
                    positive_comments, negative_comments = get_top_comments(chat_raw_filtered, sentiment_data)

                    # 4. Analyze Description Sentiment
                    description_sentiment = "N/A"
                    if desc_clean:
                        if tokenizer_loaded is None:
                             st.warning("Sentiment model failed to load. Skipping description analysis.", icon="⚠️")
                        else:
                             with st.spinner('📄 Analyzing description sentiment...'):
                                 description_sentiment, _ = analyze_sentiment(desc_clean)

                    # 5. Store results
                    response_data = {
                        'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
                        'video_details': {'title': title},
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
                        # Store the *filtered* raw messages that were analyzed
                        "live_chat_messages_raw": chat_raw_filtered,
                        "description_sentiment": description_sentiment,
                        # Add placeholder for summary - generated on demand later
                        # 'transcript_summary': None
                    }
                    st.session_state.responses.append(response_data)
                    analysis_successful = True # Mark as successful

                except ValueError as ve: # Catch specific error from data fetching
                    st.error(f"Analysis aborted: {ve}", icon="🚨")
                    logging.error(f"Analysis ValueError for {youtube_link}: {ve}", exc_info=False)
                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {e}", icon="🚨")
                    logging.error(f"Analysis error for {youtube_link}: {e}", exc_info=True) # Log full traceback

                # Show success message only if the process completed without critical errors
                if analysis_successful:
                    st.success("✅ Analysis complete!", icon="✅")
                else:
                    # Ensure spinner closes even on error
                    st.error("Analysis could not be completed due to errors.", icon="❌")


# --- Display Results Area ---
if not st.session_state.get('responses'):
    st.info("Enter a YouTube video link above and click 'Analyze Video' to see the results.")
else:
    # Display the latest analysis (index 0 as we clear responses each time)
    response = st.session_state.responses[0] # Get the first (and only) response data

    video_details = response.get('video_details', {})
    comments = response.get('comments', {})
    # Use the filtered raw chat list stored in the response
    live_chat_messages_filtered = response.get('live_chat_messages_raw', [])
    sentiment_data = response.get('sentiment_data', [])
    video_id = response.get('video_id')
    video_title = video_details.get('title', 'Video')

    st.markdown("---")
    st.header(f"📊 Analysis Results for: {video_title}")

    tab1, tab2, tab3 = st.tabs(["📝 Video Info", "💬 Live Chat Analysis", "📜 Summary"])

    # --- Tab 1: Video Info ---
    with tab1:
        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.subheader("📄 Description")
            desc_text = response.get('description', 'N/A')
            if desc_text == 'N/A' or not desc_text.strip():
                st.info("No description available for this video.")
            else:
                with st.expander("Click to view description", expanded=False):
                     st.markdown(f"> {desc_text}", unsafe_allow_html=True) # Blockquote

            st.subheader("📈 Description Sentiment")
            sentiment_emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐", "N/A": "❓", "Error": "⚠️"}
            desc_sentiment = response.get('description_sentiment', 'N/A')
            st.markdown(f"**Overall Sentiment:** {desc_sentiment} {sentiment_emoji.get(desc_sentiment, '')}")

        with col2:
            st.subheader("🖼️ Video Thumbnail")
            thumb_url = response.get('thumbnail_url')
            if thumb_url:
                st.image(thumb_url, use_column_width=True, caption=video_title)
            else:
                st.info("Thumbnail not available.")

    # --- Tab 2: Live Chat Analysis ---
    with tab2:
        total_analyzed_comments = comments.get('total_comments', 0)

        if not live_chat_messages_filtered and total_analyzed_comments == 0:
             st.info("No live chat messages were found or could be analyzed for this video.")
        else:
            col_chat_table, col_chat_chart = st.columns([0.6, 0.4]) # Renamed columns

            with col_chat_table:
                st.subheader("🗨️ Live Chat Messages & Sentiment")
                # Ensure lists are valid and lengths match *before* creating DataFrame
                if live_chat_messages_filtered and sentiment_data and len(live_chat_messages_filtered) == len(sentiment_data):
                    df_data = {
                        'Live Chat Message': live_chat_messages_filtered,
                        'Sentiment': sentiment_data
                     }
                    df = pd.DataFrame(df_data)
                    # Display DataFrame - consider height based on number of comments?
                    display_height = min(450, max(200, len(live_chat_messages_filtered) * 35)) # Dynamic height attempt
                    st.dataframe(df, height=display_height, use_container_width=True)
                elif live_chat_messages_filtered or sentiment_data: # Mismatch or one list empty
                    st.warning("Could not display chat messages table due to data inconsistency.", icon="⚠️")
                    logging.warning(f"DataFrame display skipped. Filtered msgs: {len(live_chat_messages_filtered)}, Sentiments: {len(sentiment_data)}")
                else: # Both empty, but total_comments > 0 (shouldn't happen often now)
                     st.info("No live chat messages to display in table.")

            with col_chat_chart:
                st.subheader("📊 Sentiment Breakdown")
                if total_analyzed_comments > 0:
                    positive = comments.get('positive_comments', 0)
                    negative = comments.get('negative_comments', 0)
                    # Pass counts directly to the plotting function
                    fig = plot_sentiment_pie_chart_plotly(positive, negative, total_analyzed_comments)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Display the empty chart message if total_comments is 0
                    fig = plot_sentiment_pie_chart_plotly(0, 0, 0)
                    st.plotly_chart(fig, use_container_width=True)
                    # st.info("No comments were analyzed to show sentiment breakdown.")


            # --- Top Comments Display (Moved below columns but within Tab 2) ---
            st.markdown("---") # Separator
            st.subheader("⭐ Top Comments")
            if total_analyzed_comments > 0:
                pos_comments_list = comments.get('positive_comments_list', [])
                neg_comments_list = comments.get('negative_comments_list', [])

                if not pos_comments_list and not neg_comments_list:
                     st.caption("No specific positive or negative comments identified as 'top'.")
                else:
                    with st.expander("Show Top Positive & Negative Comments", expanded=True): # Expanded by default
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.markdown(f"<h5 style='color: #28a745;'>👍 Top Positive:</h5>", unsafe_allow_html=True)
                            if pos_comments_list:
                                for comment in pos_comments_list:
                                    st.markdown(f"<div style='background-color: rgba(40, 167, 69, 0.1); padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #28a745;'>{st.markdown.escape(comment)}</div>", unsafe_allow_html=True)
                            else:
                                st.caption("None found.")

                        with col_neg:
                            st.markdown(f"<h5 style='color: #dc3545;'>👎 Top Negative:</h5>", unsafe_allow_html=True)
                            if neg_comments_list:
                                for comment in neg_comments_list:
                                    st.markdown(f"<div style='background-color: rgba(220, 53, 69, 0.1); padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #dc3545;'>{st.markdown.escape(comment)}</div>", unsafe_allow_html=True)
                            else:
                                 st.caption("None found.")
            else:
                st.caption("No comments available to display top examples.")


    # --- Tab 3: Summary ---
    with tab3:
        st.subheader("✍️ Video Summary (via Gemini AI)")
        # st.caption("Powered by Google Gemini. Summary based on video transcript (if available).")

        summary_key = f"summary_{video_id}" # Simpler key if only one response shown
        summary_button_label = "📜 Generate Summary"

        # Check if summary already exists in the current response data within session state
        if 'transcript_summary' in st.session_state.responses[0]:
             summary_button_label = "🔄 Regenerate Summary"

        # Button to trigger summary generation
        if st.button(summary_button_label, key=summary_key):
            if not GOOGLE_API_KEY or not genai._config.api_key:
                 st.error("Gemini API key not configured. Cannot generate summary.", icon="🔑")
            else:
                with st.spinner("🔄 Generating summary with Gemini AI... This may take a few moments."):
                    summary = None # Initialize summary variable
                    try:
                        transcript = get_sub(video_id) # Fetch transcript
                        if transcript:
                            summary = get_gemini_response_with_retry(transcript) # Call Gemini
                            if summary and "Error:" not in summary:
                                # Update the specific response in session state
                                st.session_state.responses[0]['transcript_summary'] = summary
                                st.session_state.summary_generated = True # Flag success
                                # st.rerun() # Rerun might not be needed if state updates correctly
                            elif summary: # Handle "Error:" case returned from Gemini function
                                 st.error(f"Summary generation failed: {summary}", icon="🚨")
                                 st.session_state.responses[0]['transcript_summary'] = None # Clear any previous summary
                            else: # Handle None case (meaning retry failed)
                                # Error should have been shown in get_gemini_response_with_retry
                                st.session_state.responses[0]['transcript_summary'] = None # Clear any previous summary
                                pass
                        else:
                            # Error/Warning shown in get_sub
                            st.warning("Could not generate summary because the transcript is unavailable.", icon="⚠️")
                            st.session_state.responses[0]['transcript_summary'] = None # Ensure no old summary displayed

                    except Exception as e:
                        st.error(f"An unexpected error occurred during summary generation: {e}", icon="🚨")
                        logging.error(f"Summary generation error for {video_id}: {e}", exc_info=True)
                        st.session_state.responses[0]['transcript_summary'] = None # Clear summary on error

                    # Force rerun only if summary was successfully generated to update display immediately
                    if st.session_state.get('summary_generated', False):
                        st.rerun()


        # Display generated summary if it exists in the response data
        # Retrieve the latest summary state right before displaying
        current_summary = st.session_state.responses[0].get('transcript_summary')
        if current_summary:
            # Use st.markdown with escaped HTML/Markdown for safety if needed, or trust Gemini output
            st.markdown(f"<div style='background-color: #121f12; padding: 15px; border-radius: 8px; border-left: 5px solid #0d6efd;'>{current_summary}</div>", unsafe_allow_html=True)
        elif not st.session_state.get('summary_button_clicked', False): # Only show initial message if button wasn't clicked yet
             st.info("Click 'Generate Summary' to create a summary of the video transcript using AI.")
             # We need a way to know if the button was clicked but failed vs never clicked
             # This simple logic might not cover all edge cases perfectly.


# --- Footer ---
st.markdown("---")
st.caption("YouTube Live Chat Sentiment Analyzer | Summary by Google Gemini")





###################################################3
# import re
# import json
# import os
# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import yt_dlp
# from googleapiclient.discovery import build
# import logging
# import matplotlib.pyplot as plt # Keep for potential future use, but not for the main pie chart anymore
# import numpy as np
# from youtube_transcript_api import YouTubeTranscriptApi
# import pandas as pd
# import google.generativeai as genai
# import plotly.graph_objects as go # <-- Import Plotly
# from plotly.subplots import make_subplots
# import time

# # Your API Key - should be stored securely, not hardcoded
# API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"  # Replace with your actual YouTube Data API key
# GOOGLE_API_KEY = "AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"  # Replace with your Gemini API key

# # Configure logging
# logging.basicConfig(filename='app.log', level=logging.ERROR,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# # Configure Gemini API
# genai.configure(api_key=GOOGLE_API_KEY)

# MODEL_PATH = ""  # Set this to the directory if you have a folder ofr the weights, other wise it would be ""
# MODEL_FILE = "sentiment_classifier (1).pth"


# @st.cache_resource
# def load_model():
#     # Using a spinner here is good for the initial load if it's slow
#     with st.spinner("Loading Sentiment Analysis Model... This may take a moment."):
#         model_path = os.path.join(MODEL_PATH, MODEL_FILE)
#         model_id = "wonrax/phobert-base-vietnamese-sentiment"
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_id)
#             model = AutoModelForSequenceClassification.from_pretrained(model_id)
#             model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             model.to(device)
#             print(f"Model loaded successfully from {model_path} and moved to {device}")
#             return tokenizer, model
#         except Exception as e:
#             st.error(f"Fatal Error: Could not load sentiment analysis model. Please check model path and file integrity. Error: {e}", icon="🚨")
#             logging.error(f"Error loading model from {model_path}: {e}")
#             st.stop() # Stop execution if model fails to load
#             return None, None


# def analyze_sentiment(text):
#     tokenizer, model = load_model()
#     if tokenizer is None or model is None:
#         # Error already shown in load_model, maybe a simpler message here
#         st.warning("Sentiment analysis model not available.", icon="⚠️")
#         return "Error", [0, 0, 0]

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     tokenizer.padding_side = "left"
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

#     sentiment_labels = ["Negative", "Neutral", "Positive"]
#     predicted_class = np.argmax(predictions)
#     sentiment_label = sentiment_labels[predicted_class]

#     return sentiment_label, predictions


# def preprocess_model_input_str(text, video_title=""):
#     if not text:
#         return ""
#     regex_pattern = r"(http|www).*(\/|\/\/)\s?|[-()+*&^%$#!@\";<>\/\.\?]{3,}|\n|#.*|\w*:"
#     clean_str = re.sub(r"\s{2,}", " ", re.sub(regex_pattern, " ", text)).replace(video_title, "").strip()
#     return clean_str


# def extract_video_id(url):
#     pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})")
#     match = pattern.search(url)
#     if match:
#         return match.group(1)
#     return None


# def fetch_video_description(video_id, api_key):
#     try:
#         youtube = build("youtube", "v3", developerKey=api_key)
#         response = youtube.videos().list(
#             part="snippet",
#             id=video_id
#         ).execute()

#         if not response["items"]:
#             return None
#         return response["items"][0]["snippet"]["description"]
#     except Exception as e:
#         logging.error(f"Error fetching video description: {e}")
#         return None


# def download_live_chat(video_url, video_id):
#     ydl_opts = {
#         'writesubtitles': True,
#         'skip_download': True,
#         'subtitleslangs': ['live_chat'],
#         'outtmpl': f'{video_id}',
#         'quiet': True, # Make yt-dlp less verbose in console
#         'no_warnings': True,
#     }
#     subtitle_file = f"{video_id}.live_chat.json"
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.extract_info(video_url, download=True)
#         return subtitle_file
#     except yt_dlp.utils.DownloadError as e:
#         # Handle specific yt-dlp errors, like no live chat found
#         if "live chat" in str(e).lower():
#              st.warning("Could not find live chat replay for this video. Analysis will proceed without chat data.", icon="💬")
#         else:
#             st.error(f"Error downloading video data: {e}", icon="🚨")
#         logging.error(f"Error downloading live chat: {e}")
#         return None
#     except Exception as e:
#         st.error(f"An unexpected error occurred during video data download: {e}", icon="🚨")
#         logging.error(f"Error downloading live chat: {e}")
#         return None

# def parse_jsonl(file_path):
#     data = []
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 data.append(json.loads(line))
#         return data
#     except FileNotFoundError:
#         logging.error(f"Live chat file not found: {file_path}")
#         return None
#     except json.JSONDecodeError as e:
#         logging.error(f"Error parsing JSON in live chat file: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Error opening/reading file: {e}")
#         return None

# def extract_live_chat_messages(subtitle_file):
#     messages = []
#     if not subtitle_file or not os.path.exists(subtitle_file):
#         return messages

#     data = parse_jsonl(subtitle_file)
#     if not data:
#         return messages

#     for lc in data:
#         try:
#             lc_actions = lc.get('replayChatItemAction', {}).get('actions', [])
#             for act in lc_actions:
#                 live_chat = act.get('addChatItemAction', {}).get('item', {}).get('liveChatTextMessageRenderer', None)
#                 if live_chat:
#                     runs = live_chat.get('message', {}).get('runs', []) # Safer access
#                     # Combine runs into a single message string
#                     full_message = ''.join(run.get('text', '') for run in runs)
#                     if full_message: # Ensure we don't add empty messages
#                         messages.append(full_message)
#         except Exception as e:
#             logging.warning(f"Error processing a live chat message: {str(e)}")
#             continue
#     return messages


# def get_video_details_with_chat(video_url: str, api_key: str) -> dict:
#     video_id = extract_video_id(video_url)
#     if not video_id:
#         return {"error": "Invalid YouTube URL. Could not extract video ID."}

#     # 1. Fetch Video Description
#     description = fetch_video_description(video_id, api_key)
#     if description is None:
#         description = "" # Default to empty string

#     # 2. Download and Parse Live Chat
#     subtitle_file = download_live_chat(video_url, video_id) # This now handles errors better
#     live_chat_messages = [] # Initialize as empty list
#     if subtitle_file: # Only try to parse if download succeeded
#         live_chat_messages = extract_live_chat_messages(subtitle_file)

#     # 3. Clean up the temp file
#     if subtitle_file and os.path.exists(subtitle_file):
#         try:
#             os.remove(subtitle_file)
#             logging.info(f"Deleted temporary file: {subtitle_file}")
#         except Exception as e:
#             logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")

#     return {
#         "video_id": video_id,
#         "description": description,
#         "live_chat": live_chat_messages
#     }


# def get_desc_chat(video_url, API_KEY):
#     # Note: Spinner is now outside this function call
#     # st.write(f"Analyzing video: {video_url}") # Less verbose, spinner indicates action
#     video_info = get_video_details_with_chat(video_url, API_KEY)

#     if "error" in video_info:
#         st.error(f"Error: {video_info['error']}", icon="🚨")
#         return None, [], "", [] # Return consistent types on error

#     video_id = video_info.get("video_id")
#     video_title = "Video Title Unavailable" # Default title
#     try:
#         youtube = build("youtube", "v3", developerKey=API_KEY)
#         response = youtube.videos().list(
#             part="snippet",
#             id=video_id
#         ).execute()
#         if response.get('items'):
#             video_title = response['items'][0]['snippet']['title']
#     except Exception as e:
#         st.warning(f"Could not fetch video title: {e}", icon="⚠️")
#         logging.error(f"Error fetching video title for {video_id}: {e}")

#     video_description = video_info['description']
#     video_live_chat_raw = video_info['live_chat'] # Keep raw messages for display

#     clean_description = preprocess_model_input_str(video_description, video_title)
#     clean_live_chat = [preprocess_model_input_str(chat) for chat in video_live_chat_raw if chat.strip()] # Preprocess non-empty chats

#     return clean_description, clean_live_chat, video_title, video_live_chat_raw


# def get_top_comments(live_chat_raw, sentiment_labels, top_n=3):
#     """Selects top N comments based on calculated sentiment labels."""
#     positive_comments = []
#     negative_comments = []

#     # Ensure raw chat and labels have the same length for safe iteration
#     min_len = min(len(live_chat_raw), len(sentiment_labels))

#     for i in range(min_len):
#         comment = live_chat_raw[i]
#         sentiment = sentiment_labels[i]
#         if sentiment == "Positive":
#             positive_comments.append(comment)
#         elif sentiment == "Negative":
#             negative_comments.append(comment)

#     # Return only top N, even if fewer are found
#     return positive_comments[:top_n], negative_comments[:top_n]

# # --- NEW Plotly Pie Chart Function ---
# def plot_sentiment_pie_chart_plotly(positive_count, negative_count, total_comments):
#     """Generates an interactive Plotly pie chart."""
#     if total_comments == 0:
#         # Return an empty figure or a message if no comments
#         fig = go.Figure()
#         fig.update_layout(title_text='No comments to analyze', title_x=0.5)
#         return fig

#     neutral_count = total_comments - (positive_count + negative_count)
#     labels = ['Positive', 'Negative', 'Neutral']
#     values = [positive_count, negative_count, neutral_count]
#     # More distinct colors
#     colors = ['#28a745', '#dc3545', '#6c757d'] # Green, Red, Gray

#     fig = go.Figure(data=[go.Pie(labels=labels,
#                                 values=values,
#                                 marker_colors=colors,
#                                 pull=[0.05, 0.05, 0], # Slightly pull positive and negative
#                                 hole=0.3, # Donut chart style
#                                 textinfo='percent+value', # Show percentage and count
#                                 insidetextorientation='radial',
#                                 hoverinfo='label+percent+value')]) # Tooltip info
#     fig.update_layout(
#         #title_text='Live Chat Sentiment Distribution',
#         #title_x=0.5,
#         margin=dict(l=10, r=10, t=30, b=10), # Minimal margins
#         legend_title_text='Sentiments',
#         legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5), # Horizontal legend below
#         # Transparent background can blend better with Streamlit themes
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         # Uncomment if using dark theme
#         # font_color="white"
#     )
#     return fig

# def get_sub(video_id):
#     try:
#         # Attempt to get Vietnamese first, fallback to English
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         try:
#             transcript = transcript_list.find_generated_transcript(['vi']).fetch()
#         except: # Fallback to English if Vietnamese not found
#              try:
#                  transcript = transcript_list.find_generated_transcript(['en']).fetch()
#                  st.info("Vietnamese transcript not found, using English for summary.", icon="ℹ️")
#              except:
#                  st.error(f"No suitable transcript (Vietnamese or English) found for video ID {video_id}.", icon="🚨")
#                  return None

#         concatenated_text = ' '.join([segment['text'] for segment in transcript])
#         return concatenated_text
#     except Exception as e:
#         st.error(f"Error retrieving transcript for video ID {video_id}: {e}", icon="🚨")
#         logging.error(f"Error getting subtitles for video ID {video_id}: {e}")
#         return None


# # Define the prompt for the Gemini model
# prompt = """
# Bạn là người tóm tắt video trên Youtube. Bạn sẽ lấy văn bản ghi chép
# và tóm tắt toàn bộ video và cung cấp bản tóm tắt quan trọng theo các điểm
# trong vòng 300 từ. Vui lòng cung cấp bản tóm tắt của văn bản được đưa ra ở đây:
# """

# # Define the function to get the Gemini response, with retry logic
# def get_gemini_response_with_retry(transcript_text, max_attempts=3):
#     if not transcript_text:
#         return "Error: Cannot generate summary from empty transcript."

#     # Add context to the prompt (optional but can improve results)
#     full_prompt = f"{prompt}\n\nTranscript:\n{transcript_text}"

#     for attempt in range(max_attempts):
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             # Send only the combined prompt and transcript
#             response = model.generate_content(full_prompt)
#             # Check for safety ratings or blocks if applicable/needed
#             # if response.prompt_feedback.block_reason:
#             #     logging.error(f"Gemini response blocked: {response.prompt_feedback.block_reason}")
#             #     return f"Error: Content generation blocked due to safety settings ({response.prompt_feedback.block_reason})."
#             return response.text
#         except Exception as e:
#             logging.error(f"Attempt {attempt + 1} failed to generate Gemini response: {e}")
#             if attempt < max_attempts - 1:
#                 st.warning(f"Summary generation attempt {attempt + 1} failed. Retrying...", icon="⏳")
#                 time.sleep(2 ** attempt) # Exponential backoff
#             else:
#                 st.error(f"Failed to generate summary from Gemini after {max_attempts} attempts. Error: {e}", icon="🚨")
#                 return None # Indicate final failure
#     return None # Should not be reached, but for safety

# # --- Streamlit App UI ---
# st.set_page_config(page_title="🎥 YouTube Video Analysis", layout="wide", initial_sidebar_state="collapsed")
# st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)
# st.markdown("---") # Add a visual separator

# # Initialize session state
# if 'responses' not in st.session_state:
#     st.session_state.responses = []
# if 'last_youtube_link' not in st.session_state:
#     st.session_state.last_youtube_link = ""

# # --- Input Area ---
# st.subheader("Enter YouTube Video Link")
# st.markdown("e.g., https://www.youtube.com/watch?v=ISrGxpJgLXM&t=3606s")
# youtube_link = st.text_input("🔗 Paste the YouTube video URL here:", key="youtube_link_input", label_visibility="collapsed")

# # --- Analyze Button and Processing Logic ---
# if st.button("🔍 Analyze Video", type="primary"): # Use primary button type
#     if not youtube_link or not youtube_link.strip():
#         st.warning("Please enter a YouTube video link.", icon="⚠️")
#         # Optionally clear previous results if the button is clicked with no link
#         # st.session_state.responses = []
#         # st.session_state.last_youtube_link = ""
#     elif youtube_link == st.session_state.last_youtube_link and st.session_state.responses:
#          st.info("Analysis for this video is already displayed below.", icon="ℹ️")
#     else:
#         # --- NEW: Main spinner for the whole analysis process ---
#         with st.spinner('🚀 Analyzing video... Fetching data, processing chat, and evaluating sentiment. Please wait.'):
#             st.session_state.responses = [] # Clear previous results for a new analysis
#             st.session_state.last_youtube_link = youtube_link # Store the link being analyzed
#             video_id = extract_video_id(youtube_link)

#             if video_id:
#                 try:
#                     # 1. Get Description and Chat Data
#                     desc_clean, chat_clean, title, chat_raw = get_desc_chat(youtube_link, API_KEY)

#                     if desc_clean is None: # Handle error from get_desc_chat
#                        raise ValueError("Failed to retrieve video details.")

#                     # 2. Analyze Live Chat Sentiment (if chat exists)
#                     sentiment_data = []
#                     positive_count = 0
#                     negative_count = 0
#                     if chat_clean:
#                         # Consider batching sentiment analysis if performance is an issue
#                         with st.spinner('Analyzing live chat sentiment...'): # Nested spinner for specific step
#                             for chat in chat_clean:
#                                 sentiment, _ = analyze_sentiment(chat)
#                                 sentiment_data.append(sentiment)
#                             positive_count = sum(1 for s in sentiment_data if s == "Positive")
#                             negative_count = sum(1 for s in sentiment_data if s == "Negative")
#                     total_comments = len(sentiment_data)

#                     # 3. Get Top Comments (use raw chat messages for display)
#                     # Pass the *raw* chat messages corresponding to the cleaned ones analyzed
#                     # Ensure raw chat list matches the sentiment list length if preprocessing removed items
#                     # For simplicity here, assuming chat_raw and sentiment_data correspond correctly
#                     # A more robust approach might map indices if filtering happens during cleaning.
#                     raw_chat_for_top = chat_raw[:len(sentiment_data)] # Use raw chats corresponding to analyzed ones
#                     positive_comments, negative_comments = get_top_comments(raw_chat_for_top, sentiment_data)


#                     # 4. Analyze Description Sentiment (if description exists)
#                     description_sentiment = "N/A" # Default
#                     if desc_clean:
#                          with st.spinner('Analyzing description sentiment...'): # Nested spinner
#                              description_sentiment, _ = analyze_sentiment(desc_clean)

#                     # 5. Store results
#                     response_data = {
#                         'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", # Medium quality thumb
#                         'video_details': {'title': title}, # Simplified details for now
#                         'comments': {
#                             'total_comments': total_comments,
#                             'positive_comments': positive_count,
#                             'negative_comments': negative_count,
#                             'positive_comments_list': positive_comments,
#                             'negative_comments_list': negative_comments
#                         },
#                         "description": desc_clean if desc_clean else "No description available.",
#                         "video_id": video_id,
#                         "sentiment_data": sentiment_data,
#                         "live_chat_messages_raw": chat_raw, # Store raw for display
#                         "description_sentiment": description_sentiment,
#                     }
#                     st.session_state.responses.append(response_data)
#                     st.success("Analysis complete!", icon="✅") # Indicate success after spinner

#                 except Exception as e:
#                     st.error(f"An error occurred during analysis: {e}", icon="🚨")
#                     logging.error(f"Analysis error for {youtube_link}: {e}", exc_info=True) # Log full traceback
#             else:
#                 st.error("Invalid YouTube URL provided. Please check the link and try again.", icon="🔗")


# # --- Display Results Area ---
# if not st.session_state.responses:
#     st.info("Enter a YouTube video link above and click 'Analyze Video' to see the results.")
# else:
#     # Only display the latest analysis if needed, or loop as before
#     # For now, loop through all stored (usually just one after the logic change)
#     for idx, response in enumerate(st.session_state.responses):
#         video_details = response.get('video_details', {})
#         comments = response.get('comments', {})
#         live_chat_messages = response.get('live_chat_messages_raw', []) # Use raw for display
#         sentiment_data = response.get('sentiment_data', [])
#         video_id = response.get('video_id')

#         st.markdown("---")
#         st.header(f"📊 Analysis Results for: {video_details.get('title', 'Video')}")

#         tab1, tab2, tab3 = st.tabs(["📝 Video Info", "💬 Live Chat Analysis", "📜 Summary"])

#         # --- Tab 1: Video Info ---
#         with tab1:
#             col1, col2 = st.columns([0.6, 0.4]) # Give text slightly more space

#             with col1:
#                 st.subheader("📄 Description")
#                 # Use an expander for potentially long descriptions
#                 with st.expander("Click to view description", expanded=False):
#                      st.markdown(f"> {response.get('description', 'N/A')}", unsafe_allow_html=True) # Blockquote style

#                 st.subheader("📈 Description Sentiment")
#                 sentiment_emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐", "N/A": "❓", "Error": "⚠️"}
#                 desc_sentiment = response.get('description_sentiment', 'N/A')
#                 st.markdown(f"**Overall Sentiment:** {desc_sentiment} {sentiment_emoji.get(desc_sentiment, '')}")


#             with col2:
#                 st.subheader("🖼️ Video Thumbnail")
#                 thumb_url = response.get('thumbnail_url')
#                 if thumb_url:
#                     st.image(thumb_url, use_column_width=True, caption=video_details.get('title', 'Video Thumbnail'))
#                 else:
#                     st.info("Thumbnail not available.")

#         # --- Tab 2: Live Chat Analysis ---
#         with tab2:
#             if not live_chat_messages and comments.get('total_comments', 0) == 0:
#                  st.info("No live chat messages were found or could be analyzed for this video.")
#             else:
#                 col1, col2 = st.columns([0.6, 0.4]) # Adjusted column widths

#                 with col1:
#                     st.subheader("🗨️ Live Chat Messages & Sentiment")
#                     if live_chat_messages and sentiment_data:
#                         # Ensure lists have the same length for DataFrame creation
#                         min_len = min(len(live_chat_messages), len(sentiment_data))
#                         df_data = {
#                             'Live Chat Message': live_chat_messages[:min_len],
#                             'Sentiment': sentiment_data[:min_len]
#                          }
#                         df = pd.DataFrame(df_data)
#                         # Use st.dataframe for better display control
#                         st.dataframe(df, height=450, use_container_width=True) # Slightly taller, use container width
#                     elif comments.get('total_comments', 0) > 0:
#                         st.warning("Sentiment data might be missing for some chat messages.", icon="⚠️")
#                     else:
#                          st.info("No live chat messages to display.")


#                 with col2:
#                     st.subheader("📊 Sentiment Breakdown")
#                     if comments and 'total_comments' in comments:
#                         total = comments['total_comments']
#                         positive = comments['positive_comments']
#                         negative = comments['negative_comments']
#                         neutral = total - positive - negative

#                         if total > 0:
#                              # Display Pie Chart using Plotly
#                              fig = plot_sentiment_pie_chart_plotly(positive, negative, total)
#                              st.plotly_chart(fig, use_container_width=True) # Key: use container width

#                              # # Display Metrics using st.metric for a cleaner look
#                              # st.metric(label="Total Comments Analyzed", value=f"{total}")

#                              # pos_perc = (positive / total) * 100 if total > 0 else 0
#                              # neg_perc = (negative / total) * 100 if total > 0 else 0
#                              # neu_perc = (neutral / total) * 100 if total > 0 else 0 # Calculate neutral percentage

#                              # st.metric(label="😊 Positive", value=f"{positive}", delta=f"{pos_perc:.1f}%")
#                              # st.metric(label="😠 Negative", value=f"{negative}", delta=f"{neg_perc:.1f}%")
#                              # st.metric(label="😐 Neutral", value=f"{neutral}", delta=f"{neu_perc:.1f}%")

#                         else:
#                             st.info("No comments were analyzed.")
#                     else:
#                          st.warning("Comment statistics not available.")

#                 # --- Top Comments Display (Moved below columns but within Tab 2) ---
#                 st.markdown("---") # Separator
#                 st.subheader("⭐ Top Comments")
#                 if comments and comments['total_comments'] > 0:
#                     # Using st.expander instead of checkbox for a cleaner look
#                     with st.expander("Show Top Positive & Negative Comments", expanded=False):
#                         col_pos, col_neg = st.columns(2)
#                         with col_pos:
#                             st.markdown(f"<h5 style='color: #28a745;'>👍 Top Positive Comments:</h5>", unsafe_allow_html=True)
#                             if comments['positive_comments_list']:
#                                 for comment in comments['positive_comments_list']:
#                                     st.markdown(f"<div style='background-color: #121f12; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #28a745;'>{comment}</div>", unsafe_allow_html=True)
#                             else:
#                                 st.caption("No positive comments found.")

#                         with col_neg:
#                             st.markdown(f"<h5 style='color: #dc3545;'>👎 Top Negative Comments:</h5>", unsafe_allow_html=True)
#                             if comments['negative_comments_list']:
#                                 for comment in comments['negative_comments_list']:
#                                     st.markdown(f"<div style='background-color: #121f12; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #dc3545;'>{comment}</div>", unsafe_allow_html=True)
#                             else:
#                                  st.caption("No negative comments found.")
#                 else:
#                     st.caption("No comments available to display top examples.")


#         # --- Tab 3: Summary ---
#         with tab3:
#             st.subheader("✍️ Video Summary (via Gemini AI)")
#             st.markdown("Please try again for 2-3 times until the summary works")
#             # Button to trigger summary generation
#             summary_key = f"summary_{video_id}_{idx}" # Unique key per video/response
#             summary_button_label = "📜 Generate Summary"
#             if 'transcript_summary' in response:
#                 summary_button_label = "🔄 Regenerate Summary" # Change label if summary exists

#             if st.button(summary_button_label, key=summary_key):
#                 with st.spinner("🔄 Generating summary with Gemini AI... This might take a few seconds."):
#                     try:
#                         transcript = get_sub(video_id) # Fetch transcript
#                         if transcript:
#                             summary = get_gemini_response_with_retry(transcript) # Call Gemini
#                             if summary:
#                                 # Update the specific response in session state
#                                 st.session_state.responses[idx]['transcript_summary'] = summary
#                                 st.rerun() # Rerun to display the new summary immediately
#                             else:
#                                 # Error handled within get_gemini_response_with_retry
#                                 pass
#                         else:
#                             # Error handled within get_sub
#                             st.warning("Could not generate summary because the transcript is unavailable.", icon="⚠️")
#                     except Exception as e:
#                         st.error(f"An unexpected error occurred during summary generation: {e}", icon="🚨")
#                         logging.error(f"Summary generation error for {video_id}: {e}", exc_info=True)

#             # Display generated summary if it exists in the response
#             if 'transcript_summary' in response:
#                 st.markdown(f"<div style='background-color: #121f12; padding: 15px; border-radius: 8px; border-left: 5px solid #0d6efd;'>{response['transcript_summary']}</div>", unsafe_allow_html=True)
#             else:
#                 st.info("Click 'Generate Summary' to create a summary of the video transcript using AI.")

# # Optional: Add a footer
# st.markdown("---")
# st.caption("YOUTUBE LIVECHAT SENTIMENT | TRAN THU HIEN | Summary by Google Gemini")
