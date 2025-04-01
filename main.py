import re
import json
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yt_dlp
from googleapiclient.discovery import build
import logging
# import matplotlib.pyplot as plt # No longer needed for pie chart
import numpy as np
# Import specific exceptions from youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go # Import Plotly
# from plotly.subplots import make_subplots # Not used, can remove if desired
import time

# --- Configuration ---
# Your API Key - should be stored securely, not hardcoded
# Consider using Streamlit secrets or environment variables
# Example: API_KEY = st.secrets["YOUTUBE_API_KEY"]
API_KEY = "YOUR_YOUTUBE_API_KEY"  # <---- REPLACE WITH YOUR YOUTUBE API KEY
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY" # <---- REPLACE WITH YOUR GEMINI API KEY

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, # Set to INFO for more details if needed
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# Configure Gemini API
try:
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GEMINI_API_KEY": # Check if key is provided and not the placeholder
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("Google Gemini API configured successfully.")
    else:
        st.warning("Gemini API key is missing or is the placeholder. Summary generation will be disabled.", icon="🔑")
        logging.warning("Gemini API key not provided or is placeholder.")
        GOOGLE_API_KEY = None # Set to None to easily check later

except Exception as e:
    st.error(f"Failed to configure Google Gemini API. Please check your API key. Error: {e}", icon="🔑")
    logging.error(f"Gemini API configuration error: {e}")
    GOOGLE_API_KEY = None # Ensure it's None on failure


# Model Configuration
MODEL_PATH = ""  # Set this to the directory if you have a folder for the weights, otherwise it would be ""
MODEL_FILE = "sentiment_classifier (1).pth" # Ensure this file exists at the expected location


# --- Model Loading ---
@st.cache_resource # Cache the loaded model and tokenizer
def load_model():
    # Using a spinner here is good for the initial load if it's slow
    with st.spinner("⏳ Loading Sentiment Analysis Model... This might take a moment on first run."):
        model_path = os.path.join(MODEL_PATH, MODEL_FILE)
        model_id = "wonrax/phobert-base-vietnamese-sentiment"
        try:
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found at '{model_path}'. Please ensure the path and filename are correct. You might need to download it first.")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            # Load the state dict; ensure strict=False if loading a partial model or fine-tuned head
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            print(f"✅ Sentiment model loaded successfully from {model_path} to {device}") # Keep print for console feedback
            logging.info(f"Sentiment model loaded successfully from {model_path} to {device}")
            return tokenizer, model
        except FileNotFoundError as fnf_error:
            st.error(f"🚨 Fatal Error: {fnf_error}", icon="🔥")
            logging.error(f"Model loading error: {fnf_error}")
            st.stop() # Stop execution if model file is missing
        except Exception as e:
            st.error(f"🚨 Fatal Error: Could not load sentiment analysis model. Check model files and dependencies. Error: {e}", icon="🔥")
            logging.exception("Unhandled error during model loading:") # Log full traceback
            st.stop() # Stop execution if model fails to load
            return None, None

# --- Core Functions ---
def analyze_sentiment(text):
    """Analyzes the sentiment of a single text string."""
    tokenizer, model = load_model() # Get cached model
    if tokenizer is None or model is None:
        # Error already shown in load_model, maybe a simpler message here or just return error
        # st.warning("Sentiment analysis model not available.", icon="⚠️")
        return "Error", [0, 0, 0] # Return default/error state

    if not text or not isinstance(text, str) or not text.strip():
         # logging.debug("analyze_sentiment called with empty or invalid input.") # Optional debug log
         return "Neutral", [0, 1, 0] # Handle empty or invalid input gracefully

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure model is on the correct device (might be redundant with caching but safe)
    # model.to(device) # This might be slightly inefficient if called repeatedly, rely on cache_resource
    tokenizer.padding_side = "left" # Set padding side

    try:
        # Reduce max_length if hitting memory limits, but 256 is usually reasonable
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(device) # Pad to max_length
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply softmax to get probabilities
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_class = np.argmax(predictions) # Get index of max probability
        sentiment_label = sentiment_labels[predicted_class]

        return sentiment_label, predictions.tolist() # Return list for consistency
    except Exception as e:
        logging.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}")
        # Don't show error to user for every failed analysis, return default
        return "Error", [0, 0, 0]

def preprocess_model_input_str(text, video_title=""):
    """Cleans text for sentiment analysis."""
    if not text or not isinstance(text, str):
        return ""
    # Improved regex: handles more URL variations, repeated punctuation/symbols, newlines, hashtags, mentions, RTs
    # Group 1: Mentions, Hashtags, RT, URLs
    # Group 2: Multiple newlines
    # Group 3: Repeated punctuation/symbols (2 or more)
    regex_pattern = r'(@\w+|#\w+|RT\s|https?://\S+|www\.\S+)|(\n+)|([.,!?;*&^%$#@\"<>(){}\[\]\\/\|~`\-=_+]{2,})'
    # First, remove URLs, hashtags, mentions, RTs (replace with nothing)
    # Then replace multiple newlines and repeated symbols with a single space
    clean_str = re.sub(regex_pattern, lambda m: "" if m.group(1) else " " if m.group(2) else " " if m.group(3) else m.group(0), text)
    # Remove video title if present (case-insensitive) - do this *after* other cleaning
    if video_title:
        clean_str = re.sub(re.escape(video_title), "", clean_str, flags=re.IGNORECASE)
    # Normalize whitespace (replace multiple spaces with one) and strip leading/trailing space
    clean_str = re.sub(r"\s{2,}", " ", clean_str).strip()
    return clean_str


def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    if not url or not isinstance(url, str):
        logging.warning("extract_video_id called with invalid URL.")
        return None
    # Comprehensive regex for various YouTube URL patterns
    # Handles: youtube.com/watch?v=..., youtu.be/..., youtube.com/embed/..., youtube.com/v/...
    # Also handles URLs within playlists (e.g., youtube.com/watch?v=...&list=...)
    pattern = re.compile(r'(?:youtube(?:-nocookie)?\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)([^"&?/\s]{11})')
    match = pattern.search(url)
    if match:
        # logging.debug(f"Extracted video ID {match.group(1)} from URL {url}")
        return match.group(1)
    else:
        logging.warning(f"Could not extract video ID from URL: {url}")
        return None

def fetch_video_details(video_id, api_key):
    """Fetches video title and description using YouTube Data API."""
    if not video_id:
        logging.warning("fetch_video_details called with missing video_id.")
        return "Title Unavailable", "Description Unavailable", "Error: Missing Video ID"
    if not api_key or api_key == "YOUR_YOUTUBE_API_KEY":
        logging.error("fetch_video_details called with missing or placeholder YouTube API key.")
        return "API Key Error", "API Key Error", "Error: YouTube API Key not configured."

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response.get("items"):
            logging.warning(f"YouTube API: Video not found for ID {video_id}")
            # It's possible the video exists but is private/deleted, API returns empty items
            return "Video Not Found", "Video might be private, deleted, or ID is incorrect.", "Error: Video not found via API."

        snippet = response["items"][0].get("snippet", {})
        title = snippet.get("title", "Title Unavailable")
        description = snippet.get("description", "No description provided.")
        logging.info(f"Fetched details for video ID {video_id}: Title='{title[:30]}...'")
        return title, description, None # Return None for error if successful

    except Exception as e:
        # Check for common API errors (e.g., quota exceeded)
        error_reason = "Unknown API Error"
        if hasattr(e, 'resp') and hasattr(e.resp, 'status'):
             if e.resp.status == 403:
                 error_reason = "API Quota Exceeded or Key Invalid"
             elif e.resp.status == 404:
                  error_reason = "Video Not Found (API 404)" # Should be caught by empty items check usually
             else:
                 error_reason = f"API HTTP Error {e.resp.status}"
        st.warning(f"⚠️ Could not fetch video details from YouTube API ({error_reason}). Check API key, quotas, and video ID. Error: {e}", icon="📡")
        logging.error(f"Error fetching video details for {video_id} ({error_reason}): {e}", exc_info=True)
        return "API Error", "API Error", f"Error: {error_reason}"

def download_live_chat(video_url, video_id):
    """Downloads live chat replay JSON file using yt-dlp. Returns filename or None."""
    ydl_opts = {
        'writesubtitles': True,
        'subtitleslangs': ['live_chat'], # Specify live_chat language
        'skip_download': True, # Don't download the video
        'outtmpl': f'{video_id}.%(ext)s', # Use extension placeholder - yt-dlp adds .live_chat.json
        'quiet': True, # Suppress yt-dlp console output
        'no_warnings': True,
        'ignoreerrors': True, # Continue on download errors (like no chat found) - essential!
        'socket_timeout': 30, # Set a timeout for network operations (seconds)
        'retries': 3, # Retry downloads a few times
        'fragment_retries': 3,
    }
    # Define the expected output filename based on yt-dlp's default for live_chat
    subtitle_file = f"{video_id}.live_chat.json"

    # --- Clean up any old file first ---
    if os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
            logging.info(f"Removed existing chat file before download: {subtitle_file}")
        except OSError as e:
             # This is usually not critical, log as warning
             logging.warning(f"Could not remove old chat file {subtitle_file} before download: {e}")

    logging.info(f"Attempting to download live chat for {video_id} from {video_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info without downloading the video itself.
            # This process triggers the subtitle download if available.
            # download=False is crucial here.
            info_dict = ydl.extract_info(video_url, download=False)

            # --- Check if the expected subtitle file exists AFTER extract_info ---
            if os.path.exists(subtitle_file):
                logging.info(f"Live chat successfully downloaded to: {subtitle_file}")
                return subtitle_file # Return filename if successful
            else:
                # File doesn't exist, try to determine why using info_dict
                is_live = info_dict.get('is_live', False)
                # 'was_live' might be useful, but 'live_status' is often more reliable
                live_status = info_dict.get('live_status') # Can be 'is_live', 'was_live', 'is_upcoming', 'post_live' etc.

                if is_live or live_status == 'is_live':
                    st.info("This appears to be an ongoing live stream. Live chat replay is usually available only after the stream ends.", icon="🔴")
                    logging.info(f"Skipping chat download for ongoing live stream: {video_url}")
                elif live_status == 'is_upcoming':
                     st.info("This is a scheduled premiere or upcoming stream. Live chat replay will be available after it airs.", icon="📅")
                     logging.info(f"Skipping chat download for upcoming stream: {video_url}")
                # If it wasn't live and isn't live now, maybe no chat existed or it wasn't enabled/archived
                elif live_status in ['was_live', 'post_live'] or info_dict.get('was_live'):
                     # It was live, but the file wasn't created -> Chat likely not available/archived
                     st.info("Live chat replay not found. The stream ended, but chat might not have been enabled or archived by the creator.", icon="💬")
                     logging.info(f"Live chat subtitle file not found after extract_info for ended stream {video_id}. Chat likely unavailable.")
                else: # General "not found" message if status is unclear
                     st.info("Live chat replay not found or unavailable for this video (it might not have had a live chat, or it's too old).", icon="💬")
                     logging.info(f"Live chat subtitle file not found after extract_info for {video_id}. Status: {live_status}")
                return None # Indicate no file was created/found

    except yt_dlp.utils.DownloadError as e:
        error_str = str(e).lower()
        # Provide more specific user feedback based on common yt-dlp errors
        if "live chat was not found" in error_str or "requested format not available" in error_str or "no closed captions found" in error_str:
            st.info("Live chat replay not found or unavailable for this video.", icon="💬")
        elif "this live event will begin" in error_str or "premiere" in error_str:
             st.info("This is a scheduled premiere or upcoming stream. Live chat replay will be available after it airs.", icon="📅")
        elif "private video" in error_str or "login required" in error_str:
             st.warning("Cannot download chat: Video is private or requires login.", icon="🔒")
        elif "video unavailable" in error_str:
             st.warning("Cannot download chat: Video is unavailable.", icon="🚫")
        else:
            # Show a concise version of other download errors
            concise_error = str(e).split(': ERROR: ')[-1].splitlines()[0]
            st.warning(f"⚠️ Could not download live chat data: {concise_error}", icon="📡")
        logging.error(f"yt-dlp DownloadError for {video_url}: {e}") # Log the full error
        return None
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred during chat download: {e}", icon="🔥")
        logging.exception(f"Unexpected error downloading chat for {video_url}:") # Log full traceback
        return None


def parse_jsonl(file_path):
    """Parses a JSONL (JSON Lines) file. Returns list of dicts or None on error."""
    data = []
    if not os.path.exists(file_path):
        logging.error(f"JSONL file not found for parsing: {file_path}")
        # Don't show st message here, calling function should handle None return
        return None
    try:
        line_count = 0
        parsed_count = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                line = line.strip() # Remove leading/trailing whitespace
                if not line: continue # Skip empty lines
                line_count += 1
                try:
                    data.append(json.loads(line))
                    parsed_count += 1
                except json.JSONDecodeError as json_e:
                     # Log warning for corrupted lines but continue parsing others
                     logging.warning(f"Skipping invalid JSON line {line_num+1} in {file_path}: {json_e}. Line: '{line[:100]}...'")
                     continue # Skip this line
        if line_count == 0:
             logging.warning(f"JSONL file was empty or contained only whitespace: {file_path}")
        elif parsed_count == 0 and line_count > 0:
             logging.error(f"Failed to parse any valid JSON from {line_count} non-empty lines in {file_path}.")
             # Maybe the file isn't JSONL?
             st.warning("⚠️ Downloaded chat file could not be parsed (invalid format?).", icon="❓")
             return None # Indicate failure if no lines could be parsed
        else:
             logging.info(f"Parsed {parsed_count} records from {line_count} non-empty lines in {file_path}")
        return data
    except Exception as e:
        st.error(f"🚨 Error reading or processing chat file: {e}", icon="🔥")
        logging.error(f"Error opening/reading JSONL file {file_path}: {e}", exc_info=True)
        return None

def extract_live_chat_messages(subtitle_file):
    """Extracts message texts from parsed live chat data. Cleans up the file."""
    messages_raw = [] # Store the original, uncleaned messages
    if not subtitle_file or not os.path.exists(subtitle_file):
         logging.error(f"extract_live_chat_messages called with invalid or non-existent file: {subtitle_file}")
         return messages_raw # Return empty list

    raw_data = parse_jsonl(subtitle_file) # Use the improved parsing function

    if raw_data is None: # Parsing failed critically (e.g., couldn't open file, zero valid lines)
        logging.error(f"Parsing failed for subtitle file: {subtitle_file}. No messages extracted.")
        # Don't clean up the file here, might be needed for debugging if parse_jsonl failed
        return messages_raw # Return empty list
    if not raw_data: # Parsing succeeded but file was empty or had no relevant actions
        logging.warning(f"No data entries found after parsing subtitle file: {subtitle_file}")
        # Cleanup can proceed if parsing technically worked but found nothing
    else:
        message_count = 0
        processed_ids = set() # To avoid duplicates if any strange structure exists

        for entry_num, entry in enumerate(raw_data):
            try:
                # Navigate through the typical YouTube live chat JSON structure
                # Look within 'replayChatItemAction' first, then maybe others if needed
                replay_action = entry.get('replayChatItemAction', {})
                actions = replay_action.get('actions', [])
                if not actions: continue # Skip if no actions in this entry

                for action in actions:
                     # Could be 'addChatItemAction', 'addLiveChatTickerItemAction', etc.
                     # Focus on 'addChatItemAction' which contains most messages
                     add_chat_item = action.get('addChatItemAction', {})
                     item = add_chat_item.get('item', {})
                     if not item: continue # Skip if no item in action

                     message_renderer = item.get('liveChatTextMessageRenderer') # Standard text messages
                     paid_renderer = item.get('liveChatPaidMessageRenderer') # Super Chats
                     # member_renderer = item.get('liveChatMembershipItemRenderer') # Membership milestones/messages
                     # sticker_renderer = item.get('liveChatPaidStickerRenderer') # Super Stickers

                     msg_id = None
                     message_parts = None
                     prefix = ""

                     if message_renderer:
                         msg_id = message_renderer.get('id')
                         message_parts = message_renderer.get('message', {}).get('runs', [])
                         prefix = ""
                     elif paid_renderer:
                         msg_id = paid_renderer.get('id')
                         # Paid messages might have text in 'message' or just be the amount/user
                         message_parts = paid_renderer.get('message', {}).get('runs', []) # Check if text exists
                         prefix = "[Super Chat] " # Mark super chats

                     # Can add elif for member_renderer etc. if needed

                     if message_parts and (not msg_id or msg_id not in processed_ids):
                         # Concatenate text from all 'runs' (handles emojis, links within message)
                         full_message = ''.join(part.get('text', '') for part in message_parts).strip()

                         if full_message: # Only add non-empty messages
                             messages_raw.append(f"{prefix}{full_message}")
                             if msg_id: processed_ids.add(msg_id)
                             message_count += 1

            except Exception as e:
                # Log error for a specific entry but continue with others
                logging.warning(f"Error processing chat entry #{entry_num+1} in {subtitle_file}: {e} - Entry snippet: {str(entry)[:200]}", exc_info=False) # Set exc_info=True for full traceback if needed
                continue

        logging.info(f"Extracted {message_count} messages from {len(raw_data)} entries in {subtitle_file}")

    # --- Cleanup: Moved inside this function ---
    # Always try to remove the file after attempting extraction, regardless of success/failure within loop
    if os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
            logging.info(f"Cleaned up temporary chat file: {subtitle_file}")
        except OSError as e:
            # Log warning if cleanup fails, but don't crash the app
            logging.warning(f"Could not remove temporary chat file {subtitle_file} after processing: {e}")

    return messages_raw


def get_combined_video_data(video_url: str, api_key: str) -> dict:
    """Fetches video details and live chat, handling errors. Returns a dict."""
    # Initialize result structure
    result = {
        "video_id": None,
        "title": "Unavailable",
        "description": "Unavailable",
        "live_chat_raw": [],
        "error": None # Stores the *first* critical error encountered
    }
    logging.info(f"Starting data retrieval for URL: {video_url}")

    # --- 1. Get Video ID ---
    video_id = extract_video_id(video_url)
    if not video_id:
        result["error"] = "Invalid YouTube URL or could not extract video ID."
        logging.error(f"Failed to extract video ID from URL: {video_url}")
        return result # Stop early if no ID
    result["video_id"] = video_id
    logging.info(f"Extracted video ID: {video_id}")

    # --- 2. Fetch Video Title and Description ---
    # Use a spinner context for this potentially slow API call
    with st.spinner("🔗 Fetching video details from YouTube API..."):
        try:
            title, description, api_error = fetch_video_details(video_id, api_key)
            result["title"] = title
            result["description"] = description
            if api_error:
                # Store the API error, but continue to try fetching chat
                # User already warned by fetch_video_details
                result["error"] = api_error # Record the first error
                logging.warning(f"API error encountered ({api_error}), but proceeding to attempt chat download.")
        except Exception as e:
             # Catch unexpected errors during the API call itself
             logging.error(f"Unhandled exception during fetch_video_details for {video_id}: {e}", exc_info=True)
             result["error"] = "An unexpected error occurred while fetching video details."
             result["title"] = "Error"
             result["description"] = "Error"
             # Continue to attempt chat download even if details failed unexpectedly

    # --- 3. Download and Parse Live Chat ---
    # Proceed even if there was an API error fetching details, chat might still work
    subtitle_file = None # Define before try block
    try:
        # No spinner needed here, download_live_chat shows st.info/st.warning
        logging.info(f"Attempting live chat download for {video_id}...")
        subtitle_file = download_live_chat(video_url, video_id)

        if subtitle_file and os.path.exists(subtitle_file):
            # Only show parsing spinner if download was successful
            with st.spinner("Processing downloaded chat data..."):
                # Extract messages (this function now also handles cleanup on success)
                raw_messages = extract_live_chat_messages(subtitle_file)
                result["live_chat_raw"] = raw_messages
                logging.info(f"Successfully extracted {len(raw_messages)} raw chat messages for {video_id}.")
                # File should be cleaned up by extract_live_chat_messages now

        elif subtitle_file and not os.path.exists(subtitle_file):
             # This case should ideally not happen if download_live_chat checks existence
             logging.error(f"Logic Error: download_live_chat returned file '{subtitle_file}' but it doesn't exist.")

        # If subtitle_file is None, download_live_chat already informed user/logged why
        elif subtitle_file is None:
             logging.info(f"Live chat download did not produce a file for {video_id}.")


    except Exception as e:
         # Catch unexpected errors during the download/extraction process
         logging.error(f"Unhandled exception during chat download/processing for {video_id}: {e}", exc_info=True)
         st.error(f"🚨 An unexpected error occurred while processing the live chat.", icon="🔥")
         # Record error if none exists yet
         if not result["error"]:
              result["error"] = "An unexpected error occurred during chat processing."

    # --- Failsafe Cleanup (Just in Case) ---
    # If extract_live_chat_messages failed after download OR if download succeeded
    # but something else went wrong before extraction cleanup ran.
    finally:
         if subtitle_file and os.path.exists(subtitle_file):
              try:
                  os.remove(subtitle_file)
                  logging.warning(f"Failsafe Cleanup: Removed temporary chat file {subtitle_file} as it still existed.")
              except Exception as e:
                   logging.error(f"Failsafe cleanup failed for {subtitle_file}: {e}")

    # --- Final Logging ---
    log_msg = f"Finished data retrieval for {video_id}. "
    log_msg += f"Title: {'OK' if result['title'] not in ['Unavailable', 'API Error', 'Error'] else 'Failed'}. "
    log_msg += f"Chat messages: {len(result['live_chat_raw'])}. "
    if result['error']:
        log_msg += f"Error Recorded: {result['error']}"
    logging.info(log_msg)

    return result


def get_top_comments(live_chat_raw, sentiment_labels, top_n=3):
    """Selects top N positive and negative RAW comments based on calculated sentiment labels."""
    positive_comments = []
    negative_comments = []

    # Ensure we don't go out of bounds if analysis failed for some comments
    # sentiment_labels corresponds to the successfully analyzed *cleaned* comments.
    # live_chat_raw contains *all* extracted raw comments.
    # We assume the first N raw comments correspond to the N sentiment labels.
    num_analyzed = len(sentiment_labels)
    if num_analyzed == 0:
        return [], [] # No sentiments, no top comments

    logging.debug(f"Selecting top comments from {len(live_chat_raw)} raw messages with {num_analyzed} sentiment labels.")

    # Iterate only up to the number of available sentiment labels
    for i in range(num_analyzed):
        # Get the ORIGINAL raw comment corresponding to this sentiment label
        if i < len(live_chat_raw): # Safety check
            comment = live_chat_raw[i]
            sentiment = sentiment_labels[i]
            if sentiment == "Positive":
                positive_comments.append(comment)
            elif sentiment == "Negative":
                negative_comments.append(comment)
        else:
             # This shouldn't happen if logic is correct, but log if it does
             logging.warning(f"Mismatch between sentiment label count ({num_analyzed}) and raw chat count ({len(live_chat_raw)}) at index {i}.")


    # Return only top N, even if fewer are found
    return positive_comments[:top_n], negative_comments[:top_n]

# --- Plotly Pie Chart Function ---
def plot_sentiment_pie_chart_plotly(positive_count, negative_count, total_comments):
    """Generates an interactive Plotly pie chart for sentiment distribution."""
    if total_comments <= 0: # Check for zero or negative total
        logging.info("plot_sentiment_pie_chart_plotly called with zero or negative total comments.")
        fig = go.Figure()
        fig.update_layout(
            # title_text='No Comments Analyzed', # Title handled by st.subheader
            # title_x=0.5,
            annotations=[dict(text='No Analyzed Comments', showarrow=False, font_size=14)],
             paper_bgcolor='rgba(0,0,0,0)', # Transparent background
             plot_bgcolor='rgba(0,0,0,0)', # Transparent plot area
             height=300 # Give it some height even when empty
             )
        return fig

    # Calculate neutral count safely
    neutral_count = max(0, total_comments - (positive_count + negative_count))
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_count, negative_count, neutral_count]
    # Define colors (consider colorblind accessibility if needed)
    colors = ['#28a745', '#dc3545', '#6c757d'] # Green, Red, Gray (Bootstrap colors)

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,
                                marker_colors=colors,
                                # Pull slices slightly out if they have value > 0
                                pull=[0.05 if v > 0 else 0 for v in values],
                                hole=0.35, # Makes it a donut chart
                                textinfo='percent+value', # Show percentage and raw count on slices
                                insidetextorientation='radial', # Adjust text orientation inside slices
                                hoverinfo='label+percent+value', # Info shown on hover
                                name='' # Avoid showing trace name 'trace 0' in hover
                                )])

    # Further styling
    fig.update_traces(
        textfont_size=12,
        marker=dict(line=dict(color='#FFFFFF', width=1.5)) # White outline for slices
    )
    fig.update_layout(
        # title_text='Live Chat Sentiment Distribution', # Title handled by st.subheader now
        # title_x=0.5, # Center title if using it
        margin=dict(l=10, r=10, t=30, b=40), # Adjust margins (more top margin)
        legend_title_text='Sentiments',
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom", y=-0.15, # Position below chart
            xanchor="center", x=0.5
            ),
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot area
        height=350, # Consistent height
        # font_color="white" # Uncomment if using Streamlit dark theme explicitly and colors don't contrast
    )
    logging.info(f"Generated sentiment pie chart: Pos={positive_count}, Neg={negative_count}, Neu={neutral_count}, Total={total_comments}")
    return fig

# --- Transcript and Summary Functions ---
def get_sub(video_id):
    """Retrieves transcript text, preferring Vietnamese, falling back to English."""
    if not video_id:
        logging.warning("get_sub called with no video_id.")
        return None, "Error: Missing Video ID"
    logging.info(f"Attempting to retrieve transcript for video ID: {video_id}")
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_data = None
        chosen_lang = None
        languages_to_try = ['vi', 'en'] # Prioritize Vietnamese

        # Log available transcripts for debugging
        try:
            available_langs = [t.language for t in transcript_list]
            logging.info(f"Available transcript languages for {video_id}: {available_langs}")
        except Exception as log_e:
             logging.warning(f"Could not log available languages for {video_id}: {log_e}")


        for lang in languages_to_try:
             try:
                 # Try finding a manually created or generated transcript in the language
                 transcript = transcript_list.find_transcript([lang])
                 # Check if it's translatable (useful if we only find auto-generated English)
                 # if transcript.is_translatable and lang != 'vi':
                 #     transcript = transcript.translate('vi') # Example: Auto-translate to Vietnamese
                 #     chosen_lang = 'vi (auto-translated from en)'
                 # else:
                 transcript_data = transcript.fetch()
                 chosen_lang = lang
                 logging.info(f"Found transcript in '{chosen_lang}' for {video_id}.")
                 if lang != languages_to_try[0]: # Inform if fallback language was used
                     st.info(f"ℹ️ Vietnamese transcript not found, using '{lang}' transcript for summary.", icon="🌐")
                 break # Stop searching once found
             except NoTranscriptFound:
                 logging.info(f"No transcript found for language '{lang}' for {video_id}.")
                 continue # Try next language
             except Exception as find_exc:
                  # Catch other errors during find/fetch for a specific language
                  logging.error(f"Error finding/fetching transcript for lang '{lang}' for {video_id}: {find_exc}", exc_info=True)
                  continue # Try next language

        if not transcript_data:
            st.warning(f"⚠️ No suitable transcript (Vietnamese or English) found for video ID {video_id}.", icon="📜")
            logging.warning(f"No suitable transcript found in {languages_to_try} for {video_id}.")
            return None, "Error: No suitable transcript found."

        # Concatenate text segments
        concatenated_text = ' '.join([segment['text'] for segment in transcript_data if 'text' in segment])
        if not concatenated_text.strip():
             st.warning(f"⚠️ Found transcript ('{chosen_lang}') but it appears to be empty.", icon="📜")
             logging.warning(f"Transcript for {video_id} ('{chosen_lang}') was empty after concatenating segments.")
             return None, f"Error: Transcript ('{chosen_lang}') is empty."

        logging.info(f"Successfully retrieved and concatenated transcript for {video_id} ('{chosen_lang}', length: {len(concatenated_text)}).")
        return concatenated_text, None # Return text and None for error

    except TranscriptsDisabled:
        st.warning(f"🚫 Transcripts are disabled by the uploader for video ID {video_id}.", icon="🔒")
        logging.warning(f"Transcripts disabled for video ID {video_id}.")
        return None, "Error: Transcripts disabled for this video."
    except Exception as e: # Catch any other exceptions from list_transcripts etc.
        st.error(f"🚨 Error retrieving transcript list for video ID {video_id}: {e}", icon="🔥")
        logging.exception(f"Unhandled error getting subtitles for video ID {video_id}:")
        return None, f"Error: Failed to retrieve transcript list ({e})"


# Define the prompt for the Gemini model (Consider making this configurable)
GEMINI_PROMPT = """
Bạn là một trợ lý AI chuyên tóm tắt video trên YouTube. Dựa vào bản ghi đầy đủ dưới đây,
hãy tạo một bản tóm tắt súc tích bằng tiếng Việt (khoảng 250-300 từ), nêu bật những điểm chính và ý chính của video.
Trình bày rõ ràng, mạch lạc, có thể dùng gạch đầu dòng hoặc đoạn văn nếu phù hợp. Tránh thêm lời chào hay kết luận không cần thiết.

Bản ghi:
"""

# Function to get the Gemini response, with retry logic
def get_gemini_response_with_retry(transcript_text, max_attempts=3):
    """Generates summary using Gemini API with retry logic."""
    if not transcript_text or not isinstance(transcript_text, str) or not transcript_text.strip():
        logging.warning("Attempted to generate summary from empty or invalid transcript.")
        return "Error: Cannot generate summary from empty transcript."
    if not GOOGLE_API_KEY: # Check if the key is configured (set to None if missing/placeholder)
         logging.error("Gemini summary generation skipped: API key not configured.")
         # User already warned by the button being disabled, return error string
         return "Error: Gemini API key not configured."

    full_prompt = f"{GEMINI_PROMPT}\n{transcript_text}"
    logging.info(f"Generating Gemini summary. Prompt length: {len(full_prompt)} chars.")

    for attempt in range(max_attempts):
        try:
            # Choose the appropriate model - Flash is faster and often sufficient for summarization
            # Use gemini-1.5-pro for potentially higher quality but slower/more expensive results
            model = genai.GenerativeModel("gemini-1.5-flash") # or "gemini-1.5-pro"

            # Consider adding safety settings to block harmful content if needed:
            # safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            # ]
            # response = model.generate_content(full_prompt, safety_settings=safety_settings)

            response = model.generate_content(full_prompt)

            # --- Check for blocked content or empty response ---
            # Accessing response.text directly might raise an exception if blocked.
            # Check prompt_feedback first if available.
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else 'N/A'
                 logging.error(f"Gemini response blocked. Reason: {block_reason}. Ratings: {safety_ratings}")
                 # Return a user-friendly error
                 return f"Error: Content generation blocked due to safety settings (Reason: {block_reason}). Please try modifying the content or contact support."

            # Check if response.parts is empty (another way content might be missing)
            if not response.parts:
                 logging.error("Gemini response has no parts (empty response). Feedback: %s", response.prompt_feedback)
                 return "Error: Received an empty response from the AI. The content might be blocked or the model failed to generate."

            # If not blocked and parts exist, try accessing the text
            summary_text = response.text # This should now be safer

            if not summary_text or not summary_text.strip():
                 logging.warning(f"Gemini generated an empty or whitespace-only summary (Attempt {attempt+1}).")
                 # Don't retry for empty generation, return error
                 return "Error: AI generated an empty summary."

            logging.info(f"Gemini summary generated successfully (Attempt {attempt+1}). Length: {len(summary_text)} chars.")
            return summary_text.strip() # Return the generated text, stripped

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to generate Gemini response: {e}", exc_info=True)
            # Check for specific API errors if possible (e.g., quota, invalid key)
            error_str = str(e).lower()
            if "api key not valid" in error_str:
                 st.error("🚨 Gemini Error: Invalid API Key. Please check your GOOGLE_API_KEY.", icon="🔑")
                 return "Error: Invalid Gemini API Key." # Stop retrying if key is bad
            elif "quota" in error_str:
                 st.error("🚨 Gemini Error: API Quota Exceeded. Please check your usage limits.", icon="💰")
                 return "Error: Gemini API Quota Exceeded." # Stop retrying if quota is hit

            # General retry logic for other errors
            if attempt < max_attempts - 1:
                st.warning(f"⏳ Summary generation attempt {attempt + 1} failed. Retrying in {1.5 ** attempt:.1f}s...", icon="⚠️")
                time.sleep(1.5 ** attempt) # Exponential backoff
            else:
                st.error(f"🚨 Failed to generate summary from Gemini after {max_attempts} attempts. Please try again later. Error: {e}", icon="🔥")
                # Return None or a final error message after last attempt
                return f"Error: Failed to generate summary after {max_attempts} attempts ({e})."

    return "Error: Failed to generate summary after multiple attempts." # Should ideally be caught earlier

# --- Streamlit App UI ---
st.set_page_config(page_title="🎥 YouTube Video Analysis", layout="wide", initial_sidebar_state="auto") # Sidebar can be useful
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)
st.markdown("---") # Visual separator

# Initialize session state variables if they don't exist
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None # Stores results for the *single* currently analyzed video
if 'last_youtube_link' not in st.session_state:
    st.session_state.last_youtube_link = "" # Tracks the most recently analyzed link

# --- Input Area ---
# Use columns for better layout on wider screens
input_col, button_col = st.columns([4, 1])
with input_col:
    st.subheader("🔗 Enter YouTube Video Link")
    youtube_link = st.text_input(
        "Paste the YouTube video URL here:",
        key="youtube_link_input",
        label_visibility="collapsed",
        placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example placeholder
    )
with button_col:
    st.text("") # Add space to align button vertically
    st.text("")
    analyze_button = st.button("🔍 Analyze Video", type="primary", use_container_width=True) # Primary button, full width


# --- Analyze Button and Processing Logic ---
if analyze_button:
    # Basic URL validation (simple check)
    is_valid_url_simple = youtube_link and ("youtube.com/watch?v=" in youtube_link or "youtu.be/" in youtube_link)

    if not youtube_link or not youtube_link.strip():
        st.warning("⚠️ Please enter a YouTube video link.", icon="💡")
    elif not is_valid_url_simple:
         st.warning("⚠️ Please enter a valid YouTube video URL (e.g., https://www.youtube.com/watch?v=...).", icon="🔗")
    elif youtube_link == st.session_state.last_youtube_link and st.session_state.analysis_results:
         st.info("ℹ️ Analysis for this video is already displayed below. Enter a new link to analyze another.", icon="🔄")
    elif API_KEY == "YOUR_YOUTUBE_API_KEY" or not API_KEY: # Check if placeholder or empty
         st.error("🚨 Error: YouTube Data API Key is missing or is the placeholder. Please configure it in the script.", icon="🔑")
    else:
        # --- Main Spinner for the entire analysis process ---
        main_spinner_placeholder = st.empty() # Placeholder for the spinner
        main_spinner_placeholder.info('🚀 Starting analysis... Fetching data and processing, please wait.', icon="⏳")

        st.session_state.analysis_results = None # Clear previous results for a new analysis
        st.session_state.last_youtube_link = youtube_link # Store the link being analyzed
        analysis_successful = False # Flag to track success
        response_data = {} # Initialize response data dict

        try:
            # --- 1. Get Video ID and Basic Details (Title, Description, Raw Chat) ---
            # Function now incorporates spinners for sub-tasks
            combined_data = get_combined_video_data(youtube_link, API_KEY)

            # Check for critical errors stopping analysis
            if combined_data.get("error") and "API Key Error" in combined_data["error"]:
                 # Error already shown by fetch_video_details
                 raise ValueError("Critical Error: YouTube API Key issue.")
            elif combined_data.get("error") and "Missing Video ID" in combined_data["error"]:
                 # Error already shown by extract_video_id via get_combined_video_data
                 raise ValueError("Critical Error: Could not get Video ID.")
            # Allow continuing even if other API errors occurred (e.g., quota, video not found)


            # --- Extract data (even if some parts failed) ---
            video_id = combined_data.get("video_id") # Should exist if we passed the checks above
            title = combined_data.get("title", "Unavailable")
            desc_raw = combined_data.get("description", "Unavailable")
            chat_raw = combined_data.get("live_chat_raw", [])

            # Initial population of response data
            response_data = {
                'video_id': video_id,
                'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg" if video_id else None,
                'video_details': {'title': title},
                'description': desc_raw,
                'live_chat_messages_raw': chat_raw,
                'live_chat_messages_clean_count': 0,
                'comments': {'total_comments': 0, 'positive_comments': 0, 'negative_comments': 0, 'positive_comments_list': [], 'negative_comments_list': []},
                'sentiment_data': [], # Stores sentiment labels ["Positive", "Negative", ...]
                'description_sentiment': "N/A",
                'transcript_summary': None, # Placeholder for summary
                'analysis_error': combined_data.get("error") # Store non-critical errors reported by get_combined_video_data
            }


            # --- 2. Preprocess Text Data ---
            # Only proceed if video_id exists (it should if we reached here)
            if video_id:
                with st.spinner("🧹 Cleaning text data..."):
                    desc_clean = preprocess_model_input_str(desc_raw, title)
                    # Pass title to preprocess chat too & filter empty results
                    chat_clean = [msg for msg in (preprocess_model_input_str(chat, title) for chat in chat_raw) if msg]
                    response_data['live_chat_messages_clean_count'] = len(chat_clean)
                    logging.info(f"Cleaned {len(chat_clean)} chat messages from {len(chat_raw)} raw messages.")


                # --- 3. Analyze Sentiments (Description & Chat) ---
                # Analyze Description
                if desc_clean:
                    with st.spinner("🤔 Analyzing description sentiment..."):
                        desc_sentiment_label, _ = analyze_sentiment(desc_clean)
                        response_data['description_sentiment'] = desc_sentiment_label
                        logging.info(f"Description sentiment: {desc_sentiment_label}")


                # Analyze Chat (if cleaned messages available)
                if chat_clean:
                    sentiment_results = []
                    num_to_analyze = len(chat_clean)
                    analysis_progress = st.progress(0)
                    status_text = st.empty()

                    # Determine a reasonable step for progress updates to avoid overwhelming Streamlit
                    progress_step = max(1, num_to_analyze // 20) # Update progress roughly 20 times

                    with st.spinner(f"📊 Analyzing sentiment for {num_to_analyze} chat messages..."):

                        for i, chat in enumerate(chat_clean):
                            sentiment_label, _ = analyze_sentiment(chat)
                            # Only append if analysis didn't return "Error"
                            if sentiment_label != "Error":
                                sentiment_results.append(sentiment_label)
                            # Else: Skip adding 'Error' to the list used for stats/top comments

                            # Update progress bar and status text periodically
                            if (i + 1) % progress_step == 0 or (i + 1) == num_to_analyze:
                                progress_percentage = (i + 1) / num_to_analyze
                                analysis_progress.progress(progress_percentage)
                                status_text.text(f"📊 Analyzing chat sentiment... {i+1}/{num_to_analyze} ({progress_percentage:.0%})")

                        analysis_progress.empty() # Remove progress bar
                        status_text.empty() # Remove status text

                    response_data['sentiment_data'] = sentiment_results # Store only non-error results
                    total_analyzed = len(sentiment_results) # Count based on successful analyses

                    # Calculate counts based on the valid sentiment results
                    positive_count = sum(1 for s in sentiment_results if s == "Positive")
                    negative_count = sum(1 for s in sentiment_results if s == "Negative")

                    response_data['comments']['positive_comments'] = positive_count
                    response_data['comments']['negative_comments'] = negative_count
                    response_data['comments']['total_comments'] = total_analyzed # Total *successfully* analyzed

                    logging.info(f"Chat sentiment analysis complete: Analyzed={total_analyzed}, Pos={positive_count}, Neg={negative_count}. (Started with {num_to_analyze} cleaned messages)")


                # --- 4. Get Top Comments (using raw chat messages aligned with sentiment results) ---
                if response_data['comments']['total_comments'] > 0:
                     # Pass the raw messages and the *successful* sentiment labels
                     pos_comments, neg_comments = get_top_comments(
                         response_data['live_chat_messages_raw'], # Original raw messages
                         response_data['sentiment_data']          # Labels from successful analyses
                     )
                     response_data['comments']['positive_comments_list'] = pos_comments
                     response_data['comments']['negative_comments_list'] = neg_comments
                     logging.info(f"Selected top comments: {len(pos_comments)} positive, {len(neg_comments)} negative.")


                # --- 5. Store final results in session state ---
                st.session_state.analysis_results = response_data
                analysis_successful = True # Mark as successful

            else: # Should not happen if URL validation and ID extraction work
                st.error("🚨 Analysis stopped: Could not obtain Video ID.", icon="🔥")


        except ValueError as ve: # Catch specific critical errors raised earlier
            st.error(f"🚨 Analysis Halted: {ve}", icon="🛑")
            logging.error(f"Analysis ValueError for {youtube_link}: {ve}")
            st.session_state.analysis_results = None # Ensure no partial results displayed
            st.session_state.last_youtube_link = "" # Reset last link

        except Exception as e:
            st.error(f"🚨 An unexpected error occurred during the analysis pipeline: {e}", icon="🔥")
            logging.exception(f"Analysis pipeline error for {youtube_link}:") # Log full traceback
            st.session_state.analysis_results = None # Clear potentially incomplete results
            st.session_state.last_youtube_link = ""

        finally:
             main_spinner_placeholder.empty() # Remove the main spinner regardless of outcome
             if analysis_successful:
                st.success("✅ Analysis complete! Results are shown below.", icon="🎉")
                st.balloons()
             elif st.session_state.analysis_results is None: # Check if results were cleared due to error
                 st.error("❌ Analysis could not be completed due to errors. Check logs for details.", icon="💔")


# --- Display Results Area ---
if not st.session_state.analysis_results:
    # Display initial message with example only if no analysis has been run yet
    if not st.session_state.last_youtube_link:
        st.info("Enter a YouTube video link above and click 'Analyze Video' to see the results.\n e.g: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    # If last_youtube_link exists but results are None, it means analysis failed, error shown above.
else:
    # Display results for the latest analysis
    response = st.session_state.analysis_results # Get the single stored response dictionary
    idx = 0 # Index used for summary button key uniqueness

    # --- Safely get data from the response dictionary ---
    video_details = response.get('video_details', {})
    comments = response.get('comments', {}) # Contains counts and top comment lists
    live_chat_messages_raw = response.get('live_chat_messages_raw', [])
    sentiment_data = response.get('sentiment_data', []) # List of sentiment labels ["Pos", "Neg", ...]
    video_id = response.get('video_id')
    desc_raw = response.get('description', 'N/A')
    current_summary = response.get('transcript_summary') # Get summary state if previously generated
    analysis_error = response.get('analysis_error') # Get any non-critical errors stored during fetch

    st.markdown("---")
    st.header(f"📊 Analysis Results for: {video_details.get('title', 'Video')}")

    # Display non-critical error message if one was stored
    if analysis_error:
        st.warning(f"⚠️ Note: {analysis_error}", icon="❗")

    # --- Tabs for Results ---
    tab1, tab2, tab3 = st.tabs(["📝 Video Info", "💬 Live Chat Analysis", "📜 Summary"])

    # --- Tab 1: Video Info ---
    with tab1:
        col1, col2 = st.columns([0.6, 0.4]) # Text column wider

        with col1:
            st.subheader("📄 Description")
            # Use an expander for potentially long descriptions
            # Expand by default if description is short or contains "N/A" or "Error"
            expand_desc = len(desc_raw) < 300 or "N/A" in desc_raw or "Error" in desc_raw or "Unavailable" in desc_raw
            with st.expander("Click to view description", expanded=expand_desc):
                 # ***** FIX APPLIED HERE: Preserve line breaks visually *****
                 # Replace newline characters with HTML line breaks
                 formatted_desc = desc_raw.replace('\n', '<br>')
                 # Display raw description using markdown with blockquote style
                 st.markdown(f"<blockquote style='background-color:#f8f9fa; border-left: 5px solid #ccc; padding: 10px; margin: 10px 0;'>{formatted_desc}</blockquote>", unsafe_allow_html=True)

            st.subheader("📈 Description Sentiment")
            sentiment_emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐", "N/A": "❓", "Error": "⚠️"}
            desc_sentiment = response.get('description_sentiment', 'N/A')
            st.markdown(f"**Overall Sentiment:** {desc_sentiment} {sentiment_emoji.get(desc_sentiment, '')}")


        with col2:
            st.subheader("🖼️ Video Thumbnail")
            thumb_url = response.get('thumbnail_url')
            if thumb_url:
                st.image(thumb_url, use_column_width='auto', caption=video_details.get('title', 'Video Thumbnail'))
            else:
                st.info("Thumbnail could not be loaded (maybe invalid Video ID?).")

    # --- Tab 2: Live Chat Analysis ---
    with tab2:
        total_analyzed = comments.get('total_comments', 0) # Successfully analyzed comments
        total_raw = len(live_chat_messages_raw) # Total messages extracted
        clean_count = response.get('live_chat_messages_clean_count', 0) # Messages remaining after cleaning

        # --- Display informative messages based on chat data availability and analysis results ---
        if total_raw == 0 and not analysis_error: # Only show if no other error message explains lack of chat
             st.info("ℹ️ No live chat messages were found or downloaded for this video. (Chat may be disabled, unavailable, or the video had no chat).")
        elif total_raw > 0 and clean_count == 0:
             st.info(f"ℹ️ Found {total_raw} raw chat entries, but none contained processable text after cleaning (e.g., only emojis, links, or system messages).")
        elif total_raw > 0 and clean_count > 0 and total_analyzed == 0:
             st.warning(f"⚠️ Found {clean_count} processable text messages, but sentiment analysis failed for all of them. Check model or logs.", icon="❗")
        # Only show captions if analysis was partially or fully successful
        elif total_analyzed > 0:
            if total_analyzed < clean_count:
                 st.caption(f"ℹ️ Analyzed sentiment for **{total_analyzed}** out of {clean_count} cleaned text messages. ({clean_count - total_analyzed} messages might have caused analysis errors).")
            elif total_raw > clean_count : # If some raw messages were filtered during cleaning
                 st.caption(f"ℹ️ Found {total_raw} raw chat entries, processed and analyzed **{total_analyzed}** text messages.")
            else: # Raw = Clean = Analyzed
                 st.caption(f"ℹ️ Successfully analyzed sentiment for all **{total_analyzed}** processed text messages.")


        # --- Display results only if there are analyzed comments ---
        if total_analyzed > 0:
            col1, col2 = st.columns([0.6, 0.4]) # Adjust ratio: DataFrame | Chart & Metrics

            with col1:
                st.subheader("🗨️ Live Chat Messages & Sentiment")
                # Check if both raw messages and corresponding sentiment data exist
                if live_chat_messages_raw and sentiment_data:
                    # Align raw messages with sentiment results for display in DataFrame
                    # Use the length of sentiment_data as it represents successful analyses
                    display_limit = len(sentiment_data)
                    df_data = {
                        # Show the ORIGINAL raw message
                        'Live Chat Message': live_chat_messages_raw[:display_limit],
                        # Show the corresponding sentiment label
                        'Detected Sentiment': sentiment_data # Already has the correct length
                     }
                    df = pd.DataFrame(df_data)
                    # Use container width, set reasonable height, hide default index
                    st.dataframe(df, height=400, use_container_width=True, hide_index=True)
                # This 'else' shouldn't be reachable if total_analyzed > 0, but as a fallback:
                # else:
                #    st.warning("Data inconsistency: Analysis count > 0 but data missing for DataFrame.")

            with col2:
                st.subheader("📊 Sentiment Breakdown")
                if comments: # Check if comments dict exists
                    positive = comments.get('positive_comments', 0)
                    negative = comments.get('negative_comments', 0)
                    # Calculate neutral based on total *analyzed*
                    neutral = max(0, total_analyzed - positive - negative)

                    # Display Pie Chart using Plotly
                    fig = plot_sentiment_pie_chart_plotly(positive, negative, total_analyzed)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display Metrics using st.metric for a clean look
                    st.metric(label="Total Comments Analyzed", value=f"{total_analyzed}")

                    # Calculate percentages safely (avoid division by zero)
                    pos_perc = (positive / total_analyzed) * 100 if total_analyzed > 0 else 0
                    neg_perc = (negative / total_analyzed) * 100 if total_analyzed > 0 else 0
                    neu_perc = (neutral / total_analyzed) * 100 if total_analyzed > 0 else 0

                    # Use columns for metrics for better spacing on wide layouts
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        # Use delta_color to indicate positive/negative nature - green is default "normal"
                        st.metric(label="😊 Positive", value=f"{positive}", delta=f"{pos_perc:.1f}%", delta_color="normal")
                    with m_col2:
                        # "inverse" typically maps to red
                        st.metric(label="😠 Negative", value=f"{negative}", delta=f"{neg_perc:.1f}%", delta_color="inverse")
                    with m_col3:
                        # "off" shows delta without color
                        st.metric(label="😐 Neutral", value=f"{neutral}", delta=f"{neu_perc:.1f}%", delta_color="off")

                else: # Should not happen if total_analyzed > 0
                    st.info("Sentiment statistics unavailable.")

            # --- Top Comments Display (Moved below columns but within Tab 2) ---
            st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space
            st.subheader("⭐ Top Comments Examples")
            if comments: # Check comments dict again
                # Use expander to keep the UI cleaner initially
                with st.expander("Show Top Positive & Negative Comments", expanded=False):
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown(f"<h5 style='color: #28a745; font-weight: bold;'>👍 Top Positive ({len(comments.get('positive_comments_list', []))} shown):</h5>", unsafe_allow_html=True)
                        pos_list = comments.get('positive_comments_list', [])
                        if pos_list:
                            for i, comment in enumerate(pos_list):
                                # Basic escaping happens when st.markdown renders the raw comment string
                                # Then embed in styled div. Added unique key for potential updates.
                                st.markdown(f"<div key='pos-{i}' style='background-color: #e9f7ef; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid #28a745; color: black; font-size: 0.9em;'>{comment}</div>", unsafe_allow_html=True)
                        else:
                            st.caption("No positive comments found among analyzed messages.")

                    with col_neg:
                        st.markdown(f"<h5 style='color: #dc3545; font-weight: bold;'>👎 Top Negative ({len(comments.get('negative_comments_list', []))} shown):</h5>", unsafe_allow_html=True)
                        neg_list = comments.get('negative_comments_list', [])
                        if neg_list:
                            for i, comment in enumerate(neg_list):
                                st.markdown(f"<div key='neg-{i}' style='background-color: #fdeded; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid #dc3545; color: black; font-size: 0.9em;'>{comment}</div>", unsafe_allow_html=True)
                        else:
                             st.caption("No negative comments found among analyzed messages.")
            # else: No analyzed comments, message handled earlier

        # --- Message if no analysis results to show in this tab ---
        elif total_raw > 0 and total_analyzed == 0: # Handled by warning/info above
             pass # Avoid redundant messages
        elif total_raw == 0 and not analysis_error: # Handled by info above
             pass


    # --- Tab 3: Summary ---
    with tab3:
        st.subheader("✍️ Video Summary (via Gemini AI)")

        summary_key = f"summary_{video_id}_{idx}" # Unique key per video/analysis instance
        summary_button_label = "📜 Generate Summary"
        if current_summary and "Error:" not in current_summary: # Change label if summary exists and is not an error
            summary_button_label = "🔄 Regenerate Summary"

        # Disable button if Gemini API key is missing/placeholder
        disable_summary_button = not GOOGLE_API_KEY

        # Button to trigger summary generation
        if st.button(summary_button_label, key=summary_key, type="secondary", disabled=disable_summary_button):
            with st.spinner("🔄 Fetching transcript and generating summary with Gemini AI... Please wait."):
                summary_result = None # Reset summary variable
                summary_error = None # Reset error variable
                try:
                    # Fetch transcript (returns text, error_message)
                    transcript, transcript_error = get_sub(video_id)

                    if transcript_error:
                         # Show error from get_sub (already logged by the function)
                         st.error(f"🚨 Transcript Error: {transcript_error}", icon="📜")
                    elif transcript:
                        # If transcript fetched successfully, call Gemini
                        summary_result = get_gemini_response_with_retry(transcript)
                        if summary_result and "Error:" in summary_result:
                             # Handle errors returned explicitly by Gemini function
                             summary_error = summary_result # Store the error message
                             st.error(f"⚠️ Summary Generation Failed: {summary_error}", icon="🤖")
                             summary_result = None # Ensure no error text is displayed as summary
                        elif not summary_result:
                             # Handle case where Gemini function returned None (e.g., max retries failed)
                             summary_error = "Summary generation failed after multiple attempts."
                             # Error message already shown by get_gemini_response_with_retry
                        else:
                            # SUCCESS! Update the specific response in session state
                            st.session_state.analysis_results['transcript_summary'] = summary_result
                            # Don't rerun immediately, let the display logic below handle it
                            # st.rerun() # Rerun can sometimes cause issues with spinner/state
                            current_summary = summary_result # Update local variable for immediate display

                    # else: transcript was None but no transcript_error (shouldn't happen with new get_sub)
                    #      Or get_sub already showed error via st.warning/st.error

                except Exception as e:
                    summary_error = f"An unexpected error occurred: {e}"
                    st.error(f"🚨 An unexpected error occurred during summary generation: {e}", icon="🔥")
                    logging.exception(f"Summary generation pipeline error for {video_id}:")

                # Update session state with error if one occurred during generation attempt
                if summary_error:
                     st.session_state.analysis_results['transcript_summary'] = summary_error


        # Display informative message if button is disabled
        if disable_summary_button:
             st.warning("⚠️ Summary generation is disabled because the Google Gemini API key is not configured or is the placeholder value.", icon="🔑")

        # --- Display generated summary (or error) ---
        # Check the potentially updated current_summary or the value from session state
        summary_to_display = current_summary or st.session_state.analysis_results.get('transcript_summary')

        if summary_to_display:
            if "Error:" in summary_to_display: # Check for explicit error messages stored
                 # Display stored error messages clearly
                 st.warning(f"ℹ️ {summary_to_display}", icon="🚫") # Use warning/info for stored errors
            else:
                 # ***** FIX APPLIED HERE: Preserve line breaks visually *****
                 formatted_summary = summary_to_display.replace('\n', '<br>')
                 # Added styling for better readability
                 st.markdown(f"<div style='background-color: #eaf4ff; padding: 15px; border-radius: 8px; border-left: 5px solid #0d6efd; color: black; font-family: sans-serif; line-height: 1.6;'>{formatted_summary}</div>", unsafe_allow_html=True)
        elif not disable_summary_button: # Only show prompt if button is enabled and no summary/error exists yet
            st.info("Click 'Generate Summary' to create a summary of the video transcript using AI (requires transcript availability).")

# --- Footer ---
st.markdown("---")
st.caption("YouTube Analysis App | Sentiment powered by `wonrax/phobert-base-vietnamese-sentiment` | Summary by Google Gemini | Live Chat via `yt-dlp`")
# import re
# import json
# import os
# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import yt_dlp
# from googleapiclient.discovery import build
# import logging
# import matplotlib.pyplot as plt
# import numpy as np
# from youtube_transcript_api import YouTubeTranscriptApi
# import pandas as pd
# import google.generativeai as genai  # Import the Gemini library

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
#     model_path = os.path.join(MODEL_PATH, MODEL_FILE)  # Full path to the .pth file
#     # tokenizer_path = MODEL_PATH  # Tokenizer usually saved in the same directory as the model
#     model_id = "wonrax/phobert-base-vietnamese-sentiment"
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         # **Important:** Replace with the correct model class if needed
#         model = AutoModelForSequenceClassification.from_pretrained(model_id) #Or RobertaForSequenceClassification

#         # Load the state dictionary from the saved .pth file
#         # Try strict=False *only* if you understand the implications
#         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

#         # Move model to GPU if available
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model.to(device)
#         print(f"Model loaded successfully from {model_path} and moved to {device}")
#         return tokenizer, model
#     except Exception as e:
#         st.error(f"Error loading model from {model_path}: {e}")
#         logging.error(f"Error loading model from {model_path}: {e}")
#         return None, None  # Return None values to indicate loading failure


# def analyze_sentiment(text):
#     tokenizer, model = load_model()
#     if tokenizer is None or model is None:
#         st.error("Model loading failed. Sentiment analysis is unavailable.")
#         return "Error", [0, 0, 0]  # Return a default sentiment and scores

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     tokenizer.padding_side = "left"
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]  # Move to CPU and convert to NumPy array

#     sentiment_labels = ["Negative", "Neutral", "Positive"]
#     predicted_class = np.argmax(predictions)  # Get index of max value
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

# # Helper function to fetch video description from YouTube API
# def fetch_video_description(video_id, api_key):
#     try:
#         youtube = build("youtube", "v3", developerKey=api_key)
#         response = youtube.videos().list(
#             part="snippet",
#             id=video_id
#         ).execute()

#         if not response["items"]:
#             return None  # Video not found
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
#     }
#     subtitle_file = f"{video_id}.live_chat.json"
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.extract_info(video_url, download=True)  # Download the live chat
#         return subtitle_file  # Return the filename to be parsed
#     except Exception as e:
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
#         return messages  # Return empty list if no file or file doesn't exist

#     data = parse_jsonl(subtitle_file)  # Use the parsing function
#     if not data:
#         return messages  # Return empty list if parsing failed

#     for lc in data:
#         try:
#             lc_actions = lc.get('replayChatItemAction', {}).get('actions', [])
#             for act in lc_actions:
#                 live_chat = act.get('addChatItemAction', {}).get('item', {}).get('liveChatTextMessageRenderer', None)
#                 if live_chat:
#                     runs = live_chat['message']['runs']
#                     for run in runs:
#                         messages.append(run['text'])
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
#         description = ""  # Handle cases where description retrieval fails

#     # 2. Download and Parse Live Chat
#     subtitle_file = download_live_chat(video_url, video_id)
#     live_chat_messages = extract_live_chat_messages(subtitle_file)

#     # 3. Clean up the temp file
#     if subtitle_file and os.path.exists(subtitle_file):
#         try:
#             os.remove(subtitle_file)
#             logging.info(f"Deleted temporary file: {subtitle_file}")
#         except Exception as e:
#             logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")

#     # Return the data
#     return {
#         "video_id": video_id,  # Include the video_id here
#         "description": description,
#         "live_chat": live_chat_messages
#     }


# def get_desc_chat(video_url, API_KEY):
#     st.write(f"Analyzing video: {video_url}")
#     video_info = get_video_details_with_chat(video_url, API_KEY)

#     if "error" in video_info:
#         st.error(f"Error: {video_info['error']}")
#         return "", [], [], {}

#     # Extract the video_id from video_info
#     video_id = video_info.get("video_id")

#     # Use the video_id to construct the URL to fetch the title
#     try:
#         youtube = build("youtube", "v3", developerKey=API_KEY)  # Ensure API_KEY is correctly passed
#         response = youtube.videos().list(
#             part="snippet",
#             id=video_id
#         ).execute()
#         video_title = response['items'][0]['snippet']['title']
#     except Exception as e:
#         logging.error(f"Error fetching video title: {e}")
#         video_title = "Video Title Unavailable"  # Fallback title

#     video_description = video_info['description']
#     video_live_chat = video_info['live_chat']

#     clean_description = preprocess_model_input_str(video_description, video_title)
#     clean_live_chat = [preprocess_model_input_str(live_chat) for live_chat in video_live_chat]

#     return clean_description, clean_live_chat, video_title, video_info['live_chat']


# def get_top_comments(live_chat, sentiment_labels, top_n=3):
#     """
#     Selects the top N positive and negative comments based on sentiment scores.
#     Sentiment labels are passed so we don't need to analyze them multiple times.
#     """
#     positive_comments = []
#     negative_comments = []

#     for i, comment in enumerate(live_chat):
#         if sentiment_labels[i] == "Positive":
#             positive_comments.append(comment)
#         elif sentiment_labels[i] == "Negative":
#             negative_comments.append(comment)

#     return positive_comments[:top_n], negative_comments[:top_n]



# def plot_sentiment_pie_chart(positive_count, negative_count, total_comments):
#     labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
#     sizes = [positive_count, negative_count, total_comments - (positive_count + negative_count)]
#     colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
#     explode = (0.1, 0, 0)

#     fig, ax = plt.subplots()
#     ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
#     ax.axis('equal')
#     return fig



# def get_sub(video_id):
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["vi"])
#         data = []
#         for segment in transcript:
#             text = segment['text']
#             start = segment['start']
#             duration = segment['duration']
#             data.append([video_id, start, start + duration, text])

#         df = pd.DataFrame(data, columns=['video_id', 'start_time', 'end_time', 'text'])
#         concatenated_text = ' '.join(df['text'].astype(str))
#         return concatenated_text
#     except Exception as e:
#         logging.error(f"Error getting subtitles for video ID {video_id}: {e}")
#         return None

# # Define the prompt for the Gemini model
# prompt = """
# Bạn là người tóm tắt video trên Youtube. Bạn sẽ lấy văn bản ghi chép
# và tóm tắt toàn bộ video và cung cấp bản tóm tắt quan trọng theo các điểm
# trong vòng 300 từ. Vui lòng cung cấp bản tóm tắt của văn bản được đưa ra ở đây:
# """

# # Define the function to get the Gemini response
# def get_gemini_response(transcript_text):
#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the model
#         response = model.generate_content(transcript_text + prompt)
#         return response.text
#     except Exception as e:
#         logging.error(f"Error generating Gemini response: {e}")
#         return None

# # Function to create and display the sentiment analysis visualization
# def display_sentiment_visualization(video_description, video_live_chat):
#     sentiment_labels = ["Negative", "Neutral", "Positive"]

#     # Analyze comments
#     comments_results = []
#     for comment in video_live_chat:
#         sentiment_label, scores = analyze_sentiment(comment)
#         comments_results.append(
#             {
#                 "Text": comment,
#                 "Sentiment": sentiment_label,
#                 **{
#                     label: scores[i] * 100
#                     for i, label in enumerate(sentiment_labels)
#                 },
#             }
#         )

#     # Analyze description
#     sentiment_label, description_scores = analyze_sentiment(video_description)
#     description_scores = description_scores * 100

#     # Create visualization
#     fig = make_subplots(
#         rows=2, cols=1, subplot_titles=("Description Analysis", "Comments Analysis")
#     )

#     # Description visualization
#     fig.add_trace(
#         go.Bar(
#             name="Description Sentiment", x=sentiment_labels, y=description_scores
#         ),
#         row=1,
#         col=1,
#     )

#     # Comments visualization
#     for i, label in enumerate(sentiment_labels):
#         scores = [result[label] for result in comments_results]
#         fig.add_trace(
#             go.Bar(name=label, x=list(range(1, len(scores) + 1)), y=scores),
#             row=2,
#             col=1,
#         )

#     fig.update_layout(height=700, barmode="group")
#     st.plotly_chart(fig)

#     # Display results
#     st.subheader("Description Analysis")
#     st.write(
#         f"**Overall Sentiment:** {sentiment_labels[np.argmax(description_scores)]}"
#     )
#     st.write(
#         f"**Scores:** {', '.join([f'{label}: {description_scores[i]:.2f}%' for i, label in enumerate(sentiment_labels)])}"
#     )
#     st.write(f"**Text:** {video_description}")

#     st.subheader("Comments Analysis")
#     comments_df = pd.DataFrame(comments_results)
#     st.dataframe(comments_df)


# # Setup Streamlit app
# st.set_page_config(page_title="🎥 YouTube Video Sentiment and Summarization")
# st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment and Summarization 🎯</h1>", unsafe_allow_html=True)

# # Initialize session state
# if 'responses' not in st.session_state:
#     st.session_state.responses = []

# # Unique key for text input
# youtube_link = st.text_input("🔗 Enter YouTube Video Link Below:", key="youtube_link_input")

# # Clear the display when a new URL is entered
# if youtube_link and 'last_youtube_link' in st.session_state and youtube_link != st.session_state.last_youtube_link:
#     st.empty()  # Clear all elements on the page

# # Store the current YouTube link
# st.session_state.last_youtube_link = youtube_link

# # Add Submit URL button below the URL input field
# if st.button("🔍 Analyze Video"):
#     if youtube_link.strip() == "":
#         st.session_state.responses = []
#         st.write("The video link has been removed. All previous responses have been cleared.")
#     else:
#         with st.spinner('Collecting video information...'):
#             video_id = extract_video_id(youtube_link)
#             if video_id:
#                 try:
#                     clean_description, clean_live_chat, video_title, live_chat_messages = get_desc_chat(youtube_link, API_KEY)

#                     # Analyze sentiment for all live chat messages (batched)
#                     sentiment_data = []
#                     for chat in clean_live_chat:
#                         sentiment, _ = analyze_sentiment(chat)
#                         sentiment_data.append(sentiment)

#                     positive_count = sum(1 for s in sentiment_data if s == "Positive")
#                     negative_count = sum(1 for s in sentiment_data if s == "Negative")
#                     total_comments = len(sentiment_data)

#                     # Get top comments directly, passing in the sentiment labels we already calculated
#                     positive_comments, negative_comments = get_top_comments(live_chat_messages, sentiment_data)

#                     # Analyze description sentiment
#                     description_sentiment, description_scores = analyze_sentiment(clean_description)

#                     response = {
#                         'thumbnail_url': f"http://img.youtube.com/vi/{video_id}/0.jpg",
#                         'video_details': {
#                             'title': video_title,
#                             'channel_title': None,
#                             'view_count': None,
#                             'upload_date': None,
#                             'duration': None,
#                             'like_count': None,
#                             'dislike_count': None
#                         },
#                         'comments': {
#                             'total_comments': total_comments,
#                             'positive_comments': positive_count,
#                             'negative_comments': negative_count,
#                             'positive_comments_list': positive_comments,
#                             'negative_comments_list': negative_comments
#                         },
#                         "description": clean_description,
#                         "video_id": video_id,  # Store video ID
#                         "sentiment_data": sentiment_data, # Store so table can be loaded.
#                         "live_chat_messages": live_chat_messages,
#                         "description_sentiment": description_sentiment, # Store sentiment of description
#                     }
#                     st.session_state.responses.append(response)

#                 except Exception as e:
#                     st.error(f"Error: {e}")
#             else:
#                 st.error("Invalid YouTube URL")


# # Display stored responses using tabs
# for idx, response in enumerate(st.session_state.responses):
#     video_details = response.get('video_details')
#     comments = response.get('comments')
#     live_chat_messages = response.get('live_chat_messages')
#     sentiment_data = response.get('sentiment_data')

#     st.header(f"Analysis of Video #{idx+1}")

#     # Create tabs
#     tab1, tab2, tab3 = st.tabs(["Video Info", "Live Chat Analysis", "Summary"])

#     with tab1:
#         # Page 1: Video Information
#         st.markdown("<h2 style='text-align: center; color: #FF4500;'>📹 Video Title:</h2>", unsafe_allow_html=True)
#         st.markdown(f"<p style='text-align: center;'>{video_details['title']}</p>", unsafe_allow_html=True)

#         st.image(response['thumbnail_url'], use_column_width=True)

#         st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📝 Description:</h2>", unsafe_allow_html=True)
#         st.markdown(f"<p style='text-align: center;'>{response['description']}</p>", unsafe_allow_html=True)

#         st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📊 Description Sentiment:</h2>", unsafe_allow_html=True)
#         st.markdown(f"<p style='text-align: center;'>{response['description_sentiment']}</p>", unsafe_allow_html=True)

#     with tab2:
#         # Page 2: Live Chat Analysis
#         st.markdown("<h2 style='text-align: center; color: #FF4500;'>💬 Live Chat Sentiment:</h2>", unsafe_allow_html=True)
#         if live_chat_messages is not None and sentiment_data is not None:
#             df = pd.DataFrame({'Live Chat': live_chat_messages, 'Sentiment': sentiment_data})
#             st.dataframe(df)  # Use st.dataframe for a DataFrame
#         else:
#             st.write("No live chat data available.")  # Handle case where no data

#         if comments:
#             st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>💬 Total Comments:</h2>", unsafe_allow_html=True)
#             st.markdown(f"<p style='text-align: center;'>{comments['total_comments']}</p>", unsafe_allow_html=True)

#             # Plot and display pie chart for comments sentiment
#             fig = plot_sentiment_pie_chart(comments['positive_comments'], comments['negative_comments'], comments['total_comments'])
#             st.pyplot(fig)

#             st.markdown(f"<h2 style='text-align: center; color: #32CD32;'>👍 Positive Comments:</h2>", unsafe_allow_html=True)
#             st.markdown(f"<p style='text-align: center;'>{comments['positive_comments']} ({(comments['positive_comments']/comments['total_comments'])*100:.2f}%)</p>", unsafe_allow_html=True)

#             st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>👎 Negative Comments:</h2>", unsafe_allow_html=True)
#             st.markdown(f"<p style='text-align: center;'>{comments['negative_comments']} ({(comments['negative_comments']/comments['total_comments'])*100:.2f}%)</p>", unsafe_allow_html=True)

#             # Use st.session_state to maintain the state of the toggle
#             if f"show_comments_{idx}" not in st.session_state:
#                 st.session_state[f"show_comments_{idx}"] = False

#             # Add a toggle button to show/hide the top comments
#             st.session_state[f"show_comments_{idx}"] = st.checkbox("Show Top Comments", key=f"toggle_comments_{idx}", value=st.session_state[f"show_comments_{idx}"])

#             if st.session_state[f"show_comments_{idx}"]:
#                 st.markdown(f"<h2 style='text-align: center; color: #32CD32;'>👍 Top 3 Positive Comments:</h2>", unsafe_allow_html=True)
#                 for comment in comments['positive_comments_list']:
#                     st.markdown(f"<div style='background-color: #DFF0D8; padding: 10px; border-radius: 5px; color: black;'>{comment}</div>", unsafe_allow_html=True)

#                 st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>👎Top 3 Negative Comments:</h2>", unsafe_allow_html=True)
#                 for comment in comments['negative_comments_list']:
#                     st.markdown(f"<div style='background-color: #F2DEDE; padding: 10px; border-radius: 5px; color: black;'>{comment}</div>", unsafe_allow_html=True)
#         else:
#             st.write("No comment data available.") # Handle case where no comments

#     with tab3:
#         # Page 3: Summary
#         # Button to generate summary
#         if 'transcript_summary' not in response:
#             if st.button("📜 Generate Summary", key=f"summarize_{idx}"):
#                 with st.spinner("Generating summary..."):
#                     try:  # Add try-except block for more robust error handling
#                         video_id = response["video_id"]  # Get video ID from the response
#                         print(f"Attempting to retrieve transcript for video ID: {video_id}") # Debugging line
#                         transcript = get_sub(video_id)
#                         if transcript:
#                             summary = get_gemini_response(transcript)  # Call Gemini
#                             if summary:
#                                 response['transcript_summary'] = summary
#                                 st.session_state.responses[idx] = response
#                             else:
#                                 st.error("Failed to generate summary from Gemini.")
#                         else:
#                             st.error("Failed to retrieve transcript.")
#                     except Exception as e:
#                         st.error(f"An error occurred during summary generation: {e}")

#         # Display generated summary
#         if 'transcript_summary' in response:
#             st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>📜 Summary:</h2>", unsafe_allow_html=True)
#             st.markdown(f"<div style='background-color: #F0F8FF; padding: 10px; border-radius: 5px; color: black;'>{response['transcript_summary']}</div>", unsafe_allow_html=True)
#         else:
#             st.write("No summary generated yet. Click 'Generate Summary' to create one.") # Handle no summary
