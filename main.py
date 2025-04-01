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
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go # Import Plotly
# from plotly.subplots import make_subplots # Not used, can remove if desired
import time

# --- Configuration ---
# Your API Key - should be stored securely, not hardcoded
# Consider using Streamlit secrets or environment variables
# Example: API_KEY = st.secrets["YOUTUBE_API_KEY"]
API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"  # Replace with your actual YouTube Data API key
GOOGLE_API_KEY = "AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"  # Replace with your Gemini API key

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google Gemini API. Please check your API key. Error: {e}", icon="🔑")
    logging.error(f"Gemini API configuration error: {e}")
    # Optionally st.stop() here if Gemini is essential

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
                 raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the path is correct.")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            # Load the state dict; ensure strict=False if loading a partial model or fine-tuned head
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            print(f"✅ Sentiment model loaded successfully from {model_path} to {device}")
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
         return "Neutral", [0, 1, 0] # Handle empty or invalid input gracefully

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device) # Ensure model is on the correct device (might be redundant with caching)
    tokenizer.padding_side = "left" # Set padding side

    try:
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
    regex_pattern = r'(@\w+|#\w+|RT\s|https?://\S+|www\.\S+)|(\n+)|([.,!?;*&^%$#@\"<>(){}\[\]\\/\|~`\-=_+]{2,})'
    # First, remove URLs, hashtags, mentions, RTs
    clean_str = re.sub(regex_pattern, lambda m: "" if m.group(1) else " " if m.group(2) else " " if m.group(3) else m.group(0), text)
    # Remove video title if present
    if video_title:
        clean_str = clean_str.replace(video_title, "")
    # Normalize whitespace and strip
    clean_str = re.sub(r"\s{2,}", " ", clean_str).strip()
    return clean_str


def extract_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    if not url or not isinstance(url, str):
        return None
    # Comprehensive regex for various YouTube URL patterns
    pattern = re.compile(r'(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)([^"&?/\s]{11})')
    match = pattern.search(url)
    return match.group(1) if match else None

def fetch_video_details(video_id, api_key):
    """Fetches video title and description using YouTube Data API."""
    if not video_id or not api_key:
        return "Title Unavailable", "Description Unavailable"
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response.get("items"):
            return "Video Not Found", "" # Handle case where video doesn't exist
        snippet = response["items"][0].get("snippet", {})
        title = snippet.get("title", "Title Unavailable")
        description = snippet.get("description", "No description provided.")
        return title, description
    except Exception as e:
        st.warning(f"⚠️ Could not fetch video details from YouTube API: {e}", icon="📡")
        logging.error(f"Error fetching video details for {video_id}: {e}")
        return "API Error", "API Error"


def download_live_chat(video_url, video_id):
    """Downloads live chat replay JSON file using yt-dlp."""
    ydl_opts = {
        'writesubtitles': True,
        'subtitleslangs': ['live_chat'], # Specify live_chat language
        'skip_download': True, # Don't download the video
        'outtmpl': f'{video_id}', # Output filename template
        'quiet': True, # Suppress yt-dlp console output
        'no_warnings': True,
        'ignoreerrors': True, # Continue on download errors (like no chat found)
        'socket_timeout': 30, # Set a timeout for network operations
    }
    subtitle_file = f"{video_id}.live_chat.json"

    # Clean up any old file first
    if os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
        except OSError as e:
             logging.warning(f"Could not remove old chat file {subtitle_file}: {e}")


    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True) # Trigger download

            # Check if subtitles were actually downloaded
            downloaded_sub_path = ydl.prepare_filename(info_dict, outtmpl=f'{video_id}.live_chat.json')

            if not os.path.exists(downloaded_sub_path):
                 # Check common reasons for failure in info_dict if possible
                 is_live = info_dict.get('is_live', False)
                 if is_live:
                     st.info("This appears to be an ongoing live stream. Live chat replay is usually available only after the stream ends.", icon="🔴")
                     logging.info(f"Skipping chat download for ongoing live stream: {video_url}")
                 else:
                     st.info("Live chat replay not found or unavailable for this video.", icon="💬")
                     logging.info(f"Live chat subtitle file not found after download attempt for {video_id}")
                 return None # Indicate no file was created

        return subtitle_file # Return filename if successful

    except yt_dlp.utils.DownloadError as e:
        # Provide more specific feedback if possible
        if "requested format not available" in str(e).lower() or "live chat" in str(e).lower():
            st.info("Live chat replay not found or unavailable for this video.", icon="💬")
        elif "Premieres in" in str(e):
             st.info("This is a scheduled premiere. Live chat replay will be available after it airs.", icon="📅")
        else:
            st.warning(f"⚠️ Could not download live chat data: {str(e).splitlines()[-1]}", icon="📡") # Show concise error
        logging.error(f"yt-dlp DownloadError for {video_url}: {e}")
        return None
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred during chat download: {e}", icon="🔥")
        logging.exception(f"Unexpected error downloading chat for {video_url}:")
        return None

def parse_jsonl(file_path):
    """Parses a JSONL (JSON Lines) file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as json_e:
                     logging.warning(f"Skipping invalid JSON line {line_num+1} in {file_path}: {json_e}")
                     continue # Skip corrupted lines
        return data
    except FileNotFoundError:
        logging.error(f"Live chat file not found for parsing: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error opening/reading JSONL file {file_path}: {e}")
        return None

def extract_live_chat_messages(subtitle_file):
    """Extracts message texts from parsed live chat data."""
    messages = []
    raw_data = parse_jsonl(subtitle_file)
    if not raw_data:
        return messages # Return empty list if parsing failed or file was empty

    message_count = 0
    processed_ids = set() # To avoid duplicates if any strange structure exists

    for entry in raw_data:
        try:
            # Navigate through the typical YouTube live chat JSON structure
            replay_action = entry.get('replayChatItemAction', {})
            actions = replay_action.get('actions', [])
            if not actions: continue

            for action in actions:
                 add_chat_item = action.get('addChatItemAction', {})
                 item = add_chat_item.get('item', {})
                 message_renderer = item.get('liveChatTextMessageRenderer') # Most common type

                 if message_renderer:
                     msg_id = message_renderer.get('id')
                     if msg_id in processed_ids: continue # Skip duplicates

                     message_parts = message_renderer.get('message', {}).get('runs', [])
                     full_message = ''.join(part.get('text', '') for part in message_parts).strip()

                     if full_message: # Only add non-empty messages
                         messages.append(full_message)
                         processed_ids.add(msg_id)
                         message_count += 1
                 # Can add handlers for other message types like stickers, superchats if needed
                 # elif item.get('liveChatPaidMessageRenderer'): ...
                 # elif item.get('liveChatMembershipItemRenderer'): ...

        except Exception as e:
            logging.warning(f"Error processing a live chat entry: {e} - Entry: {str(entry)[:200]}") # Log problematic entry
            continue

    logging.info(f"Extracted {message_count} messages from {subtitle_file}")
    return messages


def get_combined_video_data(video_url: str, api_key: str) -> dict:
    """Fetches video details and live chat, handling errors."""
    result = {"video_id": None, "title": None, "description": None, "live_chat_raw": [], "error": None}
    video_id = extract_video_id(video_url)
    if not video_id:
        result["error"] = "Invalid YouTube URL or could not extract video ID."
        return result
    result["video_id"] = video_id

    # 1. Fetch Video Title and Description
    title, description = fetch_video_details(video_id, api_key)
    result["title"] = title
    result["description"] = description
    if title == "API Error": # Propagate API error
        result["error"] = "Failed to fetch video details from YouTube API."
        # Decide if you want to continue trying to get chat or stop here
        # return result

    # 2. Download and Parse Live Chat
    subtitle_file = None
    try:
        subtitle_file = download_live_chat(video_url, video_id)
        if subtitle_file:
            with st.spinner("Parsing downloaded chat data..."): # Feedback for potentially large files
                raw_messages = extract_live_chat_messages(subtitle_file)
                result["live_chat_raw"] = raw_messages
        # If subtitle_file is None, it means download failed or chat wasn't found; user already informed.
    finally:
        # 3. Clean up the temp file regardless of parsing success
        if subtitle_file and os.path.exists(subtitle_file):
            try:
                os.remove(subtitle_file)
                logging.info(f"Deleted temporary chat file: {subtitle_file}")
            except Exception as e:
                logging.warning(f"Could not delete temporary chat file {subtitle_file}: {str(e)}")

    return result


def get_top_comments(live_chat_raw, sentiment_labels, top_n=3):
    """Selects top N positive and negative comments based on calculated sentiment labels."""
    positive_comments = []
    negative_comments = []

    # Ensure raw chat and labels have the same length for safe iteration
    min_len = min(len(live_chat_raw), len(sentiment_labels))

    for i in range(min_len):
        comment = live_chat_raw[i] # Get the original raw comment
        sentiment = sentiment_labels[i]
        if sentiment == "Positive":
            positive_comments.append(comment)
        elif sentiment == "Negative":
            negative_comments.append(comment)

    # Return only top N, even if fewer are found
    return positive_comments[:top_n], negative_comments[:top_n]

# --- Plotly Pie Chart Function ---
def plot_sentiment_pie_chart_plotly(positive_count, negative_count, total_comments):
    """Generates an interactive Plotly pie chart for sentiment distribution."""
    if total_comments == 0:
        # Return an empty figure with a message if no comments
        fig = go.Figure()
        fig.update_layout(
            title_text='No Comments Analyzed',
            title_x=0.5,
            annotations=[dict(text='No data available', showarrow=False)],
             paper_bgcolor='rgba(0,0,0,0)',
             plot_bgcolor='rgba(0,0,0,0)'
             )
        return fig

    neutral_count = max(0, total_comments - (positive_count + negative_count)) # Ensure non-negative
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_count, negative_count, neutral_count]
    # Define colors (consider accessibility)
    colors = ['#28a745', '#dc3545', '#6c757d'] # Green, Red, Gray

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,
                                marker_colors=colors,
                                pull=[0.05 if v > 0 else 0 for v in values], # Pull slices with value > 0
                                hole=0.35, # Donut chart style
                                textinfo='percent+value', # Show percentage and count
                                insidetextorientation='auto', # Let Plotly decide text orientation
                                hoverinfo='label+percent+value', # Tooltip info
                                name='' # Avoid showing trace name in hover
                                )])
    fig.update_traces(textfont_size=12, marker=dict(line=dict(color='#000000', width=1))) # Add outline
    fig.update_layout(
        # title_text='Live Chat Sentiment Distribution', # Title handled by st.subheader now
        # title_x=0.5,
        margin=dict(l=10, r=10, t=10, b=10), # Minimal margins
        legend_title_text='Sentiments',
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5), # Horizontal legend below chart
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        height=350, # Fixed height for consistency
        # font_color="white" # Uncomment if using Streamlit dark theme explicitly
    )
    return fig

# --- Transcript and Summary Functions ---
def get_sub(video_id):
    """Retrieves transcript text, preferring Vietnamese, falling back to English."""
    if not video_id: return None
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        languages_to_try = ['vi', 'en'] # Prioritize Vietnamese

        for lang in languages_to_try:
             try:
                 # Try finding a manually created or generated transcript in the language
                 transcript = transcript_list.find_transcript([lang]).fetch()
                 if lang != languages_to_try[0]: # Inform if fallback language was used
                     st.info(f"Vietnamese transcript not found, using {lang} transcript.", icon="ℹ️")
                 break # Stop searching once found
             except Exception:
                 continue # Try next language

        if not transcript:
            st.warning(f"No suitable transcript (Vietnamese or English) found for video ID {video_id}.", icon="📜")
            return None

        concatenated_text = ' '.join([segment['text'] for segment in transcript])
        return concatenated_text

    except TranscriptsDisabled:
        st.warning(f"Transcripts are disabled for video ID {video_id}.", icon="🚫")
        logging.warning(f"Transcripts disabled for video ID {video_id}.")
        return None
    except NoTranscriptFound:
         st.warning(f"Could not find any transcript for video ID {video_id}.", icon="❓")
         logging.warning(f"No transcript found for video ID {video_id}")
         return None
    except Exception as e:
        st.error(f"🚨 Error retrieving transcript for video ID {video_id}: {e}", icon="🔥")
        logging.exception(f"Unhandled error getting subtitles for video ID {video_id}:")
        return None


# Define the prompt for the Gemini model (Consider making this configurable)
GEMINI_PROMPT = """
Bạn là một trợ lý AI chuyên tóm tắt video trên YouTube. Dựa vào bản ghi đầy đủ dưới đây,
hãy tạo một bản tóm tắt súc tích, nêu bật những điểm chính và ý chính của video.
Giới hạn tóm tắt trong khoảng 250-300 từ. Trình bày dưới dạng các gạch đầu dòng nếu phù hợp.

Bản ghi:
"""

# Function to get the Gemini response, with retry logic
def get_gemini_response_with_retry(transcript_text, max_attempts=3):
    """Generates summary using Gemini API with retry logic."""
    if not transcript_text or not isinstance(transcript_text, str) or not transcript_text.strip():
        logging.warning("Attempted to generate summary from empty or invalid transcript.")
        return "Error: Cannot generate summary from empty transcript."
    if not GOOGLE_API_KEY:
         return "Error: Gemini API key not configured."

    full_prompt = f"{GEMINI_PROMPT}\n{transcript_text}"

    for attempt in range(max_attempts):
        try:
            # Choose the appropriate model - Flash is faster and cheaper for summarization
            model = genai.GenerativeModel("gemini-1.5-flash")
            # Consider adding safety settings if needed:
            # safety_settings = {...}
            # response = model.generate_content(full_prompt, safety_settings=safety_settings)
            response = model.generate_content(full_prompt)

            # Basic check for blocked content (can be made more robust)
            if not response.parts:
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
                 logging.error(f"Gemini response blocked. Reason: {block_reason}")
                 return f"Error: Content generation blocked due to safety settings ({block_reason})."

            return response.text # Return the generated text

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to generate Gemini response: {e}")
            if attempt < max_attempts - 1:
                st.warning(f"⏳ Summary generation attempt {attempt + 1} failed. Retrying...", icon="⚠️")
                time.sleep(1.5 ** attempt) # Exponential backoff
            else:
                st.error(f"🚨 Failed to generate summary from Gemini after {max_attempts} attempts. Error: {e}", icon="🔥")
                return None # Indicate final failure
    return None # Should not be reached

# --- Streamlit App UI ---
st.set_page_config(page_title="🎥 YouTube Video Analysis", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)
st.markdown("---") # Visual separator

# Initialize session state variables
if 'responses' not in st.session_state:
    st.session_state.responses = [] # Stores analysis results for potentially multiple videos
if 'last_youtube_link' not in st.session_state:
    st.session_state.last_youtube_link = "" # Tracks the most recently analyzed link

# --- Input Area ---
st.subheader("🔗 Enter YouTube Video Link")
youtube_link = st.text_input(
    "Paste the YouTube video URL here:",
    key="youtube_link_input",
    label_visibility="collapsed",
    placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example placeholder
)

# --- Analyze Button and Processing Logic ---
if st.button("🔍 Analyze Video", type="primary", use_container_width=True): # Primary button, full width
    if not youtube_link or not youtube_link.strip():
        st.warning("⚠️ Please enter a YouTube video link.", icon="💡")
    elif youtube_link == st.session_state.last_youtube_link and st.session_state.responses:
         st.info("ℹ️ Analysis for this video is already displayed below. Enter a new link to analyze another.", icon="🔄")
    else:
        # --- Main Spinner for the entire analysis process ---
        main_spinner_placeholder = st.empty() # Placeholder for the spinner
        main_spinner_placeholder.info('🚀 Starting analysis... Please wait.', icon="⏳")

        st.session_state.responses = [] # Clear previous results for a new analysis
        st.session_state.last_youtube_link = youtube_link # Store the link being analyzed
        analysis_successful = False # Flag to track success

        try:
            # 1. Get Video ID and Basic Details (Title, Description, Raw Chat)
            with st.spinner("🔗 Validating link and fetching video data..."):
                 combined_data = get_combined_video_data(youtube_link, API_KEY)

            if combined_data.get("error"):
                 st.error(f"🚨 {combined_data['error']}", icon="🔥")
            else:
                 video_id = combined_data["video_id"]
                 title = combined_data["title"]
                 desc_raw = combined_data["description"]
                 chat_raw = combined_data["live_chat_raw"]

                 # 2. Preprocess Text Data
                 with st.spinner("🧹 Cleaning text data..."):
                     desc_clean = preprocess_model_input_str(desc_raw, title)
                     # Filter out empty strings after cleaning chat messages
                     chat_clean = [msg for msg in (preprocess_model_input_str(chat) for chat in chat_raw) if msg]


                 # 3. Analyze Sentiments (Description & Chat)
                 sentiment_data = []
                 positive_count = 0
                 negative_count = 0
                 description_sentiment = "N/A" # Default

                 # Analyze Description
                 if desc_clean:
                     with st.spinner("🤔 Analyzing description sentiment..."):
                         description_sentiment, _ = analyze_sentiment(desc_clean)

                 # Analyze Chat (if available)
                 if chat_clean:
                     with st.spinner(f"📊 Analyzing sentiment for {len(chat_clean)} chat messages... (this may take time)"):
                         # Consider parallel processing or batching for large chats if needed
                         for i, chat in enumerate(chat_clean):
                              sentiment, _ = analyze_sentiment(chat)
                              sentiment_data.append(sentiment)
                              # Optional: update spinner progress
                              # st.spinner(f"📊 Analyzing sentiment for chat messages... ({i+1}/{len(chat_clean)})")
                         positive_count = sum(1 for s in sentiment_data if s == "Positive")
                         negative_count = sum(1 for s in sentiment_data if s == "Negative")

                 total_comments = len(sentiment_data) # Based on successfully analyzed chat messages

                 # 4. Get Top Comments (use raw chat messages for display)
                 # We need to align raw chat with sentiment results.
                 # Assuming chat_clean corresponds index-wise to the *start* of chat_raw.
                 # A more robust approach might involve IDs if available.
                 # For now, use the cleaned chat count to slice raw chat.
                 raw_chat_for_top = chat_raw[:len(sentiment_data)]
                 positive_comments, negative_comments = get_top_comments(raw_chat_for_top, sentiment_data)


                 # 5. Store results in session state
                 response_data = {
                     'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", # Medium quality thumb
                     'video_details': {'title': title if title else "Video"}, # Use fetched title
                     'comments': {
                         'total_comments': total_comments,
                         'positive_comments': positive_count,
                         'negative_comments': negative_count,
                         'positive_comments_list': positive_comments,
                         'negative_comments_list': negative_comments
                     },
                     "description": desc_raw if desc_raw else "No description available.", # Store raw description
                     "video_id": video_id,
                     "sentiment_data": sentiment_data,
                     "live_chat_messages_raw": chat_raw, # Store raw chat for display
                     "live_chat_messages_clean_count": len(chat_clean), # Store count of cleaned messages for info
                     "description_sentiment": description_sentiment,
                     'transcript_summary': None # Placeholder for summary
                 }
                 st.session_state.responses.append(response_data)
                 analysis_successful = True # Mark as successful

        except Exception as e:
            st.error(f"🚨 An unexpected error occurred during the analysis pipeline: {e}", icon="🔥")
            logging.exception(f"Analysis pipeline error for {youtube_link}:") # Log full traceback

        finally:
             main_spinner_placeholder.empty() # Remove the main spinner
             if analysis_successful:
                st.success("✅ Analysis complete! Results are shown below.", icon="🎉")
             else:
                 st.error("❌ Analysis could not be completed due to errors.", icon="💔")


# --- Display Results Area ---
if not st.session_state.responses:
    # ***** CHANGE HERE: Added example URL *****
    st.info("Enter a YouTube video link above and click 'Analyze Video' to see the results. e.g: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
else:
    # Display results for the latest analysis (index 0, as we clear responses each time)
    # If you want to keep history, loop through st.session_state.responses
    response = st.session_state.responses[0] # Get the single stored response
    idx = 0 # Index for keys

    video_details = response.get('video_details', {})
    comments = response.get('comments', {})
    live_chat_messages_raw = response.get('live_chat_messages_raw', [])
    sentiment_data = response.get('sentiment_data', [])
    video_id = response.get('video_id')
    desc_raw = response.get('description', 'N/A')

    st.markdown("---")
    st.header(f"📊 Analysis Results for: {video_details.get('title', 'Video')}")

    # --- Tabs for Results ---
    tab1, tab2, tab3 = st.tabs(["📝 Video Info", "💬 Live Chat Analysis", "📜 Summary"])

    # --- Tab 1: Video Info ---
    with tab1:
        col1, col2 = st.columns([0.6, 0.4]) # Text column wider

        with col1:
            st.subheader("📄 Description")
            # Use an expander for potentially long descriptions
            with st.expander("Click to view description", expanded=(len(desc_raw) < 300)): # Expand short descriptions
                 # Display raw description using markdown with blockquote for style
                 st.markdown(f"> {desc_raw.replace('\n', '<br>')}", unsafe_allow_html=True) # Preserve line breaks visually

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
                st.info("Thumbnail could not be loaded.")

    # --- Tab 2: Live Chat Analysis ---
    with tab2:
        total_analyzed = comments.get('total_comments', 0)
        total_raw = len(live_chat_messages_raw)
        clean_count = response.get('live_chat_messages_clean_count', 0)

        if total_raw == 0:
             st.info("ℹ️ No live chat messages were found or downloaded for this video.")
        elif total_analyzed == 0 and clean_count > 0:
             st.warning("⚠️ Found live chat messages, but sentiment analysis could not be performed on any of them (e.g., non-text messages or errors).")
        elif total_analyzed == 0 and clean_count == 0:
             st.info("ℹ️ Live chat messages were found, but none contained processable text after cleaning.")
        else:
            if total_analyzed < clean_count:
                 st.caption(f"ℹ️ Analyzed sentiment for {total_analyzed} out of {clean_count} cleaned text messages. Some messages might have caused analysis errors.")
            elif total_raw > clean_count:
                 st.caption(f"ℹ️ Found {total_raw} raw chat entries, processed {clean_count} text messages for sentiment analysis.")

            col1, col2 = st.columns([0.6, 0.4]) # Adjust ratio as needed

            with col1:
                st.subheader("🗨️ Live Chat Messages & Sentiment")
                if live_chat_messages_raw and sentiment_data:
                    # Align raw messages with sentiment results for display
                    display_messages = live_chat_messages_raw[:len(sentiment_data)]
                    df_data = {
                        'Live Chat Message': display_messages,
                        'Detected Sentiment': sentiment_data[:len(display_messages)] # Ensure length match
                     }
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, height=400, use_container_width=True, hide_index=True) # Use container width, hide index
                else:
                     st.info("No sentiment data available to display alongside messages.")


            with col2:
                st.subheader("📊 Sentiment Breakdown")
                if comments and total_analyzed > 0:
                    positive = comments['positive_comments']
                    negative = comments['negative_comments']
                    neutral = max(0, total_analyzed - positive - negative) # Ensure non-negative

                    # Display Pie Chart using Plotly
                    fig = plot_sentiment_pie_chart_plotly(positive, negative, total_analyzed)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display Metrics using st.metric for a clean look
                    st.metric(label="Total Comments Analyzed", value=f"{total_analyzed}")

                    pos_perc = (positive / total_analyzed) * 100 if total_analyzed > 0 else 0
                    neg_perc = (negative / total_analyzed) * 100 if total_analyzed > 0 else 0
                    neu_perc = (neutral / total_analyzed) * 100 if total_analyzed > 0 else 0

                    # Use columns for metrics for better spacing on wide layouts
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric(label="😊 Positive", value=f"{positive}", delta=f"{pos_perc:.1f}%", delta_color="normal") # Green delta is default
                    with m_col2:
                        st.metric(label="😠 Negative", value=f"{negative}", delta=f"{neg_perc:.1f}%", delta_color="inverse") # Red delta
                    with m_col3:
                        st.metric(label="😐 Neutral", value=f"{neutral}", delta=f"{neu_perc:.1f}%", delta_color="off") # No color delta

                else:
                    st.info("No comment sentiment statistics available.")

            # --- Top Comments Display (Moved below columns but within Tab 2) ---
            st.markdown("<br>", unsafe_allow_html=True) # Add some space
            st.subheader("⭐ Top Comments Examples")
            if comments and total_analyzed > 0:
                # Use expander
                with st.expander("Show Top Positive & Negative Comments", expanded=False):
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown(f"<h5 style='color: #28a745; font-weight: bold;'>👍 Top Positive:</h5>", unsafe_allow_html=True)
                        pos_list = comments.get('positive_comments_list', [])
                        if pos_list:
                            for comment in pos_list:
                                # Added color: black; and slight escape for safety
                                st.markdown(f"<div style='background-color: #e9f7ef; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid #28a745; color: black; font-size: 0.9em;'>{st.markdown(comment).strip()}</div>", unsafe_allow_html=True)
                        else:
                            st.caption("No positive comments found.")

                    with col_neg:
                        st.markdown(f"<h5 style='color: #dc3545; font-weight: bold;'>👎 Top Negative:</h5>", unsafe_allow_html=True)
                        neg_list = comments.get('negative_comments_list', [])
                        if neg_list:
                            for comment in neg_list:
                                # Added color: black; and slight escape for safety
                                st.markdown(f"<div style='background-color: #fdeded; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid #dc3545; color: black; font-size: 0.9em;'>{st.markdown(comment).strip()}</div>", unsafe_allow_html=True)
                        else:
                             st.caption("No negative comments found.")
            else:
                st.caption("No comments available to display top examples.")


    # --- Tab 3: Summary ---
    with tab3:
        st.subheader("✍️ Video Summary (via Gemini AI)")

        summary_key = f"summary_{video_id}_{idx}" # Unique key per video/response
        summary_button_label = "📜 Generate Summary"
        current_summary = response.get('transcript_summary') # Get current summary state

        if current_summary:
            summary_button_label = "🔄 Regenerate Summary"

        # Button to trigger summary generation
        if st.button(summary_button_label, key=summary_key, type="secondary"):
            with st.spinner("🔄 Fetching transcript and generating summary with Gemini AI..."):
                summary = None # Reset summary variable
                try:
                    transcript = get_sub(video_id) # Fetch transcript
                    if transcript:
                        summary = get_gemini_response_with_retry(transcript) # Call Gemini
                        if summary and "Error:" not in summary:
                            # Update the specific response in session state
                            st.session_state.responses[idx]['transcript_summary'] = summary
                            st.rerun() # Rerun to display the new summary immediately
                        elif summary: # Handle Gemini error messages
                             st.error(f"⚠️ Gemini Error: {summary}", icon="🤖")
                        # else: Gemini function already showed error
                    # else: get_sub function already showed error/warning

                except Exception as e:
                    st.error(f"🚨 An unexpected error occurred during summary generation: {e}", icon="🔥")
                    logging.exception(f"Summary generation error for {video_id}:")

        # Display generated summary if it exists
        if current_summary:
            if "Error:" in current_summary:
                 st.warning(f"⚠️ Could not display summary: {current_summary}", icon="🚫")
            else:
                 # Added color: black; styling
                 st.markdown(f"<div style='background-color: #eaf4ff; padding: 15px; border-radius: 8px; border-left: 5px solid #0d6efd; color: black; font-family: sans-serif; line-height: 1.6;'>{current_summary.replace('\n', '<br>')}</div>", unsafe_allow_html=True)
        else:
            st.info("Click 'Generate Summary' to create a summary of the video transcript using AI (requires transcript availability).")

# --- Footer Removed ---
# st.markdown("---")
# st.caption("YouTube Analysis App | Sentiment: PhoBERT | Summary: Google Gemini")
# Optional: Add a footer
st.markdown("---")
st.caption("YouTube Analysis App | Sentiment powered by PhoBERT | Summary by Google Gemini")
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
