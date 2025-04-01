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

# MODEL_ID = "wonrax/phobert-base-vietnamese-sentiment"
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
# from plotly.subplots import make_subplots # No longer needed with Matplotlib
# import plotly.graph_objects as go # No longer needed with Matplotlib

# --- Configuration ---
# Your API Key - should be stored securely, e.g., using Streamlit secrets or environment variables
API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"  # Replace with your actual YouTube Data API key
GOOGLE_API_KEY = "AIzaSyArb6Eme11X4tl8mhreEQUfRLkTjqTP59I"  # Replace with your Gemini API key

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, # Changed to INFO for better tracking
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logging.info("Gemini API configured successfully.")
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    logging.error(f"Failed to configure Gemini API: {e}")
    # Optionally exit or disable Gemini features if configuration fails
    # st.stop()

MODEL_PATH = ""  # Set this to the directory if you have a folder for the weights, otherwise it would be ""
MODEL_FILE = "sentiment_classifier (1).pth" # Make sure this path is correct relative to your script

# --- Model Loading ---
@st.cache_resource # Cache the loaded model and tokenizer
def load_model():
    model_path = os.path.join(MODEL_PATH, MODEL_FILE)
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    try:
        if not os.path.exists(model_path):
             st.error(f"Model file not found at path: {os.path.abspath(model_path)}")
             logging.error(f"Model file not found at path: {os.path.abspath(model_path)}")
             return None, None

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

        # Load the state dictionary from the saved .pth file
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=map_location)

        # Handle potential key mismatches (e.g., if saved with DataParallel)
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict, strict=False)

        # Try loading directly first, if it fails, try adjusting keys
        try:
             model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logging.warning(f"Strict loading failed ({e}), trying non-strict or adjusting keys...")
            try:
                # Attempt non-strict loading
                model.load_state_dict(state_dict, strict=False)
                logging.warning("Loaded model state dict with strict=False. Check model architecture compatibility.")
            except Exception as load_err:
                 st.error(f"Error loading model state_dict even with adjustments: {load_err}")
                 logging.error(f"Error loading model state_dict even with adjustments: {load_err}")
                 return None, None


        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logging.info(f"Model loaded successfully from {model_path} and moved to {device}")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model/tokenizer from {model_id} or file {model_path}: {e}")
        logging.error(f"Error loading model/tokenizer from {model_id} or file {model_path}: {e}")
        return None, None

# --- Sentiment Analysis ---
def analyze_sentiment(text):
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.error("Model loading failed. Sentiment analysis is unavailable.")
        return "Error", np.array([0.0, 0.0, 0.0]) # Return numpy array for consistency

    if not text or not isinstance(text, str) or text.strip() == "":
        logging.warning("Received empty or invalid text for sentiment analysis.")
        return "Neutral", np.array([0.0, 1.0, 0.0]) # Return neutral for empty input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # No need for tokenizer.padding_side = "left" with single inputs usually
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device) # Reduced max_length
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_class_idx = np.argmax(predictions)
        sentiment_label = sentiment_labels[predicted_class_idx]

        return sentiment_label, predictions # predictions are scores [neg, neu, pos]
    except Exception as e:
        logging.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}")
        st.warning(f"Could not analyze sentiment for a piece of text: {e}")
        return "Error", np.array([0.0, 0.0, 0.0])

# --- Text Preprocessing ---
def preprocess_model_input_str(text, video_title=""):
    if not text or not isinstance(text, str):
        return ""
    # Slightly simplified regex
    regex_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.\S+|#\w+|\@\w+|[\n\r]+|[:;=]{2,}"
    clean_str = re.sub(regex_pattern, " ", text)
    clean_str = re.sub(r"\s{2,}", " ", clean_str) # Replace multiple spaces with one
    if video_title:
        clean_str = clean_str.replace(video_title, "") # Remove title if present
    return clean_str.strip()

# --- YouTube Data Fetching ---
def extract_video_id(url):
    # Regex to handle various YouTube URL formats
    pattern = re.compile(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*')
    match = pattern.search(url)
    if match:
        return match.group(1)
    # Handle short URLs
    pattern_short = re.compile(r'youtu\.be\/([0-9A-Za-z_-]{11})')
    match_short = pattern_short.search(url)
    if match_short:
        return match_short.group(1)
    return None

def fetch_video_details(video_id, api_key):
    """Fetches title and description using YouTube Data API."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(
            part="snippet", # Only need snippet for title and description
            id=video_id
        )
        response = request.execute()

        if not response["items"]:
            logging.error(f"Video not found with ID: {video_id}")
            return None, "Video Title Unavailable", "Video Description Unavailable"

        snippet = response["items"][0]["snippet"]
        title = snippet.get("title", "Video Title Unavailable")
        description = snippet.get("description", "") # Default to empty string if missing
        return video_id, title, description
    except Exception as e:
        logging.error(f"Error fetching video details for {video_id}: {e}")
        st.error(f"Error fetching video details: {e}")
        return video_id, "Video Title Unavailable", "Video Description Unavailable"


def download_live_chat(video_url, video_id):
    """Downloads live chat replay if available."""
    subtitle_file = f"{video_id}.live_chat.json"
    # Delete existing file if it exists to ensure fresh download
    if os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
        except OSError as e:
            logging.warning(f"Could not delete existing chat file {subtitle_file}: {e}")

    ydl_opts = {
        'writesubtitles': True,
        'subtitleslangs': ['live_chat'], # Explicitly request live chat
        'skip_download': True, # Don't download the video
        'outtmpl': f'{video_id}', # Base name for files
        'quiet': True,
        'ignoreerrors': True, # Continue even if chat download fails
        'logtostderr': False, # Don't print yt-dlp logs to stderr/console
    }

    try:
        logging.info(f"Attempting to download live chat for {video_id} from {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=False) # Get info first
            # Check if live chat is available before attempting download
            if result and result.get('is_live') and not result.get('was_live'):
                 logging.warning(f"Video {video_id} is currently live, live chat replay may not be available yet.")
                 # return None # Don't try downloading if it's live and not finished

            # Now attempt download
            ydl.download([video_url])


        if os.path.exists(subtitle_file):
            logging.info(f"Live chat downloaded successfully to {subtitle_file}")
            return subtitle_file
        else:
            # Check common reasons for failure
            if result and not result.get('subtitles'):
                 logging.warning(f"No subtitles (including live chat) found for video {video_id}.")
            else:
                 logging.warning(f"Live chat file {subtitle_file} not found after download attempt for {video_id}. Live chat might be disabled or unavailable.")
            return None # Return None if download didn't produce the file
    except yt_dlp.utils.DownloadError as e:
        # Specifically catch download errors which often indicate chat unavailability
        if "live chat replay is not available" in str(e).lower():
            logging.warning(f"Live chat replay not available for video {video_id}.")
        else:
            logging.error(f"yt-dlp DownloadError for {video_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Generic error downloading live chat for {video_id}: {e}")
        return None

def parse_jsonl(file_path):
    """Parses a JSONL file, returning a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as json_err:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {json_err}")
                    continue # Skip malformed lines
        return data
    except FileNotFoundError:
        logging.error(f"Live chat file not found during parsing: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error opening/reading file {file_path}: {e}")
        return None

def extract_live_chat_messages(subtitle_file):
    """Extracts message texts from the parsed live chat JSONL data."""
    messages = []
    if not subtitle_file or not os.path.exists(subtitle_file):
        logging.warning(f"Subtitle file missing or not provided: {subtitle_file}")
        return messages # Return empty list if no file

    data = parse_jsonl(subtitle_file)
    if data is None: # Check if parsing failed
        return messages # Return empty list

    for entry in data:
        try:
            # Navigate the nested structure common in YouTube chat replays
            replay_action = entry.get('replayChatItemAction', {})
            actions = replay_action.get('actions', [])
            for action in actions:
                add_chat_item = action.get('addChatItemAction', {})
                item = add_chat_item.get('item', {})
                msg_renderer = item.get('liveChatTextMessageRenderer') or item.get('liveChatPaidMessageRenderer') # Handle regular and paid messages

                if msg_renderer:
                    message_content = msg_renderer.get('message')
                    if message_content and 'runs' in message_content:
                        full_message = "".join(run.get('text', '') for run in message_content['runs'] if 'text' in run)
                        if full_message.strip(): # Ensure message isn't just whitespace
                             messages.append(full_message.strip())
                    elif 'simpleText' in message_content: # Sometimes it's simpler
                         simple_text = message_content.get('simpleText', '').strip()
                         if simple_text:
                              messages.append(simple_text)

        except Exception as e:
            logging.warning(f"Error processing a live chat entry: {entry} - Error: {str(e)}")
            continue # Skip to the next entry if one causes an error
    return messages

# --- Data Aggregation ---
def get_video_data(video_url: str, api_key: str) -> dict:
    """Fetches video details and live chat, returning a structured dictionary."""
    video_id = extract_video_id(video_url)
    if not video_id:
        logging.error(f"Invalid YouTube URL: {video_url}")
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    logging.info(f"Processing video ID: {video_id}")

    # 1. Fetch Video Title & Description
    video_id, title, description = fetch_video_details(video_id, api_key)
    # if title == "Video Title Unavailable": # Check if fetch failed
    #     return {"error": f"Could not fetch details for video ID: {video_id}"}

    # 2. Download and Parse Live Chat
    subtitle_file = download_live_chat(video_url, video_id)
    live_chat_messages = [] # Default to empty list
    if subtitle_file:
        live_chat_messages = extract_live_chat_messages(subtitle_file)
        # 3. Clean up the temp file *after* processing
        try:
            os.remove(subtitle_file)
            logging.info(f"Deleted temporary chat file: {subtitle_file}")
        except Exception as e:
            logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")
    else:
         logging.warning(f"No subtitle file generated or found for {video_id}. Proceeding without live chat.")


    # 4. Preprocess text
    clean_description = preprocess_model_input_str(description, title)
    # Clean live chat messages *only if* they exist
    clean_live_chat = [preprocess_model_input_str(msg) for msg in live_chat_messages] if live_chat_messages else []


    # Return the structured data
    return {
        "video_id": video_id,
        "title": title,
        "description_raw": description,
        "description_clean": clean_description,
        "live_chat_raw": live_chat_messages,
        "live_chat_clean": clean_live_chat,
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/0.jpg" if video_id else None,
        "error": None # Indicate success
    }


# --- Analysis & Visualization ---

def analyze_all_sentiments(description_clean, live_chat_clean):
    """Analyzes sentiment for description and all live chat messages."""
    # Analyze description
    desc_sentiment_label, desc_sentiment_scores = analyze_sentiment(description_clean)

    # Analyze live chat messages
    chat_sentiments = [] # List of tuples: (label, scores_array)
    chat_labels = [] # List of labels only
    if live_chat_clean: # Only analyze if chat messages exist
        # Using st.progress for visual feedback on chat analysis
        progress_bar = st.progress(0, text="Analyzing live chat sentiment...")
        total_chats = len(live_chat_clean)
        for i, chat in enumerate(live_chat_clean):
            label, scores = analyze_sentiment(chat)
            chat_sentiments.append((label, scores))
            chat_labels.append(label)
            # Update progress bar
            progress = (i + 1) / total_chats
            progress_text = f"Analyzing live chat sentiment... ({i+1}/{total_chats})"
            # Check if progress bar exists before updating (in case of errors)
            if progress_bar:
                 try:
                      progress_bar.progress(progress, text=progress_text)
                 except Exception as e:
                      logging.warning(f"Error updating progress bar: {e}")
                      progress_bar = None # Stop trying to update if it fails

        if progress_bar:
            progress_bar.empty() # Remove progress bar when done
    else:
        logging.info("No clean live chat messages to analyze.")


    # Calculate counts for pie chart
    positive_count = sum(1 for label in chat_labels if label == "Positive")
    negative_count = sum(1 for label in chat_labels if label == "Negative")
    neutral_count = sum(1 for label in chat_labels if label == "Neutral")
    error_count = sum(1 for label in chat_labels if label == "Error")
    total_analyzed = len(chat_labels)

    if error_count > 0:
        st.warning(f"Could not analyze sentiment for {error_count} chat messages.")

    return {
        "description_sentiment": desc_sentiment_label,
        "description_scores": desc_sentiment_scores,
        "chat_sentiment_labels": chat_labels, # List of 'Positive', 'Negative', 'Neutral', 'Error'
        "chat_sentiments_full": chat_sentiments, # List of ('Label', [neg, neu, pos] scores)
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_analyzed_chats": total_analyzed,
    }

def get_top_comments(live_chat_raw, chat_sentiment_labels, top_n=3):
    """Selects top N positive/negative comments based on pre-calculated labels."""
    positive_comments = []
    negative_comments = []

    # Ensure lists are of the same length
    if len(live_chat_raw) != len(chat_sentiment_labels):
        logging.error("Mismatch between raw chat messages and sentiment labels count.")
        # Fallback: return empty lists or handle error appropriately
        return [], []

    for i, comment in enumerate(live_chat_raw):
        sentiment = chat_sentiment_labels[i]
        if sentiment == "Positive":
            positive_comments.append(comment)
        elif sentiment == "Negative":
            negative_comments.append(comment)

    # Return the top N, no sorting needed here as order might be relevant (time)
    # If specific score-based sorting is needed, 'chat_sentiments_full' would be required
    return positive_comments[:top_n], negative_comments[:top_n]


def plot_sentiment_pie_chart(positive_count, negative_count, neutral_count, total_comments):
    """Generates a Matplotlib pie chart for sentiment distribution."""
    if total_comments == 0:
        # Return an empty figure or a message if no comments
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No chat comments to analyze', horizontalalignment='center', verticalalignment='center')
        ax.axis('off') # Hide axes
        return fig

    labels = []
    sizes = []
    colors = []
    explode = []

    # Only include slices with data
    if positive_count > 0:
        labels.append(f'😊 Positive ({positive_count})')
        sizes.append(positive_count)
        colors.append('#DFF0D8') # Light green
        explode.append(0.1 if positive_count >= max(negative_count, neutral_count) else 0) # Explode largest slice slightly
    if negative_count > 0:
        labels.append(f'😠 Negative ({negative_count})')
        sizes.append(negative_count)
        colors.append('#F2DEDE') # Light red
        explode.append(0.1 if negative_count > max(positive_count, neutral_count) else 0)
    if neutral_count > 0:
        labels.append(f'😐 Neutral ({neutral_count})')
        sizes.append(neutral_count)
        colors.append('#EAEAEA') # Light gray
        explode.append(0.1 if neutral_count >= max(positive_count, negative_count) else 0)

    # Ensure explode has the correct length
    while len(explode) < len(sizes):
         explode.append(0) # Should not happen with the logic above, but safe fallback
    if sum(explode) > 0.1: # Avoid exploding multiple slices if counts are equal
         max_idx = sizes.index(max(sizes))
         explode = [0.1 if i == max_idx else 0 for i in range(len(sizes))]


    fig, ax = plt.subplots(figsize=(6, 4)) # Adjust figure size if needed
    ax.pie(sizes, explode=tuple(explode), labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=140, pctdistance=0.85) # Show percentage inside slice
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Live Chat Sentiment Distribution", pad=20)
    return fig


# --- Summarization ---
def get_sub(video_id):
    """Fetches video transcript using youtube_transcript_api."""
    try:
        # Try Vietnamese first, fallback to English
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        target_langs = ['vi', 'en'] # Prioritize Vietnamese
        transcript = None
        for lang in target_langs:
             try:
                  transcript = transcript_list.find_generated_transcript([lang])
                  logging.info(f"Found transcript in language: {lang} for {video_id}")
                  break # Stop after finding the first available language in our list
             except Exception:
                  continue # Try next language

        if not transcript:
            logging.warning(f"No transcript found in {target_langs} for video ID {video_id}. Trying any available.")
            # Fallback: try fetching any available transcript
            try:
                transcript = transcript_list.find_generated_transcript(transcript_list.languages)
                logging.info(f"Found transcript in language: {transcript.language} for {video_id}")
            except Exception as e:
                 logging.error(f"Could not find any transcript for video ID {video_id}: {e}")
                 st.warning(f"Could not retrieve transcript for this video ({e}).")
                 return None


        full_transcript = transcript.fetch()
        concatenated_text = ' '.join([segment['text'] for segment in full_transcript])
        logging.info(f"Transcript retrieved successfully for {video_id} (length: {len(concatenated_text)} chars)")
        return concatenated_text
    except Exception as e:
        logging.error(f"Error getting transcript for video ID {video_id}: {e}")
        st.error(f"Error retrieving transcript: {e}")
        return None

# Define the prompt for the Gemini model (Updated as per original)
GEMINI_PROMPT = """
Bạn là người tóm tắt video trên Youtube. Bạn sẽ lấy văn bản ghi chép
và tóm tắt toàn bộ video và cung cấp bản tóm tắt quan trọng theo các điểm
trong vòng 300 từ. Vui lòng cung cấp bản tóm tắt của văn bản được đưa ra ở đây:
"""

def get_gemini_response(transcript_text):
    """Generates summary using Google Gemini API."""
    if not transcript_text:
        logging.error("Cannot generate summary from empty transcript.")
        return None
    if not GOOGLE_API_KEY or genai._config.api_key is None: # Check if Gemini configured
         st.error("Gemini API key not configured. Cannot generate summary.")
         logging.error("Gemini API key not configured.")
         return None
    try:
        # Specify a model known for summarization, ensure it's available/correct
        # Check available models if 'gemini-1.5-flash' causes issues
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = transcript_text + "\n\n" + GEMINI_PROMPT # Combine transcript and prompt
        response = model.generate_content(full_prompt)
        logging.info(f"Gemini summary generated successfully.")
        return response.text # Extract the text part of the response
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        st.error(f"Error generating summary via Gemini: {e}")
        return None


# --- Streamlit App UI ---
st.set_page_config(page_title="YT Analysis", layout="wide") # Use wide layout
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)

# Initialize session state if keys don't exist
if 'video_data' not in st.session_state:
    st.session_state.video_data = None # Stores raw fetched data
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None # Stores sentiment results
if 'summary' not in st.session_state:
    st.session_state.summary = None # Stores generated summary
if 'last_youtube_link' not in st.session_state:
    st.session_state.last_youtube_link = ""
if 'show_top_comments' not in st.session_state:
    st.session_state.show_top_comments = False # State for the checkbox

# --- Input Form ---
with st.form("youtube_url_form"):
    youtube_link = st.text_input("🔗 Enter YouTube Video Link:", key="youtube_link_input", value=st.session_state.last_youtube_link)
    submitted = st.form_submit_button("🔍 Analyze Video")

    if submitted:
        # Clear previous results if new link is different or empty
        if not youtube_link or youtube_link != st.session_state.last_youtube_link:
            st.session_state.video_data = None
            st.session_state.analysis_results = None
            st.session_state.summary = None
            st.session_state.show_top_comments = False # Reset checkbox state
            st.session_state.last_youtube_link = youtube_link # Update last link

        if not youtube_link:
            st.warning("Please enter a YouTube video link.")
            st.stop() # Stop execution if link is empty

        st.session_state.last_youtube_link = youtube_link # Store link even if analysis fails

        with st.spinner('🔄 Processing video... Please wait.'):
            # 1. Fetch Video Data
            with st.spinner('⬇️ Fetching video details and live chat...'):
                 video_data = get_video_data(youtube_link, API_KEY)
                 st.session_state.video_data = video_data # Store fetched data

            if video_data and video_data.get("error"):
                st.error(video_data["error"])
                st.session_state.video_data = None # Clear data on error
                st.stop()
            elif not video_data or not video_data.get("video_id"):
                st.error("Failed to retrieve essential video data.")
                st.session_state.video_data = None # Clear data on error
                st.stop()

            # 2. Perform Sentiment Analysis (only if data fetch succeeded)
            with st.spinner('📊 Analyzing sentiments...'):
                analysis_results = analyze_all_sentiments(
                    video_data.get("description_clean", ""),
                    video_data.get("live_chat_clean", [])
                )
                st.session_state.analysis_results = analysis_results # Store analysis results

            # Clear summary from previous runs
            st.session_state.summary = None
            st.success("✅ Analysis complete! View results in the tabs below.")
            # Rerun to update the display sections below the form
            st.rerun()


# --- Display Results using Tabs (only if data exists) ---
if st.session_state.video_data and st.session_state.analysis_results:
    # Retrieve data from session state for easier access
    video_data = st.session_state.video_data
    analysis = st.session_state.analysis_results

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Video Info", "💬 Live Chat Analysis", "📝 Summary"])

    # --- Tab 1: Video Info ---
    with tab1:
        st.markdown("### Video Details")
        if video_data.get("thumbnail_url"):
             st.image(video_data["thumbnail_url"], caption=video_data.get('title', 'Video Thumbnail'), use_column_width=True)
        else:
             st.warning("Could not load video thumbnail.")

        st.markdown(f"**Title:** {video_data.get('title', 'N/A')}")

        st.markdown("**Description:**")
        # Use expander for potentially long descriptions
        with st.expander("Click to view/hide description", expanded=False):
             st.markdown(f"> {video_data.get('description_raw', 'No description available.')}") # Display raw description

        st.markdown("**Description Sentiment:**")
        desc_sentiment = analysis.get('description_sentiment', 'N/A')
        st.markdown(f"**Sentiment:** `{desc_sentiment}`")
        # Optional: Show scores if needed
        # desc_scores = analysis.get('description_scores')
        # if desc_scores is not None:
        #     st.write(f"Scores (Neg, Neu, Pos): {[f'{s:.2f}' for s in desc_scores]}")


    # --- Tab 2: Live Chat Analysis ---
    with tab2:
        st.markdown("### Live Chat Sentiment Analysis")

        raw_chat = video_data.get("live_chat_raw", [])
        chat_labels = analysis.get("chat_sentiment_labels", [])
        total_analyzed = analysis.get("total_analyzed_chats", 0)

        if not raw_chat:
            st.info("ℹ️ No live chat messages were found or downloaded for this video.")
        else:
            # Combine chat and labels for DataFrame
            if len(raw_chat) == len(chat_labels):
                chat_df_data = {'Live Chat Message': raw_chat, 'Sentiment': chat_labels}
                chat_df = pd.DataFrame(chat_df_data)

                st.markdown("**Live Chat Messages and Sentiments:**")
                # Use st.dataframe for better table display, limit height
                st.dataframe(chat_df, height=300, use_container_width=True)
            else:
                st.warning("Mismatch between number of chat messages and analyzed sentiments. Cannot display table reliably.")
                logging.error(f"Chat/Label count mismatch: {len(raw_chat)} messages, {len(chat_labels)} labels for {video_data.get('video_id')}")


            st.markdown("**Sentiment Distribution:**")
            # Plot and display pie chart
            fig = plot_sentiment_pie_chart(
                analysis.get('positive_count', 0),
                analysis.get('negative_count', 0),
                analysis.get('neutral_count', 0),
                total_analyzed
            )
            st.pyplot(fig)

            st.markdown("**Top Comments:**")
            # Get top comments using pre-calculated labels
            positive_comments, negative_comments = get_top_comments(
                raw_chat,
                chat_labels,
                top_n=3
            )

            # Use columns for better layout
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h4 style='color: #28a745;'>👍 Top Positive Comments ({len(positive_comments)}):</h4>", unsafe_allow_html=True)
                if positive_comments:
                    for comment in positive_comments:
                        st.markdown(f"<div style='background-color: #DFF0D8; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black;'>{comment}</div>", unsafe_allow_html=True)
                else:
                    st.write("No positive comments found.")

            with col2:
                st.markdown(f"<h4 style='color: #dc3545;'>👎 Top Negative Comments ({len(negative_comments)}):</h4>", unsafe_allow_html=True)
                if negative_comments:
                    for comment in negative_comments:
                        st.markdown(f"<div style='background-color: #F2DEDE; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black;'>{comment}</div>", unsafe_allow_html=True)
                else:
                    st.write("No negative comments found.")

    # --- Tab 3: Summary ---
    with tab3:
        st.markdown("### Video Summary (Generated by AI)")

        video_id = video_data.get("video_id")

        # Display summary if already generated and stored
        if st.session_state.summary:
            st.markdown(f"<div style='background-color: #F0F8FF; padding: 10px; border-radius: 5px; color: black;'>{st.session_state.summary}</div>", unsafe_allow_html=True)
            # Option to regenerate
            if st.button("🔄 Regenerate Summary", key="regenerate_summary"):
                 st.session_state.summary = None # Clear current summary
                 st.rerun() # Rerun to trigger generation logic below
        else:
             # Button to generate summary (only shown if not generated yet)
             if st.button("📜 Generate Summary", key="generate_summary"):
                 with st.spinner("⏳ Generating summary... This may take a moment."):
                     # 1. Get Transcript
                     transcript = get_sub(video_id)
                     if transcript:
                         # 2. Get Gemini Response
                         summary_text = get_gemini_response(transcript)
                         if summary_text:
                             st.session_state.summary = summary_text # Store the summary
                             st.rerun() # Rerun to display the summary immediately
                         else:
                             st.error("❌ Failed to generate summary from the transcript.")
                             # Keep summary as None
                     else:
                         st.error("❌ Failed to retrieve transcript. Cannot generate summary.")
                         # Keep summary as None
             else:
                  st.info("ℹ️ Click the button above to generate an AI summary of the video transcript (if available).")
