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
    model_path = os.path.join(MODEL_PATH, MODEL_FILE)
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the path is correct.")
        logging.error(f"Model file not found at {model_path}.")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

        # Load the state dictionary
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        except RuntimeError as e:
            st.warning(f"RuntimeError loading state_dict (possibly mismatched keys, trying strict=False): {e}")
            logging.warning(f"RuntimeError loading state_dict (trying strict=False): {e}")
            # If strict=False was already intended, this warning is informational.
            # If you expected strict=True to work, investigate the mismatch.
        except Exception as e:
             st.error(f"Error loading model state_dict from {model_path}: {e}")
             logging.error(f"Error loading model state_dict from {model_path}: {e}")
             return None, None # Indicate failure

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logging.info(f"Model loaded successfully from {model_path} and moved to {device}")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model/tokenizer ({model_id}): {e}")
        logging.error(f"Error loading model/tokenizer ({model_id}): {e}")
        return None, None

# --- Sentiment Analysis ---
def analyze_sentiment(text):
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.error("Model loading failed. Sentiment analysis is unavailable.")
        return "Error", [0.33, 0.33, 0.33] # Return default neutral-ish scores on error

    if not text or not isinstance(text, str) or not text.strip():
         return "Neutral", [0.0, 1.0, 0.0] # Handle empty or invalid input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure model is on the correct device (might be needed if cache expires differently)
    model.to(device)
    # Tokenizer padding side matters for some models, ensure consistency if needed
    # tokenizer.padding_side = "left" # Uncomment if PhoBERT needs left padding specifically

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device) # Reduced max_length slightly
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_class_index = np.argmax(predictions)
        sentiment_label = sentiment_labels[predicted_class_index]

        return sentiment_label, predictions # Return raw probabilities
    except Exception as e:
        logging.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}")
        st.warning(f"Could not analyze sentiment for a piece of text: {e}")
        return "Error", [0.33, 0.33, 0.33]


def preprocess_model_input_str(text, video_title=""):
    if not text or not isinstance(text, str):
        return ""
    # Slightly refined regex
    regex_pattern = r"https?://\S+|www\.\S+|#\w+|\@\w+|[:;=8][\-o\*\']?[\)\]\(\[dDpP/\\*]" # Keep emojis basic
    # Remove URLs, hashtags, mentions, basic emoticons
    clean_str = re.sub(regex_pattern, "", text)
    # Remove excessive punctuation/symbols (3 or more)
    clean_str = re.sub(r"[^\w\s\u00C0-\u1FFF]{3,}", " ", clean_str) # Allow Vietnamese characters
    # Remove line breaks and excessive whitespace
    clean_str = re.sub(r"\s+", " ", clean_str).strip()
    # Optionally remove video title if it contaminates description/comments
    if video_title:
       clean_str = clean_str.replace(video_title, "").strip()
    return clean_str

# --- YouTube Data Fetching ---
def extract_video_id(url):
    # More robust regex covering various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu\.be\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_video_details(video_id, api_key):
    """Fetches title, description, and potentially other details."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet", # Add 'statistics' here if you need view counts etc.
            id=video_id
        ).execute()

        if not response.get("items"):
            logging.warning(f"Video details not found for ID: {video_id}")
            return None # Video not found or private

        snippet = response["items"][0].get("snippet", {})
        # statistics = response["items"][0].get("statistics", {}) # Uncomment if needed

        details = {
            'title': snippet.get('title', 'Title Not Available'),
            'description': snippet.get('description', ''),
            'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url') or snippet.get('thumbnails', {}).get('default', {}).get('url'), # Get high-res thumbnail
            # Add more details if needed
            # 'channel_title': snippet.get('channelTitle'),
            # 'upload_date': snippet.get('publishedAt'),
            # 'view_count': statistics.get('viewCount'),
            # 'like_count': statistics.get('likeCount'),
        }
        return details
    except Exception as e:
        logging.error(f"Error fetching video details for {video_id}: {e}")
        st.error(f"Error fetching video details: {e}")
        return None

def download_live_chat(video_url, video_id):
    """Downloads live chat replay to a JSONL file."""
    output_template = f'{video_id}.%(ext)s' # Use extractor's extension
    ydl_opts = {
        'writesubtitles': True,        # Request subtitles/chat
        'subtitleslangs': ['live_chat'],# Specifically ask for live chat
        'skip_download': True,         # Don't download the video itself
        'outtmpl': output_template,    # File naming template
        'quiet': True,                 # Suppress yt-dlp console output
        'no_warnings': True,           # Suppress warnings
        'logtostderr': False,          # Don't log errors to console (we log manually)
        'ignoreerrors': True,          # Try to continue on non-fatal errors
    }
    subtitle_file = f"{video_id}.live_chat.json"
    # Ensure old file doesn't interfere
    if os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
        except OSError as e:
            logging.warning(f"Could not remove old chat file {subtitle_file}: {e}")

    try:
        logging.info(f"Attempting to download live chat for {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=True) # Download the chat
            # Check if subtitles were actually downloaded
            requested_subs = result.get('requested_subtitles')
            if requested_subs and 'live_chat' in requested_subs:
                # yt-dlp might save with a different exact name, find it
                actual_file = requested_subs['live_chat'].get('filepath')
                if actual_file and os.path.exists(actual_file):
                    logging.info(f"Live chat downloaded to {actual_file}")
                    # Rename to expected format if different
                    if actual_file != subtitle_file:
                         os.rename(actual_file, subtitle_file)
                    return subtitle_file
                else:
                    logging.warning(f"yt-dlp reported success but chat file not found for {video_id}.")
                    return None
            else:
                logging.warning(f"Live chat replay not available or download failed for {video_id}.")
                return None # No live chat subtitle track found/downloaded
    except yt_dlp.utils.DownloadError as e:
         # Specifically catch download errors (like 'no live chat found')
         logging.warning(f"yt-dlp download error for {video_id}: {e}")
         st.warning("Live chat replay might not be available for this video.")
         return None
    except Exception as e:
        logging.error(f"Unexpected error downloading live chat for {video_id}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during live chat download.")
        return None


def parse_jsonl(file_path):
    """Parses a JSON Lines file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                     logging.warning(f"Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {e}")
                     continue # Skip corrupted lines
        return data
    except FileNotFoundError:
        logging.error(f"Live chat file not found for parsing: {file_path}")
        st.error(f"Could not find the downloaded chat file: {os.path.basename(file_path)}")
        return None
    except Exception as e:
        logging.error(f"Error opening/reading file {file_path}: {e}")
        st.error(f"Error reading the chat file: {e}")
        return None

def extract_live_chat_messages(subtitle_file):
    """Extracts message texts from parsed live chat data."""
    messages = []
    authors = [] # Optional: track authors
    timestamps = [] # Optional: track timestamps

    if not subtitle_file or not os.path.exists(subtitle_file):
        logging.warning(f"Subtitle file path invalid or file does not exist: {subtitle_file}")
        return [], [], [] # Return empty lists

    parsed_data = parse_jsonl(subtitle_file)
    if parsed_data is None:
        # Error already logged by parse_jsonl
        return [], [], []

    for entry in parsed_data:
        # Structure can vary, check common paths for chat messages
        actions = entry.get('replayChatItemAction', {}).get('actions', [])
        for action in actions:
            item = action.get('addChatItemAction', {}).get('item', {})
            message_renderer = item.get('liveChatTextMessageRenderer') or item.get('liveChatPaidMessageRenderer') # Handle regular and paid messages

            if message_renderer:
                # Extract message text (can be fragmented in 'runs')
                message_content = message_renderer.get('message', {})
                runs = message_content.get('runs', [])
                full_message = "".join(run.get('text', '') for run in runs if 'text' in run).strip()

                if full_message: # Only add non-empty messages
                    messages.append(full_message)
                    # Optional: Extract author and timestamp
                    authors.append(message_renderer.get('authorName', {}).get('simpleText', 'Unknown Author'))
                    timestamps.append(message_renderer.get('timestampText', {}).get('simpleText', '')) # Timestamp string like "1:23:45"

    logging.info(f"Extracted {len(messages)} messages from {subtitle_file}")
    return messages, authors, timestamps # Return all lists


def cleanup_temp_file(file_path):
    """Safely removes a temporary file."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logging.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logging.warning(f"Error deleting temporary file {file_path}: {str(e)}")


# --- Data Aggregation ---
def get_video_analysis_data(video_url: str, api_key: str) -> dict:
    """Fetches details, downloads/parses chat, returns structured data."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    analysis_data = {
        "video_id": video_id,
        "details": None,
        "live_chat_messages": [],
        "live_chat_authors": [],
        "live_chat_timestamps": [],
        "error": None,
        "chat_file": None, # Store temp file path for cleanup
    }

    # 1. Fetch Video Details (Title, Description, Thumbnail)
    video_details = fetch_video_details(video_id, api_key)
    if not video_details:
        analysis_data["error"] = "Could not retrieve video details. Video might be private or deleted."
        # Allow proceeding without details if needed, or return here
        # return analysis_data
    analysis_data["details"] = video_details

    # 2. Download Live Chat
    subtitle_file = download_live_chat(video_url, video_id)
    analysis_data["chat_file"] = subtitle_file # Store for cleanup regardless of success

    # 3. Parse Live Chat (only if download succeeded)
    if subtitle_file:
        messages, authors, timestamps = extract_live_chat_messages(subtitle_file)
        analysis_data["live_chat_messages"] = messages
        analysis_data["live_chat_authors"] = authors # Store if needed later
        analysis_data["live_chat_timestamps"] = timestamps # Store if needed later
        if not messages:
             st.info("Live chat file was found but contained no messages or failed to parse.")
    else:
        # No warning needed here if download_live_chat already gave one
        logging.info(f"No live chat file to parse for {video_id}.")
        # st.info("Live chat replay not available or download failed.") # Optional user message

    # 4. Return collected data
    return analysis_data

# --- Analysis & Visualization Helpers ---

def get_top_comments(live_chat_messages, sentiment_labels, top_n=3):
    """Selects top N positive and negative comments based on pre-calculated sentiments."""
    positive_comments = []
    negative_comments = []

    # Check if lists have the same length
    if len(live_chat_messages) != len(sentiment_labels):
        logging.error(f"Mismatch between chat messages ({len(live_chat_messages)}) and sentiment labels ({len(sentiment_labels)})")
        st.warning("Could not reliably determine top comments due to data mismatch.")
        return [], []

    for i, comment in enumerate(live_chat_messages):
        sentiment = sentiment_labels[i] # Get the pre-calculated sentiment
        if sentiment == "Positive":
            positive_comments.append(comment)
        elif sentiment == "Negative":
            negative_comments.append(comment)
        # Neutral comments are ignored for 'top' lists

    # Return the top N, or fewer if not enough comments exist
    return positive_comments[:top_n], negative_comments[:top_n]


def plot_sentiment_pie_chart(positive_count, negative_count, neutral_count):
    """Generates a sentiment pie chart using Matplotlib."""
    total = positive_count + negative_count + neutral_count
    if total == 0:
        st.write("No comments with sentiment data to plot.")
        return None # Cannot plot if no data

    labels = [f'😊 Positive ({positive_count})', f'😠 Negative ({negative_count})', f'😐 Neutral ({neutral_count})']
    sizes = [positive_count, negative_count, neutral_count]
    # Filter out zero-sized slices to avoid plotting issues
    filtered_labels = [label for i, label in enumerate(labels) if sizes[i] > 0]
    filtered_sizes = [size for size in sizes if size > 0]

    if not filtered_sizes: # Check again after filtering
        st.write("No comments with positive, negative or neutral sentiment to plot.")
        return None

    colors = ['#90EE90', '#FFB6C1', '#D3D3D3'] # Light Green, Light Pink, Light Grey
    # Match colors to filtered data
    filtered_colors = []
    if positive_count > 0: filtered_colors.append(colors[0])
    if negative_count > 0: filtered_colors.append(colors[1])
    if neutral_count > 0: filtered_colors.append(colors[2])

    # Explode the largest slice slightly if desired
    explode = [0] * len(filtered_sizes) # No explosion by default
    # if filtered_sizes:
    #    explode[np.argmax(filtered_sizes)] = 0.1 # Explode largest slice

    fig, ax = plt.subplots(figsize=(6, 4)) # Adjust figure size
    ax.pie(filtered_sizes, explode=explode, labels=filtered_labels, colors=filtered_colors,
           autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.title("Live Chat Sentiment Distribution", fontsize=12) # Optional title
    return fig


# --- Summarization (Gemini) ---
def get_sub(video_id):
    """Fetches video transcript using youtube_transcript_api."""
    try:
        # Try fetching Vietnamese first, fallback to English
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        target_langs = ['vi', 'en']
        transcript = None
        for lang in target_langs:
             try:
                  transcript = transcript_list.find_generated_transcript([lang])
                  logging.info(f"Found '{lang}' transcript for {video_id}")
                  break # Stop after finding the first preferred language
             except Exception:
                  continue # Try next language

        if not transcript:
             logging.warning(f"No suitable transcript found (tried {target_langs}) for {video_id}")
             st.warning("Could not find a Vietnamese or English transcript for this video.")
             return None

        transcript_data = transcript.fetch()
        concatenated_text = ' '.join([segment['text'] for segment in transcript_data])
        logging.info(f"Successfully fetched and concatenated transcript for {video_id} ({len(concatenated_text)} chars)")
        return concatenated_text
    except Exception as e:
        logging.error(f"Error getting subtitles for video ID {video_id}: {e}", exc_info=True)
        st.error(f"An error occurred while fetching the transcript: {e}")
        return None

# Define the prompt for the Gemini model (Consider making it more robust)
prompt_template = """
You are a helpful assistant specializing in summarizing YouTube video transcripts.
Please provide a concise summary (around 200-300 words) of the key points discussed in the following transcript.
Focus on the main topics and conclusions. Present the summary as bullet points or a short paragraph.

Transcript:
"{transcript_text}"

Summary:
"""

def get_gemini_response(transcript_text):
    """Generates summary using Gemini API."""
    if not transcript_text:
        st.error("Cannot generate summary from an empty transcript.")
        return None
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY":
         st.error("Gemini API Key is not configured. Cannot generate summary.")
         return None

    try:
        model = genai.GenerativeModel("gemini-1.5-flash") # Use a specific model
        full_prompt = prompt_template.format(transcript_text=transcript_text[:20000]) # Limit transcript length if needed
        response = model.generate_content(full_prompt)
        logging.info("Successfully generated summary using Gemini.")
        return response.text
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}", exc_info=True)
        st.error(f"An error occurred while generating the summary with Gemini: {e}")
        return None


# --- Streamlit App UI ---
st.set_page_config(page_title="🎥 YouTube Analysis", layout="wide") # Use wide layout
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Analysis 🎯</h1>", unsafe_allow_html=True)

# Initialize session state for storing results if it doesn't exist
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = [] # Store results for multiple videos if needed
if 'last_youtube_link' not in st.session_state:
    st.session_state.last_youtube_link = ""


# --- Input Area ---
new_youtube_link = st.text_input("🔗 Enter YouTube Video Link:", key="youtube_link_input", placeholder="e.g., https://www.youtube.com/watch?v=...")

# Clear results if URL is cleared or changed significantly (optional)
if new_youtube_link != st.session_state.last_youtube_link:
     st.session_state.analysis_results = [] # Clear previous results on new link
     st.session_state.last_youtube_link = new_youtube_link
     # Clear generated summary state if needed

analyze_button = st.button("🔍 Analyze Video")

# --- Processing Logic ---
if analyze_button and new_youtube_link:
    video_id = extract_video_id(new_youtube_link)
    if video_id:
        st.session_state.analysis_results = [] # Clear previous before new analysis
        with st.spinner('📊 Analyzing video... Please wait.'):
            try:
                # 1. Get Base Data (Details, Chat)
                raw_data = get_video_analysis_data(new_youtube_link, API_KEY)

                # Handle potential errors from data fetching
                if raw_data.get("error"):
                    st.error(raw_data["error"])
                    # Clean up temp file even if there was an error fetching details
                    cleanup_temp_file(raw_data.get("chat_file"))

                else:
                    processed_result = {"video_id": video_id} # Start building result dict

                    # --- Process Details ---
                    video_details = raw_data.get("details")
                    if video_details:
                        processed_result['title'] = video_details.get('title', 'N/A')
                        processed_result['thumbnail_url'] = video_details.get('thumbnail_url')
                        processed_result['description_raw'] = video_details.get('description', '')
                        # Preprocess and analyze description
                        processed_result['description_clean'] = preprocess_model_input_str(processed_result['description_raw'], processed_result['title'])
                        desc_sentiment, _ = analyze_sentiment(processed_result['description_clean'])
                        processed_result['description_sentiment'] = desc_sentiment
                    else:
                        processed_result['title'] = 'Title Not Available'
                        processed_result['thumbnail_url'] = None
                        processed_result['description_raw'] = ''
                        processed_result['description_clean'] = ''
                        processed_result['description_sentiment'] = 'N/A'

                    # --- Process Live Chat ---
                    live_chat_messages = raw_data.get("live_chat_messages", [])
                    processed_result['live_chat_raw'] = live_chat_messages

                    if live_chat_messages:
                        st.write(f"Found {len(live_chat_messages)} live chat messages. Analyzing sentiment...")
                        chat_sentiments = []
                        cleaned_chats = []
                        # Analyze sentiment for each chat message
                        # Consider batching for performance if > 1000s messages
                        prog_bar = st.progress(0)
                        for i, chat in enumerate(live_chat_messages):
                            cleaned_chat = preprocess_model_input_str(chat)
                            cleaned_chats.append(cleaned_chat) # Store cleaned chat for display?
                            sentiment, _ = analyze_sentiment(cleaned_chat)
                            chat_sentiments.append(sentiment)
                            prog_bar.progress((i + 1) / len(live_chat_messages))

                        processed_result['live_chat_sentiments'] = chat_sentiments
                        processed_result['live_chat_cleaned'] = cleaned_chats # Optional

                        # Calculate counts
                        positive_count = chat_sentiments.count("Positive")
                        negative_count = chat_sentiments.count("Negative")
                        neutral_count = chat_sentiments.count("Neutral")
                        processed_result['sentiment_counts'] = {
                            "positive": positive_count,
                            "negative": negative_count,
                            "neutral": neutral_count,
                            "total": len(chat_sentiments)
                        }

                        # Get Top Comments
                        pos_top, neg_top = get_top_comments(live_chat_messages, chat_sentiments)
                        processed_result['top_positive_comments'] = pos_top
                        processed_result['top_negative_comments'] = neg_top
                    else:
                        st.info("No live chat messages found or extracted.")
                        processed_result['live_chat_sentiments'] = []
                        processed_result['sentiment_counts'] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
                        processed_result['top_positive_comments'] = []
                        processed_result['top_negative_comments'] = []

                    # Add the complete result to session state
                    st.session_state.analysis_results.append(processed_result)
                    st.success("✅ Analysis Complete!")

            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                logging.error(f"Analysis failed for URL {new_youtube_link}: {e}", exc_info=True)
            finally:
                 # Ensure cleanup happens even if analysis fails midway
                 cleanup_temp_file(raw_data.get("chat_file") if 'raw_data' in locals() else None)

    elif new_youtube_link: # Input provided but ID extraction failed
        st.error("❌ Invalid YouTube URL. Please enter a valid link.")

# --- Display Results using Tabs ---
if not st.session_state.analysis_results:
    st.info("Enter a YouTube video URL and click 'Analyze Video' to see the results.")
else:
    # Display analysis for the *first* (and likely only) result in the list
    # If you want to handle multiple analyses, you'd loop here or select one
    result = st.session_state.analysis_results[0]
    video_id = result['video_id']

    st.header(f"Analysis Results for: {result.get('title', video_id)}")

    # Create the tabs
    tab1, tab2, tab3 = st.tabs(["📊 Video Info", "💬 Live Chat Analysis", "📝 Summary"])

    # --- Tab 1: Video Info ---
    with tab1:
        st.subheader("📹 Video Thumbnail & Details")
        col1, col2 = st.columns([1, 2]) # Ratio for image and text
        with col1:
            if result.get('thumbnail_url'):
                st.image(result['thumbnail_url'], width=250) # Adjust width as needed
            else:
                st.write("(No thumbnail available)")
        with col2:
            st.markdown(f"**Title:** {result.get('title', 'N/A')}")
            # Add other details if fetched:
            # st.write(f"**Channel:** {result.get('channel_title', 'N/A')}")
            # st.write(f"**Uploaded:** {result.get('upload_date', 'N/A')}")

        st.subheader("📝 Description")
        st.markdown(result.get('description_raw', "No description found.")) # Show raw description

        st.subheader("🤔 Description Sentiment")
        sentiment = result.get('description_sentiment', 'N/A')
        if sentiment == "Positive":
             st.success(f"**Sentiment:** {sentiment}")
        elif sentiment == "Negative":
             st.error(f"**Sentiment:** {sentiment}")
        elif sentiment == "Neutral":
             st.info(f"**Sentiment:** {sentiment}")
        else:
             st.warning(f"**Sentiment:** {sentiment}") # N/A or Error

    # --- Tab 2: Live Chat Analysis ---
    with tab2:
        st.subheader("💬 Live Chat Messages & Sentiment")
        chat_messages = result.get('live_chat_raw', [])
        chat_sentiments = result.get('live_chat_sentiments', [])
        sentiment_counts = result.get('sentiment_counts', {"total": 0})

        if chat_messages and chat_sentiments and len(chat_messages) == len(chat_sentiments):
            # Create DataFrame for the table
            chat_df = pd.DataFrame({
                'Message': chat_messages,
                'Sentiment': chat_sentiments
            })
            st.dataframe(chat_df, height=300, use_container_width=True) # Show table
            st.caption(f"Total Analyzed Messages: {sentiment_counts.get('total', 0)}")
        elif chat_messages:
            st.warning("Could not display chat sentiments (data mismatch). Showing raw messages.")
            st.dataframe(pd.DataFrame({'Message': chat_messages}), height=300, use_container_width=True)
        else:
            st.info("No live chat messages were found or extracted for this video.")

        # Display Pie Chart if data exists
        st.subheader("📈 Sentiment Distribution (Pie Chart)")
        if sentiment_counts.get('total', 0) > 0:
            fig = plot_sentiment_pie_chart(
                sentiment_counts.get('positive', 0),
                sentiment_counts.get('negative', 0),
                sentiment_counts.get('neutral', 0)
            )
            if fig:
                st.pyplot(fig)
            else:
                st.info("Not enough sentiment data to create a pie chart.")
        else:
            st.info("No sentiment data available to plot.")


        # Display Top 3 Comments
        st.subheader("🏆 Top Comments")
        top_positive = result.get('top_positive_comments', [])
        top_negative = result.get('top_negative_comments', [])

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("<h4 style='color: #28a745;'>👍 Top Positive</h4>", unsafe_allow_html=True)
            if top_positive:
                for comment in top_positive:
                    st.markdown(f"<div style='background-color: #e9f7ef; border-left: 5px solid #28a745; padding: 8px; margin-bottom: 5px; border-radius: 3px;'>{comment}</div>", unsafe_allow_html=True)
            else:
                st.info("No positive comments found.")

        with col_neg:
            st.markdown("<h4 style='color: #dc3545;'>👎 Top Negative</h4>", unsafe_allow_html=True)
            if top_negative:
                for comment in top_negative:
                    st.markdown(f"<div style='background-color: #fceded; border-left: 5px solid #dc3545; padding: 8px; margin-bottom: 5px; border-radius: 3px;'>{comment}</div>", unsafe_allow_html=True)
            else:
                st.info("No negative comments found.")

    # --- Tab 3: Summary ---
    with tab3:
        st.subheader("📜 Video Summary (Generated by AI)")

        # Use a unique key for the summary in session state based on video_id
        summary_key = f"summary_{video_id}"

        # Display summary if already generated
        if summary_key in st.session_state:
            st.markdown(f"<div style='background-color: #e7f3fe; border-left: 5px solid #0d6efd; padding: 10px; border-radius: 3px;'>{st.session_state[summary_key]}</div>", unsafe_allow_html=True)
            # Optionally add a button to regenerate
            if st.button("🔄 Regenerate Summary", key=f"regen_summary_{video_id}"):
                 del st.session_state[summary_key] # Remove old summary
                 st.rerun() # Rerun to trigger generation below

        # Show generate button if no summary exists yet
        elif st.button("✍️ Generate Summary", key=f"generate_summary_{video_id}"):
            with st.spinner("⏳ Generating summary..."):
                transcript = get_sub(video_id)
                if transcript:
                    summary = get_gemini_response(transcript)
                    if summary:
                        st.session_state[summary_key] = summary # Store the generated summary
                        st.rerun() # Rerun the script to display the summary
                    else:
                        st.error("❌ Failed to generate summary.")
                        # Keep the button visible for another try
                else:
                    st.error("❌ Could not retrieve transcript to generate summary.")
                    # Keep the button visible
        else:
             st.info("Click the button above to generate an AI summary of the video transcript.")

# --- Footer or additional info ---
st.markdown("---")
st.caption("Sentiment analysis powered by PhoBERT. Summarization by Google Gemini.")
