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
    """Loads the sentiment analysis model and tokenizer."""
    model_full_path = os.path.join(MODEL_PATH, MODEL_FILE)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID) # Or appropriate class

        # Load the state dictionary from the saved .pth file
        # Use strict=False only if necessary and you understand the implications
        # It might mean the saved weights don't perfectly match the model architecture.
        model.load_state_dict(torch.load(model_full_path, map_location=torch.device('cpu')), strict=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logging.info(f"Model loaded successfully from {model_full_path} and moved to {device}")
        return tokenizer, model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_full_path}. Please ensure the file exists.")
        logging.error(f"Model file not found at {model_full_path}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model from {model_full_path}: {e}")
        logging.error(f"Error loading model from {model_full_path}: {e}")
        return None, None


def analyze_sentiment(text):
    """Analyzes the sentiment of a given text using the loaded model."""
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.error("Model loading failed. Sentiment analysis is unavailable.")
        return "Error", [0.0, 0.0, 0.0] # Return default error state

    if not text or text.isspace():
        return "Neutral", [0.0, 1.0, 0.0] # Treat empty/whitespace as Neutral

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure model is on the correct device (might be moved back to CPU by Streamlit caching)
    model.to(device)

    try:
        # Tokenize and predict
        tokenizer.padding_side = "left" # Usually set during training, ensure consistency
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device) # Reduced max_length slightly
        with torch.no_grad():
            outputs = model(**inputs)
            # Ensure outputs are moved to CPU before converting to NumPy
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        sentiment_labels = ["Negative", "Neutral", "Positive"] # Make sure this order matches model output
        predicted_class_index = np.argmax(predictions)
        sentiment_label = sentiment_labels[predicted_class_index]

        return sentiment_label, predictions.tolist() # Return scores as a list
    except Exception as e:
        logging.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}")
        st.warning(f"Could not analyze sentiment for a piece of text: {e}")
        return "Error", [0.0, 0.0, 0.0]


def preprocess_model_input_str(text, video_title=""):
    """Cleans text by removing URLs, excessive punctuation, newlines, etc."""
    if not text:
        return ""
    # Keep Vietnamese characters, remove URLs, excessive symbols, newlines
    text = re.sub(r'(?:https?|ftp):\/\/[\n\S]+', ' ', text) # More robust URL removal
    text = re.sub(r'[-<>*&^%$#@";.,!?()\']{3,}', ' ', text) # Remove 3+ consecutive symbols
    text = re.sub(r'\n+', ' ', text) # Replace newlines with space
    text = re.sub(r'#\w+', ' ', text) # Remove hashtags
    text = re.sub(r'\w*:', ' ', text) # Remove word: patterns
    text = re.sub(r'\s{2,}', ' ', text).strip() # Remove extra spaces
    if video_title:
        text = text.replace(video_title, "").strip() # Remove title if present
    return text


def extract_video_id(url):
    """Extracts the YouTube video ID from various URL formats."""
    # Regex covers youtube.com/watch, youtu.be/, youtube.com/embed/, youtube.com/v/
    pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})")
    match = pattern.search(url)
    if match:
        return match.group(1)
    logging.warning(f"Could not extract video ID from URL: {url}")
    return None

def fetch_video_details(video_id, api_key):
    """Fetches video title and description using YouTube Data API."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response.get("items"):
            logging.error(f"No video found with ID: {video_id}")
            return None, None # Video not found
        snippet = response["items"][0]["snippet"]
        return snippet.get("title", "Title Not Found"), snippet.get("description", "")
    except Exception as e:
        logging.error(f"Error fetching video details for {video_id}: {e}")
        st.error(f"Could not fetch video details using YouTube API: {e}")
        return None, None


def download_live_chat(video_url, video_id):
    """Downloads the live chat replay as a JSON file using yt-dlp."""
    ydl_opts = {
        'writesubtitles': True,
        'subtitleslangs': ['live_chat'], # Request live chat specifically
        'skip_download': True,           # Don't download the video
        'outtmpl': f'{video_id}',        # Base filename for output
        'quiet': True,                   # Suppress yt-dlp console output
        'noprogress': True,
        'noplaylist': True,              # Don't process as playlist
        'socket_timeout': 30,            # Set timeout
        # 'verbose': True,                 # Uncomment for debugging yt-dlp issues
        'no_warnings': True,             # Suppress some yt-dlp warnings
        'ignoreerrors': True,            # Try to continue on errors
        'format': 'best',                # Needed sometimes even with skip_download
        'forcejson': True,               # Ensure JSON format for chat
    }
    subtitle_file = f"{video_id}.live_chat.json"

    # Clean up old file if it exists
    if os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
        except OSError as e:
            logging.warning(f"Could not remove pre-existing chat file {subtitle_file}: {e}")

    try:
        logging.info(f"Attempting to download live chat for {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)

        if os.path.exists(subtitle_file):
            logging.info(f"Live chat downloaded successfully: {subtitle_file}")
            return subtitle_file
        else:
            logging.warning(f"yt-dlp finished but live chat file {subtitle_file} not found. Live chat might be unavailable.")
            return None
    except yt_dlp.utils.DownloadError as e:
        # Specifically catch download errors (like no live chat available)
        logging.warning(f"yt-dlp DownloadError for {video_id}: {e}. Likely no live chat available.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading live chat for {video_id}: {e}")
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
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {e} - Line: {line[:100]}")
                    continue
        return data
    except FileNotFoundError:
        logging.error(f"Live chat file not found during parsing: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error opening/reading live chat file {file_path}: {e}")
        return None


def extract_live_chat_messages(subtitle_file):
    """Extracts message texts from the parsed live chat JSON data."""
    messages = []
    if not subtitle_file:
        logging.info("No subtitle file provided for chat extraction.")
        return messages

    data = parse_jsonl(subtitle_file)
    if not data:
        logging.warning(f"Parsing failed or returned no data for {subtitle_file}.")
        return messages

    message_count = 0
    for entry in data:
        try:
            # Structure can vary slightly, check common paths
            actions = entry.get('replayChatItemAction', {}).get('actions', [])
            for action in actions:
                 item = action.get('addChatItemAction', {}).get('item', {})
                 message_renderer = item.get('liveChatTextMessageRenderer') or item.get('liveChatPaidMessageRenderer') # Handle paid messages too

                 if message_renderer and 'message' in message_renderer:
                    message_parts = message_renderer['message'].get('runs', [])
                    full_message = ''.join(part.get('text', '') for part in message_parts if 'text' in part).strip()
                    if full_message:
                        messages.append(full_message)
                        message_count += 1
        except Exception as e:
            logging.warning(f"Error processing a live chat entry: {e} - Entry: {str(entry)[:200]}")
            continue

    logging.info(f"Extracted {message_count} messages from {subtitle_file}")
    return messages


def get_desc_chat(video_url, api_key):
    """Fetches description and live chat, returning cleaned versions and raw chat."""
    st.write(f"Analyzing video: {video_url}") # Keep user informed
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL. Could not extract video ID.")
        return None, [], "Invalid URL", [] # Return consistent types

    # 1. Fetch Video Title and Description
    video_title, video_description = fetch_video_details(video_id, api_key)
    if video_title is None: # Check if fetching failed
         st.error("Failed to retrieve video details from YouTube API.")
         return None, [], "Title Unavailable", []

    # 2. Download and Parse Live Chat
    subtitle_file = download_live_chat(video_url, video_id)
    raw_live_chat_messages = extract_live_chat_messages(subtitle_file) # Pass the filename

    # 3. Clean up the temp chat file
    if subtitle_file and os.path.exists(subtitle_file):
        try:
            os.remove(subtitle_file)
            logging.info(f"Deleted temporary chat file: {subtitle_file}")
        except Exception as e:
            logging.warning(f"Could not delete temporary file {subtitle_file}: {e}")

    # 4. Preprocess Text
    clean_description = preprocess_model_input_str(video_description or "", video_title)
    # Preprocess only non-empty messages
    clean_live_chat = [preprocess_model_input_str(msg) for msg in raw_live_chat_messages if msg]

    logging.info(f"Retrieved: Title='{video_title}', Desc Length={len(clean_description)}, Raw Chat Count={len(raw_live_chat_messages)}, Clean Chat Count={len(clean_live_chat)}")
    return clean_description, clean_live_chat, video_title, raw_live_chat_messages


def get_top_comments(live_chat, sentiment_labels, top_n=3):
    """Selects the top N positive and negative comments based on pre-calculated sentiment."""
    positive_comments = []
    negative_comments = []

    if len(live_chat) != len(sentiment_labels):
        logging.error("Mismatch between number of comments and sentiment labels in get_top_comments.")
        return [], [] # Return empty lists on mismatch

    for i, comment in enumerate(live_chat):
        sentiment = sentiment_labels[i]
        if sentiment == "Positive":
            positive_comments.append(comment)
        elif sentiment == "Negative":
            negative_comments.append(comment)

    # No need to sort by score here as scores aren't directly available, just labels
    # Return the first N found
    return positive_comments[:top_n], negative_comments[:top_n]


def plot_sentiment_pie_chart(positive_count, negative_count, total_comments):
    """Generates a Matplotlib pie chart for sentiment distribution."""
    if total_comments <= 0:
        # Handle case with no comments
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No comments to analyze', horizontalalignment='center', verticalalignment='center')
        ax.axis('off') # Hide axes
        return fig

    neutral_count = total_comments - positive_count - negative_count
    # Ensure neutral is not negative due to potential rounding or errors elsewhere
    neutral_count = max(0, neutral_count)

    labels = []
    sizes = []
    colors = []
    explode = [] # To slightly separate slices

    if positive_count > 0:
        labels.append(f'😊 Positive ({positive_count})')
        sizes.append(positive_count)
        colors.append('#32CD32') # Green
        explode.append(0.05)
    if negative_count > 0:
        labels.append(f'😠 Negative ({negative_count})')
        sizes.append(negative_count)
        colors.append('#FF6347') # Red
        explode.append(0.05)
    if neutral_count > 0:
        labels.append(f'😐 Neutral ({neutral_count})')
        sizes.append(neutral_count)
        colors.append('#D3D3D3') # Grey
        explode.append(0)

    if not sizes: # If all counts were zero (should be caught by total_comments check, but safe)
         fig, ax = plt.subplots()
         ax.text(0.5, 0.5, 'No sentiment data', horizontalalignment='center', verticalalignment='center')
         ax.axis('off')
         return fig

    fig, ax = plt.subplots(figsize=(6, 4)) # Adjust figure size if needed
    ax.pie(sizes,
           labels=labels,
           colors=colors,
           autopct='%1.1f%%',
           startangle=90,
           pctdistance=0.85, # Position percentage inside slices
           explode=explode) # Apply explode effect

    # Draw circle for doughnut chart effect (optional)
    # centre_circle = plt.Circle((0,0),0.70,fc='white')
    # fig.gca().add_artist(centre_circle)

    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Live Chat Sentiment Distribution", pad=20)
    # plt.tight_layout() # Adjust layout
    return fig


def get_sub(video_id):
    """Retrieves and concatenates the transcript for a given video ID."""
    try:
        # Try fetching Vietnamese first, fall back to English if needed
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            target_langs = ['vi', 'en'] # Prioritize Vietnamese
            transcript = None
            for lang in target_langs:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    logging.info(f"Found '{lang}' transcript for {video_id}")
                    break # Found one, stop searching
                except Exception:
                    continue # Try next language

            if not transcript:
                 logging.warning(f"No 'vi' or 'en' transcript found for {video_id}. Attempting any available.")
                 # If specific languages not found, try getting any transcript
                 transcript = transcript_list.find_generated_transcript(transcript_list.languages)

            transcript_data = transcript.fetch()

        except Exception as find_err:
             logging.error(f"Could not find any transcript for video ID {video_id}: {find_err}")
             st.warning(f"Could not find any transcript for video ID {video_id}.")
             return None


        if not transcript_data:
            logging.warning(f"Transcript fetched but is empty for video ID {video_id}")
            st.warning(f"Transcript appears to be empty for video ID {video_id}.")
            return None

        # Concatenate text segments
        concatenated_text = ' '.join([segment['text'] for segment in transcript_data if 'text' in segment])
        logging.info(f"Successfully retrieved and concatenated transcript for {video_id}. Length: {len(concatenated_text)}")
        return concatenated_text

    except Exception as e:
        logging.error(f"Error getting subtitles for video ID {video_id}: {e}")
        st.error(f"An error occurred while fetching the transcript: {e}")
        return None


# Define the prompt for the Gemini model (Vietnamese)
prompt = """
Bạn là một trợ lý tóm tắt video YouTube. Nhiệm vụ của bạn là đọc bản ghi đầy đủ của video được cung cấp
và tạo ra một bản tóm tắt ngắn gọn, tập trung vào các điểm chính và nội dung quan trọng nhất.
Bản tóm tắt nên được trình bày dưới dạng các gạch đầu dòng hoặc một đoạn văn mạch lạc, trong khoảng 150-300 từ.
Vui lòng tóm tắt nội dung từ bản ghi sau đây:
"""

def get_gemini_response(transcript_text):
    """Generates a summary using the Gemini API."""
    if not transcript_text or transcript_text.isspace():
        logging.warning("Transcript text is empty, cannot generate summary.")
        return "Error: Transcript was empty."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash") # Or other suitable model like gemini-pro
        full_prompt = transcript_text + "\n\n" + prompt # Combine transcript and prompt
        logging.info(f"Sending text length {len(full_prompt)} to Gemini for summarization.")
        # Add safety settings if needed
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
        # response = model.generate_content(full_prompt, safety_settings=safety_settings)
        response = model.generate_content(full_prompt)

        # Check for safety blocks or empty response
        if not response.parts:
            logging.warning("Gemini response was blocked or empty.")
            # Try to get candidate information if available
            try:
                 block_reason = response.prompt_feedback.block_reason
                 logging.warning(f"Gemini content blocked. Reason: {block_reason}")
                 return f"Error: Content generation blocked by safety filters (Reason: {block_reason})."
            except Exception:
                 return "Error: Content generation failed (empty response)."

        logging.info("Successfully received summary from Gemini.")
        return response.text
    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        st.error(f"Failed to generate summary using Gemini: {e}")
        return f"Error during summary generation: {e}"

# --- Streamlit App UI and Logic ---

# Setup Streamlit app
st.set_page_config(page_title="🎥 YouTube Video Sentiment & Summary", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = [] # Stores the analysis results for the current video
if 'last_youtube_link' not in st.session_state:
     st.session_state.last_youtube_link = ""

# Input field
youtube_link = st.text_input(
    "🔗 Enter YouTube Video Link:",
    key="youtube_link_input",
    value=st.session_state.get("last_youtube_link", ""),
    placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)

# Placeholder for results - allows clearing/updating content below the input field
results_placeholder = st.container()

# Analyze Button Logic
if st.button("🔍 Analyze Video", key="analyze_button"):
    submitted_link = youtube_link.strip()

    if not submitted_link:
        st.session_state.responses = []
        st.session_state.last_youtube_link = ""
        results_placeholder.warning("Input cleared. Enter a YouTube URL.")
    elif submitted_link != st.session_state.last_youtube_link:
        # New link entered, clear previous results and analyze
        st.session_state.responses = [] # Clear previous results
        st.session_state.last_youtube_link = submitted_link # Store the new link

        with results_placeholder: # Show spinner within the results area
            with st.spinner('🧠 Analyzing video... Fetching data, analyzing sentiment...'):
                video_id = extract_video_id(submitted_link)
                if video_id:
                    try:
                        # Ensure the correct API_KEY from config is used
                        clean_description, clean_live_chat, video_title, live_chat_messages = get_desc_chat(submitted_link, API_KEY)

                        if video_title == "Invalid URL" or video_title == "Title Unavailable":
                             # Error handled within get_desc_chat, stop processing
                             st.error(f"Could not process video: {video_title}")

                        else:
                            # --- Sentiment Analysis ---
                            # Analyze description
                            desc_sentiment_label, _ = analyze_sentiment(clean_description)
                            logging.info(f"Description Sentiment: {desc_sentiment_label}")

                            # Analyze live chat messages (only if chat exists)
                            chat_sentiment_labels = []
                            if clean_live_chat:
                                for chat_text in clean_live_chat:
                                    label, _ = analyze_sentiment(chat_text)
                                    chat_sentiment_labels.append(label)
                                logging.info(f"Analyzed {len(chat_sentiment_labels)} live chat messages.")
                            else:
                                logging.info("No clean live chat messages to analyze.")


                            # --- Calculate Stats ---
                            positive_count = sum(1 for s in chat_sentiment_labels if s == "Positive")
                            negative_count = sum(1 for s in chat_sentiment_labels if s == "Negative")
                            total_analyzed_comments = len(chat_sentiment_labels)

                            # --- Get Top Comments ---
                            # Use raw messages for display, map sentiments using index
                            top_positive, top_negative = get_top_comments(live_chat_messages, chat_sentiment_labels)

                            # --- Store Results ---
                            response = {
                                'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", # Medium quality thumbnail
                                'video_details': { 'title': video_title },
                                'comments': {
                                    'total_comments': total_analyzed_comments,
                                    'positive_comments': positive_count,
                                    'negative_comments': negative_count,
                                    'positive_comments_list': top_positive,
                                    'negative_comments_list': top_negative
                                },
                                "description": clean_description or "No description provided.",
                                "video_id": video_id,
                                "sentiment_data": chat_sentiment_labels, # Store labels for the table
                                "live_chat_messages": live_chat_messages, # Store raw for table/top comments
                                "description_sentiment": desc_sentiment_label,
                                # Add 'transcript_summary': None # Initialize summary field
                            }
                            st.session_state.responses = [response] # Store the single, new response
                            logging.info(f"Analysis complete for video ID: {video_id}")

                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred during analysis: {e}")
                        logging.exception("Analysis pipeline failed.") # Log full traceback
                        st.session_state.responses = [] # Clear results on major error
                else:
                    st.error("❌ Invalid YouTube URL provided.")
                    st.session_state.responses = [] # Clear results on invalid URL
    # No 'else' needed: if button clicked with same URL, just redisplay existing results below

# --- Display Results ---
with results_placeholder:
    if not st.session_state.responses:
        # Show only if the button hasn't been clicked yet or if cleared
        if not st.session_state.get("analyze_button"):
             st.info("✨ Enter a YouTube URL and click 'Analyze Video' to see the magic!")
    else:
        # Displaying the latest (and only) response
        response = st.session_state.responses[0]
        idx = 0 # Index is always 0 now

        video_details = response.get('video_details', {})
        comments_data = response.get('comments', {}) # Renamed for clarity
        live_chat_messages = response.get('live_chat_messages', [])
        sentiment_data = response.get('sentiment_data', []) # Chat sentiment labels

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Video Info", "💬 Live Chat Analysis", "📝 Summary"])

        # --- Tab 1: Video Info ---
        with tab1:
            col1, col2 = st.columns([1, 2]) # Image column, Text column

            with col1:
                if response.get('thumbnail_url'):
                    st.image(response['thumbnail_url'], use_column_width='always', caption="Video Thumbnail")
                else:
                    st.warning("Thumbnail not available.")

            with col2:
                st.markdown(f"#### 📹 Video Title:")
                st.markdown(f"**{video_details.get('title', 'N/A')}**")

                st.markdown("---")

                st.markdown(f"#### 📝 Description:")
                desc_text = response.get('description', 'No description provided.')
                if len(desc_text) > 300: # Use expander for long descriptions
                    with st.expander("View Full Description"):
                        st.markdown(f"{desc_text}")
                else:
                    st.markdown(f"{desc_text}")


                st.markdown("---")

                st.markdown(f"#### 📊 Description Sentiment:")
                st.markdown(f"**{response.get('description_sentiment', 'N/A')}**")

        # --- Tab 2: Live Chat Analysis ---
        with tab2:
            st.markdown("<h3 style='text-align: center; color: #FF4500;'>Live Chat Sentiment Analysis</h3>", unsafe_allow_html=True)

            total_comments = comments_data.get('total_comments', 0)

            if total_comments > 0:
                # Layout for Pie Chart and Stats
                col_chart, col_stats = st.columns([2, 1]) # Give chart more space

                with col_chart:
                    st.markdown("<h4 style='text-align: center;'>Sentiment Distribution</h4>", unsafe_allow_html=True)
                    try:
                        fig = plot_sentiment_pie_chart(
                            comments_data.get('positive_comments', 0),
                            comments_data.get('negative_comments', 0),
                            total_comments
                        )
                        st.pyplot(fig, use_container_width=True)
                    except Exception as plot_err:
                         st.error(f"Could not generate pie chart: {plot_err}")
                         logging.error(f"Pie chart generation error: {plot_err}")


                with col_stats:
                    st.markdown("<h4 style='text-align: center;'>Statistics</h4>", unsafe_allow_html=True)
                    pos_comments = comments_data.get('positive_comments', 0)
                    neg_comments = comments_data.get('negative_comments', 0)
                    neu_comments = total_comments - pos_comments - neg_comments

                    st.metric(label="Total Comments Analyzed", value=f"{total_comments}")
                    st.metric(label="👍 Positive", value=f"{pos_comments} ({(pos_comments/total_comments)*100:.1f}%)" if total_comments else "0 (0.0%)")
                    st.metric(label="👎 Negative", value=f"{neg_comments} ({(neg_comments/total_comments)*100:.1f}%)" if total_comments else "0 (0.0%)")
                    st.metric(label="😐 Neutral", value=f"{neu_comments} ({(neu_comments/total_comments)*100:.1f}%)" if total_comments else "0 (0.0%)")


                st.markdown("---") # Visual separator

                # Live Chat Table and Top Comments Section
                st.markdown("#### 📜 Live Chat Messages & Detected Sentiment")
                if live_chat_messages and sentiment_data and len(live_chat_messages) == len(sentiment_data):
                    try:
                         # Create DataFrame using raw messages and calculated sentiments
                        df = pd.DataFrame({
                            'Live Chat Message': live_chat_messages,
                            'Sentiment': sentiment_data
                        })
                        st.dataframe(df, use_container_width=True, height=300) # Set height for scrollable table
                    except Exception as df_err:
                         st.error(f"Could not display chat dataframe: {df_err}")
                         logging.error(f"Chat dataframe error: {df_err}")

                else:
                    st.info("No live chat messages found or sentiment data mismatch.")


                # Top Comments Expander
                with st.expander("👀 View Top 3 Positive/Negative Comments"):
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown(f"<h5 style='color: #32CD32;'>👍 Top Positive Comments:</h5>", unsafe_allow_html=True)
                        pos_list = comments_data.get('positive_comments_list', [])
                        if pos_list:
                            for comment in pos_list:
                                st.markdown(f"<div style='background-color: #E8F5E9; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black; font-size: 0.9em;'>{comment}</div>", unsafe_allow_html=True)
                        else:
                            st.write("No positive comments found.")

                    with col_neg:
                        st.markdown(f"<h5 style='color: #FF6347;'>👎 Top Negative Comments:</h5>", unsafe_allow_html=True)
                        neg_list = comments_data.get('negative_comments_list', [])
                        if neg_list:
                            for comment in neg_list:
                                st.markdown(f"<div style='background-color: #FFEBEE; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black; font-size: 0.9em;'>{comment}</div>", unsafe_allow_html=True)
                        else:
                            st.write("No negative comments found.")
            else:
                st.info("ℹ️ No live chat comments were found or analyzed for this video.")


        # --- Tab 3: Summary ---
        with tab3:
            st.markdown("<h3 style='text-align: center; color: #1E90FF;'>📜 AI Generated Summary (from Transcript)</h3>", unsafe_allow_html=True)

            video_id = response.get("video_id")
            summary_key = f"summarize_{video_id}" # Unique key for the button

            # Button to generate summary
            if 'transcript_summary' not in response:
                if st.button("✨ Generate Summary", key=summary_key):
                    with st.spinner("🧘‍♀️ Generating summary... This might take a minute."):
                        try:
                            logging.info(f"Attempting to retrieve transcript for summary: {video_id}")
                            transcript = get_sub(video_id)
                            if transcript:
                                summary = get_gemini_response(transcript)
                                if summary and not summary.startswith("Error:"):
                                    # Update the specific response in session state
                                    st.session_state.responses[idx]['transcript_summary'] = summary
                                    logging.info(f"Summary generated successfully for {video_id}.")
                                    # Force rerun to display the summary immediately
                                    st.rerun()
                                elif summary: # Handle specific errors from Gemini
                                     st.error(f"❌ {summary}") # Display the error message from Gemini
                                     logging.error(f"Gemini summary generation failed for {video_id}: {summary}")
                                else:
                                    st.error("❌ Failed to generate summary from the AI model (received empty response).")
                                    logging.error(f"Gemini summary generation failed for {video_id} (empty response).")
                            else:
                                st.warning("⚠️ Could not retrieve transcript. Summary cannot be generated.")
                                logging.warning(f"Transcript retrieval failed for {video_id}, summary skipped.")
                                # Optionally store a 'no transcript' message
                                st.session_state.responses[idx]['transcript_summary'] = "Transcript not available."
                                st.rerun()

                        except Exception as e:
                            st.error(f"❌ An error occurred during summary generation: {e}")
                            logging.exception(f"Summary generation exception for {video_id}") # Log traceback

            # Display generated summary (or status)
            if 'transcript_summary' in response:
                 summary_text = response['transcript_summary']
                 if summary_text == "Transcript not available.":
                      st.warning("⚠️ Transcript not available for this video, summary cannot be generated.")
                 elif summary_text.startswith("Error:"):
                      st.error(f"❌ {summary_text}") # Show error if generation failed previously
                 else:
                      st.markdown("#### Summary:")
                      st.markdown(f"<div style='background-color: #E3F2FD; padding: 15px; border-radius: 8px; border: 1px solid #BBDEFB; color: black;'>{summary_text}</div>", unsafe_allow_html=True)
            else:
                 # If button hasn't been clicked yet
                 st.info("📄 Click the 'Generate Summary' button above to create a summary from the video transcript (if available).")

# Optional: Footer or additional info
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey; font-size: 0.8em;'>Sentiment Analysis & Summarization Tool</p>", unsafe_allow_html=True)
