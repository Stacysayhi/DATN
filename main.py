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
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini API Configuration ---
gemini_configured_successfully = False # Flag to track configuration status
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY": # Check if key is placeholder/missing
    st.warning("⚠️ Gemini API key is missing or is a placeholder. Summarization feature will be disabled.")
    logging.warning("Gemini API key is missing or is a placeholder.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("Gemini API configured successfully.")
        gemini_configured_successfully = True # Set flag on success
    except Exception as e:
        st.error(f"🚨 Failed to configure Gemini API: {e}. Summarization will be unavailable.")
        logging.error(f"Failed to configure Gemini API: {e}")
        # Keep gemini_configured_successfully = False

MODEL_PATH = ""
MODEL_FILE = "sentiment_classifier (1).pth"

# --- Model Loading ---
@st.cache_resource
def load_model():
    # ... (keep existing load_model function)
    model_path = os.path.join(MODEL_PATH, MODEL_FILE)
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    try:
        if not os.path.exists(model_path):
             st.error(f"Model file not found at path: {os.path.abspath(model_path)}")
             logging.error(f"Model file not found at path: {os.path.abspath(model_path)}")
             return None, None

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=map_location)

        try:
             model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logging.warning(f"Strict loading failed ({e}), trying non-strict or adjusting keys...")
            try:
                model.load_state_dict(state_dict, strict=False)
                logging.warning("Loaded model state dict with strict=False. Check model architecture compatibility.")
            except Exception as load_err:
                 st.error(f"Error loading model state_dict even with adjustments: {load_err}")
                 logging.error(f"Error loading model state_dict even with adjustments: {load_err}")
                 return None, None

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
    # ... (keep existing analyze_sentiment function)
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        # Error already shown in load_model
        logging.error("Model not loaded, cannot analyze sentiment.")
        return "Error", np.array([0.0, 0.0, 0.0])

    if not text or not isinstance(text, str) or text.strip() == "":
        # logging.warning("Received empty or invalid text for sentiment analysis.")
        return "Neutral", np.array([0.0, 1.0, 0.0]) # Return neutral for empty input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # No need to move model to device every time if it's already done in load_model and cached
    # model.to(device)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        sentiment_labels = ["Negative", "Neutral", "Positive"]
        predicted_class_idx = np.argmax(predictions)
        sentiment_label = sentiment_labels[predicted_class_idx]

        return sentiment_label, predictions
    except Exception as e:
        logging.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}")
        st.warning(f"Could not analyze sentiment for a piece of text.") # Avoid showing full error to user
        return "Error", np.array([0.0, 0.0, 0.0])


# --- Text Preprocessing ---
def preprocess_model_input_str(text, video_title=""):
    # ... (keep existing preprocess_model_input_str function)
    if not text or not isinstance(text, str):
        return ""
    regex_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.\S+|#\w+|\@\w+|[\n\r]+|[:;=]{2,}"
    clean_str = re.sub(regex_pattern, " ", text)
    clean_str = re.sub(r"\s{2,}", " ", clean_str)
    if video_title:
        clean_str = clean_str.replace(video_title, "")
    return clean_str.strip()

# --- YouTube Data Fetching ---
def extract_video_id(url):
    # ... (keep existing extract_video_id function)
    pattern = re.compile(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*')
    match = pattern.search(url)
    if match:
        return match.group(1)
    pattern_short = re.compile(r'youtu\.be\/([0-9A-Za-z_-]{11})')
    match_short = pattern_short.search(url)
    if match_short:
        return match_short.group(1)
    return None

def fetch_video_details(video_id, api_key):
    # ... (keep existing fetch_video_details function)
    if not api_key or api_key == "YOUR_YOUTUBE_API_KEY":
        st.error("YouTube API Key is missing or is a placeholder. Cannot fetch video details.")
        logging.error("YouTube API Key missing or placeholder.")
        return video_id, "Error: API Key Missing", "Error: API Key Missing"
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if not response["items"]:
            logging.error(f"Video not found with ID: {video_id}")
            return video_id, "Video Not Found", "Video Not Found"
        snippet = response["items"][0]["snippet"]
        title = snippet.get("title", "Video Title Unavailable")
        description = snippet.get("description", "")
        return video_id, title, description
    except Exception as e:
        logging.error(f"Error fetching video details for {video_id}: {e}")
        st.error(f"Error fetching video details: Check API key and video ID.")
        return video_id, "Error Fetching Title", "Error Fetching Description"


def download_live_chat(video_url, video_id):
    # ... (keep existing download_live_chat function)
    subtitle_file = f"{video_id}.live_chat.json"
    if os.path.exists(subtitle_file):
        try: os.remove(subtitle_file)
        except OSError as e: logging.warning(f"Could not delete existing chat file {subtitle_file}: {e}")

    ydl_opts = {
        'writesubtitles': True, 'subtitleslangs': ['live_chat'],
        'skip_download': True, 'outtmpl': f'{video_id}', 'quiet': True,
        'ignoreerrors': True, 'logtostderr': False,
    }
    try:
        logging.info(f"Attempting to download live chat for {video_id} from {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=False)
            # Optional: Check if live and not finished (might be too strict)
            # if result and result.get('is_live') and not result.get('was_live'):
            #      logging.warning(f"Video {video_id} is currently live, live chat replay may not be available yet.")
            #      return None
            ydl.download([video_url])

        if os.path.exists(subtitle_file):
            logging.info(f"Live chat downloaded successfully to {subtitle_file}")
            return subtitle_file
        else:
            logging.warning(f"Live chat file {subtitle_file} not found after download attempt for {video_id}. Live chat might be disabled, unavailable, or the video is ongoing.")
            return None
    except yt_dlp.utils.DownloadError as e:
        if "live chat replay is not available" in str(e).lower():
            logging.warning(f"Live chat replay not available for video {video_id}.")
        else:
            logging.error(f"yt-dlp DownloadError for {video_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Generic error downloading live chat for {video_id}: {e}")
        return None

def parse_jsonl(file_path):
    # ... (keep existing parse_jsonl function)
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try: data.append(json.loads(line))
                except json.JSONDecodeError as json_err:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {line.strip()} - Error: {json_err}")
                    continue
        return data
    except FileNotFoundError:
        logging.error(f"Live chat file not found during parsing: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error opening/reading file {file_path}: {e}")
        return None

def extract_live_chat_messages(subtitle_file):
    # ... (keep existing extract_live_chat_messages function)
    messages = []
    if not subtitle_file or not os.path.exists(subtitle_file):
        logging.warning(f"Subtitle file missing or not provided: {subtitle_file}")
        return messages

    data = parse_jsonl(subtitle_file)
    if data is None: return messages

    for entry in data:
        try:
            replay_action = entry.get('replayChatItemAction', {})
            actions = replay_action.get('actions', [])
            for action in actions:
                add_chat_item = action.get('addChatItemAction', {})
                item = add_chat_item.get('item', {})
                msg_renderer = item.get('liveChatTextMessageRenderer') or item.get('liveChatPaidMessageRenderer')

                if msg_renderer:
                    message_content = msg_renderer.get('message')
                    if message_content and 'runs' in message_content:
                        full_message = "".join(run.get('text', '') for run in message_content['runs'] if 'text' in run)
                        if full_message.strip(): messages.append(full_message.strip())
                    elif message_content and 'simpleText' in message_content:
                         simple_text = message_content.get('simpleText', '').strip()
                         if simple_text: messages.append(simple_text)
        except Exception as e:
            logging.warning(f"Error processing a live chat entry: {entry} - Error: {str(e)}")
            continue
    return messages

# --- Data Aggregation ---
def get_video_data(video_url: str, api_key: str) -> dict:
    # ... (keep existing get_video_data function)
    video_id = extract_video_id(video_url)
    if not video_id:
        logging.error(f"Invalid YouTube URL: {video_url}")
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    logging.info(f"Processing video ID: {video_id}")
    video_id, title, description = fetch_video_details(video_id, api_key)
    if "Error:" in title: # Check for errors from fetch_video_details
         return {"error": f"{title}. Could not fetch details for video ID: {video_id}"}


    subtitle_file = download_live_chat(video_url, video_id)
    live_chat_messages = []
    if subtitle_file:
        live_chat_messages = extract_live_chat_messages(subtitle_file)
        try:
            os.remove(subtitle_file)
            logging.info(f"Deleted temporary chat file: {subtitle_file}")
        except Exception as e:
            logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")
    else:
         logging.info(f"No subtitle file generated or found for {video_id}. Proceeding without live chat.") # Changed to info level

    clean_description = preprocess_model_input_str(description, title)
    clean_live_chat = [preprocess_model_input_str(msg) for msg in live_chat_messages] if live_chat_messages else []

    return {
        "video_id": video_id, "title": title,
        "description_raw": description, "description_clean": clean_description,
        "live_chat_raw": live_chat_messages, "live_chat_clean": clean_live_chat,
        "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/0.jpg" if video_id else None,
        "error": None
    }

# --- Analysis & Visualization ---
def analyze_all_sentiments(description_clean, live_chat_clean):
    # ... (keep existing analyze_all_sentiments function)
    desc_sentiment_label, desc_sentiment_scores = analyze_sentiment(description_clean)

    chat_sentiments = []
    chat_labels = []
    if live_chat_clean:
        progress_bar = st.progress(0, text="Analyzing live chat sentiment...")
        total_chats = len(live_chat_clean)
        for i, chat in enumerate(live_chat_clean):
            label, scores = analyze_sentiment(chat)
            chat_sentiments.append((label, scores))
            chat_labels.append(label)
            progress = (i + 1) / total_chats
            progress_text = f"Analyzing live chat sentiment... ({i+1}/{total_chats})"
            if progress_bar:
                 try: progress_bar.progress(progress, text=progress_text)
                 except Exception as e:
                      logging.warning(f"Error updating progress bar: {e}")
                      progress_bar = None
        if progress_bar: progress_bar.empty()
    else:
        logging.info("No clean live chat messages to analyze.")

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
        "chat_sentiment_labels": chat_labels,
        "chat_sentiments_full": chat_sentiments,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_analyzed_chats": total_analyzed,
    }

def get_top_comments(live_chat_raw, chat_sentiment_labels, top_n=3):
    # ... (keep existing get_top_comments function)
    positive_comments = []
    negative_comments = []
    if len(live_chat_raw) != len(chat_sentiment_labels):
        logging.error("Mismatch between raw chat messages and sentiment labels count.")
        return [], []
    for i, comment in enumerate(live_chat_raw):
        sentiment = chat_sentiment_labels[i]
        if sentiment == "Positive": positive_comments.append(comment)
        elif sentiment == "Negative": negative_comments.append(comment)
    return positive_comments[:top_n], negative_comments[:top_n]


def plot_sentiment_pie_chart(positive_count, negative_count, neutral_count, total_comments):
    # ... (keep existing plot_sentiment_pie_chart function)
    if total_comments == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No chat comments to analyze', horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig

    labels = []
    sizes = []
    colors = []
    explode = []

    if positive_count > 0:
        labels.append(f'😊 Positive ({positive_count})')
        sizes.append(positive_count)
        colors.append('#DFF0D8')
        explode.append(0.1 if positive_count >= max(negative_count, neutral_count, default=0) else 0)
    if negative_count > 0:
        labels.append(f'😠 Negative ({negative_count})')
        sizes.append(negative_count)
        colors.append('#F2DEDE')
        explode.append(0.1 if negative_count > max(positive_count, neutral_count, default=0) else 0)
    if neutral_count > 0:
        labels.append(f'😐 Neutral ({neutral_count})')
        sizes.append(neutral_count)
        colors.append('#EAEAEA')
        explode.append(0.1 if neutral_count >= max(positive_count, negative_count, default=0) else 0)

    # Adjust explode logic slightly
    valid_explode_indices = [i for i, size in enumerate(sizes) if size > 0]
    if sum(explode) > 0.1 and len(valid_explode_indices) > 0 :
         max_val = -1
         max_idx = -1
         # Find index of the actual max value among present slices
         for i in valid_explode_indices:
             if sizes[i] > max_val:
                 max_val = sizes[i]
                 max_idx = i
         if max_idx != -1:
             explode = [0.1 if i == max_idx else 0 for i in range(len(sizes))]


    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(sizes, explode=tuple(explode) if explode else None, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=140, pctdistance=0.85)
    ax.axis('equal')
    plt.title("Live Chat Sentiment Distribution", pad=20)
    # Ensure plot doesn't get cut off in Streamlit
    plt.tight_layout()
    return fig

# --- Summarization ---
def get_sub(video_id):
    # ... (keep existing get_sub function)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        target_langs = ['vi', 'en']
        transcript = None
        found_lang = None # Track which language was found
        for lang in target_langs:
             try:
                  transcript = transcript_list.find_generated_transcript([lang])
                  found_lang = lang
                  logging.info(f"Found transcript in language: {lang} for {video_id}")
                  break
             except Exception: continue

        if not transcript:
            logging.warning(f"No transcript found in {target_langs} for video ID {video_id}. Trying any available.")
            available_langs = [t.language for t in transcript_list]
            try:
                transcript = transcript_list.find_generated_transcript(available_langs)
                found_lang = transcript.language
                logging.info(f"Found transcript in language: {found_lang} for {video_id}")
            except Exception as e:
                 logging.error(f"Could not find any transcript for video ID {video_id}: {e}")
                 st.warning(f"Could not retrieve transcript for this video. Summary unavailable.")
                 return None

        full_transcript = transcript.fetch()
        concatenated_text = ' '.join([segment['text'] for segment in full_transcript])
        logging.info(f"Transcript retrieved successfully for {video_id} (lang: {found_lang}, length: {len(concatenated_text)} chars)")
        return concatenated_text
    except Exception as e:
        logging.error(f"Error getting transcript for video ID {video_id}: {e}")
        st.error(f"Error retrieving transcript: {e}")
        return None


GEMINI_PROMPT = """
Bạn là người tóm tắt video trên Youtube. Bạn sẽ lấy văn bản ghi chép
và tóm tắt toàn bộ video và cung cấp bản tóm tắt quan trọng theo các điểm
trong vòng 300 từ. Vui lòng cung cấp bản tóm tắt của văn bản được đưa ra ở đây:
"""

def get_gemini_response(transcript_text):
    """Generates summary using Google Gemini API."""
    # **** MODIFIED SECTION ****
    global gemini_configured_successfully # Access the global flag

    if not gemini_configured_successfully: # Check the flag first
        st.error("🚨 Gemini API was not configured successfully at startup. Cannot generate summary.")
        logging.error("Attempted to generate summary, but Gemini API configuration failed earlier.")
        return None

    if not transcript_text:
        logging.error("Cannot generate summary from empty transcript.")
        st.warning("Transcript text is empty, cannot generate summary.")
        return None
    # **** END MODIFIED SECTION ****

    try:
        # Ensure the genai library itself is available
        if not genai:
             st.error("Gemini library (genai) is not available.")
             logging.error("Gemini library (genai) is not available.")
             return None

        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = transcript_text + "\n\n" + GEMINI_PROMPT
        response = model.generate_content(full_prompt)
        logging.info("Gemini summary generated successfully.")

        if hasattr(response, 'text'):
            return response.text
        else:
             # Log the response structure if text is missing
             logging.warning(f"Gemini response did not contain 'text' attribute. Response: {response}")
             st.error("Received an unexpected response structure from Gemini.")
             return None

    except Exception as e:
        logging.error(f"Error generating Gemini response: {e}")
        # Provide more specific feedback
        if "API key not valid" in str(e) or "permission denied" in str(e).lower() or "authentication" in str(e).lower():
             st.error(f"🚨 Gemini API authentication/permission error: {e}. Check API key permissions.")
        elif "model not found" in str(e).lower():
             st.error(f"🚨 Gemini model 'gemini-1.5-flash' not found or unavailable: {e}")
        elif "quota" in str(e).lower():
             st.error(f"🚨 Gemini API quota exceeded: {e}. Please check your usage limits.")
        else:
             st.error(f"🚨 An error occurred while generating the summary via Gemini: {e}")
        return None


# --- Streamlit App UI ---
st.set_page_config(page_title="YT Analysis", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment & Summarization 🎯</h1>", unsafe_allow_html=True)

# Initialize session state
if 'video_data' not in st.session_state: st.session_state.video_data = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'summary' not in st.session_state: st.session_state.summary = None
if 'last_youtube_link' not in st.session_state: st.session_state.last_youtube_link = ""

# --- Input Form ---
with st.form("youtube_url_form"):
    youtube_link = st.text_input("🔗 Enter YouTube Video Link:", key="youtube_link_input", value=st.session_state.last_youtube_link)
    submitted = st.form_submit_button("🔍 Analyze Video")

    if submitted:
        if not youtube_link or youtube_link != st.session_state.last_youtube_link:
            st.session_state.video_data = None
            st.session_state.analysis_results = None
            st.session_state.summary = None
            st.session_state.last_youtube_link = youtube_link

        if not youtube_link:
            st.warning("⚠️ Please enter a YouTube video link.")
            st.stop()

        # Check if API keys are placeholders before proceeding
        if not API_KEY or API_KEY == "YOUR_YOUTUBE_API_KEY":
             st.error("🚨 YouTube API Key is missing or is a placeholder. Please configure it in the script.")
             st.stop()
        # No need to check Gemini key here again, handled at startup

        st.session_state.last_youtube_link = youtube_link

        with st.spinner('🔄 Processing video... Please wait.'):
            with st.spinner('⬇️ Fetching video details and live chat...'):
                 video_data = get_video_data(youtube_link, API_KEY)
                 st.session_state.video_data = video_data

            if video_data and video_data.get("error"):
                st.error(f"🚨 {video_data['error']}")
                st.session_state.video_data = None
                st.stop()
            elif not video_data or not video_data.get("video_id"):
                st.error("🚨 Failed to retrieve essential video data.")
                st.session_state.video_data = None
                st.stop()

            with st.spinner('📊 Analyzing sentiments... This might take a while for long chats.'):
                analysis_results = analyze_all_sentiments(
                    video_data.get("description_clean", ""),
                    video_data.get("live_chat_clean", [])
                )
                st.session_state.analysis_results = analysis_results

            st.session_state.summary = None # Clear previous summary
            st.success("✅ Analysis complete! View results in the tabs below.")
            st.rerun()


# --- Display Results using Tabs ---
if st.session_state.video_data and st.session_state.analysis_results:
    video_data = st.session_state.video_data
    analysis = st.session_state.analysis_results

    tab1, tab2, tab3 = st.tabs(["📈 Video Info", "💬 Live Chat Analysis", "📝 Summary"])

    # --- Tab 1: Video Info ---
    with tab1:
        st.markdown("### Video Details")
        if video_data.get("thumbnail_url"):
             st.image(video_data["thumbnail_url"], caption=video_data.get('title', 'Video Thumbnail'), use_column_width=True)
        else: st.warning("Could not load video thumbnail.")

        st.markdown(f"**Title:** {video_data.get('title', 'N/A')}")
        st.markdown("**Description:**")
        with st.expander("Click to view/hide description", expanded=False):
             desc_text = video_data.get('description_raw', 'No description available.')
             st.markdown(f"> {desc_text}" if desc_text else "> No description available.")

        st.markdown("**Description Sentiment:**")
        desc_sentiment = analysis.get('description_sentiment', 'N/A')
        st.markdown(f"**Sentiment:** `{desc_sentiment}`")


    # --- Tab 2: Live Chat Analysis ---
    with tab2:
        st.markdown("### Live Chat Sentiment Analysis")
        raw_chat = video_data.get("live_chat_raw", [])
        chat_labels = analysis.get("chat_sentiment_labels", [])
        total_analyzed = analysis.get("total_analyzed_chats", 0)

        if not raw_chat:
            st.info("ℹ️ No live chat messages were found or downloaded for this video.")
        else:
            if len(raw_chat) == len(chat_labels):
                chat_df_data = {'Live Chat Message': raw_chat, 'Sentiment': chat_labels}
                chat_df = pd.DataFrame(chat_df_data)
                st.markdown("**Live Chat Messages and Sentiments:**")
                st.dataframe(chat_df, height=300, use_container_width=True)
            else:
                st.warning("⚠️ Mismatch between number of chat messages and analyzed sentiments. Cannot display table reliably.")
                logging.error(f"Chat/Label count mismatch: {len(raw_chat)} messages, {len(chat_labels)} labels for {video_data.get('video_id')}")

            st.markdown("**Sentiment Distribution:**")
            fig = plot_sentiment_pie_chart(
                analysis.get('positive_count', 0),
                analysis.get('negative_count', 0),
                analysis.get('neutral_count', 0),
                total_analyzed
            )
            st.pyplot(fig)

            st.markdown("**Top Comments (Sample):**")
            positive_comments, negative_comments = get_top_comments(raw_chat, chat_labels, top_n=3)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h4 style='color: #28a745;'>👍 Top Positive ({len(positive_comments)}):</h4>", unsafe_allow_html=True)
                if positive_comments:
                    for comment in positive_comments: st.markdown(f"<div style='background-color: #DFF0D8; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black; word-wrap: break-word;'>{comment}</div>", unsafe_allow_html=True)
                else: st.write("No positive comments found.")
            with col2:
                st.markdown(f"<h4 style='color: #dc3545;'>👎 Top Negative ({len(negative_comments)}):</h4>", unsafe_allow_html=True)
                if negative_comments:
                    for comment in negative_comments: st.markdown(f"<div style='background-color: #F2DEDE; padding: 8px; border-radius: 5px; margin-bottom: 5px; color: black; word-wrap: break-word;'>{comment}</div>", unsafe_allow_html=True)
                else: st.write("No negative comments found.")

    # --- Tab 3: Summary ---
    with tab3:
        st.markdown("### Video Summary (Generated by AI)")
        video_id = video_data.get("video_id")

        # Only show button if Gemini was configured correctly
        if not gemini_configured_successfully:
             st.warning("⚠️ Summarization unavailable because the Gemini API key was not configured correctly at startup.")
        elif st.session_state.summary:
            st.markdown(f"<div style='background-color: #F0F8FF; padding: 10px; border-radius: 5px; color: black; border: 1px solid #ADD8E6;'>{st.session_state.summary}</div>", unsafe_allow_html=True)
            if st.button("🔄 Regenerate Summary", key="regenerate_summary"):
                 st.session_state.summary = None
                 st.rerun()
        else:
             if st.button("📜 Generate Summary", key="generate_summary"):
                 with st.spinner("⏳ Generating summary... This may take a moment."):
                     transcript = get_sub(video_id)
                     if transcript:
                         summary_text = get_gemini_response(transcript) # Function now checks config flag
                         if summary_text:
                             st.session_state.summary = summary_text
                             st.rerun()
                         # else: error message already shown in get_gemini_response
                     # else: error message already shown in get_sub
             else:
                  st.info("ℹ️ Click the button above to generate an AI summary of the video transcript (requires successful Gemini API configuration).")


# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit, Transformers, yt-dlp, Google AI Gemini, and YouTube Data API.")
