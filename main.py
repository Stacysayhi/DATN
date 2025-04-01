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

                             # Display Metrics using st.metric for a cleaner look
                             st.metric(label="Total Comments Analyzed", value=f"{total}")

                             pos_perc = (positive / total) * 100 if total > 0 else 0
                             neg_perc = (negative / total) * 100 if total > 0 else 0
                             neu_perc = (neutral / total) * 100 if total > 0 else 0 # Calculate neutral percentage

                             st.metric(label="😊 Positive", value=f"{positive}", delta=f"{pos_perc:.1f}%")
                             st.metric(label="😠 Negative", value=f"{negative}", delta=f"{neg_perc:.1f}%")
                             st.metric(label="😐 Neutral", value=f"{neutral}", delta=f"{neu_perc:.1f}%")

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
                            st.markdown(f"<h5 style='color: #180b00;'>👍 Top Positive Comments:</h5>", unsafe_allow_html=True)
                            if comments['positive_comments_list']:
                                for comment in comments['positive_comments_list']:
                                    st.markdown(f"<div style='background-color: #e9f7ef; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #28a745;'>{comment}</div>", unsafe_allow_html=True)
                            else:
                                st.caption("No positive comments found.")

                        with col_neg:
                            st.markdown(f"<h5 style='color: #dc3545;'>👎 Top Negative Comments:</h5>", unsafe_allow_html=True)
                            if comments['negative_comments_list']:
                                for comment in comments['negative_comments_list']:
                                    st.markdown(f"<div style='background-color: #fdeded; padding: 8px; border-radius: 5px; margin-bottom: 5px; border-left: 3px solid #dc3545;'>{comment}</div>", unsafe_allow_html=True)
                            else:
                                 st.caption("No negative comments found.")
                else:
                    st.caption("No comments available to display top examples.")


        # --- Tab 3: Summary ---
        with tab3:
            st.subheader("✍️ Video Summary (via Gemini AI)")

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
                st.markdown(f"<div style='background-color: #180b00; padding: 15px; border-radius: 8px; border-left: 5px solid #0d6efd;'>{response['transcript_summary']}</div>", unsafe_allow_html=True)
            else:
                st.info("Click 'Generate Summary' to create a summary of the video transcript using AI.")

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
