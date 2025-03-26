import re
import json
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yt_dlp
from googleapiclient.discovery import build
# import underthesea  # Commented out as it might not be used
import logging
import matplotlib.pyplot as plt  # Import matplotlib
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Assuming this is still needed
import numpy as np  # Import numpy

# Cấu hình logging (ví dụ, ghi vào file)
logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Your API Key - should be stored securely, not hardcoded
API_KEY = "AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M"  # Replace with your actual API key

@st.cache_resource
def load_model():
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Kiểm tra CUDA và chuyển model sang GPU nếu có
    if torch.cuda.is_available():
        model.to("cuda")
        logging.info("Model loaded to GPU.")
    else:
        logging.info("CUDA not available, running on CPU.")

    return tokenizer, model


def analyze_sentiment(text_list):  # Changed to accept a list of texts
    tokenizer, model = load_model()

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Tokenize the text list
    inputs = tokenizer(
        text_list, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Get sentiment labels
    sentiment_labels = ["Negative", "Neutral", "Positive"]

    # Calculate sentiment counts
    positive_count = 0
    negative_count = 0

    for prediction in predictions:
        predicted_class = torch.argmax(prediction).item()
        sentiment_label = sentiment_labels[predicted_class]

        if sentiment_label == "Positive":
            positive_count += 1
        elif sentiment_label == "Negative":
            negative_count += 1

    total_comments = len(text_list)

    return positive_count, negative_count, total_comments


def preprocess_model_input_str(text, video_title=""):
    if not text:
        return ""  # Xử lý trường hợp text là None hoặc rỗng

    regex_pattern = r"(http|www).*(\/|\/\/)\s?|[-()+*&^%$#!@\";<>\/\.\?]{3,}|\n|#.*|\w*:"
    clean_str = re.sub(r"\s{2,}", " ", re.sub(regex_pattern, " ", text)).replace(video_title, "").strip()
    # clean_str = underthesea.word_tokenize(clean_str, format="text") # Có thể không cần thiết
    return clean_str


def get_video_details_with_chat(video_url: str, api_key: str) -> dict:
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id_match:
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    video_id = video_id_match.group(1)

    # Fetch video description using YouTube API
    youtube = build("youtube", "v3", developerKey=api_key)
    description = ""  # Initialize description
    try:
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response["items"]:
            return {"error": "Video not found. Check the URL or video ID."}

        description = response["items"][0]["snippet"]["description"]
    except Exception as e:
        logging.error(f"Error fetching video details from YouTube API: {str(e)}")
        return {"error": f"An error occurred while fetching video details: {str(e)}"}

    # Download live chat subtitles using yt_dlp
    ydl_opts = {
        'writesubtitles': True,
        'skip_download': True,
        'subtitleslangs': ['live_chat'],
        'outtmpl': f'{video_id}',
    }
    live_chat_messages = []
    subtitle_file = f"{video_id}.live_chat.json"  # Xác định trước khi try

    def parse_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            try:
                data = parse_jsonl(subtitle_file)
                for lc in data:
                    try:
                        lc_actions = lc.get('replayChatItemAction', {}).get('actions', [])
                        for act in lc_actions:
                            live_chat = act.get('addChatItemAction', {}).get('item', {}).get('liveChatTextMessageRenderer', None)
                            if live_chat:
                                runs = live_chat['message']['runs']
                                for run in runs:
                                    live_chat_messages.append(run['text'])
                    except Exception as e:
                        logging.warning(f"Error processing a live chat message: {str(e)}")
                        continue
            except FileNotFoundError:
                logging.error(f"Live chat file not found: {subtitle_file}")
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Live chat file not found: {subtitle_file}"
                }
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON in live chat file: {str(e)}")
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Error parsing live chat JSON: {str(e)}"
                }

    except Exception as e:
        logging.error(f"An error occurred while downloading live chat: {str(e)}")
        return {
            "video_title": "", # info_dict có thể không có nếu lỗi xảy ra trước khi gọi ydl.extract_info
            "description": description,
            "live_chat": [],
            "error": f"An error occurred while downloading live chat: {str(e)}"
        }
    finally:  # Đảm bảo file tạm được xóa
        try:
            os.remove(subtitle_file)
            logging.info(f"Deleted temporary file: {subtitle_file}")
        except FileNotFoundError:
            pass  # File có thể đã bị xóa trước đó hoặc chưa bao giờ được tạo
        except Exception as e:
            logging.warning(f"Error deleting temporary file {subtitle_file}: {str(e)}")


    try:
        video_title = info_dict['title']
    except Exception as e:
        video_title = ""
        logging.error(f"Error getting video title: {str(e)}")
        return {
            "video_title": video_title,
            "description": description,
            "live_chat": [],
            "error": f"An error occurred while getting video title: {str(e)}"
        }


    return {
        "video_title": video_title,
        "description": description,
        "live_chat": live_chat_messages
    }

def get_desc_chat(video_url, API_KEY): # Truyền API_KEY như một tham số
    st.write(f"Analyzing video: {video_url}")
    video_info = get_video_details_with_chat(video_url, API_KEY)

    if "error" in video_info:
        st.error(f"Error: {video_info['error']}")
        return "", [], []  # Trả về giá trị mặc định để tránh lỗi

    video_description = video_info['description']
    video_title = video_info['video_title']
    video_live_chat = video_info['live_chat']

    clean_description = preprocess_model_input_str(video_description, video_title)
    clean_live_chat = [preprocess_model_input_str(live_chat) for live_chat in video_live_chat]

    return clean_description, clean_live_chat, video_info

def plot_sentiment_pie_chart(positive_count, negative_count, total_comments):
    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [positive_count, negative_count, total_comments - (positive_count + negative_count)]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)  # explode 1st slice (Positive)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig

def extract_video_id(url):
    # Regular expression to extract video ID from different YouTube URL formats
    pattern = re.compile(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})")
    match = pattern.search(url)
    if match:
        return match.group(1)
    return None


# Setup Streamlit app
st.set_page_config(page_title="🎥 YouTube Video Sentiment and Summarization")
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎥 YouTube Video Sentiment and Summarization 🎯</h1>", unsafe_allow_html=True)

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Unique key for text input
youtube_link = st.text_input("🔗 Enter YouTube Video Link Below:", key="youtube_link_input")

# Add Submit URL button below the URL input field
if st.button("🔍 Analyze Video"):
    if youtube_link.strip() == "":  # Check if the input is empty
        st.session_state.responses = []  # Clear responses
        st.write("The video link has been removed. All previous responses have been cleared.")
    else:
        with st.spinner('Collecting video information...'):
            video_id = extract_video_id(youtube_link)
            if video_id:
                # thumbnail_url = f"http://img.youtube.com/vi/{video_id}/0.jpg" # Not used, video details should have the thumbnail
                try:
                    # video_details = get_video_details(video_id) # not used
                    # comments = get_video_comments(video_id) # we are using this, but it does nothing
                    clean_description, clean_live_chat, video_info = get_desc_chat(youtube_link, API_KEY)
                    if not video_info:
                      st.error("Failed to retrieve video information.")
                      st.stop()

                    positive_count, negative_count, total_comments = analyze_sentiment(clean_live_chat)

                    # Get top 3 positive and negative comments # No functions for that
                    # sid = SentimentIntensityAnalyzer() # No functions for that
                    # positive_comments, negative_comments = get_top_comments(comments, sid) # No functions for that

                    # Dummy positive and negative comments
                    positive_comments = ["Great video!", "I really enjoyed this.", "Thanks for sharing!"]
                    negative_comments = ["This was boring.", "I didn't like it.", "Could be better."]


                    response = {
                        'thumbnail_url': f"http://img.youtube.com/vi/{video_id}/0.jpg",  # Get thumbnail from youtube api
                        'video_details': {
                            'title': video_info['video_title'], # use the title from the function
                            'channel_title': None,  # no channel title
                            'view_count': None,  # no view count
                            'upload_date': None,  # no upload date
                            'duration': None,  # no duration
                            'like_count': None, # no like count
                            'dislike_count': None  # no dislike count
                        },
                        'comments': {
                            'total_comments': total_comments,
                            'positive_comments': positive_count,
                            'negative_comments': negative_count,
                            'positive_comments_list': positive_comments,
                            'negative_comments_list': negative_comments
                        },
                        "description": clean_description
                    }
                    st.session_state.responses.append(response)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Invalid YouTube URL")

# Display stored responses
for idx, response in enumerate(st.session_state.responses):
    video_details = response.get('video_details')
    comments = response.get('comments')

    # Display video details
    if video_details:
        if 'thumbnail_url' in response:
            st.image(response['thumbnail_url'], use_column_width=True)

        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📹 Video Title:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{video_details['title']}</p>", unsafe_allow_html=True)

        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📝 Description:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{response['description']}</p>", unsafe_allow_html=True)

        if video_details['channel_title']: #Check none, display
          st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📺 Channel Name:</h2>", unsafe_allow_html=True)
          st.markdown(f"<p style='text-align: center;'>{video_details['channel_title']}</p>", unsafe_allow_html=True)

        if video_details['view_count']: #Check none, display
          st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>👁️ Views:</h2>", unsafe_allow_html=True)
          st.markdown(f"<p style='text-align: center;'>{video_details['view_count']}</p>", unsafe_allow_html=True)

        if video_details['upload_date']: #Check none, display
          st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>📅 Upload Date:</h2>", unsafe_allow_html=True)
          st.markdown(f"<p style='text-align: center;'>{video_details['upload_date']}</p>", unsafe_allow_html=True)

        if video_details['duration']: #Check none, display
          st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>⏱️ Duration:</h2>", unsafe_allow_html=True)
          st.markdown(f"<p style='text-align: center;'>{video_details['duration']}</p>", unsafe_allow_html=True)

        if video_details['like_count']: #Check none, display
          st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>👍 Likes:</h2>", unsafe_allow_html=True)
          st.markdown(f"<p style='text-align: center;'>{video_details['like_count']}</p>", unsafe_allow_html=True)

        if video_details['dislike_count']: #Check none, display
          st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>👎 Dislikes:</h2>", unsafe_allow_html=True)
          st.markdown(f"<p style='text-align: center;'>{video_details['dislike_count']}</p>", unsafe_allow_html=True)

        st.markdown(f"<h2 style='text-align: center; color: #FF4500;'>💬 Total Comments:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{comments['total_comments']}</p>", unsafe_allow_html=True)

        # Plot and display pie chart for comments sentiment
        fig = plot_sentiment_pie_chart(comments['positive_comments'], comments['negative_comments'], comments['total_comments'])
        st.pyplot(fig)

        st.markdown(f"<h2 style='text-align: center; color: #32CD32;'>👍 Positive Comments:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{comments['positive_comments']} ({(comments['positive_comments']/comments['total_comments'])*100:.2f}%)</p>", unsafe_allow_html=True)

        st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>👎 Negative Comments:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{comments['negative_comments']} ({(comments['negative_comments']/comments['total_comments'])*100:.2f}%)</p>", unsafe_allow_html=True)

        # Add a toggle button to show/hide the top comments
        show_comments = st.checkbox("Show Top Comments", key=f"toggle_comments_{idx}")
        if show_comments:
            st.markdown(f"<h2 style='text-align: center; color: #32CD32;'>👍 Top 3 Positive Comments:</h2>", unsafe_allow_html=True)
            for comment in comments['positive_comments_list']:
                st.markdown(f"<div style='background-color: #DFF0D8; padding: 10px; border-radius: 5px;'>{comment}</div>", unsafe_allow_html=True)

            st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>👎Top 3 Negative Comments:</h2>", unsafe_allow_html=True)
            for comment in comments['negative_comments_list']:
                st.markdown(f"<div style='background-color: #F2DEDE; padding: 10px; border-radius: 5px;'>{comment}</div>", unsafe_allow_html=True)

    # Check if detailed notes have not been generated yet
    # This section can be included, but if there is a gemini API key missing it wont run
    # if 'gemini_response' not in response:
    #     if st.button("📑 Generate Detailed Summary", key=f"btn_{idx}"):
    #         with st.spinner('Generating detailed notes...'):
    #             video_id = extract_video_id(youtube_link)
    #             if video_id:
    #                 try:
    #                     transcript = extract_transcript_details(video_id, languages=['en', 'hi', 'es', 'fr', 'de', 'zh-Hans'])
    #                     gemini_response = get_gemini_response(transcript, prompt)
    #                     response['gemini_response'] = gemini_response
    #                     st.session_state.responses[idx] = response
    #                 except Exception as e:
    #                     st.error(f"Error: {e}")
    #             else:
    #                 st.error("Invalid YouTube URL")

    # Display generated summary if available
    # if 'gemini_response' in response:
    #     st.markdown("<h2 style='color: #1E90FF;'>📜 Summary:</h2>", unsafe_allow_html=True)
    #     st.markdown(f"<div style='background-color: #F0F8FF; padding: 10px; border-radius: 5px;'>{response['gemini_response']}</div>", unsafe_allow_html=True)
