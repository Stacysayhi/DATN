# from glob import glob
#from googleapiclient.discovery import build
import json
#import yt_dlp
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import underthesea

API_KEY = 'AIzaSyBhEqWTbT3v_jVr9VBr3HYKi3dEjKc83-M'


@st.cache_resource
def load_model():
    model_id = "wonrax/phobert-base-vietnamese-sentiment"
    # model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    return tokenizer, model


def analyze_sentiment(text):
    tokenizer, model = load_model()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
    return predictions.numpy()[0]


def preprocess_model_input_str(text, video_title=""):
    regex_pattern = (
        r"(http|www).*(\/|\/\/)\s?|[-()+*&^%$#!@\";<>\/\.\?]{3,}|\n|#.*|\w*:"
    )
    clean_str = (
        re.sub(r"\s{2,}", " ", re.sub(regex_pattern, " ", text))
        .replace(video_title, "")
        .strip()
    )
    clean_str = underthesea.word_tokenize(clean_str, format="text")
    return clean_str


def get_video_details_with_chat(video_url: str, api_key: str) -> dict:
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
    if not video_id_match:
        return {"error": "Invalid YouTube URL. Could not extract video ID."}

    video_id = video_id_match.group(1)

    # Fetch video description using YouTube API
    youtube = build("youtube", "v3", developerKey=api_key)
    try:
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if not response["items"]:
            return {"error": "Video not found. Check the URL or video ID."}

        description = response["items"][0]["snippet"]["description"]
    except Exception as e:
        return {"error": f"An error occurred while fetching video details: {str(e)}"}

    # Download live chat subtitles using yt_dlp
    ydl_opts = {
        'writesubtitles': True,    # Download subtitles
        'skip_download': True,    # Skip video download
        'subtitleslangs': ['live_chat'],  # Specify live chat subtitles
        # Set output filename to {video_id}.json
        'outtmpl': f'{video_id}',
    }
    live_chat_messages = []

    def parse_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            subtitle_file = f"{video_id}.live_chat.json"
            try:
                data = parse_jsonl(subtitle_file)
                for lc in data:
                    try:
                        lc_actions = lc.get(
                            'replayChatItemAction', {}).get('actions', [])
                        for act in lc_actions:
                            live_chat = act.get('addChatItemAction', {}).get(
                                'item', {}).get('liveChatTextMessageRenderer', None)
                            if live_chat:
                                runs = live_chat['message']['runs']
                                for run in runs:
                                    live_chat_messages.append(run['text'])
                    except:
                        continue
            except FileNotFoundError:
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Live chat file not found: {subtitle_file}"
                }
            except Exception as e:
                print(e)
                return {
                    "video_title": info_dict['title'],
                    "description": description,
                    "live_chat": [],
                    "error": f"Error parsing live chat: {str(e)}"
                }
    except Exception as e:
        print(e)
        return {
            "video_title": info_dict['title'],
            "description": description,
            "live_chat": [],
            "error": f"An error occurred while downloading live chat: {str(e)}"
        }

    return {
        "video_title": info_dict['title'],
        "description": description,
        "live_chat": live_chat_messages
    }


def get_desc_chat(video_url):
    st.write(f"Analyzing video: {video_url}")
    # video_description = """Thị trường chứng khoán tiếp tục lình xình trong phiên sáng khi tỷ giá vẫn neo ở mức
    # cao.\n\nTỷ giá và chứng khoán luôn có mối quan hệ nghịch chiều. Tỷ giá tăng thì chứng khoán thường điều chỉnh.
    # Trong giai đoạn cuối năm, nhiều yếu tố đang đưa tỷ giá lên mức cao, thị trường chứng khoán cũng có những ảnh hưởng nhất định.
    # \n\nCùng phân tích về cơ hội và thách thức của tỷ giá tăng trong chương trình Khớp lệnh
    # - Tài chính thịnh vượng hôm nay với chủ đề “Đô” vật.\n\n#KhoplenhVPBankS #Taichinhthinhvuong #Vimottuonglaithinhvuong"""
    #
    # video_live_chat = [
    #     "MBS xin cảm ơn quý nhà đầu tư đã theo dõi và để lại các câu hỏi cho các chuyên gia ",
    #     "Cho em hỏi mã HAG",
    #     "Mọi người cùng vào tương tác nhé. Welcome",
    #     "Thị trường xanh rồi",
    #     "Có ai có câu hỏi dành cho chuyên gia không ạ?",
    # ]
    # video_title = "KHỚP LỆNH 28/10/2024: ĐÔ VẬT"
    video_info = get_video_details_with_chat(video_url, API_KEY)
    video_description = video_info['description']
    video_title = video_info['video_title']
    video_live_chat = video_info['live_chat']
    clean_description = preprocess_model_input_str(
        video_description, video_title)
    clean_live_chat = [
        preprocess_model_input_str(live_chat) for live_chat in video_live_chat
    ]

    return clean_description, clean_live_chat


def main():
    st.title("Content Sentiment Analysis")

    # User input for video URL
    video_url = st.text_input(label="Enter video URL")

    if st.button("Analyze Content"):
        # Pass both video_url and playlist_id to get_desc_chat
        video_description, video_live_chat = get_desc_chat(video_url)

        sentiment_labels = ["Negative", "Neutral", "Positive"]

        # Analyze comments
        comments_results = []
        for comment in video_live_chat:
            scores = analyze_sentiment(comment)
            comments_results.append(
                {
                    "Text": comment,
                    "Sentiment": sentiment_labels[np.argmax(scores)],
                    **{
                        label: scores[i] * 100
                        for i, label in enumerate(sentiment_labels)
                    },
                }
            )

        # Analyze subtitle
        description_score = analyze_sentiment(video_description) * 100

        # Create visualization
        fig = make_subplots(
            rows=2, cols=1, subplot_titles=("Description Analysis", "Comments Analysis")
        )

        # Subtitle visualization
        fig.add_trace(
            go.Bar(
                name="Description Sentiment", x=sentiment_labels, y=description_score
            ),
            row=1,
            col=1,
        )

        # Comments visualization
        for i, label in enumerate(sentiment_labels):
            scores = [result[label] for result in comments_results]
            fig.add_trace(
                go.Bar(name=label, x=list(range(1, len(scores) + 1)), y=scores),
                row=2,
                col=1,
            )

        fig.update_layout(height=700, barmode="group")
        st.plotly_chart(fig)

        # Display results
        st.subheader("Description Analysis")
        st.write(
            f"**Overall Sentiment:** {sentiment_labels[np.argmax(description_score)]}"
        )
        st.write(
            f"**Scores:** {', '.join([f'{label}: {description_score[i]:.2f}%' for i, label in enumerate(sentiment_labels)])}"
        )
        st.write(f"**Text:** {video_description}")

        st.subheader("Comments Analysis")
        comments_df = pd.DataFrame(comments_results)
        st.dataframe(comments_df)


if __name__ == "__main__":
    main()
