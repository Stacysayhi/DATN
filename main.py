import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to analyze sentiment
def analyze_sentiment(text):
    # Mock sentiment analysis function
    # Returns probabilities for Negative, Neutral, Positive
    return np.random.dirichlet((1, 1, 1), size=1)[0]

def main():
    st.title("Content Sentiment Analysis")

    # User input for video URL
    video_url = st.text_input(label="Enter video URL")

    if st.button("Analyze Content"):
        # Fetch video details and live chat
        video_description, video_live_chat = get_desc_chat(video_url)

        sentiment_labels = ["Negative", "Neutral", "Positive"]

        # Analyze comments
        comments_results = []
        positive_count = 0
        negative_count = 0
        total_comments = len(video_live_chat)

        for comment in video_live_chat:
            scores = analyze_sentiment(comment)
            sentiment = sentiment_labels[np.argmax(scores)]
            
            if sentiment == "Positive":
                positive_count += 1
            elif sentiment == "Negative":
                negative_count += 1
            
            comments_results.append(
                {
                    "Text": comment,
                    "Sentiment": sentiment,
                    **{label: scores[i] * 100 for i, label in enumerate(sentiment_labels)},
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

        # Update sentiment counts
        st.write(f"**Positive Comments:** {positive_count}")
        st.write(f"**Negative Comments:** {negative_count}")
        st.write(f"**Total Comments:** {total_comments}")

if __name__ == "__main__":
    main()
