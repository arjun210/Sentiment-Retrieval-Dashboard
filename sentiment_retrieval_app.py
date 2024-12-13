# Step 1: Importing Libraries
import pandas as pd
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('opinion_lexicon')
nltk.download('punkt')

st.write("### Step 1: Libraries and Environment Ready")


# Step 2: Loading Dataset

# Function to load dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Sampling 20% of the data for faster performance
    return df.sample(frac=0.2, random_state=42).reset_index(drop=True)


# Load the dataset
file_path = 'sampled_reviews_10_percent.csv'
hotel_reviews_df = load_data(file_path)

# Displaying the dataset
st.write("### Step 2: Dataset Loaded Successfully")
st.write(hotel_reviews_df.head())


# Step 3: Data Preprocessing and Cleaning

# Function to clean text
def clean_text(review):
    review = review.lower()  # Converts to lowercase
    review = re.sub(r'[^a-z\s]', '', review)  # Removes punctuation and numbers
    return review


# Cleaning reviews
hotel_reviews_df['Cleaned_Positive_Review'] = hotel_reviews_df['Positive_Review'].apply(clean_text)
hotel_reviews_df['Cleaned_Negative_Review'] = hotel_reviews_df['Negative_Review'].apply(clean_text)

# Combining positive and negative reviews into a single column
hotel_reviews_df['Combined_Review'] = (
        hotel_reviews_df['Cleaned_Positive_Review'] + ' ' + hotel_reviews_df['Cleaned_Negative_Review']
)

# Displays the first few rows of the combined reviews
st.write("### Step 3: Data Preprocessing Completed")
st.write(hotel_reviews_df[['Positive_Review', 'Negative_Review', 'Combined_Review']].head())

# Step 4: Tokenization and Stopword Removal

from nltk.corpus import stopwords

# Gets the list of English stopwords
stop_words = set(stopwords.words('english'))


# Function to clean and tokenize reviews
def clean_and_tokenize(review):
    review = review.lower()  # Converts to lowercase
    review = re.sub(r'[^a-z\s]', '', review)  # Removes punctuation and numbers
    tokens = review.split()  # Tokenizes (split by spaces)
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return filtered_tokens


# Applying tokenization and stopword removal
hotel_reviews_df['Cleaned_Review_Tokens'] = hotel_reviews_df['Combined_Review'].apply(clean_and_tokenize)

# Displays the first few rows to check the tokenized reviews
st.write("### Step 4: Tokenization and Stopword Removal Completed")
st.write(hotel_reviews_df[['Combined_Review', 'Cleaned_Review_Tokens']].head())

# Step 5: Lemmatization

from nltk.stem import WordNetLemmatizer

# Initializes the lemmatizer
lemmatizer = WordNetLemmatizer()


# Function to lemmatize tokens without using POS tagging
def lemmatize_tokens_simplified(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


# Apply simplified lemmatization to the cleaned tokens
hotel_reviews_df['Lemmatized_Review_Tokens'] = hotel_reviews_df['Cleaned_Review_Tokens'].apply(
    lemmatize_tokens_simplified)

# Display the updated lemmatized tokens
st.write("### Step 5: Lemmatization Completed")
st.write(hotel_reviews_df[['Cleaned_Review_Tokens', 'Lemmatized_Review_Tokens']].head())

# Step 6: Sentiment Analysis Using VADER, TextBlob, and Machine Learning Classifier

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()


# Function to calculate sentiment score using VADER
def calculate_vader_sentiment_score(review):
    sentiment_scores = sid.polarity_scores(review)
    return sentiment_scores['compound']  # We use the compound score as an overall sentiment indicator


# Calculate sentiment scores for each review using VADER
hotel_reviews_df['VADER_Sentiment_Score'] = hotel_reviews_df['Combined_Review'].apply(calculate_vader_sentiment_score)


# Function to calculate sentiment score using TextBlob
def textblob_sentiment_score(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)


# Calculate sentiment scores using TextBlob
hotel_reviews_df['TextBlob_Sentiment_Score'] = hotel_reviews_df['Combined_Review'].apply(textblob_sentiment_score)


# Step 6-B: Advanced Machine Learning Classifier Integration (TF-IDF + Logistic Regression)
def classify_sentiment_with_ml(df):
    # Prepare labeled data for training
    df['Sentiment_Label'] = df['VADER_Sentiment_Score'].apply(
        lambda x: 1 if x > 0 else 0)  # 1 for Positive, 0 for Negative

    # Balance the classes using SMOTE
    smote = SMOTE(random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['Combined_Review'])
    y = df['Sentiment_Label']
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Train a Logistic Regression Classifier
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the Classifier
    y_pred = classifier.predict(X_test)
    st.write("### Machine Learning Sentiment Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Add predictions to the original DataFrame
    df['TFIDF_Sentiment_Label'] = classifier.predict(vectorizer.transform(df['Combined_Review']))

    return df


# Integrate the machine learning classifier
hotel_reviews_df = classify_sentiment_with_ml(hotel_reviews_df)

# Display the first few rows to check the sentiment scores and ML predictions
st.write("### Step 6: Sentiment Analysis Completed with Machine Learning Integration")
st.write(hotel_reviews_df[
             ['Combined_Review', 'VADER_Sentiment_Score', 'TextBlob_Sentiment_Score', 'TFIDF_Sentiment_Label']].head())

# Step 6-A: Word Cloud Visualization (Updated with Sentiment Label Creation)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure ML_Sentiment_Label exists in the dataframe
if 'ML_Sentiment_Label' not in hotel_reviews_df.columns:
    # Create a sentiment label column based on VADER Sentiment Score
    hotel_reviews_df['ML_Sentiment_Label'] = hotel_reviews_df['VADER_Sentiment_Score'].apply(
        lambda x: 'positive' if x > 0 else 'negative'
    )


# Function to create and display a word cloud
def create_wordcloud(df, sentiment_label, hotel_name=None):
    # Filter reviews based on sentiment label
    filtered_reviews = df[df['ML_Sentiment_Label'] == sentiment_label]

    # Further filter by hotel name if provided
    if hotel_name:
        filtered_reviews = filtered_reviews[filtered_reviews['Hotel_Name'] == hotel_name]

    # Combine all reviews into a single string
    text = " ".join(filtered_reviews['Combined_Review'])

    if text.strip() == "":
        st.write(f"No {sentiment_label} sentiment reviews available for {hotel_name}.")
        return

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    st.write(f"### Word Cloud for {sentiment_label.capitalize()} Sentiment Reviews")
    if hotel_name:
        st.write(f"**Hotel:** {hotel_name}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # No axes for word cloud
    st.pyplot(fig)


# Add sidebar options for word cloud visualization
st.sidebar.header("Word Cloud Visualization")
selected_hotel_wc = st.sidebar.selectbox(
    "Select Hotel for Word Cloud (Optional)",
    options=["All Hotels"] + list(hotel_reviews_df['Hotel_Name'].unique())
)

if st.sidebar.button("Show Word Cloud for Positive Sentiment"):
    hotel_name = None if selected_hotel_wc == "All Hotels" else selected_hotel_wc
    create_wordcloud(hotel_reviews_df, 'positive', hotel_name)

if st.sidebar.button("Show Word Cloud for Negative Sentiment"):
    hotel_name = None if selected_hotel_wc == "All Hotels" else selected_hotel_wc
    create_wordcloud(hotel_reviews_df, 'negative', hotel_name)

# Step 7: BM25 Ranking with Sentiment Integration

import math
from collections import defaultdict

# BM25 parameters
k1 = 1.5
b = 0.75

# Average document length
average_doc_length = hotel_reviews_df['Lemmatized_Review_Tokens'].apply(len).mean()


# Function to calculate BM25 score for a term
def bm25(term, doc, N, df):
    tf = doc.count(term)
    if tf == 0:
        return 0

    doc_length = len(doc)
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / average_doc_length))))
    return score


# Function to calculate BM25 score for a query in a document
def calculate_bm25_for_doc(query, doc_tokens):
    N = len(hotel_reviews_df)
    score = 0
    for term in query.split():
        term = term.lower()
        df = len(inverted_index.get(term, []))
        score += bm25(term, doc_tokens, N, df)
    return score


# Building the inverted index for term frequencies
inverted_index = defaultdict(set)
for index, tokens in enumerate(hotel_reviews_df['Lemmatized_Review_Tokens']):
    for token in tokens:
        inverted_index[token].add(index)

# Convert sets to lists for easier use
inverted_index = {key: list(value) for key, value in inverted_index.items()}


# Function to calculate BM25 with sentiment adjustment
def calculate_bm25_with_sentiment(query, doc_tokens, vader_score, textblob_score, intended_sentiment):
    # Calculate the original BM25 score
    bm25_score = calculate_bm25_for_doc(query, doc_tokens)

    # Adjust the score based on sentiment
    sentiment_boost = (vader_score + textblob_score) / 2  # Average VADER and TextBlob scores
    if intended_sentiment == 'positive' and sentiment_boost > 0:
        bm25_score *= (1 + sentiment_boost)
    elif intended_sentiment == 'negative' and sentiment_boost < 0:
        bm25_score *= (1 + abs(sentiment_boost))

    return bm25_score


# Example query for testing
query = "great service and location"
intended_sentiment = 'positive'  # User-specified sentiment


# Apply enhanced BM25 with sentiment adjustment
@st.cache_data
def calculate_bm25_scores(df, query, intended_sentiment):
    df['Enhanced_BM25_Score'] = df.apply(
        lambda row: calculate_bm25_with_sentiment(
            query, row['Lemmatized_Review_Tokens'], row['VADER_Sentiment_Score'], row['TextBlob_Sentiment_Score'],
            intended_sentiment
        ), axis=1
    )
    return df


# Calculate and display results
hotel_reviews_df = calculate_bm25_scores(hotel_reviews_df, query, intended_sentiment)

# Display top 5 results
st.write("### Step 7: BM25 Ranking Completed")
st.write(hotel_reviews_df[['Combined_Review', 'Enhanced_BM25_Score']].sort_values(by='Enhanced_BM25_Score',
                                                                                  ascending=False).head())


# Step 8: Negative Sentiment Handling

# Function to calculate BM25 with sentiment adjustment for negative sentiment
def calculate_bm25_with_negative_handling(query, doc_tokens, vader_score, textblob_score, intended_sentiment):
    # Calculate the original BM25 score
    bm25_score = calculate_bm25_for_doc(query, doc_tokens)

    # Adjust the score based on sentiment
    sentiment_boost = (vader_score + textblob_score) / 2  # Average VADER and TextBlob scores
    if intended_sentiment == 'positive' and sentiment_boost > 0:
        bm25_score *= (1 + sentiment_boost)
    elif intended_sentiment == 'negative' and sentiment_boost < 0:
        bm25_score *= (1 + abs(sentiment_boost))

    return bm25_score


# Apply enhanced BM25 with sentiment adjustment for negative queries
@st.cache_data
def calculate_negative_bm25_scores(df, query, intended_sentiment):
    df['Enhanced_BM25_Score_Negative'] = df.apply(
        lambda row: calculate_bm25_with_negative_handling(
            query, row['Lemmatized_Review_Tokens'], row['VADER_Sentiment_Score'], row['TextBlob_Sentiment_Score'],
            intended_sentiment
        ), axis=1
    )
    return df


# Example negative query
negative_query = "bad service and location"
intended_sentiment = 'negative'  # User-specified sentiment

# Calculate BM25 scores for negative queries
hotel_reviews_df = calculate_negative_bm25_scores(hotel_reviews_df, negative_query, intended_sentiment)

# Display top 5 results for negative sentiment analysis
st.write("### Step 8: Negative Sentiment Handling Completed")
st.write(
    hotel_reviews_df[['Combined_Review', 'Enhanced_BM25_Score_Negative']].sort_values(by='Enhanced_BM25_Score_Negative',
                                                                                      ascending=False).head())

# Step 9: Creating an Enhanced Interactive Dashboard with Streamlit


# Initialize session state for controlling search processing
if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False

# Dashboard title and description
st.write("## Enhanced Sentiment-Aware Information Retrieval System")
st.write("Enter your query below, select the sentiment, and optionally filter by hotel location or date range.")

# Define layout
query_col, sentiment_col, location_col, location_search_col = st.columns([2, 1, 1, 1])

# User input for query
with query_col:
    user_query = st.text_input("Enter your query:", key="query_input")

# User input for intended sentiment
with sentiment_col:
    intended_sentiment = st.selectbox("Select the intended sentiment:", ['positive', 'negative'], key="sentiment_input")


# Attempt to extract only the country or city part from the 'Hotel_Address'
# This assumes addresses are structured with the last part being a city or country
def extract_location(address):
    if pd.isnull(address):
        return None
    # Try to split by commas and take the last meaningful part
    parts = address.split(',')
    return parts[-1].strip() if len(parts) > 1 else address.strip()


# Apply the extraction function
hotel_reviews_df['City_or_Country'] = hotel_reviews_df['Hotel_Address'].apply(extract_location)

# Get unique cities or countries
unique_locations = hotel_reviews_df['City_or_Country'].dropna().unique()

# Dropdown input for location
with location_col:
    dropdown_location = st.selectbox("Filter by Hotel Location (Optional):",
                                     options=['All Locations'] + list(unique_locations), key="location_dropdown")

# Free-text input for location (for fuzzy matching)
with location_search_col:
    search_location = st.text_input("Search by Location (Optional):", key="search_location_input")

# Date range filter
date_col1, date_col2 = st.columns(2)
with date_col1:
    start_date = st.date_input("Start Date (Optional):", value=None, key="start_date_input")
with date_col2:
    end_date = st.date_input("End Date (Optional):", value=None, key="end_date_input")


# Run the retrieval when the user enters a query
def handle_search():
    st.session_state.search_triggered = True  # Set the flag when "Search" is clicked


st.button("Search", on_click=handle_search)

# Process logic only when "Search" is triggered
if st.session_state.search_triggered:
    if st.session_state.query_input:
        # Function to filter and calculate results based on query
        @st.cache_data
        def calculate_and_filter_results(query, df, sentiment, dropdown_loc, search_loc, start_date, end_date):
            # Calculate scores based on sentiment
            df['Enhanced_BM25_Score_UI'] = df.apply(
                lambda row: calculate_bm25_with_negative_handling(
                    query, row['Lemmatized_Review_Tokens'], row['VADER_Sentiment_Score'],
                    row['TextBlob_Sentiment_Score'], sentiment
                ), axis=1
            )

            # Filter by dropdown location if provided
            if dropdown_loc and dropdown_loc != 'All Locations':
                df = df[df['City_or_Country'].str.lower() == dropdown_loc.lower()]

            # Filter by search location if provided
            if search_loc:
                df = df[df['Hotel_Address'].str.contains(search_loc, case=False, na=False)]

            # Filter by date range if provided
            if start_date and end_date:
                df['Review_Date'] = pd.to_datetime(df['Review_Date'], errors='coerce')
                df = df[
                    (df['Review_Date'] >= pd.to_datetime(start_date)) & (df['Review_Date'] <= pd.to_datetime(end_date))]

            # Return top 5 results sorted by the enhanced BM25 score
            return df.sort_values(by='Enhanced_BM25_Score_UI', ascending=False).head(5)


        # Calculate and filter results
        top_results_ui = calculate_and_filter_results(
            st.session_state.query_input,
            hotel_reviews_df,
            st.session_state.sentiment_input,
            st.session_state.location_dropdown,
            st.session_state.search_location_input,
            st.session_state.start_date_input,
            st.session_state.end_date_input,
        )

        # Display results
        if top_results_ui.empty:
            st.warning(
                "No matching reviews for this location due to a lack of data. Please try another location or adjust your query."
            )
        else:
            st.write("### Top 5 Results with Explanations:")

            for idx, row in top_results_ui.iterrows():
                st.write(f"**Review {idx + 1}:**")
                st.write(f"Combined Review: {row['Combined_Review']}")
                st.write(f"Hotel: {row['Hotel_Name']} (Review Date: {row['Review_Date']})")
                st.write(f"VADER Sentiment Score: {row['VADER_Sentiment_Score']}")
                st.write(f"TextBlob Sentiment Score: {row['TextBlob_Sentiment_Score']}")
                st.write(f"Enhanced BM25 Score: {row['Enhanced_BM25_Score_UI']}")

                explanation = (
                    f"This review was ranked highly because it closely matches the query terms "
                    f"with an Enhanced BM25 score of {row['Enhanced_BM25_Score_UI']:.2f}. "
                )
                if intended_sentiment == "positive" and row["VADER_Sentiment_Score"] > 0:
                    explanation += f"The review sentiment is positive, as selected, with a sentiment score of {row['VADER_Sentiment_Score']:.2f}."
                elif intended_sentiment == "negative" and row["VADER_Sentiment_Score"] < 0:
                    explanation += f"The review sentiment is negative, as selected, with a sentiment score of {row['VADER_Sentiment_Score']:.2f}."
                else:
                    explanation += f"However, the sentiment score of {row['VADER_Sentiment_Score']:.2f} may not perfectly match the intended sentiment."

                st.write("Explanation:", explanation)
                st.write("---")

            # Dynamic visualization
            st.write("### Query Insights")
            st.bar_chart(top_results_ui[['Hotel_Name', 'Enhanced_BM25_Score_UI']].set_index('Hotel_Name'))
    else:
        st.warning("Please enter a query.")

# Step 9-B: Advanced Sentiment Trend Visualization
import matplotlib.pyplot as plt

st.sidebar.header("Sentiment Trend Analysis")
selected_hotel = st.sidebar.selectbox(
    "Select Hotel for Sentiment Trend Analysis",
    hotel_reviews_df['Hotel_Name'].unique()
)

# Optional date range filters
st.sidebar.write("**Filter by Date Range**")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

if st.sidebar.button("Plot Sentiment Trend"):
    def plot_advanced_sentiment_trends(hotel_name, df, start_date, end_date):
        # Filter data for the selected hotel and date range
        hotel_df = df[df['Hotel_Name'] == hotel_name]
        hotel_df['Review_Date'] = pd.to_datetime(hotel_df['Review_Date'], errors='coerce')
        hotel_df = hotel_df.dropna(subset=['Review_Date'])
        hotel_df = hotel_df[(hotel_df['Review_Date'] >= pd.to_datetime(start_date)) &
                            (hotel_df['Review_Date'] <= pd.to_datetime(end_date))]

        if hotel_df.empty:
            st.warning(f"No reviews available for {hotel_name} in the selected date range.")
            return

        # Resample sentiment scores and review counts quarterly
        hotel_df.set_index('Review_Date', inplace=True)
        sentiment_trends = hotel_df[['VADER_Sentiment_Score', 'TextBlob_Sentiment_Score']].resample('Q').mean().fillna(
            0)
        review_counts = hotel_df['Combined_Review'].resample('Q').count()

        if sentiment_trends.empty:
            st.warning(f"Not enough data for sentiment trends for {hotel_name}.")
            return

        # Determine highest positive and negative sentiments
        highest_positive = sentiment_trends['VADER_Sentiment_Score'].idxmax()
        highest_negative = sentiment_trends['VADER_Sentiment_Score'].idxmin()

        # Plot sentiment trends
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(sentiment_trends.index, sentiment_trends['VADER_Sentiment_Score'], label='VADER Sentiment', color='b')
        ax1.plot(sentiment_trends.index, sentiment_trends['TextBlob_Sentiment_Score'], label='TextBlob Sentiment',
                 color='g')
        ax1.set_title(f'Sentiment Trend for {hotel_name}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Average Sentiment Score')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Annotate highest positive and negative sentiment points
        ax1.annotate(
            f"Highest Positive\n{highest_positive.strftime('%Y-%m')} ({sentiment_trends.loc[highest_positive, 'VADER_Sentiment_Score']:.2f})",
            xy=(highest_positive, sentiment_trends.loc[highest_positive, 'VADER_Sentiment_Score']),
            xytext=(highest_positive, sentiment_trends.loc[highest_positive, 'VADER_Sentiment_Score'] + 0.1),
            arrowprops=dict(facecolor='green', arrowstyle="->"), fontsize=10)

        ax1.annotate(
            f"Highest Negative\n{highest_negative.strftime('%Y-%m')} ({sentiment_trends.loc[highest_negative, 'VADER_Sentiment_Score']:.2f})",
            xy=(highest_negative, sentiment_trends.loc[highest_negative, 'VADER_Sentiment_Score']),
            xytext=(highest_negative, sentiment_trends.loc[highest_negative, 'VADER_Sentiment_Score'] - 0.1),
            arrowprops=dict(facecolor='red', arrowstyle="->"), fontsize=10)

        # Add secondary axis for review counts
        ax2 = ax1.twinx()
        ax2.bar(review_counts.index, review_counts, alpha=0.3, label='Number of Reviews', color='orange')
        ax2.set_ylabel('Number of Reviews')
        ax2.legend(loc='upper right')

        st.pyplot(fig)

        # Display insights
        st.write(f"### Insights for {hotel_name}")
        st.write(
            f"- **Highest Positive Sentiment:** {highest_positive.strftime('%Y-%m')} with a score of {sentiment_trends.loc[highest_positive, 'VADER_Sentiment_Score']:.2f}")
        st.write(
            f"- **Highest Negative Sentiment:** {highest_negative.strftime('%Y-%m')} with a score of {sentiment_trends.loc[highest_negative, 'VADER_Sentiment_Score']:.2f}")
        st.write(f"- **Average Sentiment Score (VADER):** {sentiment_trends['VADER_Sentiment_Score'].mean():.2f}")
        st.write(f"- **Average Sentiment Score (TextBlob):** {sentiment_trends['TextBlob_Sentiment_Score'].mean():.2f}")


    # Call the function with user inputs
    plot_advanced_sentiment_trends(selected_hotel, hotel_reviews_df, start_date, end_date)
