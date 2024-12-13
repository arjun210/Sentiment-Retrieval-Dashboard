# Sentiment-Retrieval-Dashboard

## Overview
This project is an **Enhanced Sentiment-Aware Information Retrieval System** designed to retrieve and analyze customer reviews for hotels. It uses advanced Natural Language Processing (NLP) techniques to rank reviews based on user queries, sentiment analysis, and more.

The interactive dashboard is built using **Streamlit**, allowing users to filter reviews by query, sentiment, location, and date range.

## Features
- **Query Handling:** Supports long queries, typo handling, and logical operators (`and`, `or`).
- **Sentiment Analysis:** Integrates VADER, TextBlob, and Machine Learning models for sentiment classification.
- **Data Visualization:** Provides word clouds, bar charts, and trend analysis.
- **Filters:** Filters reviews by location, date range, and sentiment.
- **PDF Export:** Allows users to generate a PDF summary of top reviews.

## Installation Guide

### Prerequisites
- **Python 3.8 or above** installed.
- **Git** installed on your system.

### Steps to Set Up
1. **Clone this repository**:
git clone https://github.com/arjun210/Sentiment-Retrieval-Dashboard.git

2. **Navigate to the project directory**:
cd Sentiment-Retrieval-Dashboard

3. **Install required libraries**:
pip install -r requirements.txt

4. **Add the dataset**:
- Place the dataset file (`sampled_reviews_20_percent.csv`) in the project root directory.
- Ensure the dataset contains the following columns:
  - `Hotel_Address`
  - `Review_Date`
  - `Combined_Review`
 
5. **Run the Streamlit application**:
streamlit run sentiment_retrieval_app.py


## Usage
1. Start the application with the `streamlit run` command.
2. Interact with the dashboard:
- Enter your query in the **query box**.
- Select the intended sentiment (**positive** or **negative**).
- Use the **location filters** (dropdown or free-text search) and date range to refine results.
3. Analyze the results:
- View word clouds, bar charts, and sentiment trends.
- Download a **PDF summary** of the top-ranked reviews.

## Project Structure
- **`sentiment_retrieval_app.py`**: The main Streamlit application file.
- **`requirements.txt`**: List of dependencies required for the project.
- **Dataset File** (`sampled_reviews_20_percent.csv`): Contains hotel review data for analysis.

## Contribution
We welcome contributions! To contribute:
1. Fork this repository.
2. Create a new branch:
git checkout -b feature-branch

3. Commit your changes:
git commit -m "Your message here"

4. Push your branch:
git push origin feature-branch

5. Open a pull request.

## License
This project is licensed under the **MIT License**. Feel free to use and modify the code for personal or commercial purposes.

## FAQ
1. **What if libraries are missing?**
Install missing libraries with:
pip install -r requirements.txt


2. **Why are my queries not returning results?**
Modify your query or adjust filters (location, date range, or sentiment).

3. **Where are the PDF reports saved?**
PDF reports are saved in the root directory of the project.

4. **Can I use my dataset?**
Yes, ensure the dataset includes these columns:
- `Hotel_Address`
- `Review_Date`
- `Combined_Review`

For additional questions, feel free to open an issue in the repository.

