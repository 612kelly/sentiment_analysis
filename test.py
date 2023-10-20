import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib.font_manager import FontProperties


# nltk.download("stopwords")
# nltk.download("wordnet")

#########################################################

st.set_page_config(
    page_title="Sentiment Analaysis Dashboard", page_icon="📊", layout="wide"
)

adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

st.title('Sentiment Analaysis Dashboard')

#########################################################

DATA_URL = (r"ikea_reviews_sentiment2.parquet")
data = pd.read_parquet(DATA_URL)
data['publishedAtDate'] = pd.to_datetime(data['publishedAtDate'])
st.write(data['publishedAtDate'])
data['date'] = data['publishedAtDate'].dt.date
st.write(data['date'])
st.write("Start date", min(data['date']))
st.write("End date", max(data['date']))

# Function to load data and filter it based on language and date range
@st.cache_data
def load_and_filter_data(DATA_URL, language, store, year, start_date, end_date):
    data['date'] = data['publishedAtDate'].dt.date
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    
    # Group "Chinese_China," "Chinese_Taiwan," and "Chinese_Hongkong" into "Chinese"
    data['language'] = data['language'].replace(["Chinese_China", "Chinese_Taiwan", "Chinese_Hongkong"], "Chinese")

    # Ensure text of sent_res are the same
    data['sent_res'] = data['sent_res'].replace(["POSITIVE"], "positive")
    data['sent_res'] = data['sent_res'].replace(["NEGATIVE"], "negative")

    # Filter data based on language
    filtered_data = data[data['language'].str.lower().isin([language.lower()])]

    # Filter data by store name
    filtered_data = filtered_data[filtered_data['title'].isin([store])]
    
    # Filter data by date range
    filtered_data = filtered_data[
        (filtered_data['date'] >= start_date) &
        (filtered_data['date'] <= end_date)
    ]

    # Convert "publishedAtDate" to datetime
    # filtered_data['publishedatdate'] = pd.to_datetime(filtered_data['publishedatdate'])
    # filtered_data = filtered_data[filtered_data['publishedatdate'].dt.date == date]

    # Filter data by year
    filtered_data = filtered_data[filtered_data['publishedatdate'].dt.year == year]
    
    return filtered_data

# Slidebar filter
st.sidebar.header("Choose your filter")

languages = ["English", "Indonesian", "Chinese"]

# Filter 1 (select language)
language = st.sidebar.selectbox("Select the language type:", languages)

# Filter 2 (select stores)
store = st.sidebar.selectbox("Select the store:", options=data["title"].unique())

# Filter 3 (year)
year = st.sidebar.slider('year', 2016, 2023, 2022)

# Filter 4 (date range)
min_date = min(data['date'])
max_date = max(data['date'])

# # # Calculate default values within the range
default_start_date = min_date  # Set the default to the minimum date
default_end_date = max_date  # Set the default to the maximum date

# start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date)
# end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date)

start_date = st.sidebar.date_input("Start Date", min_value = min_date, max_value = max_date, value=default_start_date)
end_date = st.sidebar.date_input("End Date", min_value = min_date, max_value = max_date, value=default_end_date)

# date = st.sidebar.slider("Date Range", min_date, max_date)

# Load and filter data
with st.spinner('Loading data'):
    filtered_data = load_and_filter_data(DATA_URL, language, store, year, start_date, end_date)

# st.write(pd.to_datetime(filtered_data['publishedatdate']))

st.write("Done reading data")
if st.checkbox('Show filtered data'):
    st.subheader('Raw data')
    st.write(filtered_data)