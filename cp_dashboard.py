import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
import re
# from matplotlib.font_manager import FontProperties
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from dateutil.relativedelta import relativedelta
from datetime import datetime
from textblob import TextBlob


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')

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

DATA_URL = (r"ikea_reviews_sentiment3.parquet")
data = pd.read_parquet(DATA_URL)
data['publishedAtDate'] = pd.to_datetime(data['publishedAtDate'])
data['date'] = data['publishedAtDate'].dt.date

# Function to load data and filter it based on language and date range
@st.cache_data
def load_and_filter_data(DATA_URL, language, store, start_date, end_date):
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
    
    return filtered_data

# Slidebar filter
st.sidebar.header("Choose your filter")
with st.sidebar.form(key ='Form Filter'):
    # user_word = st.text_input("Enter a keyword", "habs")    
    # select_language = st.radio('Tweet language', ('All', 'English', 'French'))
    # include_retweets = st.checkbox('Include retweets in data')
    # num_of_tweets = st.number_input('Maximum number of tweets', 100)

    languages = ["English", "Indonesian", "Chinese"]

    # Filter 1 (select language)
    language = st.selectbox("Select language:", languages)

    # Filter 2 (select stores)
    store_with_most_reviews = data["title"].value_counts().idxmax()
    #store = st.selectbox("Select store:", options=data["title"].unique(), index=data["title"].unique().tolist().index(store_with_most_reviews))
    store = st.selectbox("Select store:", options=data["title"].unique())

    # Filter 3 (date range)
    min_date = min(data['date'])
    max_date = max(data['date'])

    # Calculate default values within the range
    default_start_date = max_date - relativedelta(years=2) # min_date  # Set the default to the minimum date
    default_end_date = max_date  # Set the default to the maximum date

    start_date = st.date_input("Start Date", min_value = min_date, max_value = max_date, value=default_start_date)
    end_date = st.date_input("End Date", min_value = min_date, max_value = max_date, value=default_end_date)

    if start_date > end_date:
        st.warning("Start Date cannot be after End Date. Please select a valid date range.")
        submitted1 = st.form_submit_button(label='Submit')
    else:
        submitted1 = st.form_submit_button(label='Submit')

    # submitted1 = st.form_submit_button(label = 'Submit')


# Load and filter data
with st.spinner('Loading data'):
    filtered_data = load_and_filter_data(DATA_URL, language, store, start_date, end_date)

    #st.write(filtered_data)
    
tab1, tab2, tab3 = st.tabs(["About","Overview", "Topic Modelling"])

    #########################################################

with tab1:
    st.header("About")

    st.write("This dashboard displays analysis of reviews of all IKEA Malaysia outlets obtained from Google.")
    st.write(f"Date range of data ranges from {min_date} to {max_date}.")

    st.write("You may select the filter(s) for analysis to be display on the left panel. Do click the Submit button for the analysis to run.")

    #########################################################
        
with tab2:
    
    # make 3 columns for first row of dashboard
    col1, col2, col3 = st.columns([30, 30, 30])

    #########################################################

    with col1:
        # Group data by sentiment and count occurrences
        sentiment_counts = filtered_data['sent_res'].value_counts()
        # Assign numerical values to sentiment categories
        sentiment_values = {
            "positive": 5,
            "negative": 1,
            "neutral": 3
        }

        # Calculate the overall sentiment score as a weighted average
        overall_sentiment_score = (filtered_data['sent_res'].map(sentiment_values).mean())

        # Display the overall sentiment score in a chart
        st.subheader('Overall Sentiment Level')
        #st.write(f"Overall Sentiment Score: {overall_sentiment_score}")

        # Create a gauge chart
        gauge_chart = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_sentiment_score,
            title={'text': "Overall Sentiment Level"},
            gauge={'axis': {'range': [0, 5]},
                'bar': {'color': "lightgray"},
                'steps': [
                    {'range': [0, 1.67], 'color': "#E03C32"},
                    {'range': [1.67, 3.33], 'color': "#FFD301"},
                    {'range': [3.33, 5], 'color': "#7BB662"}
                ]}))
        #gauge_chart.update_layout(width=200, height = 350)
        st.plotly_chart(gauge_chart, use_container_width=True)

    with col2:
        st.subheader("Comments' Sentiment")
        
        # Pie chart for sentiment
        pie_chart = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.3,
            #title=f'Sentiment Distribution for {language} Reviews',
            color=sentiment_counts.index,
            color_discrete_map={"positive": "#7BB662", "negative": "#E03C32", "neutral": "#FFD301"},
        )
        pie_chart.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{value} (%{percent})",
            hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
        )
        pie_chart.update_layout(showlegend=False)
        st.plotly_chart(pie_chart, use_container_width=True)

    with col3:
        total_reviews = int(filtered_data["text"].count())
        st.subheader('Number of Reviews')
        st.subheader(f"{total_reviews}")

        average_rating = round(filtered_data["stars"].mean(),1)
        star_rating = ":star:" * int(round(average_rating,0))
        st.subheader('Average Star Reviews')
        st.subheader(f"{average_rating} {star_rating}")

        # Star Analysis Chart
        # Group data by star review and count occurrences
        stars_counts = filtered_data['stars'].value_counts()

        star_bar = px.bar(filtered_data, x=stars_counts.values, y=stars_counts.index, orientation='h')
        star_bar.update_layout(height=350)
        star_bar.update_xaxes(title='Count')
        star_bar.update_yaxes(title='Overall Star Rating')
        st.plotly_chart(star_bar, use_container_width=True)

        
    #########################################################

    
    col4, col5, col6 = st.columns([45, 10, 45])

    #########################################################

    # Word Cloud
    # Function to preprocess text
    def preprocess_text(text):
        # Tokenize the text
        words = nltk.word_tokenize(text)
        
        # Remove stopwords
        try:
            words = [word for word in words if word.lower() not in stopwords.words("english")]
        except:
            nltk.download('stopwords')
            words = [word for word in words if word.lower() not in stopwords.words("english")]
        
        # Remove the word "ikea", "the", "ok", "la"
        # words = [word for word in words if word.lower() != "ikea"]
        words = [word for word in words if word.lower() not in ["ikea", "the", "ok", "la", "good", "bad"]]

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Remove single letters and apostrophes
        words = [word for word in words if len(word) > 1 and not re.match(r'^[\'\w\s]*$', word)]
        
        # Join the words back into a single string
        return " ".join(words)
        #return words

    positive_reviews = filtered_data[filtered_data['sent_res'] == 'positive']
    negative_reviews = filtered_data[filtered_data['sent_res'] == 'negative']

    # Combine positive and negative reviews text for this language
    positive_text = " ".join(positive_reviews['text'])
    negative_text = " ".join(negative_reviews['text'])
    all_text = " ".join(filtered_data['text'])

    with st.spinner('Preprocessing data for wordcloud'):
        # Preprocess the text
        try:
            preprocessed_positive_text = preprocess_text(positive_text)
        except:
            nltk.download('omw-1.4') 
            nltk.download('wordnet') 
            preprocessed_positive_text = preprocess_text(positive_text)
        preprocessed_negative_text = preprocess_text(negative_text)

    # Check if the selected language is Chinese
        if language.lower() == "chinese":
            #font_path = (r"simhei\chinese.simhei.ttf")
            font_path = "chinese.simhei.ttf"
            # the path to the Chinese font file
        else:
            font_path = None  # Use the default font for other languages

    with col4:
        with st.spinner('Plotting Wordcloud'):
            # Postive Word Cloud
            st.subheader(f'Positive Reviews')
            positive_wordcloud = WordCloud(
                background_color='white',
                font_path=font_path,  # Set font path based on language
            ).generate(preprocessed_positive_text)

            # Set the Word Cloud for positive reviews as plot3
            positive_wc = plt.figure(figsize=(10, 5))
            plt.imshow(positive_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(positive_wc)

    with col6:
        with st.spinner('Plotting Wordcloud'):
            # Negative Word Cloud
            st.subheader(f'Negative Reviews')
            negative_wordcloud = WordCloud(
                background_color='white',
                font_path=font_path,  # Set font path based on language
            ).generate(preprocessed_negative_text)

            # Set the Word Cloud for negative reviews as plot3
            negative_wc = plt.figure(figsize=(10, 5))
            plt.imshow(negative_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(negative_wc)

    #########################################################
        
    st.subheader('Polarity and Subjectivity Analysis')
    col7, col8, col9 = st.columns([45, 10, 45])
    #########################################################

    sentiments = []
    for reviews in filtered_data['text']:
        blob = TextBlob(reviews)
        sentiment_polarity = blob.sentiment.polarity
        sentiments.append(sentiment_polarity)

    # Adding sentiment to comments by creating a new list of dictionaries with comments and sentiments
    reviews_with_sentiment_polarity = []
    for i, reviews in enumerate(filtered_data['text']):
        review_dict = {
            "reviews": reviews,
            "sentiment": sentiments[i]
        }
        reviews_with_sentiment_polarity.append(review_dict)

    # Bar chart - Sentiment Analysis (Polarity)
    # Extract sentiment values from reviews_with_sentiment_polarity dictionary
    sentiments = [entry['sentiment'] for entry in reviews_with_sentiment_polarity]

    # Count the occurrences of different sentiment categories
    sentiment_counts = {
        "Positive": len([sentiment for sentiment in sentiments if sentiment > 0]),
        "Negative": len([sentiment for sentiment in sentiments if sentiment < 0]),
        "Neutral": len([sentiment for sentiment in sentiments if sentiment == 0])
    }

    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    with col7:
        polarity_analysis = px.bar(x=labels, y=values, title='Sentiment Analysis - Polarity')
        polarity_analysis.update_xaxes(title_text='Sentiment')
        polarity_analysis.update_yaxes(title_text='Count')
        polarity_analysis.update_layout(width=500)
        st.plotly_chart(polarity_analysis)

    with col9:
        polarity_dist = px.histogram(sentiments, nbins=5, title='Sentiment Distribution - Polarity')
        polarity_dist.update_xaxes(title_text='Sentiment')
        polarity_dist.update_yaxes(title_text='Count')
        polarity_dist.update_layout(width=500)
        st.plotly_chart(polarity_dist)

    #########################################################

    sentiments2 = []
    for reviews in filtered_data['text']:
        blob = TextBlob(reviews)
        sentiment_subjectivity = blob.sentiment.subjectivity
        sentiments2.append(sentiment_subjectivity)

    # Adding sentiment to comments by creating a new list of dictionaries with comments and sentiments
    reviews_with_sentiment_subjectivity = []
    for i, reviews in enumerate(filtered_data['text']):
        review_dict = {
            "reviews": reviews,
            "sentiment": sentiments2[i]
        }
        reviews_with_sentiment_subjectivity.append(review_dict)

    # Bar chart - Sentiment Analysis (Polarity)
    # Extract sentiment values from reviews_with_sentiment_polarity dictionary
    sentiments2 = [entry['sentiment'] for entry in reviews_with_sentiment_subjectivity]

    # Count the occurrences of different sentiment categories
    sentiment_counts = {
        "Subjective": len([sentiment for sentiment in sentiments2 if sentiment >0]),
        "Objective": len([sentiment for sentiment in sentiments2 if sentiment ==0])
    }

    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    with col7:
        subjectivity_analysis = px.bar(x=labels, y=values, title='Sentiment Analysis - Subjectivity')
        subjectivity_analysis.update_xaxes(title_text='Sentiment')
        subjectivity_analysis.update_yaxes(title_text='Count')
        subjectivity_analysis.update_layout(width=500)
        st.plotly_chart(subjectivity_analysis)

    with col9:
        subjectivity_dist = px.histogram(sentiments2, nbins=5, title='Sentiment Distribution - Subjectivity')
        subjectivity_dist.update_xaxes(title_text='Sentiment')
        subjectivity_dist.update_yaxes(title_text='Count')
        subjectivity_dist.update_layout(width=500)
        st.plotly_chart(subjectivity_dist)

    #########################################################

    st.subheader('Raw data')
    if language.lower() == "english":
        st.write(filtered_data[['publishedatdate', 'stars', 'text', 'res_dict', 'sent_res', 'sent_score']])
    else:
        st.write(filtered_data[['publishedatdate', 'stars', 'text', 'texttranslated', 'res_dict', 'sent_res', 'sent_score']])

    #########################################################

with tab3:
    st.subheader('Topic Modelling')

    # Using Zero-shot classification
    labels = ['car park', 'food', 'environment','customer services','price','furniture', 'queue','toilet']

    start_modelling_time = datetime.now()
    st.write(start_modelling_time)
    
    with st.spinner('Building topic modelling'):
                                                    
        # Display the dataset with the predicted categories
        selected_labels = st.multiselect("Select store:", options=labels, default = labels)

        st.write("Predicted Categories for Each Text:")
        filtered_data_class = filtered_data[filtered_data['zeroshot_class'].isin(selected_labels)]
        st.write(filtered_data_class[['text_short', 'zeroshot_class', 'sent_res']])

        category_counts = filtered_data['zeroshot_class'].value_counts()
        category_counts_sorted = category_counts.sort_values(ascending=False)

        plot_category = px.bar(x=category_counts_sorted.values, y=category_counts_sorted.index, orientation='h')
        st.plotly_chart(plot_category, use_container_width=True)
    end_modelling_time = datetime.now()

    st.write(end_modelling_time - start_modelling_time)

    def generate_word_cloud(text, title, font_path=None):
        st.subheader(title)

        # Check if the text contains at least one word
        if not text or not any(text.split()):
            st.write("No words to generate a word cloud.")
            return
    
        wordcloud = WordCloud(
            background_color='white',
            font_path=font_path,
        ).generate(text)

        # Create a figure for the word cloud and display it
        wc_figure, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return wc_figure
    
    # Find the top category labels if category counts is more than 150 else word cloud won't be able to display
    # top_labels = category_counts.index[:2]
    if (category_counts < 150).all():
        message = f"The category count is less than 150. Word cloud won't be displayed."
        st.write(message)  # Replace 'st.write' with the appropriate method to display the message in your Streamlit application
    else:
        top_labels = category_counts[category_counts >= 150].index
        # Generate and display word clouds for positive and negative sentiments by looping through the labels and create word clouds
        for label in top_labels:
            st.subheader(f'Word Clouds for {label}')
            
            # Filter the data for the current label
            label_data = filtered_data_class[filtered_data_class['zeroshot_class'] == label]

            # Separate positive and negative sentiments
            positive_text = " ".join(label_data[label_data['sent_res'] == 'positive']['text'])
            negative_text = " ".join(label_data[label_data['sent_res'] == 'negative']['text'])

            # Preprocess the text for the word clouds
            preprocessed_positive_text = preprocess_text(positive_text)
            preprocessed_negative_text = preprocess_text(negative_text)

            # Generate and display the word clouds for positive and negative sentiments
            # Create two columns for positive and negative word clouds
            col1, col2 = st.columns(2)

            with col1:
                positive_wc_figure = generate_word_cloud(preprocessed_positive_text, f'Positive', font_path=font_path)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(positive_wc_figure)

            with col2:
                negative_wc_figure = generate_word_cloud(preprocessed_negative_text, f'Negative', font_path=font_path)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(negative_wc_figure)

