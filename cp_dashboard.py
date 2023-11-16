import streamlit as st
import pandas as pd
# import numpy as np
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
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from dateutil.relativedelta import relativedelta
from datetime import datetime
from textblob import TextBlob


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')
nltk.download('omw-1.4') 
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

# @st.cache_data
def read_and_clean_data(url):

    data = pd.read_parquet(url)
    data['publishedAtDate'] = pd.to_datetime(data['publishedAtDate'])
    data['date'] = data['publishedAtDate'].dt.date

    # Group "Chinese_China," "Chinese_Taiwan," and "Chinese_Hongkong" into "Chinese"
    data['language'] = data['language'].replace(["Chinese_China", "Chinese_Taiwan", "Chinese_Hongkong"], "Chinese")
    data['language'] = data['language'].replace(["Indonesian"], "Malay")
    data['language'] = data['language'].fillna("No text")

    # Ensure text of sent_res are the same
    data['sent_res'] = data['sent_res'].replace(["POSITIVE"], "positive")
    data['sent_res'] = data['sent_res'].replace(["NEGATIVE"], "negative")

    data['sent_res'] = data['sent_res'].fillna("No text")
    return data

data = read_and_clean_data(DATA_URL)
# st.write(data)


# Function to load data and filter it based on language and date range
@st.cache_data
def load_and_filter_data(language, store, start_date, end_date):
    data['date'] = data['publishedAtDate'].dt.date
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    
    # # Group "Chinese_China," "Chinese_Taiwan," and "Chinese_Hongkong" into "Chinese"
    # data['language'] = data['language'].replace(["Chinese_China", "Chinese_Taiwan", "Chinese_Hongkong"], "Chinese")

    # # Ensure text of sent_res are the same
    # data['sent_res'] = data['sent_res'].replace(["POSITIVE"], "positive")
    # data['sent_res'] = data['sent_res'].replace(["NEGATIVE"], "negative")

    # Filter data based on language
    # filtered_data = data[data['language'].str.lower().isin([language.lower()])  ]
    filtered_data = data[data['language'].isin(language)  ]

    # Filter data by store name
    filtered_data = filtered_data[filtered_data['title'].isin(store)]
    
    # Filter data by date range
    filtered_data = filtered_data[
        (filtered_data['date'] >= start_date) &
        (filtered_data['date'] <= end_date)
    ]
    
    return filtered_data

# Slidebar filter
st.sidebar.header("Choose your filter")
with st.sidebar.form(key ='Form Filter'):

    languages = ["English", "Malay", "Chinese","No text"]
    # languages = data['language'].unique()

    # Filter 1 (select language)
    language = st.multiselect("Select language:", options= languages, default = ["English","Malay", "Chinese"])

    # Filter 2 (select stores)
    store_with_most_reviews = data["title"].value_counts().idxmax()
    #store = st.selectbox("Select store:", options=data["title"].unique(), index=data["title"].unique().tolist().index(store_with_most_reviews))
    # store = st.selectbox("Select store:", options=data["title"].unique())
    store = st.multiselect("Select store:",options =data["title"].unique(),default=data["title"].unique())

    # Filter 3 (date range)
    min_date = min(data['date'])
    max_date = max(data['date'])

    # Calculate default values within the range
    default_start_date = max_date - relativedelta(years=2) # min_date  # Set the default to the minimum date
    default_end_date = max_date  # Set the default to the maximum date

    start_date = st.date_input("Start Date", min_value = min_date, max_value = max_date, value=default_start_date, help="Earliest date is "+str(min_date))
    end_date = st.date_input("End Date", min_value = min_date, max_value = max_date, value=default_end_date, help="Latest date is "+str(max_date))

    if start_date > end_date:
        st.warning("Start Date cannot be after End Date. Please select a valid date range.")
        submitted1 = st.form_submit_button(label='Submit')
    else:
        submitted1 = st.form_submit_button(label='Submit')

    # submitted1 = st.form_submit_button(label = 'Submit')


# Load and filter data
with st.spinner('Loading data'):
    filtered_data = load_and_filter_data(language, store, start_date, end_date)

    #st.write(filtered_data)
    
tab1, tab2, tab3 = st.tabs(["About","Overview", "Topic Classification"])

    #########################################################

with tab1:
    st.header("About")

    # Problem Statement.
    st.subheader("Introduction")

    st.write("The problem addressed in this research project is the need to analyze customer reviews of IKEA Malaysia effectively using Natural Language Processing (NLP) techniques in order to identify the patterns of feedback. The challenge lies in extracting relevant and meaningful insights from a vast volume of customer feedback gathered from internal sources and scraped from online platforms like Google as it has up to tens of thousands review available. By understanding the sentiments, common themes and issues encountered by customers, businesses can obtain valuable insights on improving their products, services, and overall customer satisfaction.")
    st.write("This dashboard displays analysis of Google reviews of all IKEA Malaysia outlets obtained from Google.")
    st.write("You may select the filter(s) for analysis to be display on the left panel. Do click the Submit button for the analysis to run.")

    # Data
    
    st.subheader("Data")
    st.write("The Google reviews data were scraped using this [website](https://apify.com/compass/google-maps-reviews-scraper).")
    st.write(f"Scraped of data ranges from {min_date} to {max_date}.")

    # Model
    st.subheader("Sentiment and Topic Classification Model")
    st.write("You can see the reviews count, statistics and sentiment analsysis under the 'Overview' tab and classification for each review under the 'Topic Classification' tab.")
    st.write("The sentiment analysis model attempts to classify each review into positive or negative. This aim to understand how visitors are talking IKEA.")
    st.write("The sentiment model used is [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).")

    st.write("\n")

    st.write("The topic classification attempts to classify each review into specific classes such as price, food, car park etc. This aim to understand what topics are mainly mentioned by visitors in IKEA.")
    st.write("The topic classification model used is [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli).")

    #########################################################
        
with tab2:
    
    # make 3 columns for first row of dashboard
    col1, col2, col3 = st.columns([25, 40, 30])

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
        # pie_chart = px.pie(
        #     values=sentiment_counts.values,
        #     names=sentiment_counts.index,
        #     hole=0.3,
        #     #title=f'Sentiment Distribution for {language} Reviews',
        #     color=sentiment_counts.index,
        #     color_discrete_map={"positive": "#7BB662", "negative": "#E03C32", "neutral": "#FFD301"},
        # )
        # pie_chart.update_traces(
        #     textposition="inside",
        #     texttemplate="%{label}<br>%{value} (%{percent})",
        #     hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
        # )
        # pie_chart.update_layout(showlegend=False)
        # st.plotly_chart(pie_chart, use_container_width=True)


        #### Use sunburst for multi language/store
        df_sunburst = filtered_data.groupby(["title","language","sent_res"],dropna=False)[["date"]].count().reset_index()
        store_sent_sunburst = px.sunburst(df_sunburst, path=['title', 'language', 'sent_res'], values='date', color='sent_res',
                        color_discrete_map={'(?)':'blue', 'positive':'#7BB662', 'negative':'#E03C32','No text':"#FFD301"})
        st.plotly_chart(store_sent_sunburst, use_container_width=True)


    with col3:
        total_reviews = int(filtered_data["sent_res"].count())
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
        # Remove text containing numbers
        text = re.sub(r'\b\d+\b', '', text)

        # Replace symbols and stop words with spaces
        # text = re.sub(r'[.,\/\-\(\)]', ' ', text)
        text = re.sub(r'\b(?:ikea|the|ok|la|good|bad|etc|covid)\b', '', text, flags=re.IGNORECASE)

        # Tokenize the text
        words = nltk.word_tokenize(text)
        
        # Remove stopwords
        try:
            words = [word for word in words if word.lower() not in stopwords.words("english")]
        except:
            nltk.download('stopwords')
            words = [word for word in words if word.lower() not in stopwords.words("english")]
        
        # Removal of words
        # words = [word for word in words if word.lower() != "ikea"]
        words = [word for word in words if word.lower() not in ["ikea", "the", "ok", "la", "good", "bad"]]

        # Remove words containing "rm" and :pm"
        words = [word for word in words if "rm" not in word.lower() and "pm" not in word.lower()]

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
    all_text = " ".join(filtered_data['text'].fillna(" "))

    # st.write(positive_text)

    with st.spinner('Preprocessing data for wordcloud'):
        # Preprocess the text
        # try:
        preprocessed_positive_text = preprocess_text(positive_text)
        # except:
        #     nltk.download('omw-1.4') 
        #     nltk.download('wordnet') 
        #     preprocessed_positive_text = preprocess_text(positive_text)
        preprocessed_negative_text = preprocess_text(negative_text)


        #### Set font path to chinese for all language
        font_path = "chinese.simhei.ttf"

        # Check if the selected language is Chinese
        # if language.lower() == "chinese":
        #     #font_path = (r"simhei\chinese.simhei.ttf")
        #     font_path = "chinese.simhei.ttf"
        #     # the path to the Chinese font file
        # else:
        #     font_path = None  # Use the default font for other languages
        #     # font_path = "chinese.simhei.ttf"

    with col4:
        try:
            with st.spinner('Plotting Wordcloud'):
                # Postive Word Cloud
                positive_wordcloud = WordCloud(
                    background_color='white',
                    font_path=font_path,  # Set font path based on language
                ).generate(preprocessed_positive_text)

                # Set the Word Cloud for positive reviews as plot3
                positive_wc = plt.figure(figsize=(10, 5))
                plt.imshow(positive_wordcloud, interpolation='bilinear')
                plt.axis('off')

                st.subheader(f'Positive Reviews')
                st.pyplot(positive_wc)
        except ValueError:
            pass

    with col6:
        try:
            with st.spinner('Plotting Wordcloud'):
                # Negative Word Cloud
                negative_wordcloud = WordCloud(
                    background_color='white',
                    font_path=font_path,  # Set font path based on language
                ).generate(preprocessed_negative_text)

                # Set the Word Cloud for negative reviews as plot3
                negative_wc = plt.figure(figsize=(10, 5))
                plt.imshow(negative_wordcloud, interpolation='bilinear')
                plt.axis('off')

                st.subheader(f'Negative Reviews')
                st.pyplot(negative_wc)
        except ValueError:
            pass
    #########################################################
        

    #########################################################

    sentiments = []
    reviews_with_sentiment_polarity = []

    for reviews in filtered_data['text']:
        if reviews:
            blob = TextBlob(reviews)
            sentiment_polarity = blob.sentiment.polarity
            sentiments.append(sentiment_polarity)

            review_dict = {
                "reviews": reviews,
                "sentiment": sentiment_polarity
            }
            reviews_with_sentiment_polarity.append(review_dict)


    if sentiments: #### Check if there is no text at all

        st.subheader('Polarity and Subjectivity Analysis')
        col7, col8, col9 = st.columns([45, 10, 45])
        # Adding sentiment to comments by creating a new list of dictionaries with comments and sentiments
        # reviews_with_sentiment_polarity = []
        # for i, reviews in enumerate(filtered_data['text']):
        #     review_dict = {
        #         "reviews": reviews,
        #         "sentiment": sentiments[i]
        #     }
        #     reviews_with_sentiment_polarity.append(review_dict)

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
    reviews_with_sentiment_subjectivity = []

    for reviews in filtered_data['text']:
        if reviews:
            blob = TextBlob(reviews)
            sentiment_subjectivity = blob.sentiment.subjectivity
            sentiments2.append(sentiment_subjectivity)

            review_dict = {
                "reviews": reviews,
                "sentiment": sentiment_subjectivity
            }
            reviews_with_sentiment_subjectivity.append(review_dict)

    # # Adding sentiment to comments by creating a new list of dictionaries with comments and sentiments
    # reviews_with_sentiment_subjectivity = []
    # for i, reviews in enumerate(filtered_data['text']):
    #     review_dict = {
    #         "reviews": reviews,
    #         "sentiment": sentiments2[i]
    #     }
    #     reviews_with_sentiment_subjectivity.append(review_dict)
    if sentiments2:
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
    # if language.lower() == "english":
    #     st.write(filtered_data[['publishedatdate', 'stars', 'text', 'res_dict', 'sent_res', 'sent_score']])
    # else:
    st.write(filtered_data[['publishedatdate', 'stars', 'text', 'texttranslated', 'res_dict', 'sent_res', 'sent_score']])

    #########################################################

with tab3:
    
    st.subheader('Topic Modelling')

    #### Check if there is no text at all

    if sum(pd.notna(filtered_data['zeroshot_class'])) != 0:

        # Using Zero-shot classification
        # labels = ['car park', 'food', 'environment','customer services','price','furniture', 'queue','toilet']
        labels = filtered_data['zeroshot_class'].unique()
        # labels = ['food', 'queue']

        # start_modelling_time = datetime.now()
        # st.write(start_modelling_time)
        
        with st.spinner('Building topic modelling'):
                                                        
            # Display the dataset with the predicted categories
            selected_labels = st.multiselect("Select class:", options=labels, default = labels)

            st.write("Predicted Categories for Each Text:")
            filtered_data_class = filtered_data[filtered_data['zeroshot_class'].isin(selected_labels)]
            st.write(filtered_data_class[['text_short', 'texttranslated', 'zeroshot_class', 'sent_res']])

            category_counts = filtered_data_class['zeroshot_class'].value_counts()
            category_counts_sorted = category_counts.sort_values(ascending=False)

            plot_category = px.bar(x=category_counts_sorted.values, y=category_counts_sorted.index, orientation='h')

            # bar_chart_data = pd.DataFrame({
            #     'Category': category_counts_sorted.index,
            #     'Count': category_counts_sorted.values,
            #     'Sentiment': [filtered_data_class[filtered_data_class['zeroshot_class'] == label]['sent_res'].iloc[0] for label in category_counts_sorted.index]
            # })

            # plot_category = px.bar(
            #     bar_chart_data,
            #     x='Count',
            #     y='Category',
            #     orientation='h',
            #     color='Sentiment',
            #     labels={'Sentiment': 'Sentiment'},
            #     color_discrete_sequence=['green', 'red'],  # Adjust colors as needed
            # )
            
            plot_category.update_xaxes(title='Count')
            plot_category.update_yaxes(title='Category')
            st.plotly_chart(plot_category, use_container_width=True)

        end_modelling_time = datetime.now()

        # st.write(end_modelling_time - start_modelling_time)

        def generate_word_cloud(text, title, font_path=None):
            st.subheader(title)

            # Check if the text contains at least one word
            if not text or not any(text.split()):
                st.write("No words to generate a word cloud.")
                return None
        
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

