import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

from datetime import datetime


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

DATA_URL = (r"ikea_reviews_sentiment2.parquet")
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
    store = st.selectbox("Select store:", options=data["title"].unique(), index=data["title"].unique().tolist().index(store_with_most_reviews))

    # Filter 3 (date range)
    min_date = min(data['date'])
    max_date = max(data['date'])

    # Calculate default values within the range
    default_start_date = min_date  # Set the default to the minimum date
    default_end_date = max_date  # Set the default to the maximum date

    start_date = st.date_input("Start Date", min_value = min_date, max_value = max_date, value=default_start_date)
    end_date = st.date_input("End Date", min_value = min_date, max_value = max_date, value=default_end_date)

    submitted1 = st.form_submit_button(label = 'Submit')


# Load and filter data
with st.spinner('Loading data'):
    filtered_data = load_and_filter_data(DATA_URL, language, store, start_date, end_date)


tab1, tab2, tab3 = st.tabs(["Overview", "Topic Modelling", "About"])

with tab1:
    if st.checkbox('Show filtered data'):
        st.subheader('Raw data')
        st.write(filtered_data)

    #########################################################

    # make 3 columns for first row of dashboard
    col1, col2, col3 = st.columns([35, 30, 30])

    #########################################################

    with col1:
        total_reviews = int(filtered_data["text"].count())
        st.subheader('Number of Reviews')
        st.subheader(f"{total_reviews}")

        average_rating = round(filtered_data["stars"].mean(),1)
        star_rating = ":star:" * int(round(average_rating,0))
        st.subheader('Average Star Reviews')
        st.subheader(f"{average_rating} {star_rating}")

    with col2:
        # Star Analysis Chart
        st.subheader('Star Count')

        # Group data by star review and count occurrences
        stars_counts = filtered_data['stars'].value_counts()

        plot5 = px.bar(filtered_data, x=stars_counts.values, y=stars_counts.index, orientation='h')
        plot5.update_xaxes(title='Count')
        plot5.update_yaxes(title='Star Rating')
        st.plotly_chart(plot5, use_container_width=True)

    #with col3:
        
    #########################################################

    st.subheader('Sentiment Analysis')
    col4, col5, col6 = st.columns([45, 10, 45])

    # with col4:
    #     # Sentiment Analysis Chart
    #     # Group data by sentiment and count occurrences
    #     sentiment_counts = filtered_data['sent_res'].value_counts()

    #     # Bar chart for sentiment
    #     bar_chart = px.bar(
    #         sentiment_counts, 
    #         x=sentiment_counts.index, 
    #         y=sentiment_counts.values,
    #         labels={'x': 'sent_res', 'y': 'Count'},
    #         title=f'Sentiment Distribution for {language} Reviews')
    #     st.plotly_chart(bar_chart, use_container_width=True)

    #########################################################

    with col4:
        # Group data by sentiment and count occurrences
        sentiment_counts = filtered_data['sent_res'].value_counts()
        # Pie chart for sentiment
        pie_chart = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.3,
            title=f'Sentiment Distribution for {language} Reviews',
            color=sentiment_counts.index,
            # set the color of positive to blue and negative to orange
            color_discrete_map={"Positive": "#1F77B4", "Negative": "#FF7F0E"},
        )
        pie_chart.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>%{value} (%{percent})",
            hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
        )
        pie_chart.update_layout(showlegend=False)
        st.plotly_chart(pie_chart, use_container_width=True)

    #########################################################

    col7, col8, col9 = st.columns([45, 10, 45])

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
        words = [word for word in words if word.lower() not in ["ikea", "the", "ok", "la"]]

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
            font_path = (r"simhei\chinese.simhei.ttf")
            # the path to the Chinese font file
        else:
            font_path = None  # Use the default font for other languages

    with col7:
        with st.spinner('Plotting Wordcloud'):
            # Postive Word Cloud
            st.subheader(f'Word Cloud for Positive Reviews in {language}')
            positive_wordcloud = WordCloud(
                background_color='white',
                font_path=font_path,  # Set font path based on language
            ).generate(preprocessed_positive_text)

            # Set the Word Cloud for positive reviews as plot3
            positive_wc = plt.figure(figsize=(10, 5))
            plt.imshow(positive_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(positive_wc)

    with col9:
        with st.spinner('Plotting Wordcloud'):
            # Negative Word Cloud
            st.subheader(f'Word Cloud for Negative Reviews in {language}')
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

    # make 2 columns for second row of dashboard
    col10, col11, col12 = st.columns([45, 10, 45])

    def get_top_n_gram(filtered_data, ngram_range, n=10):
        # load the corpus and vectorizer
        corpus = filtered_data['text']
        vectorizer = CountVectorizer(
            analyzer="word", ngram_range=ngram_range
        )

        # use the vectorizer to count the n-grams frequencies
        X = vectorizer.fit_transform(corpus.astype(str).values)
        words = vectorizer.get_feature_names_out()
        words_count = np.ravel(X.sum(axis=0))

        # store the results in a dataframe
        df = pd.DataFrame(zip(words, words_count))
        df.columns = ["words", "counts"]
        df = df.sort_values(by="counts", ascending=False).head(n)
        df["words"] = df["words"].str.title()
        return df

    # def plot_n_gram(n_gram_df, title, color="#54A24B"):
    #     # plot the top n-grams frequencies in a bar chart
    #     fig = px.bar(
    #         x=n_gram_df.counts,
    #         y=n_gram_df.words,
    #         title="<b>{}</b>".format(title),
    #         text_auto=True,
    #     )
    #     fig.update_layout(plot_bgcolor="white")
    #     fig.update_xaxes(title=None)
    #     fig.update_yaxes(autorange="reversed", title=None)
    #     fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    #     return fig

    # with col10:
    #     # plot the top 10 occuring words 
    #     top_unigram = get_top_n_gram(filtered_data, ngram_range=(1, 1), n=10)
    #     unigram_plot = plot_n_gram(
    #         top_unigram, title="Top 10 Occuring Words"
    #     )
    #     unigram_plot.update_layout(height=350)
    #     st.plotly_chart(unigram_plot, use_container_width=True)

    # with col12:
    #     top_bigram = get_top_n_gram(filtered_data, ngram_range=(2, 2), n=10)
    #     bigram_plot = plot_n_gram(
    #         top_bigram, title="Top 10 Occuring Bigrams"
    #     )
    #     bigram_plot.update_layout(height=350)
    #     st.plotly_chart(bigram_plot, use_container_width=True)

    #########################################################


with tab2:
    st.subheader('Topic Modelling')

    # Using Zero-shot classification
    # Initialize the zero-shot classification pipeline

    @st.cache_data
    def get_zero_shot_model():
        # # Check if the selected language is Chinese
        # if language.lower() == "chinese":
        #     return pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli') #cant access without permission
        # else:
        #     return pipeline('zero-shot-classification', model='joeddav/distilbert-base-uncased-agnews-student')
        
        return pipeline('zero-shot-classification', model='joeddav/distilbert-base-uncased-agnews-student')
        # return pipeline('zero-shot-classification', model='joeddav/xlm-roberta-large-xnli', use_auth_token='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')

    with st.spinner("Downloading zero shot classifier"):
      classifier = get_zero_shot_model()
    
    # labels = ['retail', 'food', 'facilities']
    labels = ['car park', 'food', 'environment','services','price','furniture']
    # labels = ['car park', 'food', 'environment', 'retail']

    # # Function to predict categories and add them to the DataFrame
    # def predict_categories(text):
    #     result = classifier(text, labels, multi_class_True)
    #     predicted_category = result['labels'][0]
    #     return predicted_category

    start_modelling_time = datetime.now()
    st.write(start_modelling_time)

    @st.cache_data 
    def predict_category(text):
        return  text.apply(lambda x: classifier(x, labels)['labels'][0])
    with st.spinner('Building topic modelling'):
        # Apply the predict_categories function to all rows in the dataset
        # filtered_data['Predicted Category'] = filtered_data['text'].apply(predict_categories)
    
        filtered_data['Predicted Category'] = predict_category(filtered_data['text_short'])
                                                    
        # Display the dataset with the predicted categories
        selected_labels = st.multiselect("Select store:", options=labels, default = labels)

        st.write("Predicted Categories for Each Text:")
        filtered_data_class = filtered_data[filtered_data['Predicted Category'].isin(selected_labels)]
        st.write(filtered_data_class[['text_short', 'Predicted Category', 'sent_res']])

        category_counts = filtered_data['Predicted Category'].value_counts()
        category_counts_sorted = category_counts.sort_values(ascending=False)

        plot_category = px.bar(filtered_data, x=category_counts_sorted.values, y=category_counts_sorted.index, orientation='h')
        st.plotly_chart(plot_category, use_container_width=True)
    end_modelling_time = datetime.now()

    st.write(end_modelling_time - start_modelling_time)

    def generate_word_cloud(text, title, font_path=None):
        st.subheader(title)
        wordcloud = WordCloud(
            background_color='white',
            font_path=font_path,
        ).generate(text)

        # Create a figure for the word cloud and display it
        wc_figure, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return wc_figure
    
    # Find the top category labels if category counts is more than 100 else word cloud won't be able to display
    # top_labels = category_counts.index[:2]
    top_labels = category_counts[category_counts >= 100].index

    # Generate and display word clouds for positive and negative sentiments by looping through the labels and create word clouds
    for label in top_labels:
        st.subheader(f'Word Clouds for {label} in {language}')
        
        # Filter the data for the current label
        label_data = filtered_data_class[filtered_data_class['Predicted Category'] == label]

        # Separate positive and negative sentiments
        positive_text = " ".join(label_data[label_data['sent_res'] == 'positive']['text_short'])
        negative_text = " ".join(label_data[label_data['sent_res'] == 'negative']['text_short'])

        # Preprocess the text for the word clouds
        preprocessed_positive_text = preprocess_text(positive_text)
        preprocessed_negative_text = preprocess_text(negative_text)

        # Generate and display the word clouds for positive and negative sentiments
        # Create two columns for positive and negative word clouds
        col1, col2 = st.columns(2)

        with col1:
            positive_wc_figure = generate_word_cloud(preprocessed_positive_text, f'Positive', font_path=font_path)
            st.pyplot(positive_wc_figure)

        with col2:
            negative_wc_figure = generate_word_cloud(preprocessed_negative_text, f'Negative', font_path=font_path)
            st.pyplot(negative_wc_figure)

    #########################################################

    # #Using LDA
    # st.subheader("LDA")
    # # Preprocess the text data for topic modeling
    # filtered_tokens = filtered_data['text'].apply(preprocess_text)

    # # Initialize the CountVectorizer
    # vectorizer = CountVectorizer()

    # # Fit the vectorizer with preprocessed comments
    # token_matrix = vectorizer.fit_transform(filtered_tokens)

    # # Initialize LatentDirichletAllocation model
    # num_topics = 10  # specify the number of topics
    # lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

    # # Fit the model with the token matrix
    # lda_model.fit(token_matrix)

    # # Print the top words for each topic
    # feature_names = vectorizer.get_feature_names_out()
    # for topic_idx, topic in enumerate(lda_model.components_):
    #     top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]  # only select 10 top words of the topic
    #     st.write(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
