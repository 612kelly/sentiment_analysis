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

DATA_URL = (r"ikea_reviews_sentiment2.csv")

# Function to load data and filter it based on language and date range
@st.cache_data
def load_and_filter_data(DATA_URL, language, year):
    data = pd.read_csv(DATA_URL)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    
    # Group "Chinese_China," "Chinese_Taiwan," and "Chinese_Hongkong" into "Chinese"
    data['language'] = data['language'].replace(["Chinese_China", "Chinese_Taiwan", "Chinese_Hongkong"], "Chinese")

    # Ensure text of sent_res are the same
    data['sent_res'] = data['sent_res'].replace(["POSITIVE"], "positive")
    data['sent_res'] = data['sent_res'].replace(["NEGATIVE"], "negative")

    # Filter data based on language
    filtered_data = data[data['language'].str.lower().isin([language.lower()])]
    
    # Convert "publishedAtDate" to datetime
    filtered_data['publishedatdate'] = pd.to_datetime(filtered_data['publishedatdate'])
    
    # Filter data by year
    filtered_data = filtered_data[filtered_data['publishedatdate'].dt.year == year]
    
    return filtered_data

# Slidebar filter
st.sidebar.header("Choose your filter")

languages = ["English", "Indonesian", "Chinese"]

# Filter 1 (select language type)
language = st.sidebar.selectbox("Select the language type:", languages)

# Filter 2 (year)
year = st.sidebar.slider('year', 2016, 2023, 2022)

# Load and filter data
with st.spinner('Loading data'):
    filtered_data = load_and_filter_data(DATA_URL, language, year)
st.write("Done reading data")
if st.checkbox('Show filtered data'):
    st.subheader('Raw data')
    st.write(filtered_data)

#########################################################

# make 3 columns for first row of dashboard
col1, col2, col3 = st.columns([35, 30, 30])

#########################################################

with col1:
    # Star Analysis Chart
    # pie chart + add star filter
    st.subheader('Star Reviews Analysis')

    # Group data by star review and count occurrences
    stars_counts = filtered_data['stars'].value_counts()

    plot5 = px.bar(filtered_data, x=stars_counts.values, y=stars_counts.index, orientation='h')
    st.plotly_chart(plot5, use_container_width=True)

#########################################################

st.subheader('Sentiment Analysis')
# make 2 columns for second row of dashboard
col4, col5, col6 = st.columns([45, 10, 45])

with col4:
    # Sentiment Analysis Chart
    # Group data by sentiment and count occurrences
    sentiment_counts = filtered_data['sent_res'].value_counts()

    # Bar chart for sentiment
    bar_chart = px.bar(
        sentiment_counts, 
        x=sentiment_counts.index, 
        y=sentiment_counts.values,
        labels={'x': 'sent_res', 'y': 'Count'},
        title=f'Sentiment Distribution for {language} Reviews')
    st.plotly_chart(bar_chart, use_container_width=True)

#########################################################

with col6:
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

# make 2 columns for second row of dashboard
col7, col8, col9 = st.columns([45, 10, 45])

#########################################################

# Word Cloud
# Function to preprocess text
# remove ' and single letter
def preprocess_text(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word.lower() not in stopwords.words("english")]
    
    # Remove the word "ikea"
    words = [word for word in words if word.lower() != "ikea"]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
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
    preprocessed_positive_text = preprocess_text(positive_text)
    preprocessed_negative_text = preprocess_text(negative_text)

with col7:
    # Check if the selected language is Chinese
    if language.lower() == "chinese":
        font_path = (r"simhei\chinese.simhei.ttf")
        # the path to the Chinese font file
    else:
        font_path = None  # Use the default font for other languages

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
    # Check if the selected language is Chinese
    if language.lower() == "chinese":
        font_path = (r"simhei\chinese.simhei.ttf")
        # the path to the Chinese font file
    else:
        font_path = None  # Use the default font for other languages

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

def plot_n_gram(n_gram_df, title, color="#54A24B"):
    # plot the top n-grams frequencies in a bar chart
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig

with col10:
    # plot the top 10 occuring words 
    top_unigram = get_top_n_gram(filtered_data, ngram_range=(1, 1), n=10)
    unigram_plot = plot_n_gram(
        top_unigram, title="Top 10 Occuring Words"
    )
    unigram_plot.update_layout(height=350)
    st.plotly_chart(unigram_plot, use_container_width=True)

with col12:
    top_bigram = get_top_n_gram(filtered_data, ngram_range=(2, 2), n=10)
    bigram_plot = plot_n_gram(
        top_bigram, title="Top 10 Occuring Bigrams"
    )
    bigram_plot.update_layout(height=350)
    st.plotly_chart(bigram_plot, use_container_width=True)

#########################################################

# make 2 columns for second row of dashboard
col13, col14, col15 = st.columns([45, 10, 45])

st.subheader('Topic Modelling')

# Preprocess the text data for topic modeling
filtered_tokens = filtered_data['text'].apply(preprocess_text)

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer with preprocessed comments
token_matrix = vectorizer.fit_transform(filtered_tokens)

# Initialize LatentDirichletAllocation model
num_topics = 10  # specify the number of topics
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

# Fit the model with the token matrix
lda_model.fit(token_matrix)

# Print the top words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]  # only select 10 top words of the topic
    st.write(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
