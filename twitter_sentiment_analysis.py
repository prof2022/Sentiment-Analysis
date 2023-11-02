import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('assets/Tweets.csv')


df = data.dropna(axis=1)
target = df.airline_sentiment
tweets = df.drop(['airline_sentiment'], axis=1)


my_stop_words = ENGLISH_STOP_WORDS.union(['airline', 'airplane', '@', 'am', 'pm'])

word_tokens = [word_tokenize(tweet) for tweet in tweets.text]
print('Original tokens:', word_tokens[0])

letters_only = [[word for word in item if word.isalpha()] for item in word_tokens]
alpha_numeric = [[word for word in item if word.isalnum()] for item in word_tokens]
digits_only = [[word for word in item if word.isdigit()] for item in word_tokens]

print('letters token:', letters_only[0])
print('letters token:', alpha_numeric[0])
print('digits token:', alpha_numeric[0])

count_vectorizer_1 = CountVectorizer(stop_words=my_stop_words, token_pattern=r'\b[^\d\W][^\d\W]+\b')
count_vectorizer_2 = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
count_vectorizer_1.fit(tweets.text)
#count_vectorizer_2.fit(tweets.negativereason)

features = count_vectorizer_1.transform(tweets.text)
features_df = pd.DataFrame(features.toarray(), columns=count_vectorizer_1.get_feature_names_out())
print(features_df.head())
#print(count_vectorizer_2.get_feature_names_out())
print('Length of vectorizer: ', len(count_vectorizer_1.get_feature_names_out()))

"""# tweets['clean_text'] = tweets['text'].apply(lambda x: clean_text(x))
tweet_cloud = WordCloud(background_color='white', stopwords=my_stop_words).generate(tweets.text.iloc[2])

plt.imshow(tweet_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()"""

# stemming
porter = PorterStemmer()
WNlemmatizer = WordNetLemmatizer()

stemmed_tokens = [[porter.stem(word) for word in tweet] for tweet in word_tokens]
print(stemmed_tokens[0])
#lem_tokens = [WNlemmatizer.lemmatize(token) for token in word_tokens]

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100, token_pattern=r'\b[^\d\W][^\d\W]+\b',
                                   stop_words=ENGLISH_STOP_WORDS)

tfidf_vectorizer.fit(tweets.text)
tfidf_features = tfidf_vectorizer.transform(tweets.text)
tfidf_features_df = pd.DataFrame(tfidf_features.A, columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_features_df.head())

print(len(target))

train_X, test_X, train_Y, test_Y = train_test_split(tfidf_features_df, target, test_size=0.3, stratify=target, random_state=123)

logit = LogisticRegression(C=0.1)
logit.fit(train_X, train_Y)

y_predicted = logit.predict(train_X)
y_test_predicted = logit.predict(test_X)
train_accuracy = accuracy_score(train_Y, y_predicted)
test_accuracy = accuracy_score(test_Y, y_test_predicted)
print('train',train_accuracy)
print('test', test_accuracy)
cm = confusion_matrix(test_Y, y_test_predicted)
print(cm/len(test_Y))

prob_0 = logit.predict_proba(test_X)[:, 0]
prob_1 = logit.predict_proba(test_X)[:, 1]
prob_2 = logit.predict_proba(test_X)[:, 2]

print(prob_0, prob_1, prob_2)

postive_reviews = df[df['airline_sentiment'] == 'positive']
print(postive_reviews.head())