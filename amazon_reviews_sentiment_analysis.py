import pandas as pd
import re
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from langdetect import detect_langs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

amazon_reviews = pd.read_csv('assets/amazon_reviews_sample.csv')
amazon_reviews.columns.str.match('Unnamed')
amazon_reviews = amazon_reviews.loc[:, ~amazon_reviews.columns.str.match('Unnamed')]
print(amazon_reviews.head())


# clean text column description
def clean_text(text):
    pattern = r"[?|$|.!'{}:<>\-(#/\")&,+=]"
    text = re.sub(pattern, '', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


# df_movies['Description'] = df_movies['Description'].apply(str)
amazon_reviews['clean_reviews'] = amazon_reviews['review'].apply(lambda x: clean_text(x))

# number of words per review
word_tokens = [word_tokenize(review) for review in amazon_reviews.clean_reviews]

len_token = []
for i in range(len(word_tokens)):
    len_token.append(len(word_tokens[i]))
amazon_reviews['n_token'] = len_token

# number of sentences per reviews
sent_tokens = [sent_tokenize(review) for review in amazon_reviews.review]
len_sentences = []
for i in range(len(sent_tokens)):
    len_sentences.append(len(sent_tokens[i]))
amazon_reviews['n_sentences'] = len_sentences

# part od speach in each sentence
pos_sentences = [nltk.pos_tag(sent) for sent in sent_tokens]
len_pos_sentences = []
for i in range(len(pos_sentences)):
    len_pos_sentences.append(len(pos_sentences[i]))
amazon_reviews['n_pos'] = len_pos_sentences

languages = []
for row in range(len(amazon_reviews)):
    languages.append(detect_langs(amazon_reviews.iloc[row, 1]))

languages = [str(lang).split(':')[0][1:] for lang in languages]
amazon_reviews['language'] = languages
print(amazon_reviews.head())

# word vectorizer, BOW and tfidf
count_vectorizer = CountVectorizer(stop_words='english', max_features=1000, max_df=500, ngram_range=(2, 2))
count_vectorizer.fit(amazon_reviews.review)
X = count_vectorizer.transform(amazon_reviews.review)

features = X.toarray()
features_df = pd.DataFrame(features, columns=count_vectorizer.get_feature_names_out())

tfidf_vectorizer = TfidfVectorizer(max_features=200, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2),
                                   token_pattern=r'\b[^\d\W][^\d\W]+\b')
tfidf_vectorizer.fit(amazon_reviews.review)
X_tf = tfidf_vectorizer.transform(amazon_reviews.review)
features_tfidf = pd.DataFrame(X_tf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print('bow', features_df.head())
print('tfidf', features_tfidf.head())

print(amazon_reviews.columns)

target = amazon_reviews.score

train_X, test_X, train_Y, test_Y = train_test_split(features_tfidf, target, test_size=0.3, stratify=target,
                                                    random_state=123)
logit = LogisticRegression()
logit.fit(train_X, train_Y)

y_predicted = logit.predict(train_X)
y_test_predicted = logit.predict(test_X)
train_accuracy = accuracy_score(train_Y, y_predicted)
test_accuracy = accuracy_score(test_Y, y_test_predicted)
print('train', train_accuracy)
print('test', test_accuracy)
cm = confusion_matrix(test_Y, y_test_predicted)
print(cm / len(test_Y))

prob_0 = logit.predict_proba(test_X)[:, 0]
prob_1 = logit.predict_proba(test_X)[:, 1]

print(prob_0, prob_1)
