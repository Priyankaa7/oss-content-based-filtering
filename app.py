import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})


# CORS error handling

"""Import the dataset"""

df = pd.read_csv('oss_data.csv')
df.fillna(method='ffill', inplace=True)

"""Combining the different attributes of the dataset into a single string"""

df['content'] = df['name'].astype(str) + ' ' + df['desc'].astype(
    str) + ' ' + df['tags'] + ' ' + df['site'].astype(str) + ' ' + df['upforgrabs__link'].astype(str) + ' ' + df['stats__issue-count'].astype(str)
df['content'] = df['content'].fillna('')
df['content']

"""Tokenize content for Word2Vec"""

df['tokenized_content'] = df['content'].apply(simple_preprocess)
df['tokenized_content']

"""Training the Word2Vec model"""

model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(df['tokenized_content'])
model.train(df['tokenized_content'],
            total_examples=model.corpus_count, epochs=10)

"""Function to average word vectors for a text"""


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


"""Function to compute average word vectors for all repos"""


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(
        tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    return np.array(features)


"""Compute average word vectors for all repos"""

w2v_feature_array = averaged_word_vectorizer(
    corpus=df['tokenized_content'], model=model, num_features=100)


@app.route('/similar_repos', methods=['POST'])
def get_similar_repos():

    user_oss = request.json['query'].strip().replace(" ", "")

    # accept input with spaces and remove spaces
    oss_index = np.nan
    if ((df['tags'] == user_oss).any()):
        oss_index = df.loc[df['tags'] == user_oss].index[0]
    else:
        oss_index = df.loc[df['name'] == user_oss].index[0] if (
            (df['name'] == user_oss).any()) else np.nan

    if not np.isnan(oss_index):
        # Compute the cosine similarities between the user repo and all other repo
        user_oss_vector = w2v_feature_array[oss_index].reshape(1, -1)
        similarity_scores = cosine_similarity(
            user_oss_vector, w2v_feature_array)

        # Get the top 20 most similar repos
        similar_repos = list(enumerate(similarity_scores[0]))
        sorted_similar_repos = sorted(
            similar_repos, key=lambda x: x[1], reverse=True)[:30]

        # Print the top 20 similar repos
        res = []
        printed_names = []  # List to keep track of printed names
        for i, score in sorted_similar_repos:
            name = df.loc[i, 'name']
            if name not in printed_names:  # Check if name has already been printed
                tags = df.loc[i, 'tags']
                link = df.loc[i, 'site']
                desc = df.loc[i, 'desc']
                upforgrabs__link = df.loc[i, 'upforgrabs__link']
                sc = df.loc[i, 'stats__issue-count']
                printed_names.append(name)
                res.append({'name': name, 'tags': tags, 'link': link, 'desc': desc,
                           'upforgrabs__link': upforgrabs__link, 'stats__issue-count': sc})  # Add name to printed names list
        return jsonify(msg=res, status="success")
    else:
        return jsonify(msg="No matching repository found", status="error")


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)
