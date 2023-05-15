import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
cors = CORS(app)

# cors = CORS(app, resources={r"/*": {"origins": "*"}})


# CORS error handling

"""Import the dataset"""

df = pd.read_csv('oss_data.csv')
df.fillna(method='ffill', inplace=True)

"""Combining the different attributes of the dataset into a single string"""

df['content'] = df['name'].astype(str) + ' ' + df['desc'].astype(str) + ' ' + df['tags'] + ' ' + df['upforgrabs__link'].astype(str)
df['content'] = df['content'].fillna('')
df['content']

"""Tokenize content for Word2Vec"""

df['tokenized_content'] = df['content'].apply(simple_preprocess)
df['tokenized_content']

"""Training the Word2Vec model"""

model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(df['tokenized_content'])
model.train(df['tokenized_content'], total_examples=model.corpus_count, epochs=10)

w2v_feature_array = averaged_word_vectorizer(corpus=df['tokenized_content'], model=model, num_features=100)

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


# Endpoint to receive similar repositories
@app.route('/similar_repos', methods=['POST'])
def get_similar_repos():
    user_oss = request.json['query'].strip().replace(" ", "")

    oss_index = np.nan
    if ((df['tags'] == user_oss).any()): 
        oss_index = df.loc[df['tags'] == user_oss].index[0]
    else:
        oss_index = df.loc[df['name'] == user_oss].index[0] if ((df['name'] == user_oss).any()) else np.nan

    if not np.isnan(oss_index):
        user_oss_vector = w2v_feature_array[oss_index].reshape(1, -1)
        similarity_scores = cosine_similarity(user_oss_vector, w2v_feature_array)

        similar_repos = list(enumerate(similarity_scores[0]))
        sorted_similar_repos = sorted(similar_repos, key=lambda x: x[1], reverse=True)[:30]

        res = [] 
        printed_names = []
        for i, score in sorted_similar_repos:
            name = df.loc[i, 'name']
            if name not in printed_names:
                tags = df.loc[i, 'tags']
                link = df.loc[i, 'upforgrabs__link']
                desc = df.loc[i, 'desc']
                printed_names.append(name)
                saved_info = save_search_information(i)
                res.append({'name': name, 'tags': tags, 'link': link, 'desc': desc, 'saved_info': saved_info})

        return jsonify(msg=res, status="success")
    else:
        return jsonify(msg="No matching repository found", status="error")

def save_search_information(repository_id):
    # Specify the path to the CSV file
    csv_file = 'searched_information.csv'

    # Extract the information of the repository
    name = df.loc[repository_id, 'name']
    tags = df.loc[repository_id, 'tags']
    link = df.loc[repository_id, 'upforgrabs__link']
    desc = df.loc[repository_id, 'desc']

    # Save the information in the CSV file
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, tags, link, desc])

    # Return the saved repository information
    return {'name': name, 'tags': tags, 'link': link, 'desc': desc}

# Approach I: Output based on the dataset - upforgrabs github link    
@app.route('/repository_info/<int:repository_id>', methods=['GET'])
def get_repository_info(repository_id):
    if repository_id >= 0 and repository_id < len(df):
        name = df.loc[repository_id, 'name']
        tags = df.loc[repository_id, 'tags']
        link = df.loc[repository_id, 'upforgrabs__link']
        desc = df.loc[repository_id, 'desc']

        # Fetch extra information using the upforgrabs__link attribute
        extra_info = fetch_extra_info(link)

        return jsonify(name=name, tags=tags, link=link, desc=desc, extra_info=extra_info)
    else:
        return jsonify(msg="Invalid repository ID", status="error")

def fetch_extra_info(link):
    try:
        response = requests.get(link)
        response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
        html_content = response.text

        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract the desired information from the parsed HTML
        # Customize this code according to the structure of the webpage you're scraping
        extra_info = {
            'extra_attribute1': soup.find('tag', class_='class-name').text,
            'extra_attribute2': soup.find('tag', id='element-id').get('attribute-name'),
            # Add more attributes as needed
        }

        return extra_info

    except requests.exceptions.RequestException as e:
        # Handle request exceptions (e.g., network errors)
        print("Error during request:", e)

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (e.g., 404, 500)
        print("HTTP error occurred:", e)

    except Exception as e:
        # Handle any other exceptions
        print("An error occurred:", e)
        
    return None
  
## Approach II: Based on Github API  
# @app.route('/repository_info/<string:owner>/<string:repo>', methods=['GET'])
# def get_repository_info(owner, repo):
#     url = f'https://api.github.com/repos/{owner}/{repo}'
#     headers = {'Accept': 'application/vnd.github.v3+json'}

#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
#         repository_data = response.json()
#         name = repository_data['name']
#         description = repository_data['description']
#         tags = repository_data['topics']
#         html_url = repository_data['html_url']

#         return jsonify(name=name, description=description, tags=tags, html_url=html_url)

#     except requests.exceptions.RequestException as e:
#         # Handle request exceptions (e.g., network errors)
#         return jsonify(msg="Error during API request", status="error")

#     except requests.exceptions.HTTPError as e:
#         # Handle HTTP errors (e.g., 404, 500)
#         return jsonify(msg="Repository not found", status="error")

#     except KeyError as e:
#         # Handle missing keys in the response JSON
#         return jsonify(msg="Invalid response from GitHub API", status="error")

#     except Exception as e:
#         # Handle any other exceptions
#         return jsonify(msg="An error occurred", status="error")
    
    
# if __name__ == '__main__':
#     app.run(debug=True)
