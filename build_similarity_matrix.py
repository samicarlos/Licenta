import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

movies = pd.read_csv('dataset.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
links = pd.read_csv('ml-latest-small/links.csv')

movies = movies[['id', 'title', 'overview', 'genre']]
movies['tags'] = movies['overview']+movies['genre']
movies = movies.drop(columns=['overview', 'genre'])
movies.rename(columns={'id': 'movieId'}, inplace=True)

movies.reset_index(inplace=True)
movies.rename(columns={'index': 'id'}, inplace=True)

merged_ratings = pd.merge(ratings, links, on='movieId')
ratings = merged_ratings[['userId', 'tmdbId', 'rating']]

pickle.dump(ratings, open('merged_ratings.pkl', 'wb'))

item_profiles = {}

tfidf = TfidfVectorizer(max_features=60000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'].values.astype('U'))

feature_names = tfidf.get_feature_names_out()
print(feature_names[0])

for movie_id, tfidf_score in zip(movies['movieId'], tfidf_matrix):
    # Get the indices and corresponding IDF scores from the sparse matrix
    indices = tfidf_score.indices
    scores = tfidf_score.data

    # Sort the feature names based on the IDF scores in descending order
    sorted_indices = indices[np.argsort(scores)[::-1]]
    sorted_scores = scores[np.argsort(scores)[::-1]]
    sorted_feature_names = [feature_names[idx] for idx in sorted_indices]

    # Select the top 50 words with the highest IDF scores
    item_profile = dict(zip(sorted_feature_names, sorted_scores))
    item_profiles[movie_id] = item_profile

user_profiles = {}

user_ratings = ratings.groupby('userId')['rating'].mean().to_dict()

for user_id, rating in user_ratings.items():
    user_item_profile = {}
    word_count = {}

    # Filter ratings based on the user ID
    user_ratings = ratings[ratings['userId'] == user_id]

    rating_sum = user_ratings['rating'].sum()
    num_movies_rated = len(user_ratings)
    if num_movies_rated > 0:
        rating_avg = rating_sum / num_movies_rated
    else:
        rating_avg = 0

    # Calculate the weighted average of item profiles based on the user's ratings
    for _, row in user_ratings.iterrows():
        movie_id = row['tmdbId']
        movie_rating = row['rating']
        normalized_movie_rating = movie_rating - rating_avg #normalize the rating by substracting the avg rating of a user
        item_profile = item_profiles.get(movie_id, {})
        for word, tfidf_score in item_profile.items():
            user_item_profile[word] = user_item_profile.get(word, 0) + tfidf_score * normalized_movie_rating
            word_count[word] = word_count.get(word, 0) + 1

    # Get the weight of each item by dividing it with the number of appearances
    for word in user_item_profile:
        user_item_profile[word] /= word_count[word]

    # Sort the dictionary by values in descending order
    sorted_user_items = sorted(user_item_profile.items(), key=lambda x: x[1], reverse=True)

    # Convert the sorted items back to a dictionary
    sorted_user_item_profile = dict(sorted_user_items)

    # Add the user profile to our big dictionary that contains all user profiles
    user_profiles[user_id] = sorted_user_item_profile

sorted_feature_names = sorted(feature_names)
user_matrix = []
for user_key, user_value in user_profiles.items():
    user_list = []
    for word in sorted_feature_names:
        tfidf = user_value.get(word, 0)
        user_list.append(tfidf)
    user_matrix.append(user_list)
item_matrix = []
for item_key, item_value in item_profiles.items():
    item_list = []
    for word in sorted_feature_names:
        tfidf = item_value.get(word, 0)
        item_list.append(tfidf)
    item_matrix.append(item_list)
similarity_matrix = cosine_similarity(user_matrix, item_matrix)

# pickle.dump(movies, open('movies_list.pkl', 'wb'))
# pickle.dump(similarity_matrix, open('similarity_matrix.pkl', 'wb'))
