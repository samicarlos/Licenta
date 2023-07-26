import pickle
import numpy as np
import pandas as pd

movies = pickle.load(open("movies_list.pkl", 'rb'))
similarity = pickle.load(open("similarity_matrix.pkl", 'rb'))
ratings = pickle.load(open("merged_ratings.pkl", 'rb'))

ratings.rename(columns={'tmdbId': 'movieId'}, inplace=True)
merged_ratings = pd.merge(ratings, movies, on='movieId')
merged_ratings = merged_ratings.drop(columns=['title', 'tags'])

#Adjust indexing
merged_ratings['userId'] = merged_ratings['userId'] - 1

def recommender(user):
    similarity_scores = similarity[user]
    # Sort the similarity scores in descending order and get the top N movies
    top_indices = np.argsort(similarity_scores)[::-1][:5]
    # Retrieve the recommended movie titles using list comprehension
    recommended_movies = [movies[movies['id'] == idx]['title'].values[0] for idx in top_indices]
    return recommended_movies

def calculate_precision_recall(similarity, merged_ratings, k):

    precision = 0
    recall = 0
    average_precision = 0
    users = set(merged_ratings['userId'])
    movies = set(merged_ratings['id'])
    user_movie_ratings = merged_ratings.groupby(['userId', 'id'])['rating'].apply(list)

    for user in users:
        similarity_scores = similarity[user]
        top_indices = np.argsort(similarity_scores)[::-1]
        movies_rated = []
        ratings_sum = 0
        movies_count = 0
        for movie in movies:
            if (user, movie) in user_movie_ratings:
                movies_rated.append(movie)
                ratings_sum += user_movie_ratings[(user, movie)][0]
                movies_count += 1
        average_rating = ratings_sum / movies_count if movies_count > 0 else 0

        relevant_items = [movie for movie in movies_rated if user_movie_ratings[(user, movie)][0] > average_rating]
        recommended_items = [movie for movie in top_indices if movie in movies_rated]
        true_positives = set(relevant_items).intersection(set(recommended_items[:k]))

        if len(relevant_items) > 0:
            precision_at_k = []
            for i in range(1, min(len(recommended_items[:k]), len(relevant_items))+1):
                true_positives_at_k = set(relevant_items).intersection(set(recommended_items[:i]))
                precision_at_k.append(len(true_positives_at_k)/i)
            average_precision += np.sum(precision_at_k) / len(relevant_items)

        precision += len(true_positives)/len(recommended_items[:k]) if len(recommended_items[:k]) > 0 else 1
        recall += len(true_positives)/len(relevant_items) if len(relevant_items) > 0 else 1

    precision /= len(users)
    recall /= len(users)
    average_precision /= len(users)

    return precision, recall, average_precision

for k in [5, 10, 20, 30, 50, 1000]:  # Top-K recommendations
    precision, recall, avg_precision = calculate_precision_recall(similarity, merged_ratings, k)
    print(f"Precision@{k}: {precision}")
    print(f"Recall@{k}: {recall}")
    print(f"mAP@{k}: {avg_precision}")
    print("-----------------------")
