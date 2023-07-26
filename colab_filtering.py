import numpy as np
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split

from collections import defaultdict
from operator import itemgetter
import heapq

import os
import csv

# Load in the movie ratings and return a dataset.
def load_dataset():
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
    ratings_dataset = Dataset.load_from_file('ml-latest-small/ratings.csv', reader=reader)

    # Lookup a movie's name with it's Movielens ID as key
    movieID_to_name = {}
    with open('ml-latest-small/movies.csv', newline='', encoding='ISO-8859-1') as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)
            for row in movie_reader:
                movieID = int(row[0])
                movie_name = row[1]
                movieID_to_name[movieID] = movie_name
    # Return both the dataset and lookup dict in tuple
    return (ratings_dataset, movieID_to_name)

dataset, movieID_to_name = load_dataset()

# Build a full Surprise training set from dataset
#trainset = dataset.build_full_trainset()
trainset, testset = train_test_split(dataset, test_size=0.2)

model = KNNBasic(k=20, sim_options={
        'name': 'pearson',
        'user_based': True
    })\
    .fit(trainset)
similarity_matrix = model.compute_similarities()

# Perform cross-validation and calculate RMSE
from surprise.model_selection import cross_validate
cross_validated_results = cross_validate(KNNBasic(), dataset, measures=['RMSE'], cv=5, verbose=True)

# Extract and print the RMSE score from cross-validation results
rmse = cross_validated_results['test_rmse'].mean()
print("Average RMSE:", rmse)

predictions = model.test(testset)

def get_average_ratings(dataset):
    # Create a dictionary to store the sum of ratings and count of ratings for each user
    user_ratings_sum = {}
    user_ratings_count = {}

    # Iterate over the training set and calculate the sum and count of ratings for each user
    for user_id, _, rating in dataset:
        if user_id in user_ratings_sum:
            user_ratings_sum[user_id] += rating
            user_ratings_count[user_id] += 1
        else:
            user_ratings_sum[user_id] = rating
            user_ratings_count[user_id] = 1

    # Calculate the average rating for each user
    user_average_ratings = {}
    for user_id in user_ratings_sum:
        average_rating = user_ratings_sum[user_id] / user_ratings_count[user_id]
        user_average_ratings[user_id] = average_rating

    return user_average_ratings

user_average_ratings = get_average_ratings(testset)

def calculate_precision_recall(predictions, user_average_ratings, k):
    precision = 0
    recall = 0
    average_precision = 0
    relevant_items = {}
    recommended_items = {}
    unique_users = set()

    for prediction in predictions:
        user_id = prediction.uid
        actual_rating = prediction.r_ui
        predicted_rating = prediction.est

        if user_id not in relevant_items:
            relevant_items[user_id] = []
        if user_id not in recommended_items:
            recommended_items[user_id] = []

        if actual_rating > user_average_ratings[user_id]:
            relevant_items[user_id].append(predicted_rating)
        if predicted_rating > user_average_ratings[user_id]:
            recommended_items[user_id].append(predicted_rating)

        unique_users.add(user_id)

    for user in unique_users:
        sorted_recommended_items = sorted(recommended_items[user], key=lambda x: x, reverse=True)
        true_positives = np.intersect1d(relevant_items[user], sorted_recommended_items[:k])
        precision += len(true_positives)/len(sorted_recommended_items[:k]) if len(sorted_recommended_items[:k]) > 0 else 1
        recall += len(true_positives) / len(relevant_items[user]) if len(relevant_items[user]) > 0 else 1
        if len(relevant_items[user]) > 0:
            precision_at_k = []
            for i in range(1, len(sorted_recommended_items[:k])):
                true_positives_at_k = np.intersect1d(relevant_items[user], sorted_recommended_items[:i])
                precision_at_k.append(len(true_positives_at_k) / i)
            average_precision += np.sum(precision_at_k) / len(relevant_items[user])

    precision /= len(unique_users)
    recall /= len(unique_users)
    average_precision /= len(unique_users)

    return precision, recall, average_precision

for k in [5, 10, 20, 30, 50, 1000]:  # Top-K recommendations
    precision, recall, avg_precision = calculate_precision_recall(predictions, user_average_ratings, k)
    print(f"Precision@{k}: {precision}")
    print(f"Recall@{k}: {recall}")
    print(f"mAP@{k}: {avg_precision}")
    print("-----------------------")

#Print the average ratings for each user
for user_id, average_rating in user_average_ratings.items():
    print("User ID:", user_id, "Average Rating:", average_rating)

# Pick a random user ID
test_subject = '70'

# Get the top K items user rated
k = 20

# When using Surprise, there are RAW and INNER IDs.
# Raw IDs are the IDs, strings or numbers, you use when
# creating the trainset. The raw ID will be converted to
# an unique integer Surprise can more easily manipulate
# for computations.
#
# So in order to find an user inside the trainset, you
# need to convert their RAW ID to the INNER Id. Read
# here for more info https://surprise.readthedocs.io/en/stable/FAQ.html#what-are-raw-and-inner-ids
test_subject_iid = trainset.to_inner_uid(test_subject)

# Get the top K items we rated
test_subject_ratings = trainset.ur[test_subject_iid]
k_neighbors = heapq.nlargest(k, test_subject_ratings, key=lambda t: t[1])

candidates = defaultdict(float)

for itemID, rating in k_neighbors:
    try:
      similaritities = similarity_matrix[itemID]
      for innerID, score in enumerate(similaritities):
          candidates[innerID] += score * (rating / 5.0)
    except:
      continue

# Utility we'll use later.
def getMovieName(movieID):
  if int(movieID) in movieID_to_name:
    return movieID_to_name[int(movieID)]
  else:
      return ""

# Build a dictionary of movies the user has watched
watched = {}
for itemID, rating in trainset.ur[test_subject_iid]:
  watched[itemID] = 1

# Add items to list of user's recommendations
# If they are similar to their favorite movies,
# AND have not already been watched.
recommendations = []

position = 0
for itemID, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
  if not itemID in watched:
    recommendations.append(getMovieName(trainset.to_raw_iid(itemID)))
    position += 1
    if (position > 10): break # We only want top 10

for rec in recommendations:
  print("Movie: ", rec)