import numpy as np
import pandas as pd

def load_ratings_from_csv(file_path):
    # Load ratings from CSV file
    ratings_df = pd.read_csv(file_path)

    # Adjust user and movie IDs to start from 0
    ratings_df['userId'] = ratings_df['userId'] - 1
    # Drop rows with NaN values
    ratings_df.dropna(inplace=True)

    # Get the total number of users and movies
    num_users = ratings_df['userId'].nunique()
    num_movies = ratings_df['movieId'].nunique()

    # Get unique movie IDs
    unique_movie_ids = ratings_df['movieId'].unique()

    # Create a mapping dictionary
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids)}

    # Create a reverse mapping dictionary
    reverse_id_mapping = {new_id: old_id for old_id, new_id in id_mapping.items()}

    ratings_matrix = np.zeros((num_users, num_movies))

    for row in ratings_df.itertuples(index=False):
        user_idx = row.userId
        movie_idx = id_mapping[row.movieId]
        rating = row.rating
        ratings_matrix[user_idx, movie_idx] = rating

    ratings_matrix = np.nan_to_num(ratings_matrix, nan=0)

    return ratings_matrix, id_mapping, reverse_id_mapping

def matrix_factorization(R, latent_dim, steps, learning_rate, reg):
    num_users, num_items = R.shape
    # initialize Q, P, bias_user, and bias_item with random values
    Q = np.random.rand(num_users, latent_dim)
    P = np.random.rand(num_items, latent_dim)

    # Calculate the mean of all non-zero ratings
    mean_rating = np.mean(R[np.nonzero(R)])

    # Initialize bias_user and bias_item with real bias values
    bias_user = np.random.rand(num_users, 1) * (mean_rating / latent_dim)
    bias_item = np.random.rand(num_items, 1) * (mean_rating / latent_dim)

    # Perform gradient descent
    loss_history = []
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0 and np.random.rand() < 0.01:
                    error = R[i, j] - (np.dot(Q[i, :], P[j, :].T) + bias_user[i] + bias_item[j])
                    for k in range(latent_dim):
                        Q[i, k] += learning_rate * (2 * error * P[j, k].T - 2 * reg * Q[i, k])
                        P[j, k] += learning_rate * (2 * error * Q[i, k] - 2 * reg * P[j, k].T)
                    bias_user[i] += learning_rate * (2 * error - reg * bias_user[i])
                    bias_item[j] += learning_rate * (2 * error - reg * bias_item[j])
        error = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:
                    error += pow(R[i, j] - (np.dot(Q[i, :], P[j, :].T) + bias_user[i] + bias_item[j]), 2)
                    for k in range(latent_dim):
                        error += (reg) * (pow(Q[i, k], 2) + pow(P[j, k], 2))
                    error += (reg) * (pow(bias_user[i], 2) + pow(bias_item[j], 2))
        if (step + 1) % 100 == 0 or step == 0:
            print(step + 1, error)
        if step + 1 == 500 or step + 1 == 100 or step + 1 == 300:
            predicted_ratings = compute_predictions(Q, P, bias_user, bias_item)
            rmse = calculate_rmse(test_set, predicted_ratings[300:, 4000:])
            print(step + 1, "rmse: ", rmse)
        loss_history.append(error)

        if step > 0 and abs(loss_history[step] - loss_history[step - 1]) < 0.001:
            break

    return Q, P, bias_user, bias_item, loss_history

def compute_predictions(Q, P, bias_user, bias_item):
    predicted_ratings = np.dot(Q, P.T) + bias_user + bias_item.T
    return predicted_ratings

def calculate_rmse(actual_ratings, predicted_ratings):
    mask = actual_ratings > 0  # Only consider the entries with non-zero ratings
    squared_error = np.power(actual_ratings - predicted_ratings, 2)
    mse = np.mean(squared_error[mask])
    rmse = np.sqrt(mse)
    return rmse

def calculate_precision_recall(actual_ratings, predicted_ratings, k):
    num_users, num_items = actual_ratings.shape
    precision = 0
    recall = 0
    average_precision = 0

    for user_idx in range(num_users):
        user_ratings = actual_ratings[user_idx]  # Ratings of the current user
        user_avg_rating = np.mean(user_ratings[user_ratings > 0])  # Average rating of the current user

        actual_positives = np.where(user_ratings > user_avg_rating)[0]
        predicted_items = []
        for i in np.where(user_ratings > 0)[0]:
            if predicted_ratings[user_idx][i] > user_avg_rating:
                predicted_items.append(i)

        predicted_items_sorted = sorted(predicted_items, key=lambda x: predicted_ratings[user_idx][x], reverse=True)
        if k != 0:
            top_predicted_items = predicted_items_sorted[:k]
        else:
            top_predicted_items = predicted_items_sorted

        true_positives = np.intersect1d(actual_positives, top_predicted_items)
        false_positives = np.setdiff1d(top_predicted_items, actual_positives)

        precision += len(true_positives) / (len(true_positives) + len(false_positives)) if len(top_predicted_items) > 0 else 1
        recall += len(true_positives) / len(actual_positives) if len(actual_positives) > 0 else 1

        # Calculate Average Precision
        num_relevant_items = len(actual_positives)
        num_retrieved_items = len(top_predicted_items)
        if num_relevant_items > 0:
            # Calculate precision at each position of the retrieved items
            precision_at_k = []
            for i in range(1, num_retrieved_items + 1):
                relevant_items = np.intersect1d(actual_positives, top_predicted_items[:i])
                precision_at_k.append(len(relevant_items) / i)
            average_precision += np.sum(precision_at_k) / num_relevant_items

    precision /= num_users
    recall /= num_users
    average_precision /= num_users

    return precision, recall, average_precision

# Load ratings from CSV file
file_path = 'ml-latest-small/ratings.csv'
ratings, map, reverse_map = load_ratings_from_csv(file_path)

#Split matrix into training set and test set ~ 0.8/0.2
test_set = ratings[300:, 4000:]
training_set = np.copy(ratings)
training_set[300:, 4000:] = 0

from sklearn.model_selection import ParameterGrid

# Define the hyperparameter grid
hyperparameters = {
    'latent_dim': [7, 10, 12, 15, 20],
    'learning_rate': [0.01],
    'breg': [0.1, 0.05, 0.15],
    'asteps': [1000]
}

# Perform grid search
best_rmse = float('inf')
best_params = None

for params in ParameterGrid(hyperparameters):
    steps = params['asteps']
    latent_dim = params['latent_dim']
    learning_rate = params['learning_rate']
    reg = params['breg']
    print(latent_dim, steps, learning_rate, reg)

    # Train the matrix factorization model
    Q, P, bias_user, bias_item, loss_history = matrix_factorization(training_set, latent_dim, steps, learning_rate, reg)

    # Compute predictions
    predicted_ratings = compute_predictions(Q, P, bias_user, bias_item)

    # Calculate RMSE
    rmse = calculate_rmse(test_set, predicted_ratings[300:, 4000:])

    print("latent dim: ", latent_dim, '\nsteps: ', steps, "\nlearning_rate: ", learning_rate, "\nreg: ", reg, "\nrmse: ", rmse)

    # Calculate precision and recall
    for k in [5, 10, 20, 30, 50, 100, 0]:  # Top-K recommendations
        precision, recall, avg_precision = calculate_precision_recall(ratings, predicted_ratings, k)
        print(f"Precision@{k}: {precision}")
        print(f"Recall@{k}: {recall}")
        print(f"mAP@{k}: {avg_precision}")
    print("-----------------------")

    # Update best parameters if RMSE improves
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

print("Best RMSE:", best_rmse)
print("Best parameters:", best_params)

# # Example usage:
# user_id = 0  # User for whom to make recommendations
# top_recommendations = np.argsort(predicted_ratings[user_id])[::-1]
# for i in range(10):  # Display top 10 recommendations
#     movie_id = reverse_map[top_recommendations[i]]
#     print(f"Recommended movie {i+1}: {movie_id}")