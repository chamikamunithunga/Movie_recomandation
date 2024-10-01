# movie_recommender.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, accuracy
from surprise.model_selection import cross_validate

# Step 1: Load and Preprocess the Dataset
def load_data():
    # Using a sample movie ratings dataset (you can replace this with any other dataset)
    ratings_data = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2, 3, 3],
        'movieId': [101, 102, 103, 101, 104, 105, 101, 105],
        'rating': [4, 5, 2, 3, 4, 2, 5, 3]
    })

    print("Loaded data:")
    print(ratings_data)
    
    return ratings_data

# Step 2: Prepare the Data for Surprise
def prepare_data(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data

# Step 3: Train the SVD Model
def train_model(data):
    trainset, testset = train_test_split(data, test_size=0.25)
    
    # Use SVD (Singular Value Decomposition) algorithm
    algo = SVD()

    # Train on training data
    algo.fit(trainset)
    
    # Evaluate on test data
    predictions = algo.test(testset)
    accuracy.rmse(predictions)

    return algo

# Step 4: Make Predictions for a User
def make_predictions(algo, user_id, data, top_n=5):
    # Get all unique movie IDs
    movie_ids = data.df['movieId'].unique()

    # Get movies the user has already rated
    rated_movies = data.df[data.df['userId'] == user_id]['movieId'].values

    # Get unrated movies
    unrated_movies = [movie_id for movie_id in movie_ids if movie_id not in rated_movies]

    # Predict ratings for unrated movies
    predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_movies]
    
    # Sort predictions by estimated rating
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)

    return recommendations[:top_n]

# Step 5: Main Function to Run the Recommendation System
def main():
    # Load data
    ratings = load_data()

    # Prepare data for Surprise
    data = prepare_data(ratings)

    # Train the model
    algo = train_model(data)

    # Predict top 5 recommendations for a specific user
    user_id = 1
    print(f"\nTop 5 movie recommendations for user {user_id}:\n")
    recommendations = make_predictions(algo, user_id, data)

    for i, rec in enumerate(recommendations):
        print(f"Rank {i + 1}: Movie ID {rec.iid} with predicted rating {rec.est:.2f}")

if __name__ == "__main__":
    main()
