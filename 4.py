import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load MovieLens 100K dataset (auto-download)
ratings = tfds.load(
    "movielens/100k-ratings",
    split="train",
    shuffle_files=True
)

# Step 2: Prepare data
ratings = ratings.map(
    lambda x: (
        (x["user_id"], x["movie_id"]),
        x["user_rating"]
    )
)

# Convert to numpy for simplicity
data = list(tfds.as_numpy(ratings))

users = np.array([x[0][0] for x in data])
movies = np.array([x[0][1] for x in data])
ratings = np.array([x[1] for x in data])

# Normalize IDs
users = users.astype("int32")
movies = movies.astype("int32")

num_users = users.max() + 1
num_items = movies.max() + 1

print("Number of users:", num_users)
print("Number of movies:", num_items)

# Step 3: Build Matrix Factorization model
class Recommender(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        item_vec = self.item_embedding(inputs[:, 1])
        return tf.reduce_sum(user_vec * item_vec, axis=1)

model = Recommender(num_users, num_items)

# Step 4: Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Step 5: Train model
history = model.fit(
    np.stack([users, movies], axis=1),
    ratings,
    epochs=5,
    batch_size=256,
    validation_split=0.2
)

# Step 6: Plot loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

# Step 7: Recommendation function
def recommend(user_id, top_k=5):
    movie_ids = np.arange(num_items)
    user_ids = np.full(num_items, user_id)

    predictions = model.predict(
        np.stack([user_ids, movie_ids], axis=1),
        verbose=0
    )

    return np.argsort(-predictions)[:top_k]

# Example
print("Recommended movies for user 0:", recommend(0))
