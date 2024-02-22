from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import random

# Establish a connection to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Define authors and titles for synthetic data generation
authors = ["Alice Johnson", "Bob Smith", "Carol Lee", "Derek Hale", "Eve Chen"]
titles = [
    "The Future of Quantum Computing",
    "Exploring the Depths of Machine Learning",
    "A Journey Through the History of Mathematics",
    "Understanding the Basics of Cryptography",
    "The Impact of AI on Modern Society"
]

# Function to generate a random date
def random_date(start, end):
    """Generate a random date between `start` and `end`."""
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())),
    )

# Generate and index articles
for _ in range(5):  # Generate 5 articles
    article = {
        "title": random.choice(titles),
        "author": random.choice(authors),
        "publication_date": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1)).date().isoformat()
    }
    # Index the article in Elasticsearch
    response = es.index(index="articles", document=article)
    print(f"Indexed article ID: {response['_id']}")

print("Done indexing articles.")
