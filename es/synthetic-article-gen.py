from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import random

# Establish a connection to Elasticsearch
es = Elasticsearch("http://localhost:9200")

def generate_random_articles():

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


def generate_static_articles():
    articles = [
        {
            "title": "Exploring the Universe",
            "author": "Alex Johnson",
            "publication_date": "2024-01-01"
        },
        {
            "title": "Advances in Quantum Computing",
            "author": "Casey Smith",
            "publication_date": "2024-01-15"
        },
        {
            "title": "Artificial Intelligence and Ethics",
            "author": "Jordan Lee",
            "publication_date": "2024-02-01"
        },
        {
            "title": "Machine Learning in Healthcare",
            "author": "Jamie Garcia",
            "publication_date": "2024-02-15"
        },
        {
            "title": "The Future of Blockchain Technology",
            "author": "Robin Singh",
            "publication_date": "2024-03-01"
        },
        {
            "title": "Example Title",
            "author": "John Doe",
            "publication_date": "2024-02-21"
        }
    ]

    for article in articles:
        response = es.index(index="articles", body=article)
        print(response)
