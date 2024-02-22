from elasticsearch import Elasticsearch

# Assuming 'es' is an Elasticsearch client instance
es = Elasticsearch(["http://localhost:9200"])

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
