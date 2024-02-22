from elasticsearch import Elasticsearch

# Assuming 'es' is an Elasticsearch client instance
es = Elasticsearch(["http://localhost:9200"])

query = {
  "query": {
    "range": {
      "publication_date": {
        "gte": "2024-01-01",
        "lte": "2024-02-28"
      }
    }
  }
}

response = es.search(index="articles", body=query)
print(response)
