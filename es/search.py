from elasticsearch import Elasticsearch

# Assuming 'es' is an Elasticsearch client instance
es = Elasticsearch(["http://localhost:9200"])

query = {
  "query": {
    "match": {
      "author": "Casey Smith"
    }
  }
}

response = es.search(index="articles", body=query)
print(response)
