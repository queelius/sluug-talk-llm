import json
from elasticsearch import Elasticsearch

# Assuming 'es' is an Elasticsearch client instance
es = Elasticsearch(["http://localhost:9200"])

query = {
  "query": {
    "match": {
      "title": "quantum computing"
    }
  }
}

# print json
response = es.search(index="articles", body=query)
print(json.dumps(response, indent=2))
