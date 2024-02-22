from elasticsearch import Elasticsearch
import json
# Assuming 'es' is an Elasticsearch client instance
es = Elasticsearch(["http://localhost:9200"])	

query = {
  "size": 0,  # Don't return the actual documents
  "aggs": {
    "articles_by_author": {
      "terms": { "field": "author" }
    }
  }
}

response = es.search(index="articles", body=query)
print(json.dumps(response, indent=2))
