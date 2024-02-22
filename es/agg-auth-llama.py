from elasticsearch import Elasticsearch
import json
# Assuming 'es' is an Elasticsearch client instance
es = Elasticsearch(["http://localhost:9200"])	

#query = {
#  "size": 0,  # Don't return the actual documents
#  "aggs": {
#    "articles_by_author": {
#      "terms": { "field": "author" }
#    }
#  }
#}


query = {
  "aggs": {
    "authors": {
      "terms": {
        "field": "author",
        "order": { "_count": "desc" }
        }
    }
  }
}


query = {
  "aggs": {
    "authors": {
      "terms": {
        "field": "author",
        "order": { "_count": "desc" }
      }
    }
  }
}

response = es.search(index="articles", body=query)
print(json.dumps(response, indent=2))
