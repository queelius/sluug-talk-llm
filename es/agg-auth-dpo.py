from elasticsearch import Elasticsearch
import json
# Assuming 'es' is an Elasticsearch client instance
es = Elasticsearch(["http://localhost:9200"])	

# Define your index and type (if applicable)
index_name = "articles"
# Define the updated index mapping
index_mapping = {
    "mappings": {
        "properties": {
            "_id": {"type": "keyword", "doc_values": True}
        }
    }
}

# Send the mapping update request to Elasticsearch
res = es.indices.put_mapping(index=index_name, body=json.dumps(index_mapping))

query = {
    "size": 0,
    "aggs": {
        "authors_count": {
            "terms": {
                "field": "author.keyword",
                "size": 1000
            },
            "aggs": {
                "article_count": {
                    "cardinality": {
                        "field": "_id"
                    }
                }
            }
        }
    }
}

# Execute the query and print the results
response = es.search(index=index_name, body=query)
results = response["aggregations"]["authors_count"]["buckets"]

for result in results:
    author = result["key"]
    article_count = result["article_count"]["value"]
    print(f"{author} wrote {article_count} articles.")