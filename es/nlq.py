import json
import examples

def generate_system_message(es, index, use_examples):
    # Check if the index exists in Elasticsearch
    if index not in es.indices.get_alias("*"):
        return f"Index {index} does not exist in ElasticSearch."

    # Get the mapping for the index
    mapping = es.indices.get_mapping(index=index)

    # Convert the mapping to a pretty-printed JSON string
    mapping_json = json.dumps(mapping, indent=2)

    content = f"""
You are a world-class natural language to ElasticSearch query generator.
Your task is to convert natural language queries (NLQ) into ElasticSearch queries,
taking into account the provided index mapping. The response should be in JSON
format only.

Also, the ElasticSearch API no longer needs `field.keyword` notation.
Here is the ElasticSearch mapping for the index:

```json
{mapping_json}
```
"""

    # See if index exists in the examples database
    if use_examples:
        ex = examples.generate_example(index)
        content += ex

    content += "Generate an ElasticSearch query based on the following natural language query."

    return content