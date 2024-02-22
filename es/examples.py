import random
import json

db = {
    "gutenberg": [
        {
            "_nlq": "show me books by lincoln that contain 'gettysberg' or something like that.",
            "_query": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "author": {
                                        "query": "lincoln",
                                        "fuzziness": "AUTO"
                                    }
                                }
                            }
                        ],
                        "should": [
                            {
                                "match": {
                                    "content": {
                                        "query": "gettysberg",
                                        "fuzziness": "AUTO"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    ],
    "articles": [
        {
            "_nlq": "find articles about quantum computing.",
            "_query": {
                "query": {
                    "match": {
                        "content": {
                            "query": "quantum computing",
                            "fuzziness": "AUTO"
                        }
                    }
                }
            }
        },
        {
            "_nlq": "count the number of articles by author.",
            "_query": {
                "query": {
                    "size": 0,
                    "aggs": {
                        "articles_by_author": {
                            "terms": {"field": "author"}
                        }
                    }
                }
            }
        }
    ]
}

def generate_example(index):

    if index not in db or len(db[index]) == 0:
        return ""

    ex_index = random.randint(0, len(db[index]) - 1)
    example = db[index][ex_index]

    example_content = f"""
Here is an example of an NLQ and its corresponding ElasticSearch query for the index `{index}`:

NLQ: "{example["_nlq"]}"
Query:

```json
{json.dumps(example["_query"], indent=2)}
```
"""
    return example_content
