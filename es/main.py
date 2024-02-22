import nlq
from fastapi import FastAPI, HTTPException, Body, Path
from elasticsearch.exceptions import NotFoundError
from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field
from typing import Dict
from fastapi.responses import HTMLResponse
from fastapi import Request
from openai import OpenAI
import os
import json

app = FastAPI()

es = Elasticsearch(["http://localhost:9200"])

class NLQ(BaseModel):
    query: str = Field(default="Find articles by Lincoln that contain 'Gettysberg' or something like that.", description="The natural language query (NLQ) to process.")
    model: str = Field(default="gpt-3.5-turbo", description="The model used for querying.")
    base_url: str = Field(default="https://api.openai.com/v1", description="The base URL for the OpenAI API.")
    use_examples: bool = Field(default=False, description="Whether to use any stored examples for the specified index.")

@app.post("/mappings/{index}")
async def create_index(index: str, mappings: dict = Body(..., example={
    "properties": {
        "title": {
            "type": "text"
        },
        "description": {
            "type": "text"
        }
    }
    })):
    try:
        response = es.indices.create(index=index,
                                     body={"mappings": mappings})
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ElasticSearch Mapping: retrieve the mapping definition
@app.get("/mappings/{index}")
async def get_mapping(index: str):
    try:
        response = es.indices.get_mapping(index=index)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/indexes/{index}/nlq", summary="Process NLQ")
async def process_nlq(
    index: str,
    request: NLQ):
    """
    Perform a natural language query (NLQ) on a specified index using OpenAI's API.

    To use a local API base URL:
    ```json
    {
        "query": "Find articles about quantum computing.",
        "model": "llama2:13b",
        "base_url": "http://localhost:11434/v1",
        "use_examples": true
    }
    ```
    """
    system_message = nlq.generate_system_message(
        es, index, request.use_examples)

    print("System message:")
    print("-" * 40)
    print(system_message)
    print("-" * 40)

    system_message = {
        "role": "system",
        "content": system_message,
    }
    #print(json.dumps(system_message, indent=2))

    # User message with the natural language query (NLQ)
    user_message = {
        "role": "user",
        "content": f"The natural language query is: '{request.query}'"
    }
    #print(json.dumps(user_message, indent=2))

    client = OpenAI(
        base_url=request.base_url,
        api_key=os.getenv('OPENAI_API_KEY'))

    # The API call structure, requesting JSON-only output
    response = client.chat.completions.create(
        model=request.model,
        response_format={"type": "json_object"},
        messages=[system_message, user_message])
    choice_content = response.choices[0].message.content
    query_body = json.loads(choice_content)
    
    #print(f"{query_body}")

    response = es.search(index=index, body=query_body)
    #print(json.dumps(response, indent=2))

    # let's see if 'content' field exists in the response
    # if so, let's truncate it to 100 characters
    for hit in response['hits']['hits']:
        if "content" in hit["_source"]:
            hit["_source"]["content"] = hit["_source"]["content"][:100] + "..."

    response["_nlq"] = request.query
    response["_query"] = query_body
    response["_system_message"] = system_message.get("content")

    print(query_body)
    
    return response



# retrieve the indexes
@app.get("/indexes")
async def get_indexes():
    try:
        response = es.indices.get_alias("*")
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# function to get download top 100 gutenberg books
@app.get("/download_gutenberg/{num_books}")
async def download_gutenberg(num_books: int = 10):
    from guten import get_gutenberg
    get_gutenberg(num_books=num_books,
                  dir='./source/gutenberg')

@app.post("/indexes/{index}/query")
async def run_query(index: str, query: dict = Body(..., example={
    "size": 0,
    "aggs": {
        "articles_by_author": {
            "terms": {"field": "author"}
            }
        }
    })):
    try:
        response = es.search(index=index, body=query)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/indexes/{index}")
async def search(index: str, query: str):
    search_query = {
        "query": {
            "multi_match": { "query": query }
        }
    }
    try:
        response = es.search(index=index, body=search_query)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/indexes/{index}")
async def add_document(index: str, doc: dict = Body(...)):
    try:
        response = es.index(index=index, body=doc)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/indexes/{index}")
async def delete_index(index: str):
    try:
        response = es.indices.delete(index=index)
        return {"message": f"Index '{index}' was deleted successfully.", "response": response}
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Index '{index}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Simple Search Interface</title>
        </head>
        <body>
            <h2>Simple Search Interface</h2>
            <form id="searchForm">
                <label for="index">Index:</label>
                <input type="text" id="index" name="index" value="default_index" required><br><br>
                <label for="query">Query:</label>
                <input type="text" id="query" name="query" placeholder="Enter your search query" required><br><br>
                <button type="button" onclick="performSearch()">Search</button>
            </form>
            <div id="searchResults"></div>
            <script>
                function performSearch() {
                    const index = document.getElementById('index').value;
                    const query = document.getElementById('query').value;
                    fetch('/indexes/' + encodeURIComponent(index) + '/nlq', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: query,
                            model: 'gpt-3.5-turbo', // Adjust this as needed
                            base_url: 'https://api.openai.com/v1', // Or use your custom base_url
                            use_examples: true // Adjust based on your requirements
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        displayResults(data);
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        document.getElementById('searchResults').innerHTML = 'An error occurred. Check the console for details.';
                    });
                }
                
                function displayResults(data) {
                    const resultsElement = document.getElementById('searchResults');
                    resultsElement.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }
            </script>
        </body>
    </html>
    """
