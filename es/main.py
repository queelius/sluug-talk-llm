from fastapi import FastAPI, HTTPException, Body, Path
from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field
from typing import Dict
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import ollama
import openai
import os
import json
openai.api_key = os.getenv('OPENAI_API_KEY')

#from langchain_community.llms import Ollama
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
#from langchain.prompts import PromptTemplate

app = FastAPI()

# Assuming Elasticsearch is running on localhost:9200
es = Elasticsearch(["http://localhost:9200"])

class Query(BaseModel):
    index: str
    body: Dict

# Make sure the model path is correct for your system!
#model = lllama_cpp.Llama(
#    model_path="/home/spinoza/models/llama2_13b.gguf",
#    n_gpu_layers=-1,
#    verbose=True,  # Verbose is required to pass to the callback manager
#)    
    
#from llama_cpp import Llama
#llm = Llama(
#      model_path="/home/spinoza/models/llama2_13b.gguf",
#      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      #n_ctx=4096 # Uncomment to increase the context window
#)


# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")



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


@app.get("/indexes/{index}/nlq_openai")
async def process_nlq_openai(index: str,
                             query: str = "Show me all articles."):
    
    # get the mapping for the index
    mapping = es.indices.get_mapping(index=index)

    # convert the mapping to a pretty-printed JSON string
    mapping_json = json.dumps(mapping, indent=2)

    system_message_content = f"""
    You are a world-class natural language to ElasticSearch query generator.
    Your task is to convert natural language queries into ElasticSearch queries, taking into account the provided index mapping. The response should be in JSON format only.

    Here is the mapping for the index:

    ```json
    {mapping_json}
    ```

    Generate the ElasticSearch query based on the following natural language query.
    """    

    # strip leading spaces
    system_message_content = system_message_content.strip()

    system_message = {
        "role": "system",
        "content": system_message_content,
    }

    # pretty print system message
    print(json.dumps(system_message, indent=2))

    # User message with the natural language query (NLQ)
    user_message = {
        "role": "user",
        "content": f"The natural language query is: '{query}'"
    }

    print(json.dumps(user_message, indent=2))
    
    # The API call structure, requesting JSON-only output
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=[
            system_message,
            user_message
        ]
    )

    return response



# let's make index default to "articles", i think you have to use some pydantic magic to do this
@app.get("/indexes/{index}/nlq")
async def process_nlq(index: str,
                      query: str = "Show me all articles.",
                      model: str = "ollama2"):
    """
    We are going to use LangChain to call my language model to generate an
    ElasticSearch query from a natural language query.

    1. Call the language model
    2. Get the response
    3. Use the response to create an ElasticSearch query
    4. Run the query
    5. Return the response
    """
    try:

        
        response = es.indices.get_mapping(index=index)
        #print(response)

        mapping = response[index]
        # Convert the mapping to a pretty-printed JSON string
        mapping_json = json.dumps(mapping, indent=2)

        template = f"""
You are world class natural language to ElasticSearch query generator.
You look at the query in the context of the mapping for the index.
The index's name is {index} and the mapping is:

```json
{mapping}
```

The natural language query is:

"{query}"

ONLY give me JSON. Nothing else.
Answer: """

        result = ollama.generate(model='llama2',
                                 prompt=template,
                                 stream=False,
                                 format="json")
        
        
        result = result["response"]
        # convert the string (JSON) to a dictionary
        import json
        result = json.loads(result)

        # run the query on the elasticsearch index
        print(result)
        response = es.search(index=index, body=result)
        
        return response


    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#    response = es.indices.get_mapping(index=index)
#    if index in response:
#        mapping = response[index]['mappings']
#        return mapping
#    else:
#        raise HTTPException(status_code=404, detail=f"Index {index} not found")

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
async def get_gutenberg(num_books: int = 10):
    from guten import get_gutenberg
    get_gutenberg(num_books=num_books, dir = './source/gutenberg')


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
            "multi_match": {
                "query": query
                #"fields": ["title", "text"]
            }
        }
    }
    try:
        response = es.search(index=index, body=search_query)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# add a document to an index
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




# Web interface endpoint for search
@app.get("/websearch", response_class=HTMLResponse)
async def search_form(request: Request):
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Form</title>
    </head>
    <body>
        <h1>Search Form</h1>
        <form action="/search" method="get">
            <label for="index">Index:</label>
            <input type="text" id="index" name="index" required>
            <br><br>
            <label for="keyword">Keyword:</label>
            <input type="text" id="keyword" name="keyword" required>
            <br><br>
            <button type="submit">Search</button>
        </form>
    </body>
    </html>
    """
