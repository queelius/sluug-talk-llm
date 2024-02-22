import requests
import os
import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch

# Function to download a book from Project Gutenberg
def download_book(rank, folder):
    url = f'https://www.gutenberg.org/cache/epub/{rank}/pg{rank}.txt.utf8'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(folder, f'{rank}.txt'), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded book with rank {rank}")
            fetch_and_save_metadata(rank, folder)  # Fetch and save metadata after downloading
        else:
            print(f"Failed to download book with rank {rank}")
    except Exception as e:
        print(f"Error downloading book with rank {rank}: {e}")

# Function to fetch and save metadata
def fetch_and_save_metadata(rank, folder):
    url = f"https://www.gutenberg.org/cache/epub/{rank}/pg{rank}.rdf"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            rdf_content = response.text
            root = ET.fromstring(rdf_content)
            title = root.find(".//{http://purl.org/dc/terms/}title").text
            author = root.find(".//{http://purl.org/dc/terms/}creator/{http://www.gutenberg.org/2009/pgterms/}agent/{http://www.gutenberg.org/2009/pgterms/}name").text
            pub_date = root.find(".//{http://purl.org/dc/terms/}issued").text

            # Save metadata to file
            with open(os.path.join(folder, f'{rank}.meta'), 'w') as meta_file:
                meta_file.write(f"Title: {title}\n")
                meta_file.write(f"Author: {author}\n")
                meta_file.write(f"Publication Date: {pub_date}\n")
            print(f"Metadata for rank {rank} saved successfully.")

            # Index metadata to Elasticsearch
            index_to_elasticsearch(rank, title, author, pub_date)
        else:
            print(f"Failed to fetch metadata for rank {rank}")
    except Exception as e:
        print(f"Error fetching metadata for rank {rank}: {e}")

# Function to index metadata to Elasticsearch
def index_to_elasticsearch(rank, title, author, pub_date):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])  # Replace with your Elasticsearch host and port
    doc = {
        'title': title,
        'author': author,
        'publication_date': pub_date,
        'file_path': f'./source/gutenberg/{rank}.txt',  # Path to the book file
        'content': open(f'./source/gutenberg/{rank}.txt', 'r').read()  # Read the content of the book from file
    }
    res = es.index(index='gutenberg', id=rank, body=doc)
    print(f"Metadata indexed to Elasticsearch for rank {rank}")

def get_gutenberg(num_books = 10, dir = './source/gutenberg'):
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Download top 100 books
    for rank in range(1, num_books + 1):
        download_book(rank, dir)


