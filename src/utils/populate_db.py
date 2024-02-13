import logging
import os

import numpy as np
import spacy
import wikipedia
from neo4j import GraphDatabase
from scipy import spatial

nlp = spacy.load("en_core_web_sm")

# setup logs config
logger = logging.getLogger("fastapi")
logger.setLevel(logging.DEBUG)

uri = "bolt://neo4j:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))


# Step 1: Get articles from Wikipedia and store them in text files
def get_articles():
    wiki_pages = [
        "Artificial intelligence",
        "Natural language processing",
        "Deep learning",
        "Recurrent neural network",
        "The Invincible",
        "Solaris (novel)"
    ]
    for wiki_item in wiki_pages:
        page = wikipedia.page(wiki_item)
        with open(f"{wiki_item}.txt", "w", encoding="utf-8") as f:
            f.write(page.content)


def clean_string(input_sentences):
    """Cleans all characters like '=' and '?' from a string."""
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."
    cleaned_text = "".join([char for char in input_sentences if char in allowed_chars])
    return cleaned_text


# Step 2: Preprocess the text
def preprocess_text(input_text):
    input_text = clean_string(input_text)

    doc = nlp(input_text)
    tokens = [token.text for token in doc]

    return tokens


# Step 3: Vectorize the text chunks
def vectorize_text(input_text_list):
    result = []
    for input_text in input_text_list:
        doc = nlp(input_text)

        result.extend([x.vector for x in doc])

    return result


# Step 4: Store embeddings in the graph database
def store_embeddings_in_database(input_embeddings):
    with driver.session() as session:
        for i, embedding in enumerate(input_embeddings):
            session.run("""
                CREATE (n:TextChunk {id: $id, embedding: $embedding})
            """, id=i, embedding=embedding.tolist())


# Step 5: Search function
def search(text_query, top_k_results=5):
    # Embed the query
    query_embedding = nlp(text_query)
    top_results = []

    with driver.session() as session:
        # Retrieve embeddings from the database
        result = session.run("""
            MATCH (n:TextChunk)
            RETURN n.embedding AS embedding, n.id AS id
        """)

        for x in query_embedding:
            q_vector = x.vector

            scores = []
            for record in result:
                chunk_id = record["id"]
                chunk_embedding = np.array(record["embedding"])
                similarity = 1 - spatial.distance.cosine(q_vector, chunk_embedding)
                scores.append((chunk_id, similarity, chunk_embedding))

            # Sort scores in descending order
            scores.sort(key=lambda x: x[1], reverse=True)

            # Retrieve top K results
            top_results.extend(scores[:top_k_results])

        return top_results


def store_chunks_in_db():
    # Step 2: Preprocess the text
    path_to_txt = "./src/utils/"
    articles = os.listdir(path_to_txt)
    for article in articles:
        if article.endswith(".txt"):
            with open(f"{path_to_txt}{article}", "r", encoding="utf-8") as file:
                logger.debug('Started to read files', extra={"article": article})
                text = file.read()
                preprocessed_text = preprocess_text(text)
                chunks = [chunk for chunk in preprocessed_text if chunk.strip()]

                logger.debug('Tokenized sentences, start embedding', extra={"article": article})
                # Step 3: Vectorize the text chunks
                embeddings = vectorize_text(chunks)
                # Step 4: Store embeddings in the graph database
                store_embeddings_in_database(embeddings)
