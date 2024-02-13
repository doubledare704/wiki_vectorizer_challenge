# wiki_vectorizer_challenge
Backend challenge task to convert it to vectorized format

# Description for local build:

* Docker containers are based on python3.10 slim and latest neo4j
* To build containers locally: `docker-compose up -d`
* To populate neo4j database with vectors:
  * Run containers
  * Go to http://localhost:8000/docs
  * Request GET /populate-db with no parameters, wait until complete (it takes seconds to read from files and save vectorized data)
  * Now you may use search endpoint with 1 parameter.