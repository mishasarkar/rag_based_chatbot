import getpass
import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter the OpenAI API key: ")

os.environ["NEO4J_URI"] = getpass.getpass("Enter the Neo4j URI: ")
os.environ["NEO4J_USERNAME"] = getpass.getpass("Enter the Neo4j username: ")
os.environ["NEO4J_PASSWORD"] = getpass.getpass("Enter the Neo4j password: ")


#The below example will create a connection with a Neo4j database 
# and will populate it with example data about movies and their actors.

# instantiate the graph
graph = Neo4jGraph()

# Import movie information from the csv file and load it in the neo4j graph database
# The query below loads the movie data from a csv file and creates nodes for movies, persons, and genres.
# It also creates relationships between the nodes based on the data in the csv file.
# The query uses the LOAD CSV command to load the data from the csv file.
# The query uses the MERGE command to create nodes and relationships.
# The query uses the FOREACH command to create multiple nodes and relationships.
movies_query = """
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
   m.title = row.title,
   m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') |
   MERGE (p:Person {name:trim(director)})
   MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') |
   MERGE (p:Person {name:trim(actor)})
   MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') |
   MERGE (g:Genre {name:trim(genre)})
   MERGE (m)-[:IN_GENRE]->(g))
"""

# execute the query
graph.query(movies_query)
# refresh the schema, It is important to refresh the schema after loading data
# to make sure that the graph is aware of the new nodes and relationships
graph.refresh_schema()
# print the schema to see the new nodes and relationships
print(graph.schema)

# Now that we have the data in the graph, we can use it to answer questions.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# langchain comes with a chain that connects the graph with the language model
# instantiate the chain with the graph and the language model
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
response = chain.invoke({"query": "What was the cast of the Toy Story?"})
# print the response
response

# validating relationship direction
# The chain can also validate the relationship direction by setting the validate_cypher parameter to True.
chain = GraphCypherQAChain.from_llm(
   graph=graph, llm=llm, verbose=True, validate_cypher=True
)
# we can validate and optionally correct the relationship directions using the validate_cypher method
response = chain.invoke({"query": "What was the cast of the Toy Story?"})
response


