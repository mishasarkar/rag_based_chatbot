# Step 1 - Import all the neccessary libraries
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


# load the .env file to read the OPENAI_API_KEY
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Step 2 - Load the web page using the WebBaseLoader, 
# and create a list of documents as source information
# define a loaders array to store all the loaders
loaders = []
loader1 = WebBaseLoader("https://www.cnn.com")
#documents1 = loader1.load()
# append loader1 to the loaders array
loaders.append(loader1)

loader2 = WebBaseLoader("https://www.bbc.com/news")
#documents2 = loader2.load()
# append loader2 to the loaders array
loaders.append(loader2)

# Step 3 - Load the documents, create a for loop to load all the documents
all_documents = []
for loader in loaders:
    all_documents += loader.load()

# Step 4 - Split the documents into smaller chunks with some chunking strategy
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(all_documents)

# Optional step
#print the first 1000 characters of the texts
# print(texts[0].page_content[:1000])

# Step 5 - Create the embeddings and vector store, with text-embedding-3-large model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(texts, embeddings)

print(vector_store.index.ntotal)

# Step 6 - Create a conversational retrieval chain with gpt-40 model, 
# this can be changed to any other model like gpt-4
llm = ChatOpenAI(model="gpt-4o", temperature=0)
chatbot = ConversationalRetrievalChain.from_llm(
    llm,
    vector_store.as_retriever(),
    return_source_documents=True
)
#create a chat loop
print("Chatbot is ready. Type 'exit' to quit.")
chat_history = []

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = chatbot.invoke({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    #print the answer from the chat after few new lines
    print("\n" * 1)
    print("Chatbot:", result['answer'])
    print("\n" * 2)