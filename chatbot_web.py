import os
from dotenv import load_dotenv
# import a webpage module
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# load the .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# load a sample web page using the web loader
#loader = WebBaseLoader("https://www.bbc.com/news")
loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/benefits-and-perks.md")

docs = loader.load()
#print(docs[0].page_content[:1000])
#print(docs[0])


#split the text into smaller chunks
splitter = RecursiveCharacterTextSplitter()
splits = splitter.split_documents(docs)

# create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(splits, embeddings)

print(vector_store.index.ntotal)

# create the chatbot
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
