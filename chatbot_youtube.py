import os
from dotenv import load_dotenv
from langchain_community.document_loaders.blob_loaders.youtube_audio import (YoutubeAudioLoader) 
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import (OpenAIWhisperParser,)

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



# load the .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# set a flag to switch between local and remote mode
local = True

# check if the audio files are already downloaded
if local:
    print("Using local audio files")
    # fetch the audio files from the local directory
    local_dir = "~/Downloads/"
    # create a file system blob loader to load the m4a files
    blob_loader = FileSystemBlobLoader(local_dir,glob="**/*.m4a", show_progress=True)
    # parse the audio files to text
    parser = OpenAIWhisperParser()
    loader = GenericLoader(blob_loader, parser)
    docs = loader.load()
else:
    print("Downloading audio files")
    # Two Karpathy lecture videos
    urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]
    # Directory to save the audio files
    save_dir = "~/Downloads"
    # Transcribe the videos to text
    loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
    docs = loader.load()


# # Two Karpathy lecture videos
# urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

# # Directory to save the audio files
# save_dir = "~/Downloads"

 

# # Transcribe the videos to text
# loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
# docs = loader.load()



# combine the documents
combined_doc = [doc.page_content for doc in docs]
text = " ".join(combined_doc)

# split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)

# Build an index
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(splits, embeddings)

# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

# Ask a question
query = "Why do we need to zero out the gradient before backdrop at each step?"
qa_chain.run(query)