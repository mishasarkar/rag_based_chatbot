

# Functional blocks
## load the .env file
## Load and process the pdf
## split the text into smaller chunks
## create embeddings and vector store
## create the chatbot
## create a chat loop

## Setting up the environment
##### >conda create -n chatbot_env python=3.9
##### >conda activate chatbot_env
##### >conda install -c conda-forge faiss-cpu
##### >conda install -c conda-forge langchain
##### >pip install langchain-openai
##### >pip install langchain-community
##### >pip install pypdf
##### >pip install python-dotenv

## Usage
#### >python chatbot.py

## Results
> I have uploaded the source document star_signs.pdf and also my result, you can open the source document and 
> validate the  user results.


## Setting up module for WebBaseLoader
#### >conda install bs4

## Usage
#### >python chatbot_web.py

## Setting up module for YouTube
#### > pip install yt_dlp
#### > pip install pydub
#### > pip install librosa
#### > conda install -c conda-forge ffmpeg
## Usage
#### >python chatbot_youtube.py

## Reference 
[Community Document Loaders](https://python.langchain.com/v0.2/docs/integrations/document_loaders/)
[Embedding Models](https://python.langchain.com/v0.2/docs/integrations/text_embedding/)

## Completed Loaders
##### Pdf
##### WeBaseLoader
##### YouTube

## Next Steps
> Experiment with other embedding Models and chunking strtategies


