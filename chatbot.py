import openai
import fitz  # PyMuPDF
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up your OpenAI API key
openai.api_key = 'xx-xxx'

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Sentence-based chunking method
def sentence_based_chunking(text):
    """
    Splits text into sentences using regular expressions.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

# Fixed-size chunking method
def fixed_size_chunking(text, chunk_size=512):
    """
    Splits text into fixed-size chunks based on a specified chunk size.
    """
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Basic retrieval method using TF-IDF
def retrieve_relevant_chunks(chunks, query, top_k=3):
    """
    Retrieves the most relevant chunks based on the user's query using TF-IDF.
    """
    vectorizer = TfidfVectorizer().fit_transform(chunks)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectorizer).flatten()
    relevant_indices = similarity.argsort()[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in relevant_indices]
    return relevant_chunks

# Define a prompt template that includes the context
prompt_template = PromptTemplate(template="Context: {context}\n\nThe user says: {input}\nAI assistant replies:", input_variables=["context", "input"])

# Define the response function
def get_openai_response(prompt):
    """
    This function sends a prompt to the OpenAI API and retrieves the response.
    """
    response = openai.Completion.create(
        engine="gpt-4o",  # The model to use
        prompt=prompt,              # The prompt containing the user input
        max_tokens=150,             # The maximum number of tokens to generate
        n=1,                        # Number of responses to generate
        stop=None,                  # Stop sequence for the model
        temperature=0.7,            # Sampling temperature for response generation
    )
    return response.choices[0].text.strip()  # Return the generated response

# Define a simple chain
class OpenAIChatbotChain(ConversationChain):
    def __init__(self, prompt_template, chunks):
        super().__init__(prompt_template=prompt_template)
        self.chunks = chunks  # Store the context chunks (PDF text) in the chain

    # def _call(self, inputs):
    #     """
    #     This method is called to process the inputs and generate the response using the OpenAI API.
    #     """
    #     user_query = inputs['input']
    #     relevant_chunks = retrieve_relevant_chunks(self.chunks, user_query)  # Retrieve relevant chunks
    #     context = ' '.join(relevant_chunks)  # Combine the relevant chunks as context
    #     inputs['context'] = context  # Add the context to the inputs
    #     prompt = self.prompt_template.format(**inputs)  # Format the input using the prompt template
    #     response = get_openai_response(prompt)          # Get the response from OpenAI API
    #     return {"output": response}                     # Return the response

# Function to chat with the bot
def chat_with_bot(pdf_path):
    """
    This function manages the user interaction with the chatbot.
    It extracts text from the given PDF and uses it as context for answering questions.
    """
    context = extract_text_from_pdf(pdf_path)  # Extract text from the PDF
    chunks = fixed_size_chunking(context)  # Chunk the extracted text
    chatbot_chain = OpenAIChatbotChain(prompt_template=prompt_template, chunks=chunks)  # Initialize the chain with chunks
    # print the context 1000 characters
    print(context[:1000])
    # print the first chunk
    print(chunks[0])
    # print the last chunk
    print(chunks[-1])
    # print the chatbot_chain
    print(chatbot_chain)

    # print("You can start chatting with the bot (type 'exit' to stop):")
    # while True:
    #     user_input = input("You: ")  # Get user input
    #     if user_input.lower() == 'exit':  # Exit the chat loop if the user types 'exit'
    #         print("Goodbye!")
    #         break
    #     response = chatbot_chain({"input": user_input})  # Get the bot's response
    #     print("Bot:", response["output"])  # Display the bot's response

# Example usage: Start the chat with a given PDF file
pdf_path = 'star_signs.pdf'
chat_with_bot(pdf_path)
