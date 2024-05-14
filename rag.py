from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
MODEL_NAME=os.getenv("MODEL_NAME")
PROJECT_NAME=os.getenv("PROJECT_NAME")

from langchain_google_genai import ChatGoogleGenerativeAI



# If there is no environment variable set for the API key, you can pass the API
# key to the parameter `google_api_key` of the `ChatGoogleGenerativeAI` function:
# `google_api_key="key"`.
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY, project=PROJECT_NAME,
                 temperature=0.7, top_p=0.85)

def translate(arabic_text:str)->str:
    """
    This function takes an Arabic text and translates it to English.

    Args:
        arabic_text (str): The Arabic text to be translated.

    Returns:
        str: The English translation of the Arabic text.

    Example:
        >>> translate("ما هو اسمك؟")
        "What is your name?"
    """
    prompt = f"translate Arabic to English: {arabic_text}" 
    return llm.invoke(prompt).content

def get_pages_contents_from_pdf(resume_pdf_path:str,pages:int=3)->str:
    """
    Extracts the content of all pages from a PDF file.

    Args:
        resume_pdf_path (str): The path to the PDF file.

    Returns:
        str: The concatenated content of all pages in the PDF.

    Example:
        >>> get_pages_contents_from_pdf("./resume.pdf")
        'Page 1 content\nPage 2 content\n...'
    """
    loader = PyPDFLoader(resume_pdf_path)
    documents=loader.load()
    page_contents:str=""
    for idx,document in enumerate(documents,start=1):
        page_contents+=dict(document)["page_content"]+"\n"
        if idx==pages:
            break
    return page_contents

import re

def remove_empty_lines_and_spaces(text):
    """
    Remove empty lines and spaces from a string.

    Parameters:
        text (str): The input string.

    Returns:
        str: The string with empty lines and spaces removed.
    """
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove leading and trailing spaces from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines after removing leading and trailing spaces
    lines = [line for line in lines if line]
    
    # Join the lines back together
    cleaned_text = '\n'.join(lines)
    
    return cleaned_text



# Path to your PDF file
pdf_file_path = "documents/legislation-6294abaff20f8.pdf"

# Initialize PdfFileLoader with the path to your PDF file

# loader = PyPDFLoader(pdf_file_path)
# docs = loader.load()


final_text = get_pages_contents_from_pdf(pdf_file_path,8)
final_text = remove_empty_lines_and_spaces(final_text)

# Convert the text to LangChain's `Document` format
docs =  [Document(page_content=final_text, metadata={"source": "local"})]

# print("Finale text: ", final_text)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# If there is no environment variable set for the API key, you can pass the API
# key to the parameter `google_api_key` of the `GoogleGenerativeAIEmbeddings`
# function: `google_api_key = "key"`.

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY, project=PROJECT_NAME)


# Save to disk
vectorstore = Chroma.from_documents(
                     documents=docs,                 # Data
                     embedding=gemini_embeddings,    # Embedding model
                     persist_directory="./chroma_db" # Directory to save data
                     )
     


# Load from disk
vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",       # Directory of db
                        embedding_function=gemini_embeddings   # Embedding model
                   )
# Get the Retriever interface for the store to use later.
# When an unstructured query is given to a retriever it will return documents.
# Read more about retrievers in the following link.
# https://python.langchain.com/docs/modules/data_connection/retrievers/
#
# Since only 1 document is stored in the Chroma vector store, search_kwargs `k`
# is set to 1 to decrease the `k` value of chroma's similarity search from 4 to
# 1. If you don't pass this value, you will get a warning.
retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})





# Prompt template to query Gemini
llm_prompt_template = """أنت مساعد لمهام الإجابة عن الأسئلة.
استخدم السياق التالي للإجابة على السؤال.
إذا لم تكن تعرف الإجابة، فقط قل أنك لا تعرف.
استخدم خمس جمل كحد أقصى وحافظ على الإجابة موجزة.\n
السؤال: {question} \nالسياق: {context} \nالإجابة:"""


llm_prompt = PromptTemplate.from_template(llm_prompt_template)




# Combine data from documents to readable string format.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | llm_prompt
#     | llm
#     | StrOutputParser()
# )

# print( final_text )
# Format the prompt with the question and context
formatted_prompt = llm_prompt.format(question="ماهي بيان األسباب؟", context=final_text)

result=llm.invoke(formatted_prompt).content
print(result[::-1])





