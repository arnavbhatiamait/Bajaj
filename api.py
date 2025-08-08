import os
from typing import List,Optional
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# import weaviate
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# import pyodbc
from dotenv import load_dotenv 
load_dotenv()
def select_vector_store(db_select: str, docs, embedding):
    db_select=db_select.strip()
    db_select=db_select.lower()
    # if db_select == 'faiss':
    vectordb = FAISS.from_documents(documents=docs, embedding=embedding)
    # elif db_select == 'weaviate':
    #     client = weaviate.connect_to_local()
    #     vectordb = Weaviate.from_documents(documents=docs, embedding=embedding, client=client)
    # else:
    #     vectordb = Chroma.from_documents(documents=docs, embedding=embedding)
    return vectordb


def llm(model: str, model_option: str, api_key: str = None):
    # if model == "Open AI":
    #     llm = ChatOpenAI(model=model_option, api_key=api_key, verbose=True, temperature=0.1)
    #     embeddings = OpenAIEmbeddings(model=model_option, api_key=api_key)
    # elif model == "Ollama":
    #     llm = ChatOllama(model=model_option, verbose=True, temperature=0.1)
    #     embeddings = OllamaEmbeddings(model=model_option)
    # elif model == "Groq":
    #     llm = ChatGroq(model=model_option, api_key=api_key, verbose=True, temperature=0.1)
    #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # elif model == "Gemini":
    llm = GoogleGenerativeAI(api_key=api_key, verbose=True, temperature=0.1, model=model_option)
    # embeddings = GoogleGenerativeAIEmbeddings(model=model_option)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # else:
    #     raise ValueError(f"Unsupported model: {model}")
    return llm, embeddings

app = FastAPI()

condtion=""
with open('policyConditions.txt', 'r', encoding='utf-8') as f:
    condition = f.read()
# print(condition)



@app.post("/main")
async def main_app(
    
    files: List[UploadFile] = File(...),
    
    user_prompt: List[str] = Form(...),
    api_key: str=Form(...)
):
    # llms, embeddings = llm(model="Groq", model_option="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    # llms,embeddings=llm(model="Ollama",model_option="llama3.2",api_key="")
    
    llms,embeddings=llm(model="Gemini",model_option="gemini-2.0-flash",api_key=api_key)


    system_prompt = """You are a helpful assistant that is responsible to provide a response 
        using the provided documents. Explain in detail and provide response 
        in breif 1-2 lines text format. Provide references from the document 
        and abstain from answering if no context is provided. Be respectful and helpful to the user.
        The Previous context is: {context} 
        User prompt is: {input}  the policy conditions are: """
    system_prompt += condition
    # print(system_prompt)

    # !cratin foldr with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"folder_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    for file in files:
        file_path = os.path.join(folder_name, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

    prompt_template = ChatPromptTemplate.from_template(system_prompt)

    # llms, embeddings = llm(model=model, model_option=model_name, api_key=api_key)

    loader = PyPDFDirectoryLoader(path=folder_name)
    documents = loader.load()

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found in uploaded files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    if not docs:
        raise HTTPException(status_code=400, detail="Failed to split documents into chunks.")

    vectorstore = select_vector_store(db_select="FAISS", docs=docs, embedding=embeddings)

    output_parser = StrOutputParser()

    document_chain = create_stuff_documents_chain(
        llm=llms,
        prompt=prompt_template,
        output_parser=output_parser
    )

    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    responses=[]
    for prompt in user_prompt:
        response = retrieval_chain.invoke({"input": prompt})
        # print(response)

        response_ans = response.get('answer', '')
        responses.append(response_ans)
        # ! similarity search
        # context_docs = [doc.page_content for doc in response.get('context', [])]

    return {"output": responses}
