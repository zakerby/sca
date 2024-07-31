from langchain_community.chat_models import ChatOllama

from langchain_community.document_loaders import GitLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

class RAG:
    vector_store = None
    retriever = None
    chain = None
    
    def __init__(self, model_name: str) -> None:
        self.model = ChatOllama(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] 
                You are an expert programmer and teacher of a code repository.
                You will be asked to explain the code for a specific task in the repo.
                You will be provided with some related code snippets or documents related to the question.
                Please think the explanation step-by-step.
                Please answer the questions based on your knowledge, and you can also refer to the provided related code snippets.
                The README.md file and the repo structure are also available for your reference.
                If you need any details clarified, please ask questions until all issues are clarified.
            [/INST] </s> 
            [INST] 
                Question: {question} 
                Context: {context} 
                Answer: 
            [/INST]
            """
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please, ingest a repository first"

        return self.chain.invoke(query)
    
    def ingest(self, repo_url: str):
        print(f"Ingesting data from {repo_url}")
        files = GitLoader(clone_url=repo_url, repo_path='new_repo').load()
        print(f"Loaded {len(files)} files")
        chunks = self.text_splitter.split_documents(files)
        chunks = filter_complex_metadata(chunks)
        print(f"Split into {len(chunks)} chunks")
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        print(f"Indexed {len(self.vector_store)} chunks")
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None