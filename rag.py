from langchain_community.chat_models import ChatOllama

from langchain_community.document_loaders import GitLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

REPOS_FOLDER_PATH = "repos"

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
        # Extract the name of the repository
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        files = GitLoader(clone_url=repo_url, repo_path=f"{REPOS_FOLDER_PATH}/{repo_name}").load()
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
    
    def get_all_embeddings(self):
        # Assuming vector_store has a method to get all embeddings
        embeddings = []
        for doc in self.vector_store.get():
            embeddings.append(doc.embedding)
        return embeddings

    
    def plot_embeddings(self):
        embeddings = self.get_all_embeddings()
        # Step 2: Dimensionality reduction using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Step 3: Create a DataFrame for Plotly
        df = pd.DataFrame(reduced_embeddings, columns=['Dimension 1', 'Dimension 2'])
        fig = px.scatter(df, x='Dimension 1', y='Dimension 2', title='t-SNE plot of embeddings')
        
        return fig

    
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None