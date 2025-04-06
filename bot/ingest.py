import logging

from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Redis, Chroma, PGVector
from langchain_postgres import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

import os,argparse
import configparser



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Ingester:
    def __init__(self, config: configparser.ConfigParser, embedding_model: str = "mxbai-embed-large"): #llm_model: str = "deepseek-r1:latest",
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.vector_store = None
        self.retriever = None
        self.config = config
        #self.vector_store = PGVector(
        #  #  documents=chunks,
        #    embeddings=self.embeddings,
        #    connection=self.config['PostgreSQL']['CONNECTION_STRING'],
        #    collection_name=self.config['PostgreSQL']['INDEX_NAME'],
        #    use_jsonb=True,
        #)

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        #for doc in docs:
        #    print(doc.metadata)
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = PGVector.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            connection=self.config['PostgreSQL']['CONNECTION_STRING'],
            collection_name=self.config['PostgreSQL']['INDEX_NAME'],
            use_jsonb=True,
        )
        
        #self.vector_store.add_documents(docs)


        #self.vector_store = Redis.from_documents(
        #    documents=chunks,
        #    embedding=self.embeddings,
        #    redis_url=self.config['REDIS']['HOST'],
        #    index_name=self.config['REDIS']['INDEX_NAME'],
        #)


        logger.info("Ingestion completed. Document embeddings stored successfully.")



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--folder',
        type=str,
        default='documents',
        help='Specify the folder name (default: "documents")'
    )  
    args = parser.parse_args()
    folder = args.folder

    config = configparser.ConfigParser()
    config.read('config.ini')

    ingester = Ingester(config)

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder, file)
            ingester.ingest(file_path)

if __name__ == "__main__":
    main()