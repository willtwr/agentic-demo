from uuid import uuid4

from typing import List
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.embeddings.stella import Stella


class ChromaVectorStore:
    """Chroma Vector Storage"""
    def __init__(self, embedding_function=Stella()):
        self.embedding_function = embedding_function
        self._build_docs_splitter()
        self.build_vector_store()

    def build_vector_store(self):
        self.vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=self.embedding_function,
            persist_directory=None
        )

    def _build_docs_splitter(self, chunk_size: int = 100, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _docs_splitter(self, docs: List[Document]):
        return self.text_splitter.split_documents(docs)

    def add_documents(self, docs: List[Document]):
        doc_splits = self._docs_splitter(docs)
        uuids = [str(uuid4()) for _ in range(len(doc_splits))]
        self.vectorstore.add_documents(doc_splits, ids=uuids)

    def get_retriever(self):
        return self.vectorstore.as_retriever()
