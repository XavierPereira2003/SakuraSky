"""Chat with retrieval and embeddings."""

import os

from langchain.chains import (
    ConversationalRetrievalChain,
    FlareChain,
    OpenAIModerationChain,
    SequentialChain,
)
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.memory import ConversationBufferMemory


from dotenv import load_dotenv
load_dotenv()

path=os.path.dirname(os.getcwd())
path=os.path.join(path,'Datasets/Test.xlsx')


loader=UnstructuredExcelLoader(path,mode="elements")

docs=loader.load() 
os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_KEY")


LLM = ChatOpenAI(
    model='gpt-3.5-turbo-1106',temperature=0.3, streaming=True
)

def init_memory():
    """Initialize the memory for contextual conversation.

    We are caching this, so it won't be deleted
     every time, we restart the server.
     """
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )


MEMORY = init_memory()

def configure_retriever() -> BaseRetriever:
    """Retriever to use."""
    # Split each document documents:
    loader=UnstructuredExcelLoader(path,mode="elements")
    docs=loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    # Create embeddings and store in vectordb:
    embeddings = OpenAIEmbeddings()
    # alternatively: HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create vectordb with single call to embedding model for texts:
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        },
    )
    return retriever

   


def configure_chain(retriever: BaseRetriever, use_flare: bool = True) -> Chain:
    """Configure chain with a retriever.

    Passing in a max_tokens_limit amount automatically
    truncates the tokens when prompting your llm!
    """
    output_key = 'response' if use_flare else 'answer'
    MEMORY.output_key = output_key
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000,
    )
    return ConversationalRetrievalChain.from_llm(
        **params
    )


def configure_retrieval_chain(
        use_flare: bool = False,
        use_moderation: bool = False
) -> Chain:
    """Read documents, configure retriever, and the chain."""

    retriever = configure_retriever()
    chain = configure_chain(retriever=retriever, use_flare=use_flare)
    if not use_moderation:
        return chain

    input_variables = ["user_input"] if use_flare else ["chat_history", "question"]
    moderation_input = "response" if use_flare else "answer"
    moderation_chain = OpenAIModerationChain(input_key=moderation_input)
    return SequentialChain(
        chains=[chain, moderation_chain],
        input_variables=input_variables
    )