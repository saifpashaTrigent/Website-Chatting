from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import  CharacterTextSplitter
from langchain_community.vectorstores import  FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# model_name="gpt-3.5-turbo", temperature=0
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 0)
    document_chunks = text_splitter.split_documents(document)
    vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    retriever = vector_store.as_retriever(return_source_documents = True)
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                """Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Keep the information precise, concise and informative as you can.
                Remember your sole purpose is to generate the information for the user questions from the
                given conversation and nothing else. Do not generate unecessary answers 
                or try do not try make answers from your own. 
                Include the source of the information at the end. """,
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
