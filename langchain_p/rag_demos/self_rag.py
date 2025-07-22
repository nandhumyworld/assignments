from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


def self_rag(query, retriver):
    qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriver,
        return_source_documents=False,
        chain_type="stuff",)  # could be 'stuff', 'map_reduce', or 'refine
    return qa_chain.run(query)