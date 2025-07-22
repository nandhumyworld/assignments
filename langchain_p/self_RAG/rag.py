from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def self_rag(query, retriver):
    qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriver,
        return_source_documents=False,
        chain_type="stuff",)  # could be 'stuff', 'map_reduce', or 'refine
    return qa_chain.run(query)

class corrective_rag:
    """
    Corrective RAG implementation using RetrievalQA
    """
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query, st):
        """
        Run the corrective RAG process
        """
        context_evaluation = self.context_evaluvation(st)
    
    def context_evaluvation(st):
        prompt = PromptTemplate(template="""EVALUATE_CONTEXT: 
        Rate the following retrieved context for the given query: 
        Query: {user_query} 
        Retrieved Context: {retrieved_context} 
        Evaluation Criteria: 
            1. Relevance Score (0-1): How well does the context address the query? 
            2. Completeness Score (0-1): Does the context provide sufficient information? 
            3. Accuracy Score (0-1): Is the information factually correct? 
            4. Specificity Score (0-1): Is the context specific enough for the query? 
        Overall Quality: [EXCELLENT/GOOD/FAIR/POOR]
        - Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
        input_variables=["user_query", "retrieved_context"])

        llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.session_state.anthropic_api_key,
                       temperature=0, max_tokens=1000)
        chain = (
            prompt 
            | llm 
            | StrOutputParser()
        )
        