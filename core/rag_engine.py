import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from core.vector_store import build_vector_store, load_vector_store, get_retriever

def get_llm():
# 1. Load the local model

    model_id = os.getenv("chat_model_id")

    generator = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.1,
        return_full_text=False,
        device=-1,)

    llm = HuggingFacePipeline(pipeline=generator)
    
    return llm

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript: str):
    
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
                """You are an expert meeting assistant. Answer the user's question 
                based ONLY on the meeting transcript context provided below.

                If the answer is not found in the context, say: 
                "I could not find this information in the meeting transcript."

                Always be concise and precise. If quoting someone, mention it clearly.

                Context from meeting transcript: {context}""",
            ),
            ("human", "{question}"),
        ]
    )
    
    # full lcel pipeline
    
    rag_chain = (
        { "context": retriever | RunnableLambda(format_docs), 
          "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever()
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
                """You are an expert meeting assistant. Answer the user's question 
                based ONLY on the meeting transcript context provided below.

                If the answer is not found in the context, say: 
                "I could not find this information in the meeting transcript."

                Always be concise and precise. If quoting someone, mention it clearly.

                Context from meeting transcript: {context}""",
            ),
            ("human", "{question}"),
        ]
    )
    
    # full lcel pipeline
    
    rag_chain = (
        { "context": retriever | RunnableLambda(format_docs), 
          "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
