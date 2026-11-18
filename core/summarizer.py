from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


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

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2500,
        chunk_overlap = 200
    )
    
    return splitter.split_text(transcript)

def summarize(transcript : str) -> str :
    llm = get_llm()
    
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    
    map_chain = map_prompt | llm | StrOutputParser()
    
    chunks = split_transcript(transcript)
    
    chunks_summeries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]
    
    combined = "\n\n".join(chunks_summeries)
    
    combined_prompt = ChatPromptTemplate.from_messages(
        [("system", 
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points."),
         ("human", "{text}")]
    )
    
    combined_chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text" : x}) | combined_prompt | llm | StrOutputParser()
    
    return combined_chain.invoke(combined)

def  generate_title(transcript : str) -> str:
    llm = get_llm()
    
    template =  ChatPromptTemplate.from_messages([("system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else."),
                ("human", "{text}"),])
    
    title_chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | template | llm | StrOutputParser()
    
    return title_chain.invoke(transcript[:2000])
