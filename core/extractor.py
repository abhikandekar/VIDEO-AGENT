#Actionableitems , decision , questions 

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
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

def build_chain(system_prompt : str) -> str:
    llm = get_llm()
    
    return( RunnablePassthrough() 
           | RunnableLambda(lambda x: {"text" : x})
           | ChatPromptTemplate.from_messages(
             [("system", system_prompt),
              ("human", "{text}")])
           | llm
           | StrOutputParser()
    )
    
def extract_action_items(transcript : str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    
    return(chain.invoke(transcript))

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    
    return(chain.invoke(transcript))

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    
    return(chain.invoke(transcript))
