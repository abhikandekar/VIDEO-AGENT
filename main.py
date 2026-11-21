from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

def run_pipline(source: str, Language: str = "english") -> dict:
    print("starting AI Video Assistant")
    
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=Language)
    
    title = generate_title(transcript)
    summary = summarize(transcript)
    
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    rag_chain = build_rag_chain(transcript)
    
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }
    
if __name__ == "__main__":
    # CLI entry point
    source = input("Enter the source (YouTube URL or local file path): ")
    language = input("Enter the language (english/hinglish): ").strip().lower()
    result = run_pipline(source, language)
    
    print("\n" + "=" * 60)
    print(f"📌 TITLE: {result['title']}")
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)
    
    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("Exiting chat. Goodbye!")
            break   
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        answer = ask_question(result['rag_chain'], question)
        print(f"AI: {answer}\n") 
    