import logging
from app.graph.workflow import build_graph

logging.basicConfig(level=logging.INFO)

app_graph = build_graph()

def test_question(q: str):
    print(f"\n==============================================")
    print(f"Q: {q}")
    print(f"==============================================\n")
    # Using proper initial state dict
    res = app_graph.invoke({
        "query": q,
        "expanded_queries": [],
        "retrieved_ids": [],
        "ranked_chunks": [],
        "expanded_chunks": [],
        "groq_answer": "",
        "steps": [],
        "citations": [],
        "final_response": {}
    })
    
    final = res.get("final_response", {})
    print("ANSWER:")
    print(final.get("answer", ""))
    print("\nSTEPS:")
    for step in final.get("steps", []):
        imgs = " | ".join([img['citation'] for img in step.get("images", [])])
        print(f"[{step.get('procedure', '')}] Step {step.get('step', '?')}: {step.get('instruction', '')}")
        if imgs:
            print(f"    --> Images: {imgs}")
    
    print("\nCITATIONS:")
    for c in final.get("citations", []):
        print(f"- Page {c['page']}, Figure: {c['figure']}")

if __name__ == "__main__":
    test_question("How do I run a leak test on a blister pack, step by step?")
