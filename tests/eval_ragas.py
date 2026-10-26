import json
import os
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    # ContextPrecision,
    # AnswerRelevancy,
)
import sys
from backend.graph import system
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import OpenAI
from langchain_ollama import ChatOllama
from google import genai


# client3 = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# model = ChatOllama(model='qwen2.5:3b', reasoning=True)
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
client2 = OpenAI(
    # model="nvidia/nemotron-3-super-120b-a12b:free",
    api_key=os.environ.get("FREE_LLM_API_KEY"),
    base_url="http://127.0.0.1:31415/v1",
    # temperature=0,
    max_retries=5,
    timeout=90,
    # model_kwargs={
    #     "parallel_tool_calls": False,
    # },
)
# eval_llm = llm_factory(model='qwen2.5:3b', provider='ollama', client=model)
# eval_llm = llm_factory(model='gemini-2.5-flash-lite', provider='google', client=client2)
# eval_llm = llm_factory(model='openrouter/auto', provider='openai', client=client)
eval_llm = llm_factory(model='auto', provider='openai', client=client2, max_tokens=4096)
eval_embeddings = embedding_factory('huggingface', model='BAAI/bge-small-en-v1.5')
def run_rag_on_testset(test_set_path:str):
    """Feeds test questions to Langgraph and captures its full runtime state."""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        golden_data = (json.load(f))[:5]

    queries = []
    generated_answers = []
    retrieved_contexts = []
    ground_truths = []

    print(f"Running evaluation test on {len(golden_data)} benchmarked items...")
    for item in golden_data:
        user_query = item['user_input']
        initial_state = {"messages": [user_query]}
        config = {'configurable': {"thread_id":"evals_session_prod"}}
        final_state = system.invoke(initial_state, config=config)
        queries.append(user_query)
        generated_answers.append(final_state.get("messages", "")[-1].content) 
        ground_truths.append(item['ground_truth'])
        raw_docs = [doc.page_content for doc in final_state.get("docs", [])]  
        retrieved_contexts.append(raw_docs)
    return{
        'question':queries,
        'answer':generated_answers,
        'contexts':retrieved_contexts,
        'ground_truth':ground_truths
    }
    
def main():
    eval_file = "tests/golden_set.json"
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation set missing at {eval_file}")
        return
    metrics = [
        Faithfulness(),
        # ContextPrecision(),
        # AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings) # Added embeddings here
    ]
        
    runtime_data = run_rag_on_testset(eval_file)
    eval_dataset = Dataset.from_dict(runtime_data)
    run_config = RunConfig(max_workers=3, timeout=60)
    print("Running LLM-as-Judge evalaution matrix via Ragas: ")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics, 
        llm=eval_llm,
        # embeddings=eval_embeddings,
        run_config=run_config
    )
    print("Evaluation summary------------------------------------------------")
    print(result)
    print("---------------------------------------------------------------------")

    try:
        scores_dict = result.to_pandas().mean(numeric_only=True).to_dict()
    except Exception:
        scores_dict = {k:v for k,v in result.items() if v is not None}
    
    os.makedirs('tests/evaluation', exist_ok=True)
    with open("tests/evaluation/latest_eval_report.json", 'w') as out:
        json.dump(scores_dict, out, indent=2)
        print("Latest Eval report is available at tests/evaluation/latest_eval_report.json")
    
    # scores = dict(result)
    current_faithfulness = scores_dict.get('faithfulness', 0.0)
    # current_precision = scores_dict.get('answer_relevancy', 0.0)
    import math
    if math.isnan(current_faithfulness):
        current_faithfulness=0.0
    # if math.isnan(current_precision):
    #     current_precision=0.0
    print(f"\nCurrent system faithfulness: {current_faithfulness}")
    # print(f"\nCurrent system context precision: {current_precision}")

    THRESHOLD = 0.75
    if current_faithfulness < THRESHOLD:
        print(f"❌ REGRESSION DETECTED: Faithfulness dropped below threshold ({THRESHOLD})!")
        sys.exit(1)
    
    print("✅ Quality Gate Passed! System is stable.")
    sys.exit(0)
    

if __name__=="__main__":
    main()
