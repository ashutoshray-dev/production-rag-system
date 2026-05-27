import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import _faithfulness, _answer_relevance, _context_precision
import sys
from pathlib import Path
# root_dir = Path(os.getcwd()).parent
# sys.path.append(str(root_dir))
# current_file_path = Path(__file__).resolve()
# root_dir = current_file_path.parent.parent
# if str(root_dir) not in sys.path:
#     sys.path.append(root_dir)

# print(root_dir)
from backend.graph import system


def run_rag_on_testset(test_set_path:str):
    """Feeds test questions to Langgraph and captures its full runtime state."""
    with open(test_set_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)

    queries = []
    generated_answers = []
    retrieved_contexts = []
    ground_truths = []

    print(f"Running evaluation test on {len(golden_data)} benchmarked items...")
    for item in golden_data:
        user_query = item['user_input']
        initial_state = {"user_input": user_query}
        config = {'configurable': {"thread_id":"eval_session_prod"}}
        final_state = system.invoke(initial_state, config=config)
        queries.append(user_query)
        generated_answers.append(final_state.get("generation_output", ""))
        ground_truths.append(item['ground_truth'])
        raw_docs = [doc.page_content for doc in final_state.get("retrieved_docs", [])]
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
    
    runtime_data = run_rag_on_testset(eval_file)
    eval_dataset = Dataset.from_dict(runtime_data)
    print("Running LLM-as-Judge evalaution matrix via Ragas: ")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            _faithfulness, _answer_relevance, _context_precision
        ]
    )
    print("Evaluation summary------------------------------------------------")
    print(result)
    print("---------------------------------------------------------------------")

    with open("tests/evaluation/latest_eval_report.json", 'w') as out:
        json.dump(dict(result), out, indent=2)
        print("Latest Eval report is available at tests/evaluation/latest_eval_report.json")
    

if __name__=="__main__":
    main()

