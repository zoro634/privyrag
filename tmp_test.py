import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline.orchestrator import answer_query, preload_caches

if __name__ == "__main__":
    try:
        preload_caches()
        print("Caches loaded.", flush=True)
        res = answer_query("What is the test document about?", [])
        print("SUCCESS:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()
