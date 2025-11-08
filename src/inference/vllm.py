from langchain_community.llms.vllm import VLLM
import os

def get_model_path(args):
    MODEL_ID = None

    if args.model == "qwen-1.5b":
        MODEL_ID = os.getenv("QWEN_1.5B_PATH")
    elif args.model == "qwen-7b":
        MODEL_ID = os.getenv("QWEN_7B_PATH")
    elif args.model == "qwen-14b":
        MODEL_ID = os.getenv("QWEN_14B_PATH")
    elif args.model == "qwen-32b":
        MODEL_ID = os.getenv("QWEN_32B_PATH")
    elif args.model == "deepseek-r1-32b":
        MODEL_ID = os.getenv("DEEPSEEK_R1_32B_PATH")
    elif args.model == "deepseek-r1-7b":
        MODEL_ID = os.getenv("DEEPSEEK_R1_7B_PATH")
    elif args.model == "qwq":
        MODEL_ID = os.getenv("QWQ_32B_PATH")
    elif args.model == "deepseek-r1-14b":
        MODEL_ID = os.getenv("DEEPSEEK_R1_14B_PATH")
    elif args.model == "deepseek-r1-1.5b":
        MODEL_ID = os.getenv("DEEPSEEK_R1_1.5B_PATH")
    
    return MODEL_ID

def build_llm(args):
    MODEL_ID = get_model_path(args)
    
    return VLLM(model=MODEL_ID,
                trust_remote_code=True,  # mandatory for hf models
                tensor_parallel_size=args.num_gpus,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                vllm_kwargs={"disable_custom_all_reduce": True, "seed": args.seed, "gpu_memory_utilization": 0.90}
            )

