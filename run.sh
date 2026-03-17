uv run python scripts/my_agent.py \
    dataset_src=huggingface \
    level=1 problem_id=1 \
    eval_mode=modal gpu=L4 \
    verbose=True log=True log_prompt=True log_generated_kernel=True log_eval_result=True \
    server_type=local model_name=openai/gpt-oss-120b is_reasoning_model=False \
    api_base=https://api.tokenfactory.nebius.com/v1 \
    api_key= \
    max_tokens=130000 temperature=0.1