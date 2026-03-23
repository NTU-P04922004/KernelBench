# uv run python scripts/my_agent_local.py \
#     dataset_src=huggingface dataset_name=makora-ai/KernelMiniBench \
#     level=1 problem_id=1 \
#     eval_mode=local \
#     verbose=True log=True log_prompt=True log_generated_kernel=True log_eval_result=True \
#     server_type=local model_name=openai/gpt-oss-120b is_reasoning_model=False \
#     api_base=https://api.tokenfactory.nebius.com/v1 \
#     api_key= \
#     max_tokens=130000 temperature=0.1

# uv run python scripts/generate_samples.py \
#     run_name=gpt_120b_mini_level_1 \
#     dataset_src=huggingface dataset_name=makora-ai/KernelMiniBench level=1 \
#     server_type=local model_name=openai/gpt-oss-120b is_reasoning_model=False \
#     api_base=https://api.tokenfactory.nebius.com/v1 \
#     api_key= \
#     max_tokens=130000 temperature=0.7 reasoning_effort=medium \
#     num_workers=4

# uv run python scripts/eval_from_generations.py \
#     run_name=gpt_120b_mini_level_1 \
#     dataset_src=huggingface dataset_name=makora-ai/KernelMiniBench level=1 \
#     eval_mode=local num_gpu_devices=1 timeout=180 num_perf_trials=20 build_cache=True

uv run python scripts/benchmark_eval_analysis.py \
    run_name=gpt_120b_mini_level_1_20260319030055 \
    dataset_src=huggingface dataset_name=makora-ai/KernelMiniBench level=1 \
    hardware=H100_Modal \
    baseline=baseline_time_torch \
    output_file=bench_02.txt