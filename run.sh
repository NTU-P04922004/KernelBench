# uv run python scripts/my_agent_local.py \
#     dataset_src=huggingface dataset_name=makora-ai/KernelMiniBench \
#     level=1 problem_id=1 \
#     eval_mode=local \
#     verbose=True log=True log_prompt=True log_generated_kernel=True log_eval_result=True \
#     server_type=local model_name=openai/gpt-oss-120b is_reasoning_model=False \
#     api_base=https://api.tokenfactory.nebius.com/v1 \
#     api_key=v1.CmQKHHN0YXRpY2tleS1lMDBneHJxNGoyOXZ6MTFzM24SIXNlcnZpY2VhY2NvdW50LWUwMG40cmQwZzlwMHRybWU3OTIMCPWh480GEPfy46oCOgwI9KT7mAcQgPSWygFAAloDZTAw.AAAAAAAAAAGxDpjfhGytESbyN-EcEORx4kP7KiVEFiQn_Z6CD-vBz2HMLZ8UzYOzlPeyp5vi8hYjICZe2jAbjSo2BtoMU7AB \
#     max_tokens=130000 temperature=0.1

uv run python scripts/generate_samples.py \
    run_name=gpt_120b_mini_level_1 \
    dataset_src=huggingface dataset_name=makora-ai/KernelMiniBench level=1 \
    server_type=local model_name=openai/gpt-oss-120b is_reasoning_model=False \
    api_base=https://api.tokenfactory.nebius.com/v1 \
    api_key=v1.CmQKHHN0YXRpY2tleS1lMDBneHJxNGoyOXZ6MTFzM24SIXNlcnZpY2VhY2NvdW50LWUwMG40cmQwZzlwMHRybWU3OTIMCPWh480GEPfy46oCOgwI9KT7mAcQgPSWygFAAloDZTAw.AAAAAAAAAAGxDpjfhGytESbyN-EcEORx4kP7KiVEFiQn_Z6CD-vBz2HMLZ8UzYOzlPeyp5vi8hYjICZe2jAbjSo2BtoMU7AB \
    max_tokens=130000 temperature=0.1 \
    num_workers=4

# uv run python scripts/eval_from_generations.py \
#     run_name=gpt_120b_mini_level_1 \
#     dataset_src=huggingface dataset_name=makora-ai/KernelMiniBench level=1 \
#     eval_mode=local num_gpu_devices=1 timeout=300