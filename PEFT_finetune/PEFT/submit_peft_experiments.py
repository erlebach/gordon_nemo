from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo_launcher import run
from nemo_launcher.utils.llm import llm

def configure_finetuning_recipe(nodes: int = 1, gpus_per_node: int = 1):
    # Use LoRA-based PEFT fine-tuning from a pretrained checkpoint
    recipe = llm.llama3_8b.finetune_recipe(
        dir="/workspace/results/lora_finetune",
        name="llama3_peft",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme="lora",  # << this enables LoRA
    )
    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "HF_TOKEN_PATH": "/tokens/huggingface",  # HuggingFace auth
        "CUDA_VISIBLE_DEVICES": "0"
    }
    return run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

def run_finetuning():
    # Step 1: Optional — convert HuggingFace checkpoint
    import_ckpt = llm.llama3_8b.import_from_huggingface(
        output_dir="/workspace/results/llama3_ckpt",
        model_name="meta-llama/Meta-Llama-3-8B",
    )

    # Step 2: PEFT fine-tuning recipe
    finetune = configure_finetuning_recipe(nodes=1, gpus_per_node=1)
    executor = local_executor_torchrun(nodes=1, devices=1)

    with run.Experiment("llama3-8b-peft") as exp:
        exp.add(import_ckpt, executor=run.LocalExecutor(), name="import_from_hf")
        exp.add(finetune, executor=executor, name="peft_finetuning")
        exp.run(sequential=True, tail_logs=True)

if __name__ == "__main__":
    run_finetuning()
