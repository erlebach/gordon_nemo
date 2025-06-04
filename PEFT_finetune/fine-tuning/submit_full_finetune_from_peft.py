from nemo_launcher import run
from nemo_launcher.utils.llm import llm

def configure_full_finetune_from_peft(nodes: int = 1, gpus_per_node: int = 2):
    # Full FT with PEFT-trained weights as initialization
    recipe = llm.llama3_8b.finetune_recipe(
        dir="/workspace/results/full_finetune",
        name="llama3_full_ft",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme=None,  # FULL finetuning: disable PEFT
        init_from_nemo_model="/workspace/results/lora_finetune/checkpoints/megatron_gpt.nemo",
    )
    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 2):
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "HF_TOKEN_PATH": "/tokens/huggingface",
        "CUDA_VISIBLE_DEVICES": "0,1"
    }
    return run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

def run_full_finetune():
    full_ft = configure_full_finetune_from_peft(nodes=1, gpus_per_node=2)
    executor = local_executor_torchrun(nodes=1, devices=2)

    with run.Experiment("llama3-8b-full-from-peft") as exp:
        exp.add(full_ft, executor=executor, name="full_finetune")
        exp.run(sequential=True, tail_logs=True)

if __name__ == "__main__":
    run_full_finetune()
