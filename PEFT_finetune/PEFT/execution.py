def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_finetuning():
    import_ckpt = configure_checkpoint_conversion()
    finetune = configure_finetuning_recipe(nodes=1, gpus_per_node=1)

    executor = local_executor_torchrun(nodes=finetune.trainer.num_nodes, devices=finetune.trainer.devices)
    executor.env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    # Set this env var for model download from huggingface
    executor.env_vars["HF_TOKEN_PATH"] = "/tokens/huggingface"

    with run.Experiment("llama3-8b-peft-finetuning") as exp:
        exp.add(import_ckpt, executor=run.LocalExecutor(), name="import_from_hf") # We don't need torchrun for the checkpoint conversion
        exp.add(finetune, executor=executor, name="peft_finetuning")
        exp.run(sequential=True, tail_logs=True) # This will run the tasks sequentially and stream the logs

# Wrap the call in an if __name__ == "__main__": block to work with Python's multiprocessing module.
if __name__ == "__main__":
    run_finetuning()
