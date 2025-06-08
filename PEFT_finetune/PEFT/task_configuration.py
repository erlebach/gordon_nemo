import nemo_run as run
from nemo.collections import llm

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=llm.llama3_8b.model(),
        source="hf://meta-llama/Meta-Llama-3-8B",
        overwrite=False,
    )

def configure_finetuning_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.llama3_8b.finetune_recipe(
        dir="/checkpoints/llama3_finetuning", # Path to store checkpoints
        name="llama3_lora",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    recipe.trainer.max_steps = 100
    recipe.trainer.num_sanity_val_steps = 0

    # Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.val_check_interval = 100

    # This is currently required for LoRA/PEFT
    recipe.trainer.strategy.ddp = "megatron"

    return recipe
