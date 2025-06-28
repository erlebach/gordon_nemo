import os
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime

LOG_DIR = "rendered_jobs"
os.makedirs(LOG_DIR, exist_ok=True)

def render_sbatch(config):
    with open("nemo_job_template.slurm") as f:
        template = f.read()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_filename = f"{config['job_name']}_{timestamp}.slurm"
    output_path = Path(LOG_DIR) / job_filename

    # Optional modules block
    modules = "\n".join([f"module load {m}" for m in config.get("modules", [])])

    rendered = template.format(
        JOB_NAME=config["job_name"],
        NODES=config["nodes"],
        TASKS_PER_NODE=config["tasks_per_node"],
        GPUS_PER_NODE=config["gpus_per_node"],
        CONTAINER=config["container"],
        TRAIN_SCRIPT=config["train_script"],
        CONFIG_PATH=config["config_path"],
        CONFIG_NAME=config["config_name"],
        DATA_MOUNT= os.path.abspath("./data"),
        OUTPUT_MOUNT= os.path.abspath("./output"),
        YAML_OVERRIDES=" \\\n    ".join(config["overrides"]),
        MODULE_LOADS=modules,
    )

    with open(output_path, "w") as out:
        out.write(rendered)

    return output_path

def submit_job(sbatch_file):
    print(f"Submitting SLURM job: {sbatch_file}")
    subprocess.run(["sbatch", str(sbatch_file)])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python submit_nemo_job.py configs/my_training_job.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sbatch_script = render_sbatch(config)
    submit_job(sbatch_script)
