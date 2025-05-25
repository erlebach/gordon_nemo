#!/bin/bash

python -m exp_lightning.run_model \
  --model_path="final_base_model.nemo" \
  --data_path="data/base/sine_val.npz" \
  --output_path="data/base_model_eval_preds.npz"

    # --finetune_val_data_path="data/finetune/sine_val.npz" \
    # --output_path="data/base_model_eval_preds.npz"
