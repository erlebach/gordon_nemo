# 2025-05-21, 11:55 am
- multiblock_regressor has been implemented with Lightning module
- Data: sine wave (x[i], y[i])
  Train, validation, test data: different x[i], y[i] + noise
  Finetune data: phase-shifted in x sin wave
- Validation and testing error are small with base data. 
  Validation errors large with fine-tuned data when using trained weights based on base data. 
- The code is built from the configuraiton file `conf/multiblock_conf.yaml`
- The code uses OmegaConf.

TODO: 
- Put the data into its own Data block (Done)

# 2025-05-22, 14:30 pm
- Using Lightning, I can control checkpointing and restart. 
- Next: try profiling. 

# 2025-05-22 22:40 pm
- using tensorboard
- code runs fine
- I don't know how to use the command line yet 

TODO: 
- Implement the finetuning adapter.

# 2025-05-22: Nemo + Lora adapter
- `multiblock_regressor_nemo.py`: 
  - upgrades the base Regressor with Lora by applying the adapter. 
    The adapter is zero if `lora_rank=0` in the configuration file.
  - the configuraiton file `config/multiblock_config.yaml` now has 
    an adapter at the top level (outside within the `model` key). 

# 2025-05-24, 20:38 pm
1. lora_rank = 0: run with base_data. Base model saved to `final_base_model.nemo`
   Configuration file: config/multiblock_nemo_config.yaml
2. lora_rank = 8: run with fine-tuned data (sine wave phase-shifted)
```bash
python -m exp_lighting.multiblock_regressor_nemo --config-name=multiblock_nemo_finetune_config.yaml
```
  When fine-tuning, also read the output of the base run. 

# 2025-05-24, 23:57
TODO: 
- Run validations on the checkpointed files, stored under tensorflow. 
  Currently, no checkpointed files are stored (only the final files). 
- Analyze the schemas for the configuration file `exp_manager,` etc. I'd like to understand structure. 
  Look at examples under NeMo/examples github repository. 
- Understand: `="data/base_model_eval_preds.npz"`
  `save_every_n_epochs,` `save_every_n_train_steps,` `every_n_train_steps`
  (where in the file to place)
