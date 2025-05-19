## Files and their functions

1. **multiblock_adapter.py**
   - Core implementation file containing all adapter-related classes
   - Includes MultiBlockRegressorModel (ModelPT implementation)
   - Defines adapters and PEFT functionality

2. **multiblock_config.yaml**
   - Configuration for training the base model
   - Specifies model architecture, training parameters, and data paths

3. **multiblock_train.py**
   - Script to train the base model using Hydra
   - Creates sine wave datasets if they don't exist
   - Uses the config to initialize and train MultiBlockRegressorModel

4. **multiblock_finetune_config.yaml**
   - Configuration for fine-tuning with adapters
   - Specifies adapter parameters and paths to data/models

5. **multiblock_finetune.py**
   - Script to fine-tune using adapters with Hydra
   - Creates phase-shifted sine datasets for fine-tuning
   - Loads base model and applies adapters

6. **multiblock_evaluate.py** (with multiblock_evaluate_config.yaml)
   - Script to evaluate and visualize both models
   - Creates plots comparing base and adapter model predictions

## Usage Order

1.  Train the base model:
   ```bash
   python multiblock_train.py model.nemo_path=multiblock_base.nemo
   ```

2.  Fine-tune with adapters:
   ```bash
   python multiblock_finetune.py model.restore_path=multiblock_base.nemo
   ```

3.  Evaluate and visualize:
   ```bash
   python multiblock_evaluate.py
   ```
