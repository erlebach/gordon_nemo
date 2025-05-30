# Minimal working example for a Nemo-2 recipe
## Series of failed examples
### First working code: `dummy_recipe_mwe_8.p9y`

- dummy_recipe_mwe_1.py
- dummy_recipe_mwe_2.py
- dummy_recipe_mwe_3.py
- dummy_recipe_mwe_4.py
- dummy_recipe_mwe_5.py
- dummy_recipe_mwe_6.py
- dummy_recipe_mwe_7.py
- dummy_recipe_mwe_8.py
- dummy_recipe_mwe.py

Here's a table outlining the key changes in each iteration:

| Version                 | Key Change(s) from Previous                                                                                                                                                              | Primary Issue Addressed / Outcome                                                                                                                              |
| :---------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Initial Script**      | User-provided MWE using `@run.cli.factory`.                                                                                                                                              | Error: `ValueError` regarding return type not being `Config` or `Partial`.                                                                                     |
| **Recipe 1**            | Changed decorator from `@run.cli.factory` to `@run.autoconvert(partial=True)`.                                                                                                           | Error: `UnsupportedLanguageConstructError` (IfExp) due to conditional `accelerator` assignment directly in `pl.Trainer` call.                                |
| **Recipe 2**            | Moved conditional `accelerator` assignment to a variable within `dummy_recipe` (still under `@run.autoconvert`) before passing to `pl.Trainer`.                                            | Error: Persisted `UnsupportedLanguageConstructError` (IfExp) as conditional was still within `autoconvert`'s scope.                                            |
| **Recipe 3**            | Introduced `create_trainer_config` function to return `run.Config` (handling conditional logic inside); `dummy_recipe` (with `@run.autoconvert`) used a nested `train_fn`.                  | Error: `UnsupportedLanguageConstructError` due to nested function definition (`train_fn`).                                                                     |
| **Recipe 4**            | Moved `train_fn` to top-level (`top_level_train_fn`); `dummy_recipe` reverted to `@run.cli.factory`; model/data modules wrapped in `run.Config` when creating `run.Partial`.                 | Error: Reverted to `ValueError` (return type not `Config` or `Partial`), suggesting `@run.cli.factory` still not satisfied.                                    |
| **Recipe 5**            | Kept `@run.cli.factory`; in `dummy_recipe`, model/data modules instantiated directly (not `run.Config`) and passed as instances to `run.Partial`.                                        | Error: Persisted `ValueError` (return type not `Config` or `Partial`).                                                                                         |
| **Recipe 6**            | Kept `@run.cli.factory`; all components (trainer, model, data) wrapped as `run.Config` objects within `dummy_recipe` before creating `run.Partial`.                                      | Error: Persisted `ValueError` (return type not `Config` or `Partial`).                                                                                         |
| **Recipe 7**            | `dummy_recipe` decorator changed back to `@run.autoconvert(partial=True)`; conditional logic for `accelerator` remained inside `dummy_recipe`.                                            | Error: User reported same "error related to the conditional", implying `UnsupportedLanguageConstructError` (IfExp) with `@run.autoconvert`.                  |
| **Recipe 8**            | Kept `@run.autoconvert(partial=True)`; `accelerator_type` hardcoded to `"cpu"` in `dummy_recipe`.                                                                                        | **Success:** Code ran. Confirmed conditional logic with `@run.autoconvert` was the primary blocker.                                                              |
| **Recipe 9**            | Introduced `DummyRecipeConfig` dataclass; conditional `accelerator` logic moved to `__post_init__` of dataclass; `dummy_recipe` (with `@run.autoconvert`) used this dataclass config.       | Ran, but new issue: Socket warnings (`gai error: 8`) appeared when running offline, related to `torchrun`'s IPv6 resolution.                                 |
| **Recipe 10**           | To address socket warnings: In `pl.Trainer` config, explicitly set `devices=1` and `strategy="auto"`; in `create_local_executor`, set `ntasks_per_node=1`.                               | Socket warnings persisted, indicating `torchrun` was still attempting distributed setup causing network resolution issues.                                   |
| **Recipe 11 (Final)** | To address socket warnings: In `create_local_executor`, changed `launcher="torchrun"` to `launcher=None`.                                                                                | **Success:** Code ran as expected without socket warnings. Bypassing `torchrun` resolved the network-related issues for a single-process setup.           |

---

# `dummy_llm_recipe_mwe.py`

Wrap a very simple LLM into a recipe. The LLM 
