Use Nemo2.0 to run fine tune a pretrained model. 

On hpc at FSU: 
# The following modules could not be added via "uv add"
uv venv python 3.10
source .venv/bin/activate
uv add opencc --no-binary-package opencc
uv pip install pangu
uv pip install datasets

# Compiles opencc (on which opencc_clib) depends. This ensures C compiler compatibility. 
uv add opencc --no-binary-package opencc

# Alternatively (from Perplexity)
The error you're encountering is not actually about OpenCC anymore - it's a **PyTorch dependency resolution issue** caused by using `--no-binary` globally. When you specify `--no-binary opencc`, uv is applying the no-binary constraint to **all packages**, including PyTorch, which doesn't provide source distributions for most versions.

## **The Root Problem**

The `--no-binary` flag in uv affects all packages in the resolution, not just the one you specify. PyTorch doesn't distribute source code for most versions (only pre-built wheels), so when uv tries to resolve your dependencies with `--no-binary`, it can't find source distributions for PyTorch, causing the entire dependency resolution to fail.

## **Correct Solutions**

### **Method 1: Use UV's Package-Specific No-Binary Syntax**

```bash
uv add opencc --no-binary-package opencc
```

This targets only the OpenCC package for source compilation while allowing other packages to use pre-built wheels.

### **Method 2: Configure PyTorch Index for Your Project**

Since you're working with Nemo (which depends on PyTorch), you should properly configure PyTorch sources in your `pyproject.toml`[4]:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux'" }
]
```

Then add OpenCC without the no-binary flag:

```bash
uv add opencc==1.1.6
```

### **Method 3: Use UV Pip Interface with Specific No-Binary**

```bash
uv pip install --no-binary=opencc opencc
```

This uses uv's pip interface which has more granular control over the no-binary behavior.

### **Method 4: Install PyTorch First, Then OpenCC**

Since your project needs both PyTorch and OpenCC, install PyTorch from the proper index first:

```bash
uv add torch --index-url https://download.pytorch.org/whl/cu121
uv add opencc==1.1.6
```

## **Recommended Approach**

Given that you're working with Nemo and need both PyTorch and OpenCC, I recommend **Method 2** - properly configuring your PyTorch sources in `pyproject.toml` and then installing a compatible version of OpenCC. This approach ensures both packages work together without compilation issues.

The key insight is that your original OpenCC GLIBCXX error might be resolved simply by using a compatible version (like 1.1.6) rather than forcing source compilation, especially since the compilation approach is causing broader dependency resolution problems with PyTorch.

Citations:
[1] https://github.com/astral-sh/uv/issues/8746
[2] https://github.com/astral-sh/uv/issues/9942
[3] https://github.com/astral-sh/uv/issues/5399
[4] https://docs.astral.sh/uv/guides/integration/pytorch/
[5] https://stackoverflow.com/questions/79552680/streamlit-modulenotfounderror-no-module-named-torch-mac-os
[6] https://stackoverflow.com/questions/75514846/pip-says-version-40-8-0-of-setuptools-does-not-satisfy-requirement-of-setuptools
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC8455138/
[8] https://aronvandepol.com/posts/post_files/251206_uv/251206_uv.html
[9] https://lightning.ai/docs/pytorch/stable/starter/installation.html
[10] https://stackoverflow.com/questions/54552367/pip-cannot-find-metadata-file-environmenterror
[11] https://forum.opencv.org/t/opencv-versions-conflict/16039
[12] https://discuss.pytorch.org/t/install-pytorch-with-cuda-12-1/174294
[13] https://github.com/Lightning-AI/pytorch-lightning/discussions/14743
[14] https://pixi.sh/dev/features/pytorch/
[15] https://lightning.ai/docs/pytorch/stable/versioning.html
[16] https://github.com/astral-sh/uv/issues/12562
[17] https://github.com/astral-sh/uv/issues/10708
[18] https://pixi.sh/v0.41.3/features/pytorch/

---
Answer from Perplexity: pplx.ai/share

Perhaps freeze the pyproject.html for reprocibility? 
----------------------------------------------------------------------
Test with HuggingFace. 
- AutoModel, AutoTokenizer. 
- Models
    - MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

The following was downloaded: 
-rw-r--r--  1 erlebach  staff       874 May 26 23:10 config.json
-rw-r--r--  1 erlebach  staff       296 May 26 23:10 special_tokens_map.json
-rw-r--r--  1 erlebach  staff     52570 May 26 23:10 tokenizer_config.json
-rw-r--r--  1 erlebach  staff  17209920 May 26 23:10 tokenizer.json

- Try Llama 8B. 
- Try Bert model (older and simpler model, but not Transformer)


