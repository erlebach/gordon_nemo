#!/usr/bin/env python3
"""Test script to check Transformer Engine installation and NeMo integration."""

import sys

import torch

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)

# Test 1: Direct import of transformer_engine
try:
    import transformer_engine

    print("✓ transformer_engine imported successfully")
    print("  Version:", transformer_engine.__version__)
except ImportError as e:
    print("✗ Failed to import transformer_engine:", e)

# Test 2: Import transformer_engine.pytorch
try:
    import transformer_engine.pytorch as te

    print("✓ transformer_engine.pytorch imported successfully")
except ImportError as e:
    print("✗ Failed to import transformer_engine.pytorch:", e)

# Test 3: Check NeMo imports
try:
    import nemo

    print("✓ nemo imported successfully")
    print("  Version:", nemo.__version__)
except ImportError as e:
    print("✗ Failed to import nemo:", e)

# Test 4: Check ApexGuardDefaults
try:
    from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

    print("✓ ApexGuardDefaults imported successfully")

    apex_guard = ApexGuardDefaults()
    print("  has_transformer_engine:", apex_guard.has_transformer_engine)
    print("  has_apex:", apex_guard.has_apex)

    if not apex_guard.has_transformer_engine:
        print("  ⚠️  ApexGuardDefaults reports transformer_engine not available")
        print("  This might be a detection issue")

except ImportError as e:
    print("✗ Failed to import ApexGuardDefaults:", e)

# Test 5: Try to import TransformerLayer
try:
    from nemo.collections.nlp.modules.common.megatron.transformer import (
        TransformerLayer,
    )

    print("✓ TransformerLayer imported successfully")
except ImportError as e:
    print("✗ Failed to import TransformerLayer:", e)

print("\n" + "=" * 50)
print("SUMMARY:")
print(
    "If transformer_engine imports but ApexGuardDefaults.has_transformer_engine is False,"
)
print("this indicates a detection issue in NeMo's ApexGuardDefaults.")
print("The transformer_engine is actually available and can be used directly.")
