# Known Issues in FoNu Transformer V1

## 1. Output Dimension Mismatch

**Error**: Output dimension 512 doesn't match target vocab size 8000

**Root Cause**: The Generator layer was correctly initialized with output dimension 8000, but wasn't being applied in the forward pass.

**Impact**: Model couldn't produce valid token probabilities over the vocabulary.

## 2. Import Path Problems

**Error**: ModuleNotFoundError: No module named 'model_utils'

**Root Cause**: Inconsistent import paths when running from different directories.

**Impact**: Scripts couldn't be run reliably from different locations.

## 3. Training Instability

**Symptom**: Loss fluctuations and failure to converge

**Root Cause**: Inappropriate learning rate and batch size for low-resource data.

**Impact**: Model training was unstable and often failed to produce useful results.
