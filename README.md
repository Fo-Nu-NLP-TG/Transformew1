# FoNu Transformer V1

This repository contains our first attempt at implementing a transformer-based model for Ewe-English translation. While this implementation faced several challenges and limitations, it serves as an important learning resource and foundation for our improved implementations.

## Project Status

⚠️ **Note**: This is an archived version with known issues. For our current implementation, see [FoNu_Transformer_V2](https://github.com/FoNuNLPTG/FoNu_Transformer_V2).

## Key Challenges Encountered

- Output dimension mismatch between model (512) and target vocabulary size (8000)
- Import path issues in various execution environments
- Training instability with low-resource language pairs
- Memory management issues with large batch sizes

## Lessons Learned

- Proper dimension handling in the generator layer is critical
- Consistent import structure is necessary for reproducibility
- Low-resource languages require specialized data augmentation
- Careful hyperparameter tuning is essential for stable training

## Repository Structure

```
FoNu_Transformer_V1/
├── model_utils.py          # Core transformer components
├── encode_decode.py        # EncodeDecode model implementation
├── train_transformer.py    # Training script with known issues
├── inference.py            # Inference script
├── training_fixes/         # Documentation of attempted fixes
└── documentation/          # Additional documentation
```

## How We've Improved

The key issues in this implementation have been addressed in our newer versions by:

1. Properly handling output dimensions in the generator layer
2. Implementing robust import structures
3. Adding specialized data augmentation for low-resource languages
4. Improving training stability with gradient accumulation

## License

MIT License
