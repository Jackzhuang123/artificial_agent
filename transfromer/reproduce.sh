#!/bin/bash
set -e

# Install dependencies
pip install torch datasets tokenizers transformers pyyaml

# Train tokenizer and start training
python train.py

# Run validation tests
python -c "from model import ModernTransformer; import yaml, torch; cfg=yaml.safe_load(open('config.yaml')); m=ModernTransformer(cfg); x=torch.randint(0,8192,(2,128)); logits,_=m(x); assert logits.shape==(2,128,8192); print('Shape test passed!')"