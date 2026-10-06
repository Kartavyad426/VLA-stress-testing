"""Fine-tuning support around third_party/lerobot's lerobot_train.

  mix.py            a weighted mixture of LeRobot datasets with one schema (new data + replay)
  groot_lora.py     GR00T N1.7 specifics: bf16 load, LoRA targets, freezing, block checkpointing
  feature_cache.py  frozen-VLM feature caching for arm (a): precompute, and a cached backbone
"""
