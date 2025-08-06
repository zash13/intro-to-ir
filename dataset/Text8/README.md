---
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 90000004
    num_examples: 1
  - name: validation
    num_bytes: 5000004
    num_examples: 1
  - name: test
    num_bytes: 5000004
    num_examples: 1
  download_size: 54357043
  dataset_size: 100000012
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
