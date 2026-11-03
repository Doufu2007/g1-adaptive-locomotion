"""Small read-only inventory of the generated G1 phase artifacts."""

import json
from pathlib import Path

import numpy as np
import torch


root = Path(__file__).resolve().parents[1]
data_path = root / "results/g1_phase_dataset_20261019/phase_dataset.npz"
with np.load(data_path, allow_pickle=True) as dataset:
  print("dataset_arrays:")
  for key in dataset.files:
    value = dataset[key]
    print(f"  {key}: shape={value.shape} dtype={value.dtype}")

metadata_path = root / "results/g1_phase_dataset_20261019/dataset_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
print(f"sequence_metadata_keys={sorted(metadata['sequences'][0].keys())}")

checkpoint_path = root / "results/g1_phase_dataset_20261019/phase_encoder_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print(f"checkpoint_keys={sorted(checkpoint.keys())}")
for key in ("mean", "std"):
  value = checkpoint[key]
  print(f"{key}: type={type(value).__name__} shape={getattr(value, 'shape', None)}")
