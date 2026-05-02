import json
from pathlib import Path

import numpy as np


def main():
    seed = 42
    train_frac = 0.8
    data_root = Path('data/f110_dataset_20220209')
    labels_path = data_root / "labels.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find {labels_path}")

    labels = np.load(labels_path)
    n = len(labels)

    # Make sure all expected images exist.
    missing = []
    for i in range(n):
        if not (data_root / f"{i}.jpg").exists():
            missing.append(i)

    if missing:
        raise FileNotFoundError(f"Missing image files for indices: {missing[:20]}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(train_frac * n)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    out_dir = Path('splits')
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_indices.npy", train_indices)
    np.save(out_dir / "val_indices.npy", val_indices)

    summary = {
        "dataset": str(data_root),
        "num_total": int(n),
        "num_train": int(len(train_indices)),
        "num_val": int(len(val_indices)),
        "train_fraction": float(train_frac),
        "seed": int(seed),
    }

    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()