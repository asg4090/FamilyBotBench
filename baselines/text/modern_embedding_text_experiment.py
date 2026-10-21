#!/usr/bin/env python3
"""Modern local embedding baseline for BotSim-24 comment-only classification."""

from __future__ import annotations

# Locate the release independently of the current working directory.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = next(p for p in _ReleasePath(__file__).resolve().parents if (p / "release_support.py").is_file())
if str(_RELEASE_ROOT) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_ROOT))
from release_support import parse_release_args, apply_paths, research_cli, load_yaml, redact_error


import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from comment_text_only_experiment import load_comment_docs, split_indices, truncate_words


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_docs(
    docs: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    normalize: bool,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    for start in range(0, len(docs), batch_size):
        batch_docs = docs[start : start + batch_size]
        encoded = tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded)
        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().numpy())
        print(f"encoded {min(start + batch_size, len(docs))}/{len(docs)}", flush=True)
    return np.vstack(embeddings)


def evaluate_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(embeddings[train_idx], labels[train_idx])

    scores = {}
    preds_by_split = {}
    probs_by_split = {}
    for split_name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        pred = clf.predict(embeddings[idx])
        prob = clf.predict_proba(embeddings[idx])[:, 1]
        preds_by_split[split_name] = pred
        probs_by_split[split_name] = prob
        scores[split_name] = {
            "accuracy": accuracy_score(labels[idx], pred),
            "precision": precision_score(labels[idx], pred, zero_division=0),
            "recall": recall_score(labels[idx], pred, zero_division=0),
            "f1": f1_score(labels[idx], pred, zero_division=0),
        }
    return scores, preds_by_split["test"], probs_by_split["test"]


def save_confusion(labels: np.ndarray, idx: np.ndarray, pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(labels[idx], pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["human", "bot"])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_metric_plot(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    x = np.arange(len(test))
    width = 0.38
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - width / 2, test["accuracy"], width, label="Accuracy", color="#4C78A8")
    plt.bar(x + width / 2, test["f1"], width, label="F1", color="#E45756")
    plt.xticks(x, test["variant"], rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Modern embedding comment-only classification")
    plt.legend()
    for i, row in test.reset_index(drop=True).iterrows():
        plt.text(i - width / 2, row["accuracy"] + 0.012, f"{row['accuracy']:.3f}", ha="center", fontsize=8)
        plt.text(i + width / 2, row["f1"] + 0.012, f"{row['f1']:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--truncate-words", type=int, default=600)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--reuse-embeddings", action="store_true")
    args = parser.parse_args()

    root = _RELEASE_ROOT
    out_dir = root / "outputs" / "embedding_baseline" / slugify_model_name(args.model)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    stats, _raw_docs, _clean_docs, strict_docs = load_comment_docs(root)
    labels = stats["label"].to_numpy(dtype=int)
    docs_all = [truncate_words(doc, args.truncate_words) for doc in strict_docs]
    nonempty_idx = stats.index[~stats["empty_after_strict_clean"]].to_numpy()

    device = pick_device(args.device)
    embedding_path = out_dir / "strict_clean_truncated_embeddings.npy"
    if args.reuse_embeddings and embedding_path.exists():
        embeddings = np.load(embedding_path)
    else:
        embeddings = encode_docs(
            docs_all,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            normalize=args.normalize,
        )
        np.save(embedding_path, embeddings)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    metric_rows = []
    prediction_rows = []
    first_seed_all_pred = None
    first_seed_all_test_idx = None
    first_seed_nonempty_pred = None
    first_seed_nonempty_test_idx = None

    for seed in seeds:
        all_train_idx, all_val_idx, all_test_idx = split_indices(labels, seed)
        nonempty_train_idx, nonempty_val_idx, nonempty_test_idx = split_indices(labels, seed, nonempty_idx)
        scenarios = [
            ("all_users", all_train_idx, all_val_idx, all_test_idx),
            ("strict_nonempty_users", nonempty_train_idx, nonempty_val_idx, nonempty_test_idx),
        ]
        for scenario, train_idx, val_idx, test_idx in scenarios:
            scores, test_pred, test_prob = evaluate_embeddings(embeddings, labels, train_idx, val_idx, test_idx)
            for split, vals in scores.items():
                metric_rows.append(
                    {
                        "seed": seed,
                        "model": args.model,
                        "variant": f"{scenario}_strict_clean_truncated_{args.truncate_words}w_embedding",
                        "split": split,
                        "size": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}[split],
                        **vals,
                    }
                )
            for row_idx, pred, prob in zip(test_idx, test_pred, test_prob):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "row": int(row_idx),
                        "user_id": stats.loc[row_idx, "user_id"],
                        "true_label": "bot" if labels[row_idx] == 1 else "human",
                        "pred_label": "bot" if pred == 1 else "human",
                        "bot_probability": float(prob),
                    }
                )
            if seed == seeds[0] and scenario == "all_users":
                first_seed_all_pred = test_pred
                first_seed_all_test_idx = test_idx
            if seed == seeds[0] and scenario == "strict_nonempty_users":
                first_seed_nonempty_pred = test_pred
                first_seed_nonempty_test_idx = test_idx

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = (
        metrics[metrics["split"] == "test"]
        .groupby(["model", "variant"])
        .agg(
            runs=("seed", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            min_f1=("f1", "min"),
            max_f1=("f1", "max"),
        )
        .reset_index()
    )

    metrics.to_csv(out_dir / "embedding_metrics.csv", index=False)
    predictions.to_csv(out_dir / "embedding_predictions.csv", index=False)
    summary.to_csv(out_dir / "embedding_summary.csv", index=False)
    save_metric_plot(metrics[metrics["seed"] == seeds[0]], fig_dir / "seed0_embedding_metric_comparison.png")
    if first_seed_all_pred is not None and first_seed_all_test_idx is not None:
        save_confusion(
            labels,
            first_seed_all_test_idx,
            first_seed_all_pred,
            fig_dir / "seed0_all_users_confusion_matrix.png",
            f"{args.model}: all users, seed {seeds[0]}",
        )
    if first_seed_nonempty_pred is not None and first_seed_nonempty_test_idx is not None:
        save_confusion(
            labels,
            first_seed_nonempty_test_idx,
            first_seed_nonempty_pred,
            fig_dir / "seed0_nonempty_users_confusion_matrix.png",
            f"{args.model}: nonempty users, seed {seeds[0]}",
        )

    run_summary = {
        "model": args.model,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "truncate_words": args.truncate_words,
        "embedding_shape": list(embeddings.shape),
        "seeds": seeds,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
