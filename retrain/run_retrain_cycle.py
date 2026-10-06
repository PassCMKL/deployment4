"""
Automated retraining cycle:
  trigger check -> fetch new labeled data -> fine-tune -> evaluate -> deploy if better

Run periodically (cron / scheduled task) or on demand for a demo:
    python retrain/run_retrain_cycle.py            # only runs if a trigger fires
    python retrain/run_retrain_cycle.py --force    # simulate a trigger
    python retrain/run_retrain_cycle.py --force --dry-run   # train + evaluate, never deploy
"""

import argparse
import io
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deploy_utils as du  # noqa: E402
import logging_db  # noqa: E402
from decision import decide_deploy, evaluate_trigger  # noqa: E402
from drift_utils import apply_drift  # noqa: E402
from preprocessing import preprocess  # noqa: E402

MIN_NEW_SAMPLES = 10        # not worth a training run on fewer
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LR = 1e-4         # small: nudge the weights, don't overwrite what they know
REPLAY_RATIO = 1.0          # clean training images mixed in per new row (limits forgetting)
N_HELDOUT = 1000            # MNIST test[0:1000]; the workload generator never samples these
CONFIDENCE_WINDOW_SECONDS = 3600


# ---- model --------------------------------------------------------------------

def build_model() -> nn.Module:
    # Same architecture the backend serves (see scripts/model.py in deployment3).
    net = timm.create_model("resnet18", pretrained=False, num_classes=10)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return net


def load_current_model() -> nn.Module:
    # First ever cycle: the live model IS the baseline, so seed the pointer from it.
    if not du.WEIGHTS_PATH.exists():
        shutil.copy2(du.BASELINE_WEIGHTS_PATH, du.WEIGHTS_PATH)
    model = build_model()
    model.load_state_dict(torch.load(du.WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ---- data ----------------------------------------------------------------------

def png_bytes(pixels_or_pil) -> bytes:
    image = pixels_or_pil if isinstance(pixels_or_pil, Image.Image) else Image.fromarray(
        np.asarray(pixels_or_pil).astype(np.uint8), mode="L"
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def to_tensor(png: bytes) -> torch.Tensor:
    # The SAME preprocess() the live backend runs: training data and serving
    # data must be processed identically or the model trains on one thing and
    # is asked about another.
    return torch.from_numpy(preprocess(png)[0])


class TensorPairs(Dataset):
    def __init__(self, tensors: list[torch.Tensor], labels: list[int]):
        self.tensors, self.labels = tensors, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tensors[i], self.labels[i]


def load_mnist(train: bool):
    return datasets.MNIST(root="./data", train=train, download=True)


def logged_rows_dataset(rows: list[dict]) -> Dataset:
    return TensorPairs([to_tensor(r["image_png"]) for r in rows], [int(r["true_label"]) for r in rows])


def replay_dataset(n: int) -> Dataset:
    """Clean original-distribution images mixed into fine-tuning (experience replay)."""
    ds = load_mnist(train=True)
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    return TensorPairs([to_tensor(png_bytes(ds[i][0])) for i in idx], [int(ds[i][1]) for i in idx])


def load_eval_sets() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fixed held-out sets, identical every cycle so accuracies are comparable:
      clean   = MNIST test[0:N_HELDOUT]
      drifted = the same images through apply_drift with a fixed seed
    """
    ds = load_mnist(train=False)
    rng = np.random.default_rng(0)
    clean, drifted, labels = [], [], []
    for i in range(min(N_HELDOUT, len(ds))):
        pil, label = ds[i]
        clean.append(to_tensor(png_bytes(pil)))
        drifted.append(to_tensor(png_bytes(apply_drift(np.array(pil, dtype=np.float32), rng))))
        labels.append(int(label))
    return torch.stack(clean), torch.stack(drifted), torch.tensor(labels)


# ---- train / evaluate ------------------------------------------------------------

def fine_tune(model: nn.Module, rows: list[dict]) -> nn.Module:
    """Continue training the CURRENT weights (not from scratch) on new + replayed data."""
    new_data = logged_rows_dataset(rows)
    data = ConcatDataset([new_data, replay_dataset(int(len(new_data) * REPLAY_RATIO))])
    loader = DataLoader(data, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)

    model.train()
    for epoch in range(FINE_TUNE_EPOCHS):
        for images, labels in loader:
            if images.shape[0] < 2:  # BatchNorm cannot train on a batch of one
                continue
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(images), labels).backward()
            optimizer.step()
        print(f"  epoch {epoch + 1}/{FINE_TUNE_EPOCHS} done ({len(data)} samples)")
    model.eval()
    return model


@torch.no_grad()
def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    return (model(images).argmax(dim=1) == labels).float().mean().item()


def score(model, clean, drifted, labels) -> dict:
    return {"clean": accuracy(model, clean, labels), "drifted": accuracy(model, drifted, labels)}


# ---- deploy ----------------------------------------------------------------------

def export_onnx(model: nn.Module, path: Path) -> None:
    dummy = torch.randn(1, 1, 28, 28)
    kwargs = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
    )
    try:
        # Legacy exporter writes ONE self-contained file, matching how the
        # served model was originally produced. (The newer exporter can split
        # weights into a separate .onnx.data file the server would not find.)
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, path, **kwargs)


def validate_onnx(path: Path, model: nn.Module, sample: torch.Tensor) -> None:
    """Pre-deploy gate: the exported file must load and agree with the PyTorch model."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"input": sample.numpy()})[0]
    with torch.no_grad():
        torch_logits = model(sample).numpy()
    if not np.allclose(onnx_logits, torch_logits, atol=1e-3):
        raise RuntimeError(f"ONNX output differs from PyTorch (max diff {np.abs(onnx_logits - torch_logits).max():.5f})")


def deploy(candidate: nn.Module, sample_images: torch.Tensor, smoke_png: bytes) -> str:
    """Returns the outcome: 'deployed' | 'deployed_server_unreachable' | 'rolled_back' | 'validation_failed'."""
    try:
        export_onnx(candidate, du.ONNX_CANDIDATE_PATH)
        validate_onnx(du.ONNX_CANDIDATE_PATH, candidate, sample_images[:8])
    except Exception as exc:
        print(f"  candidate failed pre-deploy validation, NOT deploying: {exc}")
        du.ONNX_CANDIDATE_PATH.unlink(missing_ok=True)
        return "validation_failed"

    du.backup_live_model()                                   # rollback point
    os.replace(du.ONNX_CANDIDATE_PATH, du.ONNX_PATH)         # atomic swap
    tmp = du.WEIGHTS_PATH.with_name(du.WEIGHTS_PATH.name + ".tmp")
    torch.save(candidate.state_dict(), tmp)
    os.replace(tmp, du.WEIGHTS_PATH)                         # next cycle fine-tunes from here

    status = du.reload_server()
    print(f"  server reload: {status}")
    if status == "unreachable":
        print("  server not running: it will load the new model when it starts.")
        return "deployed_server_unreachable"
    if status != "reloaded" or not du.smoke_test_server(smoke_png):
        print("  post-deploy check FAILED, rolling back to the previous model")
        print(f"  {du.rollback()}")
        return "rolled_back"
    print("  post-deploy smoke test passed")
    return "deployed"


# ---- orchestration -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="simulate a trigger (demo)")
    parser.add_argument("--dry-run", action="store_true", help="train and evaluate but never deploy")
    args = parser.parse_args()

    logging_db.init_db()
    state = du.read_state()
    triggered, reason = evaluate_trigger(
        now=time.time(),
        last_retrain_time=state["last_retrain_time"],
        avg_confidence=logging_db.recent_avg_confidence(CONFIDENCE_WINDOW_SECONDS),
        force=args.force,
    )
    print(f"Trigger: {reason}")
    if not triggered:
        return

    rows = logging_db.fetch_unused_labeled_rows()
    print(f"New labeled, unused rows: {len(rows)}")
    if len(rows) < MIN_NEW_SAMPLES:
        print(f"Fewer than {MIN_NEW_SAMPLES}: skipping (nothing consumed, trigger will fire again).")
        du.append_history({"trigger": reason, "new_rows": len(rows), "outcome": "skipped_insufficient_data"})
        return

    clean, drifted, labels = load_eval_sets()
    current = load_current_model()
    current_scores = score(current, clean, drifted, labels)
    print(f"Live model      clean={current_scores['clean']:.4f} drifted={current_scores['drifted']:.4f}")

    candidate = fine_tune(load_current_model(), rows)
    cand_scores = score(candidate, clean, drifted, labels)
    print(f"Candidate model clean={cand_scores['clean']:.4f} drifted={cand_scores['drifted']:.4f}")

    # The batch has now been tried; mark it used whether or not it wins, so the
    # next cycle looks at new data instead of re-litigating this batch.
    logging_db.mark_rows_used([r["id"] for r in rows])

    better, why = decide_deploy(current=current_scores, candidate=cand_scores)
    print(f"Decision: {why}")

    outcome = "kept_current"
    if better and args.dry_run:
        outcome = "dry_run_would_deploy"
    elif better:
        smoke = np.zeros((28, 28))
        smoke[:, ::7] = 255  # fixed striped image: any valid input works for a smoke test
        outcome = deploy(candidate, drifted, png_bytes(smoke))

    du.append_history({
        "trigger": reason, "new_rows": len(rows),
        "live_clean": round(current_scores["clean"], 4), "live_drifted": round(current_scores["drifted"], 4),
        "cand_clean": round(cand_scores["clean"], 4), "cand_drifted": round(cand_scores["drifted"], 4),
        "decision": why, "outcome": outcome,
    })
    du.write_state({"last_retrain_time": time.time()})
    print(f"Outcome: {outcome}")


if __name__ == "__main__":
    main()
