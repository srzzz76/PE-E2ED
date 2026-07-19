import argparse
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from phase_reconstruction.model import EndToEndPhaseNet, TFDNet


def build_model():
    backbone = TFDNet(
        in_channel=2,
        out_channel=2,
        width=16,
        middle_blk_num=4,
        enc_blk_nums=[1, 1, 1, 4],
        dec_blk_nums=[2, 1, 1, 1],
    )
    return EndToEndPhaseNet(backbone, width=32, num_blocks=4)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary.")
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break

    state_dict = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            while key.startswith("module."):
                key = key[len("module.") :]
            state_dict[key] = value
    if not state_dict:
        raise ValueError("No model tensors were found in the checkpoint.")
    return state_dict


def checkpoint_is_deployed(state_dict):
    has_bias = any(key.endswith("dwconv.lk_origin.bias") for key in state_dict)
    has_bn = any("dwconv.origin_bn." in key for key in state_dict)
    return has_bias and not has_bn


def load_model(checkpoint_path, device, reparameterize=True):
    state_dict = extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    deployed = checkpoint_is_deployed(state_dict)
    model = build_model()
    if deployed:
        model.switch_to_deploy()
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    if reparameterize and not deployed:
        model.switch_to_deploy()
    return model


def read_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def circular_mask(height, width, device):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    return (x.square() + y.square() <= 1).float()[None, None]


def prepare_pair(frame1_path, frame2_path, device):
    frame1 = read_grayscale(frame1_path)
    frame2 = read_grayscale(frame2_path)
    if frame1.shape != frame2.shape:
        raise ValueError(f"Input shape mismatch: {frame1.shape} vs {frame2.shape}")
    divisor = 65535.0 if frame1.max() > 255 else 255.0
    inputs = np.stack(
        [frame1.astype(np.float32) / divisor, frame2.astype(np.float32) / divisor]
    )
    return torch.from_numpy(inputs)[None].to(device), divisor


def save_uint16(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.round(np.clip(image, 0, 1) * 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")


@torch.inference_mode()
def predict_pair(model, frame1_path, frame2_path, device):
    inputs, divisor = prepare_pair(frame1_path, frame2_path, device)
    mask = circular_mask(inputs.shape[-2], inputs.shape[-1], device)
    phase, _, _, _, _, _ = model(inputs, mask)
    return phase * mask, mask, divisor


def save_prediction(output_dir, filename, phase, batch_mode):
    phase = phase[0, 0].cpu().numpy()
    if batch_mode:
        path = output_dir / "phase" / filename
    else:
        path = output_dir / "phase.png"
    save_uint16(path, phase)


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def compute_l1(phase, mask, label_path, divisor, device):
    label = read_grayscale(label_path).astype(np.float32) / divisor
    if label.shape != tuple(phase.shape[-2:]):
        raise ValueError(f"Label shape mismatch: {label.shape} vs {tuple(phase.shape[-2:])}")
    label = torch.from_numpy(label)[None, None].to(device)
    return F.l1_loss(phase, label * mask).item()


def save_loss_curve(losses, names, output_dir):
    if not losses:
        return
    figure, axis = plt.subplots(figsize=(10, 4))
    indices = np.arange(len(losses))
    axis.plot(indices, losses, color="tab:blue", linewidth=1, marker="o", markersize=3)
    axis.axhline(
        np.mean(losses), color="tab:red", linestyle="--", label=f"Mean: {np.mean(losses):.6f}"
    )
    axis.set_xlabel("Sample index")
    axis.set_ylabel("L1 loss")
    axis.set_title("L1 loss of all test samples")
    axis.grid(True, alpha=0.3)
    axis.legend()
    if len(names) <= 30:
        axis.set_xticks(indices, names, rotation=60, ha="right", fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(figure)


def predict_dataset(model, data_dir, output_dir, device):
    frame1_paths = sorted((data_dir / "frame1").glob("*.png"), key=natural_key)
    if not frame1_paths:
        raise FileNotFoundError(f"No PNG files found in {data_dir / 'frame1'}")

    losses, names = [], []
    for index, frame1_path in enumerate(frame1_paths, start=1):
        frame2_path = data_dir / "frame2" / frame1_path.name
        label_path = data_dir / "phi" / frame1_path.name
        if not frame2_path.exists():
            print(f"Skip {frame1_path.name}: frame2 is missing")
            continue

        phase, mask, divisor = predict_pair(
            model, frame1_path, frame2_path, device
        )
        save_prediction(output_dir, frame1_path.name, phase, batch_mode=True)
        if label_path.exists():
            loss = compute_l1(phase, mask, label_path, divisor, device)
            losses.append(loss)
            names.append(frame1_path.stem)
            print(f"[{index}/{len(frame1_paths)}] {frame1_path.name} | L1: {loss:.6f}")
        else:
            print(f"[{index}/{len(frame1_paths)}] {frame1_path.name} | no label")

    save_loss_curve(losses, names, output_dir)
    if losses:
        print(f"Average L1: {np.mean(losses):.6f}")


def predict_single(model, args, output_dir, device):
    phase, mask, divisor = predict_pair(
        model, args.frame1, args.frame2, device
    )
    save_prediction(output_dir, "", phase, batch_mode=False)
    if args.label:
        loss = compute_l1(phase, mask, args.label, divisor, device)
        save_loss_curve([loss], [args.frame1.stem], output_dir)
        print(f"L1: {loss:.6f}")


def main(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.data_dir is None and (args.frame1 is None or args.frame2 is None):
        raise ValueError("Use --data-dir, or provide both --frame1 and --frame2.")
    if args.data_dir is not None and (args.frame1 is not None or args.frame2 is not None):
        raise ValueError("Use either --data-dir or --frame1/--frame2, not both.")

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device, args.reparameterize)
    if args.data_dir is not None:
        predict_dataset(model, args.data_dir, output_dir, device)
    else:
        predict_single(model, args, output_dir, device)
    print(f"Results saved to: {output_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="PE-E2ED inference and evaluation.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", default="prediction", type=Path)
    parser.add_argument("--data-dir", type=Path, help="Dataset with frame1/frame2/phi folders.")
    parser.add_argument("--frame1", type=Path)
    parser.add_argument("--frame2", type=Path)
    parser.add_argument("--label", type=Path, help="Optional label for single-pair inference.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-reparameterize", action="store_false", dest="reparameterize")
    parser.set_defaults(reparameterize=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
