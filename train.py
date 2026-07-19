import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml

from phase_reconstruction.data import PhaseDataset
from phase_reconstruction.losses import PhaseLoss
from phase_reconstruction.model import EndToEndPhaseNet, TFDNet
from phase_reconstruction.training import (
    barrier,
    circular_mask,
    current_stage,
    make_loaders,
    perturbation_parameters,
    reduce_sum,
    set_seed,
    setup_distributed,
    unwrap_model,
)


def train(args):
    settings = args.settings
    local_rank, rank, world_size, device = setup_distributed()
    set_seed(args.seed + rank, deterministic=args.deterministic)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    dataset = PhaseDataset(args.data_dir)
    train_loader, validation_loader, train_sampler = make_loaders(
        dataset, args.batch_size, args.workers, rank, world_size, args.seed
    )
    backbone = TFDNet(**settings["model"]["backbone"])
    model = EndToEndPhaseNet(backbone, **settings["model"]["refine"]).to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    criterion = PhaseLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate * settings["lr_scale"]
    )
    scheduler = None
    best_validation_l1 = float("inf")
    metrics_path = output_dir / "metrics.csv"
    full_metric_names = [
        "total",
        "l1",
        "sc",
        "nll",
        "nll_term",
        "logvar_term",
        "unit",
    ]
    full_metrics_path = output_dir / "full_metrics.csv"
    if rank == 0:
        with metrics_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                ["epoch", "stage", "train_loss", "validation_loss", "validation_l1"]
            )
        if args.save_full_metrics:
            with full_metrics_path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(
                    file,
                    fieldnames=[
                        "epoch",
                        "time",
                        "stage",
                        "train_total",
                        "train_raw_consist",
                        "val_total",
                        "val_l1",
                        "val_sc",
                        "val_nll",
                        "val_nll_term",
                        "val_logvar_term",
                        "val_unit",
                    ],
                ).writeheader()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        config, stage_start = current_stage(epoch, settings["stages"])
        amplitude_range, background_range, noise_std = perturbation_parameters(
            epoch, settings["perturbation"]
        )
        if epoch == stage_start and epoch != 0:
            if rank == 0:
                torch.save(
                    unwrap_model(model).state_dict(),
                    output_dir / f"{config['name']}_init.pth",
                )
            best_validation_l1 = float("inf")
            if config["name"] == "Stage2_Joint_Phase_Retrieval":
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = 2e-3 * settings["lr_scale"]
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch, eta_min=1e-6
                )

        model.train()
        train_totals = torch.zeros(3, dtype=torch.float64, device=device)
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [{config['name']}]",
            disable=rank != 0,
        )
        for noisy_images, _, phase_target in iterator:
            noisy_images = noisy_images.to(device, non_blocking=True)
            phase_target = phase_target.to(device, non_blocking=True)
            mask = circular_mask(
                len(noisy_images), phase_target.shape[-2], phase_target.shape[-1], device
            )
            physical_target = phase_target * 80.0 - 40.0
            sin_target, cos_target = torch.sin(physical_target), torch.cos(physical_target)

            optimizer.zero_grad(set_to_none=True)
            phase, _, sin_phase, cos_phase, _, uncertainty = model(noisy_images, mask)
            base_loss, _ = criterion(
                phase,
                phase_target,
                mask,
                sin_phase,
                cos_phase,
                sin_target,
                cos_target,
                uncertainty,
                config,
            )
            consistency_loss = torch.tensor(0.0, device=device)
            raw_consistency = 0.0
            if config["w_cons"] > 0:
                with torch.no_grad():
                    batch_size = len(noisy_images)
                    amplitude = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
                        *amplitude_range
                    )
                    background = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
                        *background_range
                    )
                    detached_phase = phase.detach()
                    first = amplitude * torch.cos(detached_phase * 80.0 - 40.0) + background
                    shift = torch.rand_like(first) * 2.9 + 0.1
                    second = (
                        amplitude * torch.cos(detached_phase * 80.0 - 40.0 + shift)
                        + background
                    )
                    first = torch.clamp(first + noise_std * torch.randn_like(first), 0, 1) * mask
                    second = torch.clamp(second + noise_std * torch.randn_like(second), 0, 1) * mask
                varied_phase, _, _, _, _, _ = model(torch.cat([first, second], dim=1), mask)
                consistency_loss = config["w_cons"] * F.l1_loss(
                    varied_phase * mask, detached_phase * mask
                )
                raw_consistency = consistency_loss.item() / config["w_cons"]

            total_loss = base_loss + consistency_loss
            total_loss.backward()
            optimizer.step()
            train_totals[0] += total_loss.detach().double() * len(noisy_images)
            train_totals[1] += raw_consistency * len(noisy_images)
            train_totals[2] += len(noisy_images)

        if scheduler is not None:
            scheduler.step()
        reduce_sum(train_totals)
        train_loss = (train_totals[0] / train_totals[2].clamp_min(1)).item()
        train_raw_consistency = (train_totals[1] / train_totals[2].clamp_min(1)).item()

        model.eval()
        validation_totals = torch.zeros(
            len(full_metric_names) + 1, dtype=torch.float64, device=device
        )
        with torch.no_grad():
            for noisy_images, _, phase_target in validation_loader:
                noisy_images = noisy_images.to(device, non_blocking=True)
                phase_target = phase_target.to(device, non_blocking=True)
                mask = circular_mask(
                    len(noisy_images), phase_target.shape[-2], phase_target.shape[-1], device
                )
                physical_target = phase_target * 80.0 - 40.0
                sin_target, cos_target = torch.sin(physical_target), torch.cos(physical_target)
                phase, _, sin_phase, cos_phase, _, uncertainty = model(noisy_images, mask)
                validation_loss, validation_metrics = criterion(
                    phase,
                    phase_target,
                    mask,
                    sin_phase,
                    cos_phase,
                    sin_target,
                    cos_target,
                    uncertainty,
                    config,
                )
                count = len(noisy_images)
                metric_values = [validation_metrics[name] for name in full_metric_names]
                validation_totals[:-1] += torch.tensor(
                    metric_values, dtype=torch.float64, device=device
                ) * count
                validation_totals[-1] += count

        reduce_sum(validation_totals)
        validation_averages = validation_totals[:-1] / validation_totals[-1].clamp_min(1)
        averaged_metrics = dict(zip(full_metric_names, validation_averages.tolist()))
        validation_loss = averaged_metrics["total"]
        validation_l1 = averaged_metrics["l1"]
        if rank == 0:
            torch.save(unwrap_model(model).state_dict(), output_dir / "latest.pth")
            if validation_l1 < best_validation_l1:
                best_validation_l1 = validation_l1
                torch.save(unwrap_model(model).state_dict(), output_dir / "best.pth")
            with metrics_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(
                    [epoch + 1, config["name"], train_loss, validation_loss, validation_l1]
                )
            if args.save_full_metrics:
                full_log = {
                    "epoch": epoch,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "stage": config["name"],
                    "train_total": train_loss,
                    "train_raw_consist": train_raw_consistency,
                    "val_total": averaged_metrics["total"],
                    "val_l1": averaged_metrics["l1"],
                    "val_sc": averaged_metrics["sc"],
                    "val_nll": averaged_metrics["nll"],
                    "val_nll_term": averaged_metrics["nll_term"],
                    "val_logvar_term": averaged_metrics["logvar_term"],
                    "val_unit": averaged_metrics["unit"],
                }
                with full_metrics_path.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=full_log.keys())
                    writer.writerow(full_log)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] Epoch {epoch + 1}/{args.epochs} | train={train_loss:.6f} "
                f"| validation={validation_loss:.6f} | l1={validation_l1:.6f}"
            )
        barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/train.yaml", type=Path)
    config_args, _ = config_parser.parse_known_args()
    with config_args.config.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description="Train the phase reconstruction model.")
    parser.add_argument("--config", default=config_args.config, type=Path)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=settings["epochs"])
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings["batch_size"],
        help="Batch size per GPU; the global batch size is 8 with two GPUs.",
    )
    parser.add_argument("--learning-rate", type=float, default=settings["learning_rate"])
    parser.add_argument("--workers", type=int, default=settings["workers"])
    parser.add_argument(
        "--seed",
        type=int,
        default=settings["seed"],
        help="Base random seed. The default makes repeated runs reproducible.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=settings["deterministic"],
        help=(
            "Use deterministic cuDNN behavior where possible. This can reduce "
            "training speed; by default cuDNN benchmark mode is enabled."
        ),
    )
    parser.add_argument(
        "--no-save-full-metrics",
        action="store_false",
        dest="save_full_metrics",
        help="Disable the complete per-epoch loss breakdown in full_metrics.csv.",
    )
    parser.set_defaults(save_full_metrics=settings["save_full_metrics"])
    args = parser.parse_args()
    args.settings = settings
    return args


if __name__ == "__main__":
    train(parse_args())
