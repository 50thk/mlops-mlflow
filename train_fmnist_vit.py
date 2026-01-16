# Enable Ray Train V2 APIs
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import tempfile
import uuid
import logging

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import FashionMNIST
from torchvision.models import VisionTransformer
from torchvision.transforms import Compose, Normalize, ToTensor

import mlflow
import ray
import ray.train
import ray.train.torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fmnist-train")


def init_model() -> torch.nn.Module:
    model = VisionTransformer(
        image_size=28,
        patch_size=7,
        num_layers=6,
        num_heads=4,
        hidden_dim=128,
        mlp_dim=256,
        num_classes=10,
    )
    # FashionMNIST is grayscale, so patch embedding needs 1 input channel.
    model.conv_proj = torch.nn.Conv2d(
        in_channels=1,
        out_channels=128,
        kernel_size=7,
        stride=7,
    )
    return model


def get_data_loaders(batch_size: int):
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    data_dir = '/data'
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _reduce_metrics(train_loss_sum, train_batches, test_loss_sum, test_batches, device):
    metrics = torch.tensor(
        [train_loss_sum, train_batches, test_loss_sum, test_batches],
        device=device,
        dtype=torch.float32,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    train_loss_avg = (metrics[0] / metrics[1]).item() if metrics[1] > 0 else 0.0
    test_loss_avg = (metrics[2] / metrics[3]).item() if metrics[3] > 0 else 0.0
    return train_loss_avg, test_loss_avg


def train_loop_per_worker(config):
    device = ray.train.torch.get_device()
    model = init_model().to(device)
    model = ray.train.torch.prepare_model(model)

    optimizer = Adam(model.parameters(), lr=config["lr"])
    criterion = CrossEntropyLoss()

    train_loader, test_loader = get_data_loaders(config["batch_size"])
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    test_loader = ray.train.torch.prepare_data_loader(test_loader)

    ctx = ray.train.get_context()
    world_rank = ctx.get_world_rank()

    mlflow_active = False
    if world_rank == 0:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "fmnist-vit")
        mlflow.set_experiment(experiment)
        run_name = ctx.get_experiment_name() or f"fmnist-vit-{uuid.uuid4().hex[:6]}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(config)
        mlflow_active = True

    try:
        for epoch in range(config["epochs"]):
            model.train()
            train_loss_sum = 0.0
            train_samples = 0.0

            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * images.size(0)
                train_samples += images.size(0)

            model.eval()
            test_loss_sum = 0.0
            test_samples = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss_sum += loss.item() * images.size(0)
                    test_samples += images.size(0)

            train_loss_avg, test_loss_avg = _reduce_metrics(
                train_loss_sum, train_samples, test_loss_sum, test_samples, device
            )
            metrics = {"train_loss": train_loss_avg, "test_loss": test_loss_avg}

            ray.train.report(metrics)
            if mlflow_active:
                mlflow.log_metrics(metrics, step=epoch)
                logger.info("epoch=%s train_loss=%.4f test_loss=%.4f", epoch, train_loss_avg, test_loss_avg)

        if mlflow_active:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save = model_to_save.to("cpu").eval()
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "model.pt")
                torch.save(model_to_save.state_dict(), model_path)
                # mlflow.log_artifact(model_path, artifact_path="model")
                mlflow.pytorch.log_model(model_to_save, name="fmnist-vit")
                run_id = mlflow.active_run().info.run_id
                logger.info("MLflow run_id=%s", run_id)
    finally:
        if mlflow_active:
            mlflow.end_run()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    num_workers = int(os.getenv("NUM_WORKERS", "2"))
    use_gpu = _env_bool("USE_GPU", False) and torch.cuda.is_available()
    epochs = int(os.getenv("EPOCHS", "5"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    lr = float(os.getenv("LR", "0.001"))

    train_loop_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }

    scaling_config = ray.train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
    )

    experiment_name = os.getenv("RAY_EXPERIMENT_NAME", f"fmnist-vit-{uuid.uuid4().hex[:6]}")
    run_config = ray.train.RunConfig(name=experiment_name)

    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    logger.info("Starting training: workers=%s, use_gpu=%s", num_workers, use_gpu)
    result = trainer.fit()
    logger.info("Training finished: %s", result)
