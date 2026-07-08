import argparse
import contextlib
import os
import zipfile
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from architectures.encoder import EncoderNetwork
from architectures.predictor import PredictorNetwork
from datasets.pusht_dataset import PushTDataset
from losses import SIGReg


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(device,
          data_dir: Path,
          log_dir: Path,
          save_dir: Path,
          frame_skip: int,
          lr: float,
          seed: int,
          lambd: float,
          epochs: int,
          batch_size: int,
          num_workers: int,
          persistent_workers: bool,
          pin_memory: bool,
          window_size: int,
          embedding_dim: int,
          action_dim: int,
          pred_nheads: int,
          dropout: float,
          pred_nlayers: int,
          patch_size: int,
          enc_nheads: int,
          enc_nlayers: int,
          enc_npatches: int,
          knots: int,
          num_proj: int,
          optimizer_betas: tuple[float, float],
          optimizer_weight_decay: float):

    torch.manual_seed(seed)
    if isinstance(device, str):
        on_cuda = device.startswith("cuda")
    else:
        on_cuda = device.type == "cuda"

    if on_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

    save_dir.mkdir(parents=True, exist_ok=True)

    # loading and setting up the data
    train_dataset = PushTDataset(data_dir=data_dir,
                                 frame_skip=frame_skip,
                                 window=window_size,
                                 val=False)

    val_dataset = PushTDataset(data_dir=data_dir,
                               frame_skip=frame_skip,
                               window=window_size,
                               val=True)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  persistent_workers=persistent_workers,
                                  pin_memory=pin_memory)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                persistent_workers=persistent_workers,
                                pin_memory=pin_memory)

    # network definitions
    predictor = PredictorNetwork(d_model=embedding_dim,
                                 nheads=pred_nheads,
                                 d_action=action_dim,
                                 dropout=dropout,
                                 nlayers=pred_nlayers,
                                 window=window_size).to(device)

    encoder = EncoderNetwork(patch=patch_size,
                             nheads=enc_nheads,
                             nlayers=enc_nlayers,
                             d_model=embedding_dim,
                             npatches=enc_npatches).to(device)

    print(f"networks are on device: {device}")

    # defining the losses and the optimizer
    sig_reg = SIGReg(knots=knots,
                     num_proj=num_proj).to(device)

    pred_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(params=list(predictor.parameters()) + list(encoder.parameters()),
                                  lr=lr,
                                  betas=optimizer_betas,
                                  weight_decay=optimizer_weight_decay)

    # initializing the logger
    writer = SummaryWriter(log_dir=log_dir)

    # results dict
    results = {"train_loss": [],
               "val_loss": []}

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if on_cuda
        else contextlib.nullcontext()
    )
    non_blocking = on_cuda and pin_memory

    # training loop
    for epoch in tqdm(range(epochs)):
        print(f"Epoch; {epoch}\n------------")

        train_loss = 0  # running loss sum to be averaged after each batch for logging

        encoder.train()
        predictor.train()
        for batch, (X, A, Y) in enumerate(train_dataloader):
            X = X.to(device, non_blocking=non_blocking)
            A = A.to(device, non_blocking=non_blocking)
            Y = Y.to(device, non_blocking=non_blocking)

            with autocast_ctx: # placing the forward passes into the AMP for speedups
                embeddings = encoder(X)
                pred_embeddings = predictor(embeddings, A)
                target_embeddings = encoder(Y)

                pred_loss = pred_loss_fn(pred_embeddings, target_embeddings)
                reg_loss = sig_reg.forward(embeddings.transpose(0, 1))

                loss = pred_loss + lambd * reg_loss
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss /= len(train_dataloader)

        # validation
        val_loss = 0  # running sum for logging

        encoder.eval()
        predictor.eval()

        with torch.inference_mode():
            for (X, A, Y) in val_dataloader:
                X = X.to(device, non_blocking=non_blocking)
                A = A.to(device, non_blocking=non_blocking)
                Y = Y.to(device, non_blocking=non_blocking)

                with autocast_ctx:
                    embeddings = encoder(X)
                    pred_embeddings = predictor(embeddings, A)
                    target_embeddings = encoder(Y)

                    pred_loss = pred_loss_fn(pred_embeddings, target_embeddings)
                    reg_loss = sig_reg.forward(embeddings.transpose(0, 1))

                    loss = pred_loss + lambd * reg_loss
                val_loss += loss.item()

            val_loss /= len(val_dataloader)

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_loss,
                                            "validation_loss": val_loss},
                           global_step=epoch)

        # saving the models into a single zip after each epoch
        encoder_path = save_dir / "encoder_weights.pth"
        predictor_path = save_dir / "predictor_weights.pth"
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(predictor.state_dict(), predictor_path)

        zip_path = save_dir / f"epoch_{epoch}_model.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(encoder_path, 'encoder_weights.pth')
            zipf.write(predictor_path, 'predictor_weights.pth')

        os.remove(encoder_path)
        os.remove(predictor_path)

        print(f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    writer.close()

    return results


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "default.yaml",
        help="Path to the training config YAML file.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run. Checkpoints save to <save_dir>/<run_name>/.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    paths = config["paths"]
    training = config["training"]
    model = config["model"]
    loss = config["loss"]
    optimizer_cfg = config["optimizer"]

    data_dir = PROJECT_DIR / paths["data_dir"]
    log_dir = PROJECT_DIR / paths["log_dir"]
    save_dir = PROJECT_DIR / paths["save_dir"] / args.run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = train(
        device=device,
        data_dir=data_dir,
        log_dir=log_dir,
        save_dir=save_dir,
        frame_skip=training["frame_skip"],
        lr=training["lr"],
        seed=training["seed"],
        lambd=training["lambd"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        persistent_workers=training["persistent_workers"],
        pin_memory=training["pin_memory"],
        window_size=training["window_size"],
        embedding_dim=model["embedding_dim"],
        action_dim=model["action_dim"],
        pred_nheads=model["pred_nheads"],
        dropout=model["dropout"],
        pred_nlayers=model["pred_nlayers"],
        patch_size=model["patch_size"],
        enc_nheads=model["enc_nheads"],
        enc_nlayers=model["enc_nlayers"],
        enc_npatches=model["enc_npatches"],
        knots=loss["knots"],
        num_proj=loss["num_proj"],
        optimizer_betas=tuple(optimizer_cfg["betas"]),
        optimizer_weight_decay=optimizer_cfg["weight_decay"],
    )
