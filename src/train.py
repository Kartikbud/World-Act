import torch
from torch import nn
from architectures.encoder import EncoderNetwork
from architectures.predictor import PredictorNetwork
from datasets.pusht_dataset import PushTDataset
from pathlib import Path
from torch.utils.data import DataLoader
from losses import SIGReg
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

import os
import zipfile

def train(device,
          data_dir : Path,
          log_dir : Path,
          save_dir : Path,  
          frame_skip: int = 5,
          lr : float = 3e-4,
          seed : int = 42,
          lambd : float = 0.1,
          epochs : int = 200,
          batch_size : int = 512,
          window_size : int = 3,
          embedding_dim : int = 192,
          action_dim : int = 2,
          pred_nheads : int = 16,
          dropout : float = 0.1,
          pred_nlayers : int = 6,
          patch_size : int = 14,
          enc_nheads : int = 3,
          enc_nlayers : int = 12,
          enc_npatches : int = 256,
          knots : int = 17,
          num_proj : int = 1024):

    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
                                       shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=False)

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
                                 betas=(0.9, 0.95), # these are from the paper
                                 weight_decay=0.05)

    # initializing the logger
    writer = SummaryWriter(log_dir=log_dir)

    # results dict
    results = {"train_loss" : [],
               "val_loss" : []}

    # training loop
    for epoch in tqdm(range(epochs)):
        print(f"Epoch; {epoch}\n------------")

        train_loss = 0 # running loss sum to be averaged after each batch for logging

        encoder.train()
        predictor.train()
        for batch, (X, A, Y) in enumerate(train_dataloader):
            X = X.to(device)
            A = A.to(device)
            Y = Y.to(device)

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
        val_loss = 0 # running sum for logging

        encoder.eval()
        predictor.eval()

        with torch.inference_mode():
            for (X, A, Y) in val_dataloader:
                X = X.to(device)
                A = A.to(device)
                Y = Y.to(device)
                
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
                           tag_scalar_dict={"train_loss" : train_loss,
                                            "validation_loss" : val_loss},
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


if __name__=="__main__":
    
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent    
    
    data_dir = PROJECT_DIR / "data"
    log_dir = PROJECT_DIR / "logs" 
    save_dir = PROJECT_DIR / "models"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = train(device=device,
                    data_dir=data_dir,
                    log_dir=log_dir,
                    save_dir=save_dir)

     


    