import torch
from torch import nn
from architectures.encoder import EncoderNetwork
from architectures.predictor import PredictorNetwork
from datasets.pusht_dataset import PushTDataset
from pathlib import Path
from torch.utils.data import DataLoader
from losses import SIGReg
from tqdm.auto import tqdm

def train(device,
          lr : float = 3e-4,
          seed : int = 42,
          lambd : float = 0.1,
          data_dir : Path = Path("/Users/kartikbudihal/leworld/data"),
          epochs : int = 200,
          batch_size : int = 512,
          window_size : int = 3,
          embedding_dim : int = 192):

    
    torch.manual_seed(seed)

    # loading and setting up the data 
    train_dataset = PushTDataset(data_dir=data_dir)

    val_dataset = PushTDataset(data_dir=data_dir, 
                               val=True)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=False)

    predictor = PredictorNetwork().to(device)
    encoder = EncoderNetwork().to(device)

    print(f"networks are on device: {device}")

    sig_reg = SIGReg().to(device)
    pred_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(params=list(predictor.parameters()) + list(encoder.parameters()), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch; {epoch}\n------------")

        # train loop
        train_loss = 0 # running loss sum to be averaged after each batch

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

        # test loop
        val_loss = 0

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

        print(f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")


        