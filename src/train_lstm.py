import os, argparse, math
import numpy as np
from pyspark.sql import SparkSession, functions as F
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

NYC_TZ = "America/New_York"
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(BASE, "data", "processed", "supervised")
OUT_DIR = os.path.join(BASE, "models", "lstm")

class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len=32):
        self.X, self.y, self.seq_len = X, y, seq_len
    def __len__(self): return len(self.X) - self.seq_len + 1
    def __getitem__(self, idx):
        sl = slice(idx, idx+self.seq_len)
        return torch.from_numpy(self.X[sl]).float(), torch.from_numpy(self.y[idx+self.seq_len-1:idx+self.seq_len]).float().squeeze(0)

class LSTMReg(nn.Module):
    def __init__(self, n_in, hid=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def train_one(h, Xtrain, ytrain, Xval, yval, seq_len, device, epochs=20, bs=256, lr=1e-3):
    model = LSTMReg(Xtrain.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE
    tr_ds, va_ds = SeqDataset(Xtrain,ytrain,seq_len), SeqDataset(Xval,yval,seq_len)
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False)
    best = math.inf; best_state=None
    for epoch in range(1, epochs+1):
        model.train()
        for xb,yb in tr_dl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); opt.step()
        # val
        model.eval(); mae=0; n=0
        with torch.no_grad():
            for xb,yb in va_dl:
                xb,yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mae += (pred - yb).abs().sum().item(); n += yb.numel()
        mae /= n
        print(f"[h{h}] epoch {epoch:02d} val_MAE={mae:.3f}")
        if mae < best: best = mae; best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,2,4")
    ap.add_argument("--train_end", default="2025-03-15 23:59:59")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    spark = (SparkSession.builder
             .appName("Train_LSTM")
             .config("spark.sql.session.timeZone", NYC_TZ)
             .getOrCreate())
    df = spark.read.parquet(IN_DIR).orderBy("ts")

    feature_cols = [c for c in df.columns
                    if c not in ["ts","trips_15m","is_train"] +
                               [f"y_tplus_{h}" for h in horizons]]

    # split
    train_df = df.where(F.col("ts") <= F.lit(args.train_end)).select(["ts"]+feature_cols+[f"y_tplus_{h}" for h in horizons])
    val_df   = df.where(F.col("ts") >  F.lit(args.train_end)).select(["ts"]+feature_cols+[f"y_tplus_{h}" for h in horizons])

    # to numpy (ordered)
    train_pd = train_df.toPandas()
    val_pd   = val_df.toPandas()

    # standardize features using train stats
    Xtr = train_pd[feature_cols].to_numpy(dtype=np.float32)
    Xva = val_pd[feature_cols].to_numpy(dtype=np.float32)
    mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0)+1e-6
    Xtr = (Xtr - mu)/sigma
    Xva = (Xva - mu)/sigma

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device if (args.device!="cuda" or torch.cuda.is_available()) else "cpu")

    for h in horizons:
        ytr = train_pd[f"y_tplus_{h}"].to_numpy(dtype=np.float32)
        yva = val_pd[f"y_tplus_{h}"].to_numpy(dtype=np.float32)
        model = train_one(h, Xtr, ytr, Xva, yva, args.seq_len, device, epochs=args.epochs)

        # quick eval (MAE/RMSE/WAPE)
        def eval_seq(X, y):
            ds = SeqDataset(X,y,args.seq_len)
            dl = DataLoader(ds, batch_size=512, shuffle=False)
            preds=[]; targs=[]
            model.eval()
            with torch.no_grad():
                for xb,yb in dl:
                    xb = xb.to(device)
                    pr = model(xb).cpu().numpy()
                    preds.append(pr); targs.append(yb.numpy())
            pred = np.concatenate(preds); tgt = np.concatenate(targs)
            mae = np.mean(np.abs(pred - tgt))
            rmse = np.sqrt(np.mean((pred - tgt)**2))
            wape = np.sum(np.abs(pred - tgt))/np.sum(np.abs(tgt)+1e-6)
            return mae, rmse, wape
        mae, rmse, wape = eval_seq(Xva, yva)
        print(f"[h{h}] VAL  MAE={mae:.3f}  RMSE={rmse:.3f}  WAPE={wape:.3f}")

        # save
        torch.save({"state_dict": model.state_dict(),
                    "mu": mu, "sigma": sigma,
                    "seq_len": args.seq_len,
                    "feature_cols": feature_cols},
                   os.path.join(OUT_DIR, f"lstm_h{h}.pt"))
        print(f"[OK] saved LSTM h{h} → {os.path.join(OUT_DIR, f'lstm_h{h}.pt')}")

    spark.stop()

if __name__ == "__main__":
    main()

