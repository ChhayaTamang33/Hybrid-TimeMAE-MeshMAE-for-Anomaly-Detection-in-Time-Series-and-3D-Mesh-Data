# train.py
import torch
from tqdm import tqdm
from torch import nn, optim

def pretrain(model, loader, args):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    for epoch in range(args.num_epoch_pretrain):
        total_loss = total_mse = total_ce = 0

        for x in tqdm(loader):
            x = x.to(args.device)
            out = model.pre_train_forward(x)

            rep_tgt = out["rep_mask_target"]
            rep_pred = out["rep_mask_prediction"]
            token_logits = out["token_logits"]
            token_tgt = out["token_targets"]

            mse = mse_fn(rep_pred, rep_tgt)

            B, M, K = token_logits.shape
            ce = ce_fn(
                token_logits.view(B*M, K),
                token_tgt.view(B*M)
            )

            loss = args.beta * mse + args.alpha * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.momentum_update()

            total_loss += loss.item()
            total_mse  += mse.item()
            total_ce   += ce.item()

        print(f"[{epoch+1}] Loss={total_loss:.4f} MSE={total_mse:.4f} CE={total_ce:.4f}")

    print("Pretraining finished.")
