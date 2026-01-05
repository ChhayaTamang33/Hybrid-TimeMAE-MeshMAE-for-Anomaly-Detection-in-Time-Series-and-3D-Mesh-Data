# main.py
import torch
from torch.utils.data import DataLoader
from args import args
from dataset import TimeSeriesDataset
from models.timemae import TimeMAE
from train import pretrain
from anomaly import anomaly_score

def main():

    # load the training dataset
    print("Loading the dataset!!")
    train_ds = TimeSeriesDataset(args.data_folder)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=4)
    print("Training samples: ",len(train_ds))    
    
    # model
    model = TimeMAE(args).to(args.device)

    # pretrain
    print("Pre-training started!!")
    pretrain(model, train_loader, args)
    torch.save(model.state_dict(), f"{args.save_path}/model.pth")
    print("Model Saved!")

    # load the inference dataset
    print("Loading the inference dataset!")    
    inference_ds = TimeSeriesDataset(args.data_folder_inference)
    
    # example inference on first file
    x = inference_ds[0].unsqueeze(0).to(args.device) # add batch dimension [1, T, F]
    score = anomaly_score(model, x)

    print("Anomaly score:", score.detach().cpu().numpy())

if __name__ == "__main__":
    main()
