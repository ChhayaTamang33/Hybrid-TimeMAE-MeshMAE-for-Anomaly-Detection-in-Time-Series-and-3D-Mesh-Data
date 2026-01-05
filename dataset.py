# Load all CSV files ,normalize them [T,F], Using the standard format for custom dataset.py
import torch
import pandas as pd
import glob
import os

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, folder, device="cpu"):
        self.csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print("csv_files:", self.csv_files)
        self.device = device
        
        if len(self.csv_files) == 0:
            raise ValueError("Nor CSV files found!")
        
        self.index_map = []
        for files_idx, f in enumerate(self.csv_files):
            df = pd.read_csv(f)
            num_rows = len(df)
            for row in range(num_rows):
                self.index_map.append((files_idx, row))


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        file_idx, row_idx = self.index_map[idx]
        df = pd.read_csv(self.csv_files[file_idx])

        # row = coords + all T(K) columns
        row = df.iloc[row_idx]

        # extract coordinates
        xm, ym, zm= row["Xm"], row["Ym"], row["Zm"]

        # extract all temperature columns
        temps = row.filter(like="T (K)").values.astype("float32")

        # build time series shape: [T, 4]
        T = temps.shape[0]
        coords_repeated = torch.tensor([[xm, ym, zm]] * T, dtype=torch.float32)
        temps_t = torch.tensor(temps, dtype= torch.float32).unsqueeze(1)

        x = torch.cat([coords_repeated, temps_t], dim=1)
        return x.to(self.device)


        # df = pd.read_csv(self.files[idx])
        # print("df: ",df)
        # coords = df[['Xm','Ym','Zm']].values
        # temps = df.filter(like="T (K)").values   # all temperature columns

        # # full feature vector = concat coords + temps(t)
        # # temps shape: [N, T] → we treat each row as a sample
        # # each CSV = one sample, so we take ONE row
        # coords = coords.mean(axis=0)     # [3]
        # temps = temps[0]                 # [T]

        # x = torch.tensor(
        #     [[coords[0], coords[1], coords[2], t] for t in temps],
        #     dtype=torch.float32
        # )  # final shape: [T, 4]

        # return x
