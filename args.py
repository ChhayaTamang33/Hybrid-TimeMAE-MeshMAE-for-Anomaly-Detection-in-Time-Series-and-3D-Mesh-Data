# args.py
import torch
from types import SimpleNamespace

args = SimpleNamespace(
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu",

    # model sizes
    embed_dim = 64
    attn_heads = 4,
    d_ffn = 256,
    dropout = 0.1,

    # windowing
    window_size = 16,
    num_features = 4,     # [Xm, Ym, Zm, Temp]
    max_len = 64,

    # masking / codebook
    mask_ratio = 0.5,
    codebook_size = 128,

    # architecture depth
    encoder_layers = 3,
    decoder_layers = 2,

    # training
    lr = 1e-4,
    alpha = 1.0,
    beta  = 2.0,
    momentum = 0.996,
    num_epoch_pretrain = 5,
    batch_size = 32,

    # IO
    data_folder = "./data",
    save_path = "./checkpoints",

    # Inference
    data_folder_inference = "./data/test_set"
)
