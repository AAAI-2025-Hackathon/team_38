from lstm_discriminator import LSTMDiscriminator
from lstm_generator import LSTMGenerator
import torch 
from torch.utils.data import DataLoader
import numpy as np
from dataset import BondsDataset
import torchvision
import wandb
from torch import nn 
import pandas as pd
import os 
from plot import time_series_to_plot,time_series_to_plot_real
from dotenv import load_dotenv


if __name__=='__main__':
    load_dotenv('.env')
    weights=os.getenv("model_path")
    data=pd.read_csv("final-data.csv")
    z_dim=16
    device=torch.device('cpu')
    seq_len=121
    fixed_noise=torch.randn(1,seq_len,z_dim,device=device)
    dataset=BondsDataset([np.array(data['US_10Y_Yield'])])
    generator=LSTMGenerator(in_dim=z_dim,out_dim=1,hidden_dim=256,n_layers=3).to(device)
    generator.load_state_dict(torch.load(weights+"Run_2/model_6570.pt"))
    generator.eval()

    fake=generator(fixed_noise)
    generated_plot=time_series_to_plot(fake)
