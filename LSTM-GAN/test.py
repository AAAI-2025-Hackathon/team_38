from lstm_discriminator import LSTMDiscriminator
from lstm_generator import LSTMGenerator
from casualty import CausalConvGenerator
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

def denormalize(x):
    real_data=torch.tensor(np.array(data['USA']))
    ma=torch.max(real_data)
    mi=torch.min(real_data)
    return 0.5 * (x*ma - x*mi + ma + mi)

if __name__=='__main__':
    load_dotenv('.env')
    data=pd.read_csv("data.csv")
    data_dict={
        'DATE':np.array(data['DATE']).tolist()
    }
    df=pd.DataFrame(data=data_dict)
    print(df.head())
    weights=os.getenv("model_path")
    z_dim_c=100
    z_dim_l=100
    device=torch.device('cpu')
    seq_len=432
    dataset=BondsDataset([np.array(data['USA'])])
    #inp=input("Choose Generator type: ")
    inps=['lstm','casualty']
    for inp in inps:
        if inp=='lstm':
            fixed_noise=torch.randn(1,seq_len,z_dim_l,device=device)
            generator=LSTMGenerator(in_dim=z_dim_l,out_dim=1,hidden_dim=256,n_layers=3).to(device)
            generator.load_state_dict(torch.load(weights+"Run_2/model_6570.pt"))
        elif inp=='casualty':
            fixed_noise=torch.randn(1,seq_len,z_dim_c,device=device)
            generator=CausalConvGenerator(noise_size=z_dim_c,output_size=1,n_layers=8,n_channel=10,kernel_size=8,dropout=0.2)
            generator.load_state_dict(torch.load(weights+"Run_3/model_75000.pt"))
        generator.eval()

        fake=generator(fixed_noise)
        generated_plot=time_series_to_plot(denormalize(fake))

        fake=denormalize(fake)
        fake=fake.view(seq_len,)
        fake=fake.detach().numpy()
       
        df[inp]=fake
    df.to_csv('output.csv')