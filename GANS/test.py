from lstm_discriminator import LSTMDiscriminator
from lstm_generator import LSTMGenerator
from GANS.causalty import CausalConvGenerator
import torch 
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.functional import normalize
from dataset import BondsDataset
import torchvision
import wandb
from torch import nn 
import pandas as pd
import os 
from plot import time_series_to_plot,time_series_to_plot_real
from dotenv import load_dotenv

def denormalize(x,inp):
    if inp=="US":
        real_data=torch.tensor(np.array(data[inp+"_10Y_Yield"]))
    else:
        real_data=torch.tensor(np.array(data[inp+"_Bond_Yield"]))
    ma=torch.max(real_data)
    mi=torch.min(real_data)
    print(ma)
    return 0.5 * (x*ma - x*mi + ma + mi)

if __name__=='__main__':
    load_dotenv('.env')
    data=pd.read_csv("bonds_10yr_data.csv")
    conditions=pd.read_csv("final-data.csv")
    variable_names=conditions.columns.values.tolist()
    data_dict={
        'Date':np.array(data['Date']).tolist()
    }
    df=pd.DataFrame(data=data_dict)
    weights=os.getenv("model_path")
    z_dim_c=12
    device=torch.device('cpu')
    seq_len=121
    vals=list()

    for col in variable_names[1:]:
        conditions[col]=conditions[col].fillna(conditions[col].mean())

    for col in variable_names[1:]:
        feat=np.array(conditions[col])
        vals.append(feat)
    vals_tensor=torch.tensor(np.array(vals))
    vals_tensor=normalize(vals_tensor,p=2,dim=1)
    vals_tensor=vals_tensor.view(1,seq_len,12)
    
    #inp=input("Choose Generator type: ")
    inps=['AAA','BAA','US','Junk']
    for inp in inps:
        fixed_noise=torch.add(torch.randn(1,seq_len,z_dim_c,device=device),vals_tensor)
        fixed_noise=fixed_noise.float()
        generator=CausalConvGenerator(noise_size=z_dim_c,output_size=1,n_layers=8,n_channel=10,kernel_size=8,dropout=0.2)
        generator.load_state_dict(torch.load(weights+f"{inp}/model_4000.pt"))
        generator.eval()

        fake=generator(fixed_noise)
    
        fake=denormalize(fake,inp)
        fake=fake.view(seq_len,)
        fake=fake.detach().numpy()
       
        df[inp]=fake
    df.to_csv('output.csv')