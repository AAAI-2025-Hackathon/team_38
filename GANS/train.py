from lstm_discriminator import LSTMDiscriminator
from lstm_generator import LSTMGenerator
from GANS.causalty import CausalConvDiscriminator,CausalConvGenerator
import torch 
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import numpy as np
from dataset import BondsDataset
import torchvision
import wandb
from torch import nn 
import pandas as pd
import os 
from plot import time_series_to_plot,time_series_to_plot_real
from dotenv import load_dotenv

def training_loop():
    for epoch in range(5000):
        generator.train()
        discriminator.train()
        for step,data in enumerate(dataloader):
            discriminator.zero_grad()
            real=data.to(device)
            real=real.view(real.size(0),real.size(1),1)
            
            batch_size,seq_len=real.size(0),real.size(1)
            label=torch.full((batch_size,seq_len,1),real_label,device=device)

            output=discriminator(real)
            errd_real=loss_function(output,label.float())

            errd_real.backward()
            D_x=output.mean().item()

            noise=torch.add(torch.randn(batch_size,seq_len,z_dim,device=device),vals_tensor)
            noise=noise.float()
            fake=generator(noise)

            label.fill_(fake_label)
            output=discriminator(fake)
            # print(output)
            errd_fake=loss_function(output,label.float())
            errd_fake.backward()
            D_G_z1=output.mean().item()
            errd=errd_fake+errd_real
            optimizer_d.step()

            generator.zero_grad()
            label.fill_(real_label)
            output=discriminator(fake)
            errG=loss_function(output,label.float())
            D_G_z2=output.mean().item()

            optimizer_g.step()

         #Plotting
        with torch.no_grad():
            generator.eval()
            if (epoch+1)%10==0:
                fake=generator(fixed_noise)
                real_plot=time_series_to_plot_real(dataset.denormalize(real))
                generated_plot=time_series_to_plot(dataset.denormalize(fake))
                wandb.log(
                    {"Prediction":generated_plot,
                    "Actual":real_plot}
                )
                torch.save(generator.state_dict(),os.getenv("model_path")+f"/BAA_W/model_{epoch+1}.pt")


if __name__=='__main__':
    data=pd.read_csv("bonds_10yr_weekly_data.csv")
    conditions=pd.read_csv("economic_data_final.csv")
    variable_names=conditions.columns.values.tolist()
    run_id=3
    load_dotenv('.env')
    dataset=BondsDataset([np.array(data['BAA_Bond_Yield'])])

    dataloader=DataLoader(dataset,batch_size=1)
    device=torch.device("cpu")

    seq_len=dataset[0].size(0)
    z_dim=12
    # generator=LSTMGenerator(in_dim=z_dim,out_dim=1,hidden_dim=256,n_layers=3).to(device)
    # discriminator=LSTMDiscriminator(in_dim=1,hidden_dim=256,n_layers=3).to(device)

    discriminator=CausalConvDiscriminator(input_size=1,n_layers=8,n_channel=10,kernel_size=8,dropout=0)
    generator=CausalConvGenerator(noise_size=z_dim,output_size=1,n_layers=8,n_channel=10,kernel_size=8,dropout=0.2)

    loss_function=nn.BCELoss()

    fixed_noise=torch.randn(1,seq_len,z_dim,device=device)
    real_label=1
    fake_label=0

    optimizer_g=torch.optim.Adam(generator.parameters(),lr=2e-4)
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=2e-4)

    fixed_noise = torch.randn(1, seq_len, z_dim, device=device)

    wandb.init(project="Bond Modelling GANs")
    #handlenan
    for col in variable_names[1:]:
        conditions[col]=conditions[col].fillna(conditions[col].mean())

    #Create(steps,12) matrix
    # nan_counts=conditions.isna().sum()
    # print(nan_counts)

    vals=list()
    for col in variable_names[1:]:
        feat=np.array(conditions[col])
        vals.append(feat)
    vals_tensor=torch.tensor(np.array(vals))
    vals_tensor=normalize(vals_tensor,p=2,dim=1)
    vals_tensor=vals_tensor.view(1,seq_len,12)
    # print(vals_tensor[:,:,1])
    
    training_loop()