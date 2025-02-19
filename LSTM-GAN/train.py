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

def training_loop():
    for epoch in range(150000):
        for step,data in enumerate(dataloader):
            discriminator.zero_grad()
            real=data.to(device)

            batch_size,seq_len=real.size(0),real.size(1)
            label=torch.full((batch_size,seq_len,1),real_label,device=device)

            output=discriminator(real)
            errd_real=loss_function(output,label.float())

            errd_real.backward()
            D_x=output.mean().item()

            noise=torch.randn(batch_size,seq_len,z_dim,device=device)
            fake=generator(noise)

            label.fill_(fake_label)
            output=discriminator(fake)
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
        if (epoch+1)%10==0:
            fake=generator(fixed_noise)
            real_plot=time_series_to_plot_real(dataset.denormalize(real))
            generated_plot=time_series_to_plot(dataset.denormalize(fake))
            wandb.log(
                {"Prediction":generated_plot,
                 "Actual":real_plot}
            )
            torch.save(generator.state_dict(),os.getenv("model_path")+f"/Run_{run_id}/model_{epoch}.pt")


if __name__=='__main__':
    data=pd.read_csv("final-data.csv")
    run_id=1
    load_dotenv('.env')
    dataset=BondsDataset([np.array(data['US_10Y_Yield'])])

    dataloader=DataLoader(dataset,batch_size=1)
    device=torch.device("cpu")

    seq_len=dataset[0].size(0)
    z_dim=16
    generator=LSTMGenerator(in_dim=z_dim,out_dim=1,hidden_dim=256,n_layers=3).to(device)
    discriminator=LSTMDiscriminator(in_dim=1,hidden_dim=256,n_layers=3).to(device)

    loss_function=nn.BCELoss()

    fixed_noise=torch.randn(1,seq_len,z_dim,device=device)
    real_label=1
    fake_label=0

    optimizer_g=torch.optim.Adam(generator.parameters(),lr=2e-4)
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=2e-4)

    fixed_noise = torch.randn(1, seq_len, z_dim, device=device)

    wandb.init(project="Bond Modelling GANs")

    training_loop()