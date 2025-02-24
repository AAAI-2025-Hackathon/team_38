import pandas as pd
import numpy as np

df_gan = pd.read_csv("MAE_metric/gan_llm_output.csv")
df_rl = pd.read_csv("MAE_metric/rl_llm_output.csv")
df_actual = pd.read_csv("data/Yields/bonds_10yr_data.csv")

df_actual_trimmed = df_actual[84:121]


# Get Maes for each bond type
keys = df_actual_trimmed.columns.tolist()
keys = keys[1:]

models = ['gan', 'rl']
for m in models:
    if m == 'gan':
        ds = df_gan
    else:
        ds = df_rl
    for i in keys:
        mae = np.mean(np.abs(np.array(ds[i])-np.array(df_actual_trimmed[i])))
        print(f"Mean absolute error for {m} in bond {i}: {mae}")
