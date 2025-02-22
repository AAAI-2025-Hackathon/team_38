import json
import matplotlib.pyplot as plt

# Load JSON files
file_paths = {
    "Actual": "Profit_Loss/Prices/bond_prices_actual.json",
    "RL": "Profit_Loss/Prices/bond_prices_rl.json",
    "GAN": "Profit_Loss/Prices/bond_prices.json"
}
sources = ['Actual', 'RL', 'GAN']
# Dictionary to store data
bond_data = {}

# Load data from JSON files
for key, path in file_paths.items():
    with open(path, "r") as f:
        bond_data[key] = json.load(f)

# Extract bond types from one of the datasets
bond_types = list(next(iter(bond_data['Actual'].values())).keys())

# Generate plots
plt.figure(figsize=(12, 6))
for bond_type in bond_types:

    for key in bond_data.keys():
        # Extract dates and values
        dates = sorted(bond_data[key].keys())
        values = [bond_data[key][date].get(
            bond_type, None) for date in dates]

        # Plot the data
        plt.plot(dates, values, label=key)

    plt.xlabel("Date")
    plt.ylabel("Bond Yield")
    plt.title(f"Price Dip for {bond_type}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.show()
