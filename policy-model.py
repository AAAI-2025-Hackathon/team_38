import os
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib
import wandb

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VARIABLES = [
    'Inflation_Rate', 'GDP_Growth', 'Unemployment_Rate', 'Fed_Funds_Rate',
    'Money_Supply', 'Consumer_Confidence', 'S&P_500',
    'Crude_Oil', 'Gold', 'US_Dollar_Index', 'INR_USD', 'VIX'
]

try:
    data_path = "final-data.csv"
    df = pd.read_csv(data_path)
    
    missing_cols = [col for col in VARIABLES if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in data: {missing_cols}")
        raise ValueError(f"Data file missing required columns: {missing_cols}")
    
    df = df[VARIABLES].dropna()
    
    state_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = state_scaler.fit_transform(df.values)
    initial_state = scaled_data[-1].astype(np.float32)
    
    joblib.dump(state_scaler, "market_scaler.pkl")
    
    logger.info(f"Loaded state data from {data_path} with shape: {df.shape}")
    
    volatility_array = np.std(scaled_data, axis=0)
    time_trends_array = np.mean(np.diff(scaled_data, axis=0), axis=0)
    
    logger.info(f"Computed volatility: {volatility_array}")
    logger.info(f"Computed time trends: {time_trends_array}")
    
except FileNotFoundError:
    logger.error(f"{data_path} not found. Please ensure the file exists.")
    raise
except Exception as e:
    logger.error(f"Error loading state data: {str(e)}")
    raise

try:
    bond_data_path = "bonds_10yr_data.csv"
    bond_df = pd.read_csv(bond_data_path)  
    raw_bond_yields = bond_df["BAA_Bond_Yield"].dropna().values.reshape(-1, 1)
    
    bond_scaler = MinMaxScaler(feature_range=(-1, 1))
    bond_yields_scaled = bond_scaler.fit_transform(raw_bond_yields).ravel()
    
    logger.info(f"Loaded and scaled bond yield data from {bond_data_path} with shape: {raw_bond_yields.shape}")
    
except Exception as e:
    logger.error(f"Error loading bond yields: {str(e)}")
    raise

# Market prediction environment
class MarketPredictionEnv(gym.Env):
    """
    Environment where:
      - State: 12 economic variables (scaled to [-1, 1]).
      - Action: A single value, the predicted bond yield (scaled to [-1, 1]).
      - Reward: Negative MSE between predicted bond yield and ground-truth bond yield,
                plus an economic coherence bonus/penalty.
      - Done: After 'prediction_steps' steps corresponding to the horizon.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, initial_state, prediction_horizon='1y', bond_yields_scaled=None,
                 volatility_array=None, time_trends_array=None):
        super().__init__()
        
        # Determine prediction steps based on the horizon format.
        if prediction_horizon == '1y':
            self.prediction_steps = 12
        elif prediction_horizon == '5y':
            self.prediction_steps = 60
        elif prediction_horizon == '10y':
            self.prediction_steps = 120
        else:
            try:
                # For custom horizons like '120M' or '6M'
                if prediction_horizon.endswith('M'):
                    self.prediction_steps = int(prediction_horizon[:-1])
                else:
                    raise ValueError
            except:
                raise ValueError(f"Invalid prediction horizon: {prediction_horizon}")
        
        self.horizon = prediction_horizon
        self.current_step = 0
        self.num_variables = len(VARIABLES)
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_variables,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.initial_state = np.clip(np.array(initial_state, dtype=np.float32), -1.0, 1.0)
        self.state = self.initial_state.copy()
        
        if bond_yields_scaled is None or len(bond_yields_scaled) < self.prediction_steps:
            raise ValueError("Insufficient scaled bond yield data for the specified horizon.")
        self.bond_yields_scaled = bond_yields_scaled
        
        if volatility_array is None:
            raise ValueError("Volatility array must be provided.")
        if time_trends_array is None:
            raise ValueError("Time trends array must be provided.")
        self.volatility = volatility_array
        self.time_trends = time_trends_array
        
        self.create_economic_relationships()
        
        self.history = [self.initial_state]
    
    def create_economic_relationships(self):
        """
        Compute the relationship matrix based on pairwise Pearson correlations
        from the historical data (using the global DataFrame 'df').
        """
        correlation_matrix = df[VARIABLES].corr().values
        np.fill_diagonal(correlation_matrix, 0)
        noise = np.random.normal(0, 0.02, size=correlation_matrix.shape)
        self.relationship_matrix = correlation_matrix + noise
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self.initial_state.copy()
        self.history = [self.initial_state]
        return self.state, {}
    
    def simulate_market_dynamics(self, state):
        """
        Compute market forces using the relationship matrix and add a time-dependent trend.
        """
        market_forces = np.zeros_like(state)
        for i in range(self.num_variables):
            for j in range(self.num_variables):
                market_forces[i] += state[j] * self.relationship_matrix[i, j] * 0.1
        
        trend_scale = 1.0
        trend_effect = self.time_trends * trend_scale * (self.current_step / self.prediction_steps)
        return market_forces + trend_effect
    
    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        market_dynamics = self.simulate_market_dynamics(self.state)
        
        horizon_factor = 1.0
        if self.horizon == '5y':
            horizon_factor = 1.5
        elif self.horizon == '10y':
            horizon_factor = 2.0
        
        step_factor = np.sqrt(self.current_step / 12.0)
        noise = np.random.normal(0, self.volatility * (1 + np.abs(self.state)) * horizon_factor * step_factor)
        
        next_state = np.clip(
            self.state + market_dynamics + noise,
            self.observation_space.low,
            self.observation_space.high
        )
        
        true_bond_yield_scaled = self.bond_yields_scaled[self.current_step - 1]
        predicted_bond_yield_scaled = action[0]
        prediction_error = (predicted_bond_yield_scaled - true_bond_yield_scaled) ** 2
        
        # Coherence score based on simple heuristics.
        coherence_score = self.calculate_bond_yield_coherence(predicted_bond_yield_scaled, self.state)
        
        reward = -prediction_error + 0.3 * coherence_score
        
        self.state = next_state.astype(np.float32)
        self.history.append(self.state)
        
        terminated = (self.current_step >= self.prediction_steps)
        truncated = False
        
        info = {
            "step": self.current_step,
            "prediction_error": prediction_error,
            "coherence_score": coherence_score,
            "horizon": self.horizon
        }
        return self.state, reward, terminated, truncated, info
    
    def calculate_bond_yield_coherence(self, predicted_yield_scaled, state):
        """
        Adjust coherence based on inflation and Fed Funds Rate.
        """
        coherence = 0.0
        if state[0] > 0.5 and predicted_yield_scaled < 0.0:
            coherence -= 0.5
        if state[3] > 0.5 and predicted_yield_scaled < 0.0:
            coherence -= 0.5
        if state[7] > 0.7 and predicted_yield_scaled > 0.5:
            coherence -= 0.3
        return coherence
    
    def render(self, mode='human'):
        if mode != 'human':
            raise ValueError(f"Unsupported render mode: {mode}")
        loaded_scaler = joblib.load("market_scaler.pkl")
        readable_state = loaded_scaler.inverse_transform(self.state.reshape(1, -1))[0]
        print(f"\nStep {self.current_step} ({self.horizon} horizon)")
        for i, var in enumerate(VARIABLES):
            print(f"{var}: {readable_state[i]:.4f}")

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Timesteps")
    
    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True
    
    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                episode_info = info["episode"]
                wandb.log({
                    "episode_reward": episode_info.get("r", 0),
                    "episode_length": episode_info.get("l", 0)
                })
        return True

def save_predictions(predictions, true_bond_yields_scaled, bond_scaler, horizon, output_dir="market_predictions"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    predicted_scaled = predictions[:, 0].reshape(-1, 1)
    predicted_bond_yields = bond_scaler.inverse_transform(predicted_scaled).ravel()
    
    true_bond_yields_scaled = true_bond_yields_scaled.reshape(-1, 1)
    true_bond_yields = bond_scaler.inverse_transform(true_bond_yields_scaled).ravel()
    
    steps = [f"Month {i+1}" for i in range(len(predictions))]
    df_results = pd.DataFrame({
        'Predicted_Bond_Yield': predicted_bond_yields,
        'Actual_Bond_Yield': true_bond_yields
    }, index=steps)
    filename = os.path.join(output_dir, f"bond_yield_results_{horizon}_{timestamp}.csv")
    df_results.to_csv(filename)
    logger.info(f"Bond yield prediction results for {horizon} horizon saved to {filename}")
    wandb.save(filename)

# Updated plot_predictions with transition effect.
def plot_predictions(predictions, true_bond_yields_scaled, bond_scaler, horizon, output_dir="market_predictions"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inverse-transform the scaled predictions and true bond yields.
    predicted_scaled = predictions[:, 0].reshape(-1, 1)
    predicted_bond_yields = bond_scaler.inverse_transform(predicted_scaled).ravel()
    
    true_bond_yields_scaled = true_bond_yields_scaled.reshape(-1, 1)
    actual_bond_yields = bond_scaler.inverse_transform(true_bond_yields_scaled).ravel()
    
    # Create a transition effect by prepending the initial (historical) bond yield to both series.
    initial_bond_yield = actual_bond_yields[0]
    actual_with_history = np.concatenate(([initial_bond_yield], actual_bond_yields))
    predicted_with_history = np.concatenate(([initial_bond_yield], predicted_bond_yields))
    
    steps = np.arange(len(actual_with_history))
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, actual_with_history, label='Actual Bond Yield', color='red', linestyle='--', marker='o')
    plt.plot(steps, predicted_with_history, label='Predicted Bond Yield', color='blue', linestyle='-', marker='o')
    
    # Mark the transition point (step 1) between historical and forecasted data.
    plt.axvline(x=1, color='gray', linestyle='--', label='Transition Point')
    
    errors = np.abs(predicted_with_history - actual_with_history)
    threshold = 0.2  
    outlier_indices = np.where(errors > threshold)[0]
    plt.scatter(steps[outlier_indices], predicted_with_history[outlier_indices],
                color='black', label='High Error Points', zorder=5)
    
    ci = 0.02  
    plt.fill_between(steps,
                     predicted_with_history - ci,
                     predicted_with_history + ci,
                     color='blue', alpha=0.2, label='Confidence Interval (±0.02)')
    
    plt.xlabel('Time Step (Months)')
    plt.ylabel('Bond Yield (%)')
    plt.title(f'Bond Yield Prediction vs Actual with Transition ({horizon} horizon)')
    plt.legend()
    plt.grid(True)
    filename = os.path.join(output_dir, f"bond_yield_{horizon}_{timestamp}.png")
    plt.savefig(filename)
    plt.close()
    wandb.log({f"Bond Yield Prediction ({horizon})": wandb.Image(filename)})
    logger.info(f"Bond yield prediction plot for {horizon} horizon saved to {filename}")

# Train the model for a given prediction horizon.
def train_model_for_horizon(horizon, initial_state, bond_yields_scaled, volatility_array, time_trends_array, total_timesteps=6500):
    logger.info(f"Training model for {horizon} horizon")
    
    env = MarketPredictionEnv(
        initial_state=initial_state,
        prediction_horizon=horizon,
        bond_yields_scaled=bond_yields_scaled,
        volatility_array=volatility_array,
        time_trends_array=time_trends_array
    )
    monitored_env = Monitor(env)
    
    log_path = f"logs/{horizon}/"
    os.makedirs(log_path, exist_ok=True)
    new_logger = configure(log_path, ["stdout", "csv"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = SAC(
        "MlpPolicy",
        monitored_env,
        learning_rate=wandb.config.learning_rate,
        buffer_size=wandb.config.buffer_size,
        batch_size=wandb.config.batch_size,
        tau=wandb.config.tau,
        gamma=wandb.config.gamma,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        verbose=1,
        device=device,
        policy_kwargs={"net_arch": {"pi": wandb.config.net_arch, "qf": wandb.config.net_arch}}
    )
    
    model.set_logger(new_logger)
    
    tqdm_callback = TqdmCallback(total_timesteps=total_timesteps)
    wandb_callback = WandbCallback()
    callback = CallbackList([tqdm_callback, wandb_callback])
    
    try:
        logger.info(f"Starting training for {horizon} horizon...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        logger.info(f"Training complete for {horizon} horizon.")
        
        model_path = f"models/market_model_{horizon}.zip"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        wandb.save(model_path) 
        logger.info(f"Model saved to {model_path}")
        return model
    except Exception as e:
        logger.error(f"Training failed for {horizon} horizon: {str(e)}")
        raise

# Generate predictions.
def generate_predictions(model, env, horizon):
    predictions = []
    obs, _ = env.reset()
    
    steps = env.prediction_steps
    logger.info(f"Generating {steps} predictions for {horizon} horizon...")
    
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        predictions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if step % 10 == 0:
            logger.info(f"Generated prediction for {horizon} step {step+1}/{steps}")
        if terminated or truncated:
            break
    
    return np.array(predictions)

def main():
    wandb.init(
        project="market_prediction",
        entity="aarushsinha60", 
        config={
            "total_timesteps": 70000,
            "policy": "SAC",
            "net_arch": [1024, 1024, 1024],
            "learning_rate": 1e-4,
            "buffer_size": 10000,
            "batch_size": 512,
            "gamma": 0.99,
            "tau": 0.005
        },
        sync_tensorboard=True
    )
    
    horizon = '120M'
    if horizon.endswith('M'):
        steps_needed = int(horizon[:-1])
    elif horizon == '1y':
        steps_needed = 12
    elif horizon == '5y':
        steps_needed = 60
    elif horizon == '10y':
        steps_needed = 120
    else:
        logger.error(f"Unrecognized horizon format: {horizon}")
        return
    
    if len(bond_yields_scaled) < steps_needed:
        logger.error(f"Not enough bond yield data for {horizon} horizon. Needed {steps_needed}, have {len(bond_yields_scaled)}.")
        return
    
    horizon_bond_yields_scaled = bond_yields_scaled[:steps_needed]
    
    model = train_model_for_horizon(
        horizon=horizon,
        initial_state=initial_state,
        bond_yields_scaled=horizon_bond_yields_scaled,
        volatility_array=volatility_array,
        time_trends_array=time_trends_array,
        total_timesteps=wandb.config.total_timesteps
    )
    
    env = MarketPredictionEnv(
        initial_state=initial_state,
        prediction_horizon=horizon,
        bond_yields_scaled=horizon_bond_yields_scaled,
        volatility_array=volatility_array,
        time_trends_array=time_trends_array
    )
    
    predictions = generate_predictions(model, env, horizon)
    
    save_predictions(
        predictions,
        horizon_bond_yields_scaled,
        bond_scaler,
        horizon
    )
    plot_predictions(
        predictions,
        horizon_bond_yields_scaled,
        bond_scaler,
        horizon
    )
    
    # Now, use the same RL model to predict the next 6 months.
    # Use the final state from the previous environment as the starting state.
    final_state = env.state
    additional_steps = 6  # next six months
    
    if len(bond_yields_scaled) < steps_needed + additional_steps:
        logger.error(f"Not enough bond yield data for additional {additional_steps} months prediction.")
    else:
        # Slice the bond yields for the next six months.
        six_months_yields_scaled = bond_yields_scaled[steps_needed:steps_needed+additional_steps]
        
        # Create a new environment seeded with the final state from the 120M run.
        env_6m = MarketPredictionEnv(
            initial_state=final_state,
            prediction_horizon="6M",
            bond_yields_scaled=six_months_yields_scaled,
            volatility_array=volatility_array,
            time_trends_array=time_trends_array
        )
        
        predictions_6m = generate_predictions(model, env_6m, "6M")
        
        save_predictions(
            predictions_6m,
            six_months_yields_scaled,
            bond_scaler,
            "6M"
        )
        plot_predictions(
            predictions_6m,
            six_months_yields_scaled,
            bond_scaler,
            "6M"
        )
    
    wandb.finish()
    logger.info("Bond yield prediction completed for all horizons.")

if __name__ == "__main__":
    main()
