# Run on Modal

import logging
import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from typing import List, Dict, Optional, Tuple
    import re
    """Main function to run the bond yield prediction pipeline."""
    try:

        class BondYieldDataset(Dataset):

            """Dataset class for historical bond yield data with rolling window."""

            def __init__(self, data: pd.DataFrame, window_size: int = 24, prediction_months: int = 36):
                self.data = data
                self.window_size = window_size
                self.prediction_months = prediction_months

            def __len__(self) -> int:
                return self.prediction_months

            def __getitem__(self, idx: int) -> Dict:
                start_idx = len(self.data) - \
                    self.prediction_months - self.window_size + idx
                end_idx = start_idx + self.window_size
                window = self.data.iloc[start_idx:end_idx].to_dict('records')
                return {
                    'window': window,
                    'target_month': self.data.index[end_idx] if end_idx < len(self.data) else None
                }

        def generate_prompt(window_data: List[Dict]) -> str:
            """Generate prompt for the model using historical data."""
            df = pd.DataFrame(window_data)
            past_year_str = df.to_csv(index=False)

            user_prompt = f"""Analyze the following 24 months of bond yield data and provide:
            1. Specific numeric predictions for next month's yields
            2. A single-word trading action (BUY/SELL/HOLD) for bond investments
            3. Expected volatility level (LOW/MEDIUM/HIGH)
            4. Overall risk assessment (LOW/MEDIUM/HIGH)

            Historical Data:
            {past_year_str}

            Provide your response in exactly this format:
            US_10Y_Yield: [number]
            AAA_Bond_Yield: [number]
            BAA_Bond_Yield: [number]
            Junk_Bond_Yield: [number]
            Trading_Action: [BUY/SELL/HOLD]
            Volatility: [LOW/MEDIUM/HIGH]
            Risk_Level: [LOW/MEDIUM/HIGH]"""

            system_message = """You are a Senior Financial Advisor. Provide:
            1. Trading Action:
            - SELL if you expect yields to rise significantly
            - BUY if you expect yields to fall significantly
            - HOLD if you expect yields to remain stable
            2. Volatility Assessment:
            - LOW: stable yield movements expected
            - MEDIUM: moderate yield fluctuations expected
            - HIGH: significant yield swings expected
            3. Risk Level:
            - LOW: minimal chance of significant losses
            - MEDIUM: moderate uncertainty in yield movements
            - HIGH: substantial uncertainty or potential for losses
            Provide only the numeric predictions and one-word assessments."""

            prompt_template = f"""<|im_start|>system
            {system_message}<|im_end|>
            <|im_start|>user
            {user_prompt}<|im_end|>
            <|im_start|>assistant"""

            return prompt_template

        def parse_model_output(output_text: str) -> Optional[Dict]:
            """Parse the model's output text to extract predictions and assessments."""
            try:
                patterns = {
                    "US_10Y_Yield": r"US_10Y_Yield:\s*(\d+\.?\d*)",
                    "AAA_Bond_Yield": r"AAA_Bond_Yield:\s*(\d+\.?\d*)",
                    "BAA_Bond_Yield": r"BAA_Bond_Yield:\s*(\d+\.?\d*)",
                    "Junk_Bond_Yield": r"Junk_Bond_Yield:\s*(\d+\.?\d*)",
                    "Trading_Action": r"Trading_Action:\s*(BUY|SELL|HOLD)",
                    "Volatility": r"Volatility:\s*(LOW|MEDIUM|HIGH)",
                    "Risk_Level": r"Risk_Level:\s*(LOW|MEDIUM|HIGH)"
                }

                predictions = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, output_text, re.IGNORECASE)
                    if match:
                        value = match.group(1)
                        if key in ["Trading_Action", "Volatility", "Risk_Level"]:
                            predictions[key] = value.upper()
                        else:
                            predictions[key] = float(value)
                    else:
                        logger.warning(
                            f"Could not find pattern for {key} in output: {output_text[:200]}")
                        return None

                return predictions
            except Exception as e:
                logger.error(f"Error parsing output: {e}")
                return None

        def predict_yields_batch(
            data: pd.DataFrame,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            batch_size: int = 4
        ) -> pd.DataFrame:
            """Generate batch predictions for bond yields with rolling window."""
            dataset = BondYieldDataset(
                data=data, window_size=24, prediction_months=36)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: {
                    'window': [item['window'] for item in x],
                    'target_month': [item['target_month'] for item in x]
                }
            )

            predictions = []

            for batch_idx, batch in enumerate(dataloader):
                logger.info(f"Processing batch {batch_idx + 1}")
                prompts = [generate_prompt(window)
                           for window in batch['window']]

                tokenizer.padding_side = 'left'
                inputs = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=2048
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.95,
                        top_k=40,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.pad_token_id
                    )

                tokenizer.padding_side = 'right'
                decoded_outputs = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True)

                for output, target_month in zip(decoded_outputs, batch['target_month']):
                    pred = parse_model_output(output)
                    if pred:
                        pred['Month'] = target_month
                        predictions.append(pred)
                    else:
                        logger.warning(
                            f"Failed to parse predictions for month {target_month}")

            if not predictions:
                logger.error("No valid predictions were generated!")
                return pd.DataFrame()

            predictions_df = pd.DataFrame(predictions)
            predictions_df = predictions_df.sort_values('Month')

            return predictions_df
        logger.info("Loading data...")
        past_data = pd.read_csv('/root/config/bonda_10yr_data.csv')
        past_data['Month'] = range(len(past_data))
        past_data.set_index('Month', inplace=True)
        model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
        logger.info(f"Initializing model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, revision="main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True)
        predictions_df = predict_yields_batch(past_data, model, tokenizer)
        if not predictions_df.empty:
            logger.info(
                f"Saving predictions with shape: {predictions_df.shape}")
            predictions_df.to_csv(
                "/my_vol/predicted_actual_bond_yields.csv", index=False)
            vol.commit()
            logger.info("Predictions saved successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "pandas", "nltk", "matplotlib", "tqdm", "pyarrow", "fastparquet", "datasets", "peft", "transformers", "bitsandbytes"
).add_local_file("data/Yields/bonds_10yr_data.csv", remote_path="/root/config/bonda_10yr_data.csv")

app = modal.App()
vol = modal.Volume.from_name("my-volume")


@app.function(gpu="A100-40GB:1", image=image, timeout=32400, volumes={"/my_vol": vol})
def run_training():
    main()


if __name__ == "_main_":
    run_training()
