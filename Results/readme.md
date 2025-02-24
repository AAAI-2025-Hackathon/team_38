# 📊 Results

This README presents the experimental findings of our integrated framework, which combines **synthetic data generation** via **Causal GANs** and **Deep Reinforcement Learning (SAC)** with predictive modeling using a fine-tuned **LLM** (Qwen2.5-7B). Our evaluation spans multiple key metrics: forecasting accuracy, trading signal quality, and profitability—across four bond categories (AAA, BAA, US10Y, Junk) over a _10-year period_.

---

## 📈 Evaluation Metrics

- *🧠 LLM-as-Judge Evaluation:*  
  A separate LLM (*DeepSeekR1) scores trading recommendations on a **1–5 scale*.
  
- *💰 Profit/Loss Analysis:*  
  Profitability is measured by the *percentage of months where the predicted trading signal resulted in a profit*.
  
- *📉 Mean Absolute Error (MAE):*  
  MAE quantifies the *prediction error* between synthetic and actual bond yields.
  
- *👨‍💼 Expert Evaluation:*  
  Three *financial experts* provided qualitative scores for the *realism of synthetic data* and the *validity of trading signals*.

---

## 🔍 Key Findings

### 🎯 Predictive Accuracy

- *📊 MAE Improvement:*  
  The RL-enhanced synthetic data achieves an *MAE of 0.103* for *US10Y yields, significantly lower than the **GAN-only* approach. This suggests that *reinforcement learning improves real-world bond yield replication*.

### 💡 Trading Signal Quality

- *🤖 LLM-as-Judge Scores:*  
  - *RL Method:* 3.37  
  - *GAN-only Method:* 2.87  
  - *Actual Data:* 2.58  
  🔹 RL-enhanced predictions score the *highest, indicating **superior trading recommendations*.

- *🧑‍⚖️ Expert Evaluations:*  
  Experts rated the *RL method* with an *average score of 4.67, compared to **4.17* for GAN-only and *3.67* for actual data, reinforcing the *value of our integrated approach*.

### 💵 Profitability Analysis

- *📊 Profit/Loss Rates:*  
  The *RL-based approach* achieved a *60% profit rate for AAA bonds* and *54% for BAA bonds, demonstrating its **practical utility in generating profitable trading strategies*.

### 🛠️ Ablation Studies

- *Removing the RL component* resulted in an *increased MAE of 0.3* for *US10Y yields* and *0.1* for *Junk bond yields*. 
  🔹 This underscores the *critical role of reinforcement learning* in enhancing synthetic data quality.

---

## 📊 Detailed Results

### *Table 1: Average Evaluation Scores*
| *Method* | *Avg. LLM Judge Score (↑)* |
|------------|------------------------------|
| *RL*   | *3.37*                    |
| GAN        | 2.87                         |
| Actual     | 2.58                         |

### *Table 2: MAE Scores for all 4 bond types (SAC-RL vs CausalGAN)*
| Method | Bond Type        | MAE ↓    |
|--------|------------------|----------|
| GAN    | US_10Y_Yield     | 0.437    |
| GAN    | AAA_Bond_Yield   | 0.343    |
| GAN    | BAA_Bond_Yield   | 0.372    |
| GAN    | Junk_Bond_Yield  | 0.594    |
| *RL* | US_10Y_Yield     | *0.103*|
| RL     | AAA_Bond_Yield   | 0.124    |
| RL     | BAA_Bond_Yield   | 0.174    |
| RL     | Junk_Bond_Yield  | 0.458    |

### *Table 3: Expert Evaluation Scores*
| *Method* | *Expert 1* | *Expert 2* | *Expert 3* | *Average (↑)* |
|------------|--------------|--------------|--------------|-----------------|
| *RL*   | 4.5          | 5.0          | 4.5          | *4.67*        |
| GAN        | 4.0          | 4.0          | 4.5          | *4.17*        |
| Actual     | 3.5          | 3.5          | 4.0          | *3.67*        |

---

## 📊 Visualizations

### *Figure 1: RL Reward Curves*
![BOND-Rewards](https://github.com/user-attachments/assets/17f1482f-4c5f-482a-8724-6f193d1ee837)

### *Figure 2: LLM-as-Judge Evaluation Over Time*
![eval_scores](https://github.com/user-attachments/assets/cd2345e9-c538-49d0-bd5f-977fda209fc5)

### *Figure 3: Profit vs. Loss Analysis by Bond Type*
![profit_loss_comparison (1)](https://github.com/user-attachments/assets/cf98ee28-a29b-4f3d-9319-86a1d0e2816b)

---

## ✅ Conclusion

The results confirm that *integrating reinforcement learning with Causal GANs* significantly *improves synthetic bond yield data fidelity. When used to train a **fine-tuned LLM*, this enhanced dataset results in:
- *📉 Lower forecasting errors* (as evidenced by reduced MAE),
- *📈 Superior trading signal quality* (higher LLM and expert scores),
- *💰 More profitable trading strategies* (increased profit month ratios).

These findings validate our framework’s potential for practical financial forecasting and decision-making.
