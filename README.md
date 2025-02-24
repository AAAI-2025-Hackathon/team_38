# Check-In

- Title of your submission: **Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation**
- Team Members: [Jaskaran Singh Walia](karanwalia2k3@gmail.com), [Aarush Sinha](aarush.sinha@gmail.com), [Srinitish Srinivasan](smudge0110@icloud.com), [Srihari Unnikrishnan](srihari.unnikrishnan@gmail)
- [x] All team members agree to abide by the [Hackathon Rules](https://aaai.org/conference/aaai/aaai-25/hackathon/)
- [x] This AAAI 2025 hackathon entry was created by the team during the period of the hackathon, February 17 – February 24, 2025
- [x] The entry includes a 2-minute maximum length demo video here: [Link](https://drive.google.com/drive/folders/1Lz3sJ6CW2IHBTuvUiRv7RPywtH1V-09r) 

# Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation

---

## 🚀 Project Overview

This project presents a novel approach to **predicting liquidity-aware bond yields** by integrating three cutting-edge technologies:

1. **Synthetic Data Generation with Causal GANs** – Models realistic bond yield time-series data while preserving causal dependencies.
2. **Deep Reinforcement Learning (Soft Actor-Critic - SAC)** – Enhances synthetic data quality through self-learning feedback loops.
3. **Predictive Modeling with Fine-tuned LLMs** – Utilizes a **fine-tuned Qwen2.5-7B** model to extract actionable trading signals, risk assessments, and volatility forecasts.

### 📌 Key Features
- **High-Fidelity Synthetic Bond Yield Data** with Causal GANs
- **Self-Optimizing Data Refinement** using Deep RL (SAC)
- **LLM-Driven Market Insights** for intelligent trading strategies
- **Robust Performance Evaluation** with Mean Absolute Error (MAE) metrics

---

## 🏗️ Architecture Overview

![Architecture Diagram](https://github.com/user-attachments/assets/8a072a33-0a26-4b7a-8466-d5b3e68dc628)

For detailed experimental results, refer to [Results/README.md](Results/README.md).

---

## 📂 Directory Structure

```
├── GANS/         # Causal GAN implementation for synthetic bond yield data
├── LLMs/         # Fine-tuned Qwen2.5-7B for predictive modeling
├── MAE/          # Scripts for Mean Absolute Error computation
├── Results/      # Evaluation scripts, logs, and results summary
├── DATA/         # CSV files and datasets used in the project
├── Paper.pdf     # Full research paper (Introduction, Methodology, Results)
└── README.md     # Project documentation
```

---

## 📊 Results & Visualizations

### Model Performance

- Extensive **quantitative and qualitative** evaluation results are available in the **[Results](Results/readme.md) folder**.
- Benchmarked against real-world bond yield data with **low MAE scores**.
- The **LLM predictions align with historical financial trends** and trader insights.

---

## ⚡ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AAAI-2025-Hackathon/team_38.git
cd team_38
```

### 2️⃣ Explore the Project
- **GANS:** Run synthetic data generation scripts.
- **LLMs:** Experiment with predictive models.
- **MAE:** Evaluate model performance.
- **Results:** Access detailed findings and analysis.

### 3️⃣ Run Experiments
Follow the setup and execution instructions inside each folder’s `README.md`.

---

## 📬 Contact & Support
For questions or collaborations, reach out to the team via email or check the repository for further details.

🚀 **Let’s redefine liquidity-aware bond yield prediction together!**
