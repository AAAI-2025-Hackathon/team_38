# Check-In

- Title of your submission: **Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation**
- Team Members: [Jaskaran Singh Walia](karanwalia2k3@gmail.com), [Aarush Sinha](aarush.sinha@gmail.com), [Srinitish Srinivasan](smudge0110@icloud.com), [Srihari Unnikrishnan](srihari.unnikrishnan@gmail)
- [x] All team members agree to abide by the [Hackathon Rules](https://aaai.org/conference/aaai/aaai-25/hackathon/)
- [x] This AAAI 2025 hackathon entry was created by the team during the period of the hackathon, February 17 – February 24, 2025
- [x] The entry includes a 2-minute maximum length demo video here: [Link](https://your-link.com) 



# Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation

This repository contains the code and documentation for our project. The project was developed as part of the AAAI-2025 Hackathon and is structured into several folders corresponding to the different components of our framework.

## Project Overview

Our approach combines three key components:
- **Synthetic Data Generation:** Using Causal GANs to generate realistic bond yield time-series data.
- **Reinforcement Learning (SAC):** Enhancing synthetic data quality by refining the generated data.
- **Predictive Modeling with LLMs:** Leveraging a fine-tuned Qwen2.5-7B model to generate actionable trading signals, risk assessments, and volatility projections.

An architecture diagram below provides an overview of the system design.

## Architecture Diagram
![architecture (1)](https://github.com/user-attachments/assets/8a072a33-0a26-4b7a-8466-d5b3e68dc628)

*For detailed results and additional diagrams, please refer to the [Results README](Results/README.md).*

## Directory Structure

The repository is organized as follows:

- **GANS**  
  Contains the code and notebooks for implementing the Causal GANs used in synthetic data generation.

- **LLMs**  
  Houses the code and experiments for fine-tuning and running the Large Language Models on the generated data.

- **MAE**  
  Includes scripts and notebooks for computing the Mean Absolute Error (MAE) between actual and predicted bond yields.

- **Results**  
  Contains the evaluation scripts, result logs, and the results diagram. Detailed results are described in the README inside this folder.

- **DATA**  
  Holds the CSV files and other datasets used throughout the project.


- Other files:
  - **Paper.pdf:** The full research paper with introduction, methodology, experiments, results and findings 

All code, notebooks, and data files are organized into their respective folders to ensure a clean and modular project structure.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AAAI-2025-Hackathon/team_38.git
   ```


2. **Explore the Folders:**

    Each folder contains the relevant code and notebooks for its component. For instance:
    
    - **GANS:** Use the notebooks in the GANS folder to explore synthetic data generation.
    - **LLMs:** Check out the LLMs folder for predictive modeling experiments.
    - **MAE:** Review the MAE folder for scripts computing the error metrics.

3. **Results:**
    - **Results:** Detailed results are available in the [Results folder](Results/README.md).

## Run the Experiments

Follow the instructions in the individual folder READMEs to run the experiments locally or on your preferred platform.


For any questions or further information, please refer to the respective folder documentation or contact the team.

