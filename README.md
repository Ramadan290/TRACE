# TRACE: Transformer-Reduced Autoencoder for Classification and Estimation

This repository contains three AI models built using the **TRACE** architecture, a deep learning pipeline designed for HR analytics inside the Themis HR System. These models are:

- **Attrition Risk Prediction**
- **Benefit-to-Cost Ratio (BCR) Prediction**
- **Performance Score Prediction**

---

## What is TRACE?

**TRACE** (*Transformer-Reduced Autoencoder for Classification and Estimation*) is a specialized deep learning architecture designed to improve learning from structured HR data. It overcomes the limitations of traditional MLPs by introducing two key components **before** the classification/regression head:

### 1. Transformer Encoder
- Dynamically captures interdependencies between features through self-attention.
- Helps the model focus on relationships that matter most in each sample.

### 2. Autoencoder Bottleneck
- Compresses the transformer-encoded feature set.
- Learns a latent, noise-tolerant, and generalizable representation.

### 3. MLP Prediction Head
- Dense(64) → ReLU  
- Dropout(0.3)  
- Dense(32) → ReLU  
- Final output (Softmax or Linear depending on task)

> Feeding structured HR data directly into an MLP leads to **parameter explosion** and **vanishing gradients** due to dense connections. TRACE addresses this by encoding and compressing the input before prediction, enabling deeper learning, faster convergence, and improved generalization.

---

## Model Objectives

### 1. Attrition Risk Prediction (Classification)
- **Goal**: Predict the likelihood of an employee leaving the company.
- **Type**: Multi-class classification → Output range: {0, 1, 2, 3}
- **Use Case**: Helps HR identify high-risk employees and intervene proactively.

### 2. Benefit-to-Cost Ratio (BCR) Prediction (Regression)
- **Goal**: Estimate the cost-efficiency of employee benefits and rewards.
- **Type**: Regression → Output: continuous value between 0 and 1.
- **Use Case**: Enables budget-optimized benefit planning.

### 3. Performance Score Prediction (Regression)
- **Goal**: Predict an employee’s future performance based on past behavior and traits.
- **Type**: Regression → Output: continuous performance score.
- **Use Case**: Used to support promotion decisions, training needs, or role reassignment.

---

## Input Features

### Performance & Attrition Models

| Feature            | Description                                                       |
|--------------------|-------------------------------------------------------------------|
| `age`              | Employee’s age                                                    |
| `base_salary`      | Current fixed salary                                              |
| `num_appraisals`   | Total number of appraisals received                               |
| `num_benefits`     | Number of assigned benefits                                       |
| `num_penalties`    | Number of recorded penalties                                      |
| `attendance_rate`  | Average attendance rate (0–1)                                     |
| `completion_rate`  | Average task/project completion rate (0–1)                        |
| `sentiment_score`  | Predicted sentiment score from the sentiment model (0–4)          |

---

### Benefit-to-Cost Ratio (BCR) Model

| Feature                 | Description                                                         |
|--------------------------|---------------------------------------------------------------------|
| `base_salary`            | Current fixed salary                                                |
| `num_appraisals`         | Number of evaluations received                                      |
| `num_benefits`           | Total assigned benefits                                             |
| `num_raise_requests`     | Raise requests submitted                                            |
| `completion_rate`        | Task/project completion rate (0–1)                                  |
| `past_performance_score` | Predicted or logged score from prior model                          |
| `skills`                 | Total number of validated skills acquired                           |
| `cost_of_training`       | Estimated cost of training received (numerical value in currency)   |

---

## How to Use

1. Clone the repository:
```bash
git clone https://github.com/yourusername/themis-hr-ai-models.git
cd themis-hr-ai-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open any notebook:
- `attrition_risk.ipynb`
- `benefit_to_cost_ratio.ipynb`
- `Performance_Prediciton.ipynb`


## Credits

Developed by ME as part of **Themis HR System**, integrating deep AI into modern HR pipelines.
