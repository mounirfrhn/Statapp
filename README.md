# Fairness in Job Advertisement Click Prediction

## Project Description

This project addresses a critical challenge in the ad-tech industry: delivering relevant job advertisements while ensuring fairness across demographic groups, particularly gender. Using the FairJob dataset, we investigate how predictive models can unintentionally reinforce biases present in historical data, and we explore strategies to balance *utility* (ad effectiveness) with *fairness*.

The goal is to build machine learning models that predict user clicks on job ads with high accuracy while reducing disparities in treatment between male and female users. This balance is essential to promote ethical AI practices and comply with regulations like the AI Act and GDPR.

---

## Objectives

•⁠  ⁠Understand the structure and biases in the FairJob dataset  
•⁠  ⁠Develop baseline click prediction models (Logistic Regression, Random Forest, XGBoost)  
•⁠  ⁠Evaluate model fairness using key metrics (Demographic Parity, Equal Opportunity, Disparate Impact)  
•⁠  ⁠Apply and compare bias mitigation techniques (reweighting, post-processing, fairness regularization)  
•⁠  ⁠Analyze trade-offs between fairness and model performance  
•⁠  ⁠Provide insights and recommendations for fairer ad targeting systems  

---

## Dataset

The dataset contains historical records of user interactions with job advertisements, including user features, job ad features, click labels, and a protected attribute proxy (gender). Notable challenges include strong class imbalance (clicks are rare events), user selection bias, and positional bias in ad displays.

---

## Methodology

1.⁠ ⁠*Data Exploration & Preprocessing:*  
   - Analyze data distributions, class imbalances, and feature correlations  
   - Feature engineering and normalization  
   - Train-test split preserving protected attribute proportions  

2.⁠ ⁠*Model Training:*  
   - Train logistic regression, random forest, and XGBoost models  
   - Optimize hyperparameters and evaluate on holdout sets  

3.⁠ ⁠*Fairness Evaluation:*  
   - Calculate fairness metrics across gender groups  
   - Visualize disparities and analyze model confusion matrices  

4.⁠ ⁠*Bias Mitigation:*  
   - Implement reweighting, adversarial debiasing, post-processing corrections, and fairness regularization  
   - Tune fairness penalty and assess impact on performance and fairness  

5.⁠ ⁠*Trade-off Analysis:*  
   - Plot utility vs. fairness curves  
   - Identify optimal balance points for deployment  

---

## Results

•⁠  ⁠Baseline XGBoost model achieved strong predictive performance (AUC, log-likelihood, click-rank utility).  
•⁠  ⁠Fairness evaluation revealed subtle but meaningful gender biases in predictions despite excluding gender as a feature.  
•⁠  ⁠Bias mitigation techniques successfully improved fairness metrics with minor performance degradation.  
•⁠  ⁠The fairness-utility trade-off demonstrates the feasibility of building fair and effective ad targeting models.  

---

## How to Run

1.⁠ ⁠Clone the repository:  
   ```bash
   git clone https://github.com/mounirfrhn/Statapp/.git
   cd fairjob-click-prediction
