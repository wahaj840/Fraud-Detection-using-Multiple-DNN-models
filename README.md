# Fraud Detection Using Deep Learning Models (2 Models Compared)

This project focuses on developing, evaluating, and comparing two deep learning models for detecting fraudulent transactions in credit card data. The models compared are a **Feedforward Neural Network (FNN)** and a **Hybrid Model (LSTM + Feedforward)**. Both models are tested on a highly imbalanced dataset and are assessed for their ability to detect fraudulent transactions while minimizing false positives and operational costs.

## Project Overview

- **Project Name**: Fraud Detection Using Deep Learning (2 Models Applied)
- **Author**: Ali Wahaj (20026654)

## Colab Notebook

You can access the full implementation of both models in the following Google Colab Notebook:  
[CA_2_Deep_Learning_FNL.ipynb](https://colab.research.google.com/drive/1W67K06WFNdEd5SUWf5Fknk-rxWf_aPhy)

## Project Objective

The main objective of this project is to build and evaluate two deep learning models:
1. **Feedforward Neural Network (FNN)**: A standard dense-layered model for binary classification.
2. **Hybrid Model (LSTM + Feedforward)**: Incorporates Long Short-Term Memory (LSTM) layers to capture sequential patterns in the transaction data, followed by dense layers for classification.

The models aim to detect fraudulent credit card transactions with a high degree of accuracy and precision while minimizing false positives to maintain operational efficiency.

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Number of Transactions**: 284,807
- **Fraudulent Transactions**: 492
- **Non-fraudulent Transactions**: 284,315

The dataset exhibits a severe class imbalance, with fraudulent transactions accounting for only 0.17% of the total data.

## Methodology

The project follows the **CRISP-DM methodology**, which involves:
1. **Business Understanding**: Identify the project goals and the need for accurate fraud detection models.
2. **Data Understanding**: Perform exploratory data analysis (EDA) to understand the data’s distribution and key features.
3. **Data Preparation**: Handle missing data, apply feature scaling, and address class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
4. **Modeling**: Build two deep learning models—a Feedforward Neural Network (FNN) and a Hybrid Model (LSTM + FNN).
5. **Evaluation**: Compare model performance using metrics like **accuracy**, **precision**, **recall**, **AUC-ROC**, and **F1 score**.
6. **Deployment**: Propose a plan to deploy the best-performing model in a real-time fraud detection system.

## Model Details

### Model 1: Feedforward Neural Network (FNN)

- **Architecture**:
  - Multiple dense layers with ReLU activations.
  - Dropout layers to prevent overfitting.
  - Sigmoid activation function for binary classification.
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Key Metrics**: Accuracy, Precision, Recall, AUC-ROC, F1 score
- **Result**: Achieved an **AUC** score of 0.92, with a balanced precision-recall tradeoff. This model performed well in reducing false positives while maintaining high accuracy.

### Model 2: Hybrid Model (LSTM + Feedforward)

- **Architecture**:
  - LSTM layers to process sequential transaction data.
  - Followed by dense layers for binary classification.
  - Sigmoid activation for output.
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Key Metrics**: Accuracy, Precision, Recall, AUC-ROC, F1 score
- **Result**: The **Hybrid Model** achieved a similar AUC score of 0.92 but detected more fraudulent transactions compared to the FNN model. However, this improvement came with a higher rate of false positives, which could increase operational costs.

## Results

- **Feedforward Neural Network**:
  - **AUC**: 0.92
  - **Precision**: High precision, minimizing false positives.
  - **Recall**: Balanced recall, effectively identifying fraud.
  - **Best for**: Environments where minimizing false positives is critical.
  
- **Hybrid Model (LSTM + Feedforward)**:
  - **AUC**: 0.92
  - **Precision**: Slightly lower precision due to higher false positives.
  - **Recall**: Higher recall, detecting more fraudulent transactions.
  - **Best for**: Scenarios where maximizing fraud detection is more important than minimizing false positives.

## Economic Impact

A cost-benefit analysis was conducted for both models:
- **Feedforward Neural Network**: Provides a 90% ROI by balancing the reduction of fraudulent transactions with minimal false positives.
- **Hybrid Model**: Detects more fraud cases but incurs higher costs due to false positives, potentially lowering the ROI in customer-facing environments.

## Visualizations

1. **Correlation Matrix**: Illustrates the relationships between features.
2. **Class Distribution Plot**: Highlights the severe class imbalance in the dataset.
3. **ROC-AUC Curves**: Shows the performance of each model in distinguishing between fraudulent and non-fraudulent transactions.
4. **Precision-Recall Curves**: Demonstrates the trade-off between precision and recall for both models.

## Deployment

The **Feedforward Neural Network** is recommended for deployment in a real-time fraud detection system. It offers a good balance of precision, recall, and cost-effectiveness, making it suitable for environments where minimizing false positives is crucial.

### Future Enhancements:
- Explore additional hybrid models or ensembles to improve performance.
- Implement continuous model retraining with new data to adapt to evolving fraud strategies.

## How to Run the Project

1. Download the notebook file `CA_2_Deep_Learning_FNL.ipynb` or access it directly via the [Google Colab Link](https://colab.research.google.com/drive/1W67K06WFNdEd5SUWf5Fknk-rxWf_aPhy).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


### Key Elements Included:
- **Project Overview**: Brief summary of the project’s objectives and methodology.
- **Dataset and Methods**: Information about the dataset, preprocessing steps, and modeling approach.
- **Model Architecture and Results**: Detailed breakdown of both models and their performance.
- **Economic Impact**: Information about the cost-effectiveness of the models.
- **Visualizations**: Mentions of key visual outputs like correlation matrix and precision-recall curves.
- **Deployment and Future Enhancements**: Recommendations for deploying the Feedforward Neural Network and potential future improvements.

This `README.md` provides a comprehensive, professional overview of your project, making it easy for employers to evaluate your work.
