# House Price Prediction (Kaggle). 
This project is part of the Kaggle competition **"House Prices - Advanced Regression Techniques"**.  
  
The goal is to predict house sale prices using structured tabular data and modern regression techniques with a clean, reproducible machine learning pipeline.  
  
---  
  
## Project Overview. 
  
- **Provlem type:** Regression
- **Targer Variable:** 'SalePrice'
- **Evaluation Metric:** Root Mean Squared Error (RMSE)
- **Dataset:** Ames Housing Dataset (Kaggle). 
  
This repository focuses on:  
- clean preprocessing pipelines 
- correct handling of categorical & numerical features 
- targe transformation (log-target) 
- model comparison and improvement 
  
---

## Model Approach 
    
### 1. Baseline Model
- Linear models with:  
    - 'SimpleImputer'
    - 'StandardScaler'
    - 'OneHotEncoder
- Ridge & ElasticNet Regression
- Purpose: establish a strong, interpretable baseline. 
  
Notebook: 'notebooks/01_baseline.ipynb'. 
  
---  
  
### 2. Log-Target Transformation. 
To stabilize variance and reduce the influence of extreme house prices, the target variable was transformed unsing:  
   
``` python
y_log = np.log1p(SalePrice)
``` 

Predictions were transformed back using:  
``` python
SalePrice = np.expm1(predictions)
```
   
This significantly improved model performance.  
  
Notebook: 'notebooks/02_log_target.ipynb' 
  
### Gardient Boosting Regression 
A more powerful non-linear model was trained to capture complex feature interactions.  
- Gardient Boosting Regressor 
- Same preprocessing pipeline
- Trained on log-transformed target 

Notebook: 'notebooks/03_gardient_boosting.ipynb' 
  
## Results 
*Model:*                            *Public Score:*  
Baseline (no log target)                ~0.149
Linear Model + log target               ~0.135
Gradient Boosting + log target          ~0.137

The log-target transformation provided the biggest performance gain.  
  

### Tech Stack:  
- Python
- pandas, numpy
- scikit-learn
- Jupyter Notebook
- Git & Github 


### Key Learnings 
- Clean preprocessing pipelines are crucial
- Log-transforming skewed targets can dramatically improve perfomance
- Strong Baseline matter before complex models
- Reproducibility and structure are as important as the score