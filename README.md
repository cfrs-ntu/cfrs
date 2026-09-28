# Cerebral Frailty Risk Score (CFRS)
The CFRS is a proof-of-concept dementia risk prediction tool developed using UK Biobank data. The analytical pipeline covers:
1. Predictor selection *(cfrs_predictors.R)*\
   Fits a XGBoost-accelerated failure time model with nested cross-validation and Bayesian hyperparameter optimization, and computes SHAP values on held-out folds. SHAP-based feature ranking pools standardized SHAP values across outer folds to identify the top predictors and assess ranking stability.
3. Scoring *(cfrs_scoring.R)*\
   Implement AutoScore-Survival to discretize the top 10 XGBoost-derived predictors, derives integer weights, and constructs the CFRS. Apparent and optimism-corrected performance are reported alongside held-out test-set metrics.

# Requirements
R version ≥ 4.2.0

**Open-source R packages:**\
***• XGBoost*** (https://github.com/dmlc/xgboost) \
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016.

***• SHAPforxgboost*** (https://github.com/liuyanguu/SHAPforxgboost) \
Yang Liu and Allan Just (2020). SHAPforxgboost: SHAP Plots for 'XGBoost'. R package version 0.1.0.

***• AutoScore*** (https://github.com/nliulab/AutoScore) \
Xie F, Ning Y, Yuan H, Goldstein BA, Ong MEH, Liu N, Chakraborty B. AutoScore-Survival: Developing interpretable machine learning-based time-to-event scores with right-censored survival data. Journal of Biomedical Informatics 2022; 125: 103959.


**Data:**\
*time* — time to dementia diagnosis or censoring (numeric, years)\
*dem* — dementia status (0 = censored, 1 = event)\
Predictor columns — clinical covariates, frailty index, and MRI-derived measures


