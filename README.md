# 🏠 House Price Prediction — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🌐 Live App
👉 [Click here to open the app](https://house-price-sanjay.streamlit.app)

## 📌 Problem Statement
Predict the sale price of residential homes in Ames, Iowa based on
79 explanatory variables describing every aspect of the house.

---

## 📂 Project Structure

---

## 🔧 Tech Stack
- **Python 3.12**
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Preprocessing & baseline models
- **XGBoost** — Final production model
- **Matplotlib & Seaborn** — Visualizations
- **Streamlit** — Web application deployment
- **Pickle** — Model serialization
- **Git & GitHub** — Version control

---

## 📊 Dataset
- **Source:** Kaggle — House Prices: Advanced Regression Techniques
- **Train set:** 1460 rows × 81 columns
- **Test set:** 1459 rows × 80 columns
- **Target:** SalePrice (continuous)

---

## 🚀 Project Pipeline

### 1. Data Cleaning
- Dropped columns with >50% missing values (PoolQC, Alley, etc.)
- Filled categorical nulls with 'None'
- Filled LotFrontage using neighborhood median
- Label encoded all categorical variables

### 2. Exploratory Data Analysis
- Analyzed SalePrice distribution (right skewed)
- Correlation heatmap — identified top features
- Removed 2 outliers (large area, low price)

### 3. Feature Engineering
- `TotalSF` = Basement + 1st Floor + 2nd Floor
- `TotalBath` = Full + Half bathrooms
- `HouseAge` = YrSold - YearBuilt
- `IsRemodeled` = Was house remodeled?
- `TotalPorch` = All porch areas combined

### 4. Model Training & Results

| Model | RMSE | R² Score |
|---|---|---|
| Linear Regression | 0.1278 | 0.9031 |
| Random Forest | 0.1426 | 0.8794 |
| XGBoost (default) | 0.1232 | 0.9100 |
| **XGBoost (tuned)** | **0.1168** | **0.9191** |

### 5. Hyperparameter Tuning
Used RandomizedSearchCV with 5-fold CV across 30 combinations.

**Best Parameters:**
- n_estimators: 1000
- learning_rate: 0.01
- max_depth: 4
- subsample: 0.6
- colsample_bytree: 0.6

### 6. Top 5 Most Important Features
1. OverallQual (0.234)
2. TotalSF (0.163)
3. CentralAir (0.068)
4. GarageCars (0.052)
5. KitchenAbvGr (0.045)

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/SanjayMeghwal/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app
```bash
cd app
streamlit run app.py
```

---

## 📈 App Preview
- Enter house details in the sidebar
- Click **Predict House Price**
- Get instant price prediction with range estimate

---

## 👨‍💻 Author
**Sanjay Meghwal**
- GitHub: [@SanjayMeghwal](https://github.com/SanjayMeghwal)

---

## 📄 License
This project is open source and available under the MIT License.