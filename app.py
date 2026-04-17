import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ─── Load Model and Feature Names ──────────────────────────────
@st.cache_resource
def load_model():
    with open('../models/house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_features():
    with open('../models/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return features

model        = load_model()
feature_names = load_features()

# ─── Title ─────────────────────────────────────────────────────
st.title("🏠 House Price Prediction App")
st.markdown("### Predict the sale price of a house based on its features")
st.markdown("---")

# ─── Input Section ─────────────────────────────────────────────
st.sidebar.header("🔧 Enter House Details")

def get_user_input():
    # Core features
    OverallQual   = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
    GrLivArea     = st.sidebar.number_input("Above Ground Living Area (sq ft)", 
                                             min_value=300, max_value=6000, value=1500)
    TotalBsmtSF   = st.sidebar.number_input("Total Basement Area (sq ft)", 
                                             min_value=0, max_value=3000, value=800)
    firstFlrSF    = st.sidebar.number_input("1st Floor Area (sq ft)", 
                                             min_value=300, max_value=4000, value=900)
    secondFlrSF   = st.sidebar.number_input("2nd Floor Area (sq ft)", 
                                             min_value=0, max_value=2000, value=500)
    GarageCars    = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
    GarageArea    = st.sidebar.number_input("Garage Area (sq ft)", 
                                             min_value=0, max_value=1500, value=480)
    FullBath      = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
    HalfBath      = st.sidebar.slider("Half Bathrooms", 0, 2, 1)
    BedroomAbvGr  = st.sidebar.slider("Bedrooms Above Ground", 0, 8, 3)
    YearBuilt     = st.sidebar.number_input("Year Built", 
                                             min_value=1872, max_value=2010, value=1990)
    YearRemodAdd  = st.sidebar.number_input("Year Remodeled", 
                                             min_value=1950, max_value=2010, value=1990)
    YrSold        = st.sidebar.selectbox("Year Sold", [2006,2007,2008,2009,2010])
    LotArea       = st.sidebar.number_input("Lot Area (sq ft)", 
                                             min_value=1000, max_value=50000, value=8000)

    # ─── Engineered Features ───────────────────────────────────
    TotalSF       = TotalBsmtSF + firstFlrSF + secondFlrSF
    TotalBath     = FullBath + 0.5 * HalfBath
    HouseAge      = YrSold - YearBuilt
    IsRemodeled   = 1 if YearRemodAdd != YearBuilt else 0
    TotalPorch    = 0  # default

    # ─── Build input dict with ALL features ────────────────────
    # Start with zeros for all features
    input_dict = {col: 0 for col in feature_names}

    # Fill in the values user provided
    input_dict['OverallQual']   = OverallQual
    input_dict['GrLivArea']     = GrLivArea
    input_dict['TotalBsmtSF']   = TotalBsmtSF
    input_dict['1stFlrSF']      = firstFlrSF
    input_dict['2ndFlrSF']      = secondFlrSF
    input_dict['GarageCars']    = GarageCars
    input_dict['GarageArea']    = GarageArea
    input_dict['FullBath']      = FullBath
    input_dict['HalfBath']      = HalfBath
    input_dict['BedroomAbvGr']  = BedroomAbvGr
    input_dict['YearBuilt']     = YearBuilt
    input_dict['YearRemodAdd']  = YearRemodAdd
    input_dict['YrSold']        = YrSold
    input_dict['LotArea']       = LotArea
    input_dict['TotalSF']       = TotalSF
    input_dict['TotalBath']     = TotalBath
    input_dict['HouseAge']      = HouseAge
    input_dict['IsRemodeled']   = IsRemodeled
    input_dict['TotalPorch']    = TotalPorch

    return pd.DataFrame([input_dict])

input_df = get_user_input()

# ─── Prediction ────────────────────────────────────────────────
st.subheader("📊 Input Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Overall Quality",    input_df['OverallQual'].values[0])
    st.metric("Living Area (sqft)", input_df['GrLivArea'].values[0])
    st.metric("Total SF",           input_df['TotalSF'].values[0])

with col2:
    st.metric("Garage Cars",   input_df['GarageCars'].values[0])
    st.metric("Bathrooms",     input_df['TotalBath'].values[0])
    st.metric("Bedrooms",      input_df['BedroomAbvGr'].values[0])

with col3:
    st.metric("Year Built",    input_df['YearBuilt'].values[0])
    st.metric("House Age",     input_df['HouseAge'].values[0])
    st.metric("Remodeled",     "Yes" if input_df['IsRemodeled'].values[0] else "No")

st.markdown("---")

# ─── Predict Button ────────────────────────────────────────────
if st.button("🔮 Predict House Price", use_container_width=True):
    log_prediction  = model.predict(input_df)[0]
    actual_price    = np.expm1(log_prediction)

    st.markdown("---")
    st.success(f"## 🏡 Predicted House Price: **${actual_price:,.0f}**")

    # Price range (±10%)
    low  = actual_price * 0.90
    high = actual_price * 1.10
    st.info(f"📈 Estimated Price Range: **${low:,.0f}** — **${high:,.0f}**")

    st.markdown("---")
    st.caption("⚠️ Prediction based on Ames, Iowa housing data (2006-2010). "
               "Use as reference only.")