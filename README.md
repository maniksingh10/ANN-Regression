
# 💵 ANN Regression – Salary Prediction

This project uses an Artificial Neural Network (ANN) for predicting salary using a regression model.

--- 
## 📁 Project Structure

- `data.csv` - Dataset file  
- `ann_regression.py` - ANN training script  
- `scaler.pkl` - Saved StandardScaler
- `app.py` - Streamlit deployed
- `le_gender.pkl`, `oh.pkl`, `scaler.pkl` - Saved encoders and scaler for reuse   
- `model.h5` - Trained regression model  
- `README.md` - Project documentation  

## 📊 Dataset Overview

- **Target**: Salary  
- **Features**: Age, Balance, Gender, Geography, Tenure, Credit Score, etc.

## 🔧 Preprocessing

- Filled missing values  
- One-hot encoded categorical features  
- Normalized data using `StandardScaler`  
- Train-test split

## 🧠 Model Architecture

- Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1)  
- Optimizer: Adam (lr=0.01)  
- Loss: Mean Squared Error (MSE)  
- Metric: Mean Absolute Error (MAE)  
- Includes EarlyStopping and TensorBoard  

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
