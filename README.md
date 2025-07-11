# 🚗 Car Price Prediction using Machine Learning & Optuna Hyperparameter Tuning

This project focuses on predicting the selling price of used cars using **supervised machine learning models**. The solution uses a combination of **data preprocessing, feature engineering, and advanced hyperparameter tuning** with **Optuna** to achieve high accuracy.

---

## 📌 Problem Overview

- The goal is to build a regression model to **predict used car prices** based on historical listings.
- The dataset is sourced from **Quikr Cars India**.

---

## 📊 Dataset Summary

- **Rows:** 817 car listings (after cleaning)
- **Columns:** 6 relevant features:
  | Feature       | Description                                |
  |-------------- |-------------------------------------------- |
  | `name`        | Car name and model                          |
  | `company`     | Car manufacturer                             |
  | `year`        | Year of manufacture                          |
  | `Price`       | Selling price (in INR)                       |
  | `kms_driven`  | Kilometers driven                            |
  | `fuel_type`   | Fuel type (Petrol, Diesel, etc.)             |

- The target variable is **`Price`**.

---

## 🚀 Technologies Used

- Python 🐍
- Pandas & NumPy 📊
- Scikit-learn 🤖
- XGBoost, LightGBM 🌲
- Optuna ⚙
- Matplotlib & Seaborn 📈
- Jupyter Notebook

---

## 🔍 Exploratory Data Analysis (EDA) & Data Cleaning

- ✔ Removed missing and invalid entries (e.g., rows with `year = 'sale'`).
- ✔ Converted `Price` and `kms_driven` to numeric after cleaning text like commas and "kms".
- ✔ Filled missing values in `fuel_type` using **mode**.
- ✔ Simplified car `name` to first three words (to group similar models).
- ✔ Removed extreme outliers in price (> ₹50 lakh) and kilometers (> 1.9 lakh km).
- ✔ Applied **log transformation** to target (`Price`) to normalize skewed data.

---

## ⚙ Feature Engineering & Preprocessing

- **Categorical features:** `fuel_type`, `company`, `name` → OneHotEncoder
- **Numerical features:** `kms_driven`, `year` → StandardScaler
- Combined preprocessing and model using **Scikit-learn Pipelines** for seamless processing.

---

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

| Model Name           | Library       |
|----------------------|-------------- |
| **XGBoost Regressor** | XGBoost       |
| **Random Forest**     | Scikit-learn  |
| **LightGBM**          | LightGBM      |
| **Linear Regression** | Scikit-learn  |

---

## 🔍 Hyperparameter Tuning with Optuna

- Used **Optuna** for automatic hyperparameter tuning and **model selection**.
- Tuned:
  - Preprocessing splits: `test_size`, `random_state`
  - Model-specific hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- Evaluated using **R² Score**.

---

## 📈 Best Model Performance

| Metric     | Value      |
|----------- |----------- |
| **Best R²** | **0.8803** |

- The best performing model was selected and saved as a **pipeline** including preprocessing + model.

---

## 💾 Final Model Pipeline

The best pipeline includes:
1. **Preprocessor**: OneHotEncoder + StandardScaler
2. **Best Regressor Model** (from Optuna trials)
3. Saved as:  
```bash
# 1️⃣ Clone the repository:
git clone https://github.com/Jatin-GI/car-price-prediction.git
cd car-price-prediction

# 2️⃣ (Optional) Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On Mac/Linux

# 3️⃣ Install required packages:
pip install -r requirements.txt

# 4️⃣ Run Jupyter Notebook:
jupyter notebook
```

## ✅ Key Takeaways

- Built an **end-to-end regression pipeline** combining data cleaning, feature engineering, model training, and evaluation.
- Used **Optuna** for automated **hyperparameter tuning** and **model selection**, significantly improving model performance.
- Achieved a strong **R² Score of 0.8803** on the test set (log-transformed prices).
- The pipeline is fully reusable and can be directly deployed or integrated into a larger application.
- Demonstrated the value of combining **exploratory data analysis** with **machine learning automation** for business-ready solutions.

---

## 🚀 Future Improvements

- 💻 **Deployment:** Build an interactive **Streamlit** or **Flask** web app for real-time car price predictions.
- 🔄 **Advanced Models:** Experiment with algorithms like **CatBoost**, **TabNet**, or **AutoML** solutions.
- 📊 **Explainability:** Integrate **SHAP** or **LIME** to explain model predictions to end-users.
- 🛢 **Data Expansion:** Collect more car listings to improve model robustness and generalization across different markets.
- ⏱ **Real-time Serving:** Deploy the model as a microservice using **FastAPI** or **Docker** for production-level inference.

---

## 🤝 Contributions

Contributions are warmly welcome!  
Here’s how you can help:

- ⭐ **Star this repository** if you find it useful.
- 🐛 **Report bugs** or suggest new features via issues.
- 🔀 **Submit pull requests** with improvements, better visualizations, or new models.

Please ensure your code follows best practices and includes proper explanations or documentation.

---

## 📜 License

This project is provided **for educational and research purposes only**.  
All content is shared **"as-is" without warranties or guarantees**.  
Please use responsibly, especially when applying to real-world scenarios involving financial decisions.

---

