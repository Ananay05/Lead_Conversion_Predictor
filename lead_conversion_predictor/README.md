# 🎯 Lead Conversion Prediction System

> A college Machine Learning project that predicts whether a sales lead will convert using **Logistic Regression** and **Decision Tree** classifiers, with a live **Streamlit** web app and **SQLite** database for real-time tracking.

---

## 📸 What It Does

- Takes in lead information (age, income, source, website behaviour, engagement signals)
- Predicts if the lead will **convert** and shows a probability score
- Saves every prediction to a **SQLite database**
- Has a **dashboard** to visualize all historical predictions
- Lets you compare performance between two ML models

---

## 📁 Project Structure

```
lead_conversion_predictor/
│
├── app/
│   └── app.py                  ← Streamlit web application (main entry point)
│
├── data/
│   ├── generate_data.py        ← Script to create the synthetic dataset
│   └── leads.csv               ← Generated dataset (1000 rows)
│
├── models/
│   ├── train_model.py          ← Train & save both ML models
│   └── model_artifacts.pkl     ← Saved models, scaler, encoders (auto-generated)
│
├── notebooks/
│   └── EDA.ipynb               ← Exploratory Data Analysis notebook
│
├── utils/
│   ├── __init__.py
│   ├── db_utils.py             ← SQLite helper (save/fetch predictions)
│   └── predict_utils.py        ← Load models and run inference
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone or download the project

```bash
# If using git
git clone <your-repo-url>
cd lead_conversion_predictor

# Or just unzip the folder and open terminal inside it
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Generate the dataset

```bash
cd data
python generate_data.py
cd ..
```

You should see:
```
Dataset generated: 1000 rows
Conversion rate: 48.00%
```

---

### Step 2 — Train the models

```bash
cd models
python train_model.py
cd ..
```

You should see accuracy and ROC-AUC printed for both models, and a `model_artifacts.pkl` file gets saved.

---

### Step 3 — Launch the Streamlit app

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 🔮 **Predict** | Fill in lead details and get an instant prediction |
| 📊 **Dashboard** | Visual overview of all predictions made so far |
| 🗄️ **History** | Full table of all predictions with CSV export |
| 📈 **Model Info** | Accuracy, ROC-AUC, confusion matrix for both models |

---

## 🛠️ Tech Stack

| Tool | Use |
|------|-----|
| Python 3.9+ | Core language |
| Pandas & NumPy | Data preprocessing, feature engineering |
| Scikit-learn | Logistic Regression, Decision Tree, preprocessing |
| Streamlit | Web application frontend |
| SQLite (via `sqlite3`) | Real-time prediction storage |
| Matplotlib & Seaborn | Data visualization |
| Jupyter Notebook | EDA and exploration |

---

## 📊 Features Used for Prediction

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Age of the lead |
| `income` | Numeric | Annual income |
| `lead_source` | Categorical | How they found us (Organic, Paid Ad, etc.) |
| `website_visits` | Numeric | Number of site visits |
| `time_spent_on_site` | Numeric | Minutes spent on website |
| `pages_viewed` | Numeric | Number of pages viewed |
| `email_opened` | Binary | Did they open our email? |
| `previous_interaction` | Binary | Had a prior interaction? |
| `lead_score` | Numeric | Internal score (0–100) |
| `industry` | Categorical | Industry sector |
| `follow_up_calls` | Numeric | Number of follow-up calls made |

---

## 📒 Notebooks

To run the EDA notebook:

```bash
pip install jupyter
jupyter notebook notebooks/EDA.ipynb
```

It covers:
- Dataset overview
- Target class distribution
- Numerical feature distributions
- Correlation heatmap
- Lead Source vs Conversion Rate
- Lead Score distribution by outcome

---

## 🔧 Troubleshooting

**ModuleNotFoundError?**
Make sure you're running from the project root and your virtual environment is activated.

**`model_artifacts.pkl` not found?**  
You need to run `train_model.py` first (Step 2 above).

**App crashes on first launch?**  
Make sure `leads.csv` exists in the `data/` folder (run `generate_data.py` first).

---

## 📌 Notes

- The dataset is **synthetically generated** for demo purposes. In a real project, you'd replace `leads.csv` with actual CRM data.
- Model accuracy is intentionally realistic (~59–60%) due to the random nature of synthetic data.
- The SQLite database (`predictions.db`) is auto-created inside the `app/` folder on first run.

---

## 👤 Author

**[Your Name]**  
**Roll No:** [Your Roll Number]  
**Branch:** [CSE / IT / Data Science]  
**College:** [Your College Name]  
**Subject:** Machine Learning / Data Science

---

*This project was built as part of a college course assignment. All data used is synthetic.*
