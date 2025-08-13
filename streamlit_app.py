import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Prediction App",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .main { font-family: 'Inter', sans-serif; }

/* Header */
.main-header {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg,#1f77b4,#2ca02c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-block-end: 1rem;
}

/* Description */
.main-description {
    text-align: center;
    color: #666;
    font-size: 1.1rem;
    line-height: 1.6;
    margin-block-end: 2rem;
    padding: 0 2rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg,#ffffff,#f8f9fa);
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    border-inline-start: 4px solid #1f77b4;
    margin-block-end: 1rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover { transform: translateY(-3px); }

/* Metric Title & Value */
.metric-title { font-size: 0.9rem; font-weight: 600; color: #6c757d; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #1f77b4; }
.metric-subtitle { font-size: 0.8rem; color: #95a5a6; }

/* Prediction Result Cards */
.prediction-result { padding: 2rem; border-radius: 16px; text-align: center; font-size: 1.3rem; font-weight: 600; margin: 1rem 0; }
.success-result { background: linear-gradient(135deg,#d4edda,#c3e6cb); border: 2px solid #28a745; color: #155724; }
.error-result { background: linear-gradient(135deg,#f8d7da,#f5c6cb); border: 2px solid #dc3545; color: #721c24; }

/* Sidebar */
.sidebar .sidebar-content { background-color: #1f77b4; color:white; }
.sidebar-title { font-size: 1.8rem; font-weight: 600; margin-block-end:1rem; }

/* Footer */
.footer { text-align:center; color:#6c757d; padding:2rem; background:#f8f9fa; border-block-start:3px solid #1f77b4; margin-block-start:2rem;  }

</style>
""", unsafe_allow_html=True)

# --------------------
# Load Dataset & Model
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Titanic-Dataset.csv")
    num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_resource
def load_model():
    with open("model.pkl","rb") as file:
        return pickle.load(file)

df = load_data()
model = load_model()

# --------------------
# Feature Engineering
# --------------------
if 'FamilySize' not in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
if 'IsAlone' not in df.columns:
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x==1 else 0)

# --------------------
# Sidebar Navigation
# --------------------
st.sidebar.markdown('<div class="sidebar-title">📚 Navigation</div>', unsafe_allow_html=True)
menu = st.sidebar.radio("Main Navigation",
                        ["🏠 Home","📊 Data Exploration","📈 Visualizations","🔮 Prediction","📋 Model Performance"],
                        index=0, label_visibility="collapsed")

# --------------------
# Home Page
# --------------------
if menu == "🏠 Home":
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction App</h1>', unsafe_allow_html=True)
    st.markdown('<div class="main-description">"Welcome to Titanic Explorer. Visualize. Predict. 🚢"</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-description">Dive into the Titanic dataset, uncover survival patterns, and test your own predictions with our AI-powered tool. Discover who might survive and why—all in one interactive app!</div>', unsafe_allow_html=True)

    st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg",
    use_container_width=True,
    caption="RMS Titanic - The Unsinkable Ship"
)

    st.markdown("---")
    st.header("Quick Stats Overview")
    col1, col2, col3 = st.columns(3)
    survived = df['Survived'].sum()
    total = len(df)
    died = total - survived

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Survivors</div><div class="metric-value">{survived}</div><div class="metric-subtitle">{survived/total:.1%} survival rate</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Passengers</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Deceased</div><div class="metric-value">{died}</div><div class="metric-subtitle">{died/total:.1%} death rate</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3 style="text-align:center;color:#1f77b4;">Quick Explore Sections</h3>', unsafe_allow_html=True)
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

    with exp_col1:
        st.markdown('<div class="metric-card" style="background:#00bcd4;color:white;">📊<br>Data Exploration</div>', unsafe_allow_html=True)
    with exp_col2:
        st.markdown('<div class="metric-card" style="background:#8bc34a;color:white;">📈<br>Visualizations</div>', unsafe_allow_html=True)
    with exp_col3:
        st.markdown('<div class="metric-card" style="background:#ff9800;color:white;">🔮<br>Predictions</div>', unsafe_allow_html=True)
    with exp_col4:
        st.markdown('<div class="metric-card" style="background:#e91e63;color:white;">📋<br>Model Performance</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("*Use the sidebar to navigate the app or click on the cards for a quick overview!*")


# --------------------
# Data Exploration
# --------------------
elif menu=="📊 Data Exploration":
    st.title("Data Exploration")
    st.write("Dataset preview:")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Summary statistics:")
    st.write(df.describe())
    
    st.subheader("Survival Rate")
    fig, ax = plt.subplots()
    df['Survived'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Died','Survived'], colors=['#e63946','#2a9d8f'], startangle=90, explode=[0.05,0.05], shadow=True, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)
    
# --------------------
# Visualizations
# --------------------
elif menu=="📈 Visualizations":
    st.title("Visualizations")
    st.subheader("Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("Survival by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1', ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# --------------------
# Model Prediction
# --------------------
elif menu=="🔮 Prediction":
    st.title("Predict Survival")
    pclass = st.selectbox("Passenger Class", [1,2,3])
    sex = st.selectbox("Sex", ["male","female"])
    sex = 1 if sex=="male" else 0
    age = st.slider("Age",0,80,25)
    fare = st.number_input("Fare",0.0,500.0,32.0)
    embarked = st.selectbox("Embarked", ["C","Q","S"])
    embarked_map = {"C":0,"Q":1,"S":2}
    embarked = embarked_map[embarked]
    family_size = st.slider("Family Size",1,10,1)
    is_alone = 1 if family_size==1 else 0

    if st.button("Predict"):
        try:
            features = np.array([[pclass,sex,age,fare,embarked,family_size,is_alone]])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0][pred] if hasattr(model,"predict_proba") else None
            if pred==1:
                st.markdown(f'<div class="prediction-result success-result">✅ Passenger would survive! {"Confidence: "+str(round(proba,2)) if proba else ""}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-result error-result">❌ Passenger would not survive. {"Confidence: "+str(round(proba,2)) if proba else ""}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --------------------
# Model Performance
# --------------------
elif menu=="📋 Model Performance":
    st.title("Model Performance")
    df_enc = df.copy()
    df_enc['Sex'] = df_enc['Sex'].map({'male':1,'female':0})
    df_enc['Embarked'] = df_enc['Embarked'].map({"C":0,"Q":1,"S":2})
    X = df_enc[['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone']]
    y = df_enc['Survived']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    st.markdown(f"**Accuracy:** {acc:.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test,y_pred))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.markdown('<div class="footer">Created by Kawya ❤</div>', unsafe_allow_html=True)

   



   