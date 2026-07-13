"""
Credit Card Balance Predictor
A Streamlit app that predicts a customer's credit card balance using a
Ridge Regression model trained on the Credit dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Balance Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — credit-card themed styling
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #f1f1f6 !important;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(120deg, #7b2ff7 0%, #f107a3 100%);
    border-radius: 20px;
    padding: 28px 34px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(123, 47, 247, 0.35);
}
.hero-banner h1 {
    margin: 0;
    font-size: 2.1rem;
}
.hero-banner p {
    margin: 6px 0 0 0;
    color: rgba(255,255,255,0.9);
    font-size: 1.02rem;
}

/* Virtual credit card */
.credit-card {
    width: 100%;
    max-width: 440px;
    height: 250px;
    border-radius: 20px;
    padding: 26px 28px;
    background: linear-gradient(135deg, #232526 0%, #414345 100%);
    box-shadow: 0 15px 35px rgba(0,0,0,0.45), inset 0 0 40px rgba(255,255,255,0.03);
    color: #f5f5f5;
    position: relative;
    font-family: 'Space Mono', monospace;
    margin: 0 auto 20px auto;
}
.credit-card.tier-low   { background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); }
.credit-card.tier-mid   { background: linear-gradient(135deg, #ee9ca7 0%, #ffdde1 100%); color: #3a2a2f; }
.credit-card.tier-high  { background: linear-gradient(135deg, #870000 0%, #190a05 100%); }

.card-chip {
    width: 45px; height: 34px;
    border-radius: 6px;
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    margin-bottom: 22px;
}
.card-network {
    position: absolute;
    top: 26px;
    right: 28px;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 1px;
    opacity: 0.85;
}
.card-number {
    font-size: 1.35rem;
    letter-spacing: 4px;
    margin: 10px 0 26px 0;
}
.card-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.card-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    opacity: 0.7;
    letter-spacing: 1px;
    margin-bottom: 3px;
}
.card-value {
    font-size: 0.95rem;
    font-weight: 700;
}

/* Prediction result box */
.prediction-box {
    border-radius: 18px;
    padding: 26px 30px;
    margin-top: 10px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}
.prediction-box.low   { background: linear-gradient(120deg, #11998e 0%, #38ef7d 100%); }
.prediction-box.mid   { background: linear-gradient(120deg, #f7971e 0%, #ffd200 100%); }
.prediction-box.high  { background: linear-gradient(120deg, #eb3349 0%, #f45c43 100%); }
.prediction-box h2 {
    color: #10101a !important;
    margin: 0;
    font-size: 2.6rem;
}
.prediction-box p {
    color: #10101a;
    margin: 4px 0 0 0;
    font-weight: 600;
    font-size: 1.05rem;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.65);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}

.section-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 18px;
}

div.stButton > button {
    background: linear-gradient(120deg, #7b2ff7 0%, #f107a3 100%);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    width: 100%;
    box-shadow: 0 6px 18px rgba(123, 47, 247, 0.4);
    transition: transform 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    color: white;
    border: none;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Data + model (cached so it only trains once per session)
# ------------------------------------------------------------------
NUMERIC_COLS = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']
FEATURE_COLUMNS = NUMERIC_COLS + [
    'Gender_Male', 'Student_Yes', 'Married_Yes',
    'Ethnicity_Asian', 'Ethnicity_Caucasian'
]


@st.cache_resource(show_spinner="Training the balance prediction model...")
def load_and_train():
    df = pd.read_csv("Credit_Data.csv")
    df = df.drop(columns=['ID'])

    df['Income'] = df['Income'].fillna(df['Income'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    for c in ['Student', 'Married']:
        df[c] = df[c].fillna(df[c].mode()[0])

    df_encoded = pd.get_dummies(df, columns=['Gender', 'Student', 'Married', 'Ethnicity'], drop_first=True)

    X = df_encoded.drop(columns=['Balance'])
    y = df_encoded['Balance']
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test_scaled[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    model = Ridge(alpha=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)

    test_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    importance = pd.Series(np.abs(model.coef_), index=FEATURE_COLUMNS).sort_values(ascending=False)

    return model, scaler, r2, mae, rmse, importance, df


model, scaler, r2, mae, rmse, importance, raw_df = load_and_train()


def build_input_row(income, limit, rating, cards, age, education, gender, student, married, ethnicity):
    row = {col: 0 for col in FEATURE_COLUMNS}
    row['Income'] = income
    row['Limit'] = limit
    row['Rating'] = rating
    row['Cards'] = cards
    row['Age'] = age
    row['Education'] = education
    if gender == 'Male':
        row['Gender_Male'] = 1
    if student == 'Yes':
        row['Student_Yes'] = 1
    if married == 'Yes':
        row['Married_Yes'] = 1
    if ethnicity == 'Asian':
        row['Ethnicity_Asian'] = 1
    elif ethnicity == 'Caucasian':
        row['Ethnicity_Caucasian'] = 1
    input_df = pd.DataFrame([row])[FEATURE_COLUMNS]
    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])
    return input_df


def balance_tier(predicted_balance, limit):
    """Classify predicted balance into low / mid / high risk tiers."""
    utilization = predicted_balance / limit if limit > 0 else 0
    if predicted_balance <= 300 or utilization < 0.15:
        return "low", "🟢 Low Balance", "This customer is likely to carry little to no revolving balance — a good candidate for rewards products rather than interest-bearing offers."
    elif predicted_balance <= 900 or utilization < 0.45:
        return "mid", "🟡 Moderate Balance", "This customer is likely to carry a moderate balance — a reasonable, steady interest-revenue contributor."
    else:
        return "high", "🔴 High Balance", "This customer is likely to carry a large balance — a strong interest-revenue contributor, but worth monitoring for credit risk."


# ------------------------------------------------------------------
# Sidebar — inputs
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💳 Customer Profile")
    st.caption("Fill in the customer's details to predict their credit card balance.")

    st.markdown("### 💰 Financial Info")
    income = st.slider("Annual Income ($1,000s)", 5.0, 200.0, 45.0, step=0.5,
                        help="Customer's annual income in thousands of dollars.")
    limit = st.slider("Credit Limit ($)", 500, 15000, 4700, step=50,
                       help="The customer's total credit limit.")
    rating = st.slider("Credit Rating", 90, 1000, 350, step=5,
                        help="Internal credit rating score — closely tracks credit limit.")
    cards = st.slider("Number of Credit Cards", 1, 9, 3)

    st.markdown("### 🧑 Personal Info")
    age = st.slider("Age", 18, 100, 40)
    education = st.slider("Years of Education", 5, 20, 13)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    student = st.radio("Student?", ["No", "Yes"], horizontal=True)
    married = st.radio("Married?", ["No", "Yes"], horizontal=True)
    ethnicity = st.selectbox("Ethnicity", ["Caucasian", "Asian", "African American"])

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Balance")

    with st.expander("ℹ️ About this model"):
        st.write(
            "This app uses a **Ridge Regression** model trained on 400 "
            "credit card customers. Ridge was chosen because `Limit` and "
            "`Rating` are almost perfectly correlated (r ≈ 0.997), and "
            "Ridge's regularization stabilizes predictions under that "
            "multicollinearity better than plain Linear Regression."
        )

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>💳 Credit Card Balance Predictor</h1>
    <p>Estimate a customer's expected credit card balance from their financial & demographic profile.</p>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.3], gap="large")

# ---- Left column: virtual card preview ----
with left_col:
    st.markdown("#### Customer Card Preview")

    masked_number = "•••• •••• •••• " + str(1000 + cards * 111)[-4:]
    tier_class_preview = "tier-low" if income < 40 else ("tier-mid" if income < 100 else "tier-high")

    st.markdown(f"""
    <div class="credit-card {tier_class_preview}">
        <div class="card-chip"></div>
        <div class="card-network">VISA</div>
        <div class="card-number">{masked_number}</div>
        <div class="card-row">
            <div>
                <div class="card-label">Cardholder</div>
                <div class="card-value">{gender.upper()} · {"STUDENT" if student == "Yes" else "CUSTOMER"}</div>
            </div>
            <div>
                <div class="card-label">Credit Limit</div>
                <div class="card-value">${limit:,.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="label">Income</div>
        <div class="value">${income:.0f}K</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="label">Rating</div>
        <div class="value">{rating}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="label">Age</div>
        <div class="value">{age}</div></div>""", unsafe_allow_html=True)

# ---- Right column: prediction ----
with right_col:
    st.markdown("#### Predicted Balance")

    if predict_clicked:
        input_row = build_input_row(income, limit, rating, cards, age, education,
                                     gender, student, married, ethnicity)
        prediction = model.predict(input_row)[0]
        prediction = max(prediction, 0)  # balances can't be negative
        tier, tier_label, tier_msg = balance_tier(prediction, limit)

        st.markdown(f"""
        <div class="prediction-box {tier}">
            <h2>${prediction:,.0f}</h2>
            <p>{tier_label}</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.info(tier_msg)

        utilization = min(prediction / limit, 1.0) if limit > 0 else 0
        st.markdown("**Estimated Credit Utilization**")
        st.progress(utilization)
        st.caption(f"{utilization*100:.1f}% of the ${limit:,.0f} credit limit")
    else:
        st.markdown("""
        <div class="section-card" style="text-align:center; padding: 40px 20px;">
            <p style="font-size:1.05rem; opacity:0.85;">
            👈 Fill in the customer's details in the sidebar, then click
            <b>"Predict Balance"</b> to see the result here.
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Model performance + feature importance
# ------------------------------------------------------------------
perf_col, imp_col = st.columns([1, 1.3], gap="large")

with perf_col:
    st.markdown("#### 📊 Model Performance (Test Set)")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"""<div class="metric-card"><div class="label">R² Score</div>
        <div class="value">{r2:.3f}</div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class="metric-card"><div class="label">MAE</div>
        <div class="value">${mae:,.0f}</div></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""<div class="metric-card"><div class="label">RMSE</div>
        <div class="value">${rmse:,.0f}</div></div>""", unsafe_allow_html=True)
    st.caption("Ridge Regression, tuned alpha = 0.1, evaluated on a held-out 20% test split.")

with imp_col:
    st.markdown("#### 🔑 What Drives Balance Predictions")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(importance)))
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.2)
    ax.set_xlabel("Relative Importance (|coefficient|)", color='white', fontsize=9)
    st.pyplot(fig, transparent=True)

st.markdown("---")
st.caption("Built with Streamlit · Model: Ridge Regression · Dataset: Credit Card Customers (400 rows)")
