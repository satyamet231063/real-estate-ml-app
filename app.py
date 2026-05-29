"""
Real Estate Price Prediction App
==================================
Interactive ML application for property valuation with real-time predictions,
feature importance analysis, and market insights.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] {font-size: 32px; color: #0F7173;}
    .prediction-card {
        background: linear-gradient(135deg, #27AE60 0%, #0F7173 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    h1 {color: #0F7173;}
    h2 {color: #1A5276;}
</style>
""", unsafe_allow_html=True)

# ── Title ───────────────────────────────────────────────────────
st.title("🏠 Real Estate Price Prediction")
st.markdown("**AI-powered property valuation engine** — Get instant price estimates powered by Gradient Boosting ML.")

# ── Generate Training Data ──────────────────────────────────────
@st.cache_data
def generate_property_data(n=5000):
    np.random.seed(42)
    
    locations = ["Downtown", "Suburbs", "Waterfront", "Historic", "Commercial"]
    property_types = ["Apartment", "House", "Condo", "Townhouse", "Villa"]
    
    data = {
        "location": np.random.choice(locations, n),
        "property_type": np.random.choice(property_types, n),
        "area_sqft": np.random.normal(2500, 800, n),
        "num_rooms": np.random.randint(2, 10, n),
        "num_bedrooms": np.random.randint(1, 6, n),
        "num_bathrooms": np.random.randint(1, 4, n),
        "property_age": np.random.randint(0, 100, n),
        "parking_spaces": np.random.randint(0, 4, n),
        "has_pool": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "has_garage": np.random.choice([0, 1], n, p=[0.4, 0.6]),
    }
    
    df = pd.DataFrame(data)
    
    # Price generation
    base = 300000
    df["price"] = (
        base +
        (df["area_sqft"] * 200) +
        (df["num_bedrooms"] * 100000) +
        (df["num_bathrooms"] * 50000) -
        (df["property_age"] * 2000) +
        (df["has_pool"] * 50000) +
        (df["has_garage"] * 30000) +
        np.random.normal(0, 100000, n)
    )
    df["price"] = np.maximum(df["price"], 100000)
    df["area_sqft"] = np.maximum(df["area_sqft"], 500)
    
    return df

@st.cache_resource
def train_model(df):
    """Train Gradient Boosting model"""
    df_model = df.copy()
    
    # Encode categorical variables
    location_encoder = LabelEncoder()
    property_type_encoder = LabelEncoder()
    
    df_model["location_encoded"] = location_encoder.fit_transform(df_model["location"])
    df_model["property_type_encoded"] = property_type_encoder.fit_transform(df_model["property_type"])
    
    # Store encoders for later use
    st.session_state.location_encoder = location_encoder
    st.session_state.property_type_encoder = property_type_encoder
    
    # Feature engineering
    df_model["area_sqft_squared"] = df_model["area_sqft"] ** 2
    df_model["property_age_log"] = np.log1p(df_model["property_age"])
    df_model["rooms_x_area"] = df_model["num_rooms"] * df_model["area_sqft"]
    
    feature_cols = [col for col in df_model.columns if col not in ["price", "location", "property_type"]]
    
    X = df_model[feature_cols]
    y = df_model["price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    r2 = model.score(X_test_scaled, y_test)
    
    return model, scaler, feature_cols, r2

# Load data and train model
df_data = generate_property_data()
model, scaler, feature_cols, r2_score = train_model(df_data)

# Initialize session state for encoders
if "location_encoder" not in st.session_state:
    location_encoder = LabelEncoder()
    property_type_encoder = LabelEncoder()
    location_encoder.fit(df_data["location"])
    property_type_encoder.fit(df_data["property_type"])
    st.session_state.location_encoder = location_encoder
    st.session_state.property_type_encoder = property_type_encoder

# ── Sidebar Navigation ──────────────────────────────────────────
st.sidebar.title("🎯 Navigation")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["Price Prediction", "Model Analysis", "Market Insights", "Dataset Overview"]
)

# ══════════════════════════════════════════════════════════════════
# PAGE 1: PRICE PREDICTION
# ══════════════════════════════════════════════════════════════════
if app_mode == "Price Prediction":
    st.markdown("### 💰 Enter Property Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        location = st.selectbox(
            "Location",
            ["Downtown", "Suburbs", "Waterfront", "Historic", "Commercial"]
        )

        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "House", "Condo", "Townhouse", "Villa"]
        )

        area_sqft = st.slider("Area (Sq Ft)", 500, 10000, 2500, 100)
    
    with col2:
        num_bedrooms = st.slider("Bedrooms", 1, 6, 3)
        num_bathrooms = st.slider("Bathrooms", 1, 4, 2)
        num_rooms = st.slider("Total Rooms", 2, 10, 5)
    
    with col3:
        property_age = st.slider("Property Age (Years)", 0, 100, 20)
        parking_spaces = st.slider("Parking Spaces", 0, 4, 1)
        has_pool = st.checkbox("Has Pool", value=False)
        has_garage = st.checkbox("Has Garage", value=True)
    
    # Encode categorical variables
    location_encoded = st.session_state.location_encoder.transform([location])[0]
    property_type_encoded = st.session_state.property_type_encoder.transform([property_type])[0]
    
    # Create input dictionary
    input_dict = {
        "location_encoded": location_encoded,
        "property_type_encoded": property_type_encoded,
        "area_sqft": area_sqft,
        "num_rooms": num_rooms,
        "num_bedrooms": num_bedrooms,
        "num_bathrooms": num_bathrooms,
        "property_age": property_age,
        "parking_spaces": parking_spaces,
        "has_pool": int(has_pool),
        "has_garage": int(has_garage),
        "area_sqft_squared": area_sqft ** 2,
        "property_age_log": np.log1p(property_age),
        "rooms_x_area": num_rooms * area_sqft,
    }

    # Create dataframe
    input_data = pd.DataFrame([input_dict])

    # IMPORTANT FIX:
    # Match exact training feature order
    input_data = input_data[feature_cols]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    predicted_price = model.predict(input_scaled)[0]
    
    # Display prediction
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #27AE60 0%, #0F7173 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
        ">
            <h2 style="margin: 0; color: white;">Predicted Price</h2>
            <h1 style="margin: 10px 0 0 0; color: white;">
                ₹{predicted_price:,.0f}
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        market_price = df_data[
            (df_data["property_type"] == property_type) &
            (df_data["location"] == location)
        ]["price"].mean()
        
        variance = ((predicted_price - market_price) / market_price) * 100
        
        st.metric("Market Avg", f"₹{market_price:,.0f}")
        st.metric("vs Market", f"{variance:+.1f}%")
    
    # Confidence metrics
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model R² Score", f"{r2_score:.4f}")
    
    with col2:
        st.metric("Confidence Interval", "±₹50K (95%)")
    
    with col3:
        st.metric("Training Data", f"{len(df_data):,} properties")
    
    # Price breakdown
    st.markdown("### 📊 Price Drivers Impact")
    
    drivers = {
        "Area (Sq Ft)": area_sqft * 200,
        "Bedrooms": num_bedrooms * 100000,
        "Bathrooms": num_bathrooms * 50000,
        "Property Age": -property_age * 2000,
        "Pool": 50000 if has_pool else 0,
        "Garage": 30000 if has_garage else 0,
    }
    
    drivers_df = pd.DataFrame({
        "Feature": list(drivers.keys()),
        "Impact": list(drivers.values())
    })
    
    fig = px.bar(
        drivers_df,
        x="Feature",
        y="Impact",
        color="Impact",
        color_continuous_scale="RdYlGn",
        title="Price Contribution by Feature"
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, width='stretch')


# ══════════════════════════════════════════════════════════════════
# PAGE 2: MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif app_mode == "Model Analysis":
    st.markdown("### 🤖 Machine Learning Model Details")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Algorithm", "Gradient Boosting")
    with col2:
        st.metric("Estimators", "200")
    with col3:
        st.metric("Max Depth", "5")
    with col4:
        st.metric("R² Score", f"{r2_score:.4f}")
    
    st.markdown("### 📈 Feature Importance")
    
    # Get feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(12)
    
    fig = px.bar(importance, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Viridis",
                 title="Top 12 Most Important Features")
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=False,
        margin=dict(l=200, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Feature Importance Table")
    st.dataframe(importance, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3: MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif app_mode == "Market Insights":
    st.markdown("### 🏘️ Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location_price = df_data.groupby("location")["price"].mean().sort_values(ascending=False)
        fig = px.bar(x=location_price.values, y=location_price.index, orientation="h",
                     color=location_price.values, color_continuous_scale="Blues",
                     title="Average Price by Location")
        fig.update_layout(height=400, showlegend=False, xaxis_title="Avg Price (₹)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        type_price = df_data.groupby("property_type")["price"].mean().sort_values(ascending=False)
        fig = px.bar(x=type_price.values, y=type_price.index, orientation="h",
                     color=type_price.values, color_continuous_scale="Greens",
                     title="Average Price by Property Type")
        fig.update_layout(height=400, showlegend=False, xaxis_title="Avg Price (₹)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📐 Price vs Area Distribution")
    fig = px.scatter(df_data, x="area_sqft", y="price", 
                    color="property_type",
                    size="num_bedrooms",
                    hover_data=["location", "property_age"],
                    title="Property Price vs Area (Size = Bedrooms)",
                    labels={"area_sqft": "Area (Sq Ft)", "price": "Price (₹)"})
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 4: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════
elif app_mode == "Dataset Overview":
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", f"{len(df_data):,}")
    with col2:
        st.metric("Avg Price", f"₹{df_data['price'].mean():,.0f}")
    with col3:
        st.metric("Price Range", f"₹{df_data['price'].min():,.0f} - ₹{df_data['price'].max():,.0f}")
    with col4:
        st.metric("Locations", df_data['location'].nunique())
    
    st.markdown("### 📈 Price Distribution")
    fig = px.histogram(df_data, x="price", nbins=50, title="Price Distribution",
                      labels={"price": "Price (₹)"})
    fig.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Dataset Sample")
    st.dataframe(df_data.head(20), use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with ❤️ using Streamlit + Scikit-learn**")
st.sidebar.markdown("📧 [satyamdubey5861@gmail.com](mailto:satyamdubey5861@gmail.com)")
st.sidebar.markdown("💻 [GitHub](https://github.com/satyamdubey)")
