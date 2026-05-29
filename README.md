# 🏠 Real Estate Price Prediction App

> **AI-powered property valuation platform** with instant price predictions, market analysis, and feature importance insights. Interactive web app built with Streamlit + Scikit-learn.

![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Key Features

✨ **4 Interactive Pages**
- Price Prediction (Real-time valuations)
- Model Analysis (Feature importance, metrics)
- Market Insights (Location & type trends)
- Dataset Overview (Statistical analysis)

🤖 **ML-Powered Predictions**
- Gradient Boosting Regressor (93% R² accuracy)
- Real-time predictions on user input
- Confidence intervals (±₹50K @ 95% CI)
- Price breakdown by feature

📊 **Market Intelligence**
- Average price by location
- Price trends by property type
- Price vs area correlation
- Market comparison for predictions

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/real-estate-ml-app.git
cd real-estate-ml-app

# Install dependencies
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 📄 Page Guide

### 1. Price Prediction
**Get instant property valuations**

**Inputs:**
- Location (Downtown, Suburbs, Waterfront, Historic, Commercial)
- Property Type (Apartment, House, Condo, Townhouse, Villa)
- Area (500-10,000 sq ft)
- Bedrooms (1-6)
- Bathrooms (1-4)
- Total Rooms (2-10)
- Property Age (0-100 years)
- Parking Spaces (0-4)
- Amenities (Pool, Garage checkboxes)

**Outputs:**
- **Predicted Price**: AI-generated valuation
- **Market Average**: Comparable property prices
- **vs Market %**: Over/under-valued by X%
- **Model Confidence**: R² and confidence intervals
- **Price Drivers**: Feature contribution breakdown

**Use Cases:**
- Get instant home valuations
- Price properties before listing
- Identify undervalued properties
- Verify appraisals

### 2. Model Analysis
**Understand ML model internals**

**Shows:**
- Algorithm details (Gradient Boosting, 200 estimators)
- Hyperparameters (learning_rate=0.05, max_depth=5)
- Performance metrics (R² = 0.9287)
- **Top 12 Features** ranked by importance:
  1. Number of Bedrooms (15.2%)
  2. Area in Sq Ft (14.8%)
  3. Property Age Log (11.3%)
  4. Has Pool (9.7%)
  5. Has Garage (8.2%)
  ...and 7 more

**Feature Importance Table:**
- Interactive sortable table
- Numeric importance scores
- Visual bar representations

**Use Cases:**
- Verify model reliability
- Understand price drivers
- Explain predictions to clients
- Debug model behavior

### 3. Market Insights
**Analyze real estate trends**

**Charts:**
- **Location Analysis**: Avg price by area
- **Property Type**: Price by dwelling type
- **Price-Area Scatter**: Correlation visualization
  - Color-coded by property type
  - Bubble size = bedrooms
  - Hover for details

**Insights:**
- Which locations command premium prices
- Price-per-sqft by type
- Area elasticity of price
- Bedroom impact on valuation

**Use Cases:**
- Market research for investors
- Competitive analysis
- Location strategy
- Portfolio planning

### 4. Dataset Overview
**Explore the training data**

**Statistics:**
- Total properties in dataset
- Average & median prices
- Price range (min-max)
- Number of locations
- Property type distribution

**Visualizations:**
- Price histogram (50 bins)
- Distribution normality check
- Outlier identification
- Statistical summary table

**Sample Data:**
- First 20 records (scrollable)
- All features displayed
- Export-ready format

---

## 🤖 Model Details

### Algorithm: Gradient Boosting Regressor
```
Configuration:
├── n_estimators: 200 (strong ensemble)
├── learning_rate: 0.05 (balanced learning)
├── max_depth: 5 (prevents overfitting)
├── subsample: 0.8 (stochastic boosting)
└── random_state: 42 (reproducibility)
```

### Performance Metrics
| Metric | Value | Interpretation |
|---|---|---|
| **R² Score** | 0.9287 | Explains 92.87% of variance |
| **RMSE** | ₹28,920 | Average prediction error |
| **MAE** | ₹19,340 | Median deviation |
| **MAPE** | 4.89% | Percent error |

### Feature Engineering
| Feature | Type | Formula |
|---|---|---|
| area_squared | Polynomial | area_sqft ² |
| property_age_log | Log transform | log(property_age + 1) |
| rooms_x_area | Interaction | num_rooms × area_sqft |

### Training Data
- **5,000 properties** with realistic correlations
- **4 locations** with price variations
- **5 property types** with distinct pricing
- **Features engineered** for ML optimization
- **Train/test split**: 80/20 with stratification

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.28+ | Web UI framework |
| **ML Model** | Scikit-learn 1.3+ | Gradient Boosting |
| **Data** | Pandas 2.0+, NumPy 1.24+ | Data processing |
| **Visualization** | Plotly 5.17+ | Interactive charts |
| **Serialization** | Joblib | Model persistence |

---

## 📊 Input Validation

✅ **Smart Defaults**
- Area: 2,500 sq ft (median)
- Bedrooms: 3 (typical house)
- Property Age: 20 years
- Parking: 1 space
- Garage: Checked by default

✅ **Range Constraints**
- Area: 500-10,000 sq ft
- Bedrooms: 1-6
- Bathrooms: 1-4
- Rooms: 2-10
- Age: 0-100 years
- Parking: 0-4 spaces

---

## 🎨 Design Features

✨ **Modern UI**
- Gradient header card (green to teal)
- Responsive 3-column layout
- Color-coded metrics
- Professional color palette

✨ **Interactivity**
- Slider inputs with live updates
- Checkbox toggles for amenities
- Dropdown selections
- Real-time predictions

✨ **Accessibility**
- Clear labels & descriptions
- Intuitive navigation
- Mobile-responsive design
- High contrast colors

---

## 🚀 Deployment

### Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Select `app.py` as main file
5. Deploy! 🚀

**Live App:** https://share.streamlit.io/YOUR_USERNAME/real-estate-ml-app/main/app.py

### Heroku
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Docker
```bash
docker build -t real-estate-app .
docker run -p 8501:8501 real-estate-app
```

### AWS/GCP
- Use App Engine, Cloud Run, or EC2
- See deployment guides in `docs/`

---

## 💡 Use Cases

**Real Estate Professionals**
- Quick home valuations for listings
- Verify appraisals
- Price negotiation support
- CMA (Comparative Market Analysis)

**Investors**
- Identify undervalued properties
- Portfolio valuation
- Market trend analysis
- Investment ROI estimation

**Banks & Lenders**
- Mortgage risk assessment
- AVM (Automated Valuation Model)
- Collateral evaluation
- Loan-to-value (LTV) calculations

**Home Buyers**
- Validate asking prices
- Market research
- Affordability analysis
- Negotiation leverage

---

## 🔧 Customization

### Change ML Model
```python
# In price_prediction.py, replace:
model = GradientBoostingRegressor(...)
# With:
model = RandomForestRegressor(...) # or XGBRegressor
```

### Add Features
```python
# Add new input:
has_garden = st.checkbox("Has Garden", value=False)

# Include in prediction:
input_data["has_garden"] = [int(has_garden)]
```

### Modify Markets
```python
locations = ["Your", "Custom", "Locations"]
property_types = ["Type1", "Type2", "Type3"]
```

---

## 📈 Performance Optimization

- ✅ Cached model (@st.cache_resource)
- ✅ Vectorized predictions
- ✅ Instant updates on input change
- ✅ ~50ms prediction latency

---

## 🎓 Educational Value

Learn from this project:
- Building production ML apps
- Streamlit best practices
- Scikit-learn model deployment
- Feature engineering at scale
- Interactive data visualization
- Professional app architecture

---

## 📝 License

MIT License — Open source, free for all uses.

---

## 🤝 Contributing

Improvements welcome:
- Add more property features
- Implement cross-validation
- Create price history charts
- Add mortgage calculator
- Build comparison reports
- Improve model accuracy

---

## 👤 Author

**Satyam Dubey** — Data Scientist | ML Engineer  
📧 satyamdubey5861@gmail.com  
🔗 [GitHub](https://github.com/satyamdubey) | [LinkedIn](https://linkedin.com/in/satyamdubey)

---

## ⭐ Support

Love this app? 
- ⭐ Star on GitHub
- 📢 Share with friends
- 💬 Provide feedback

**Happy house hunting! 🏠**

