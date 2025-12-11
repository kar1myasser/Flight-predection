# Airline Flight Price Prediction - Final Project

## Project Overview

This project implements a machine learning solution to predict airline flight prices based on various flight characteristics. The project includes comprehensive data analysis, preprocessing, feature engineering, and comparison of multiple regression models to identify the best performing algorithm.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Development](#model-development)
- [Results](#results)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)

---

## Dataset Description

**File:** `airlines_flights_data.csv`

The dataset contains **300,154 flight records** with the following features:

### Original Features

| Feature | Type | Description |
|---------|------|-------------|
| `index` | Numeric | Record identifier (dropped during preprocessing) |
| `airline` | Categorical | Airline company name |
| `flight` | Categorical | Flight code (dropped during preprocessing) |
| `source_city` | Categorical | Departure city |
| `departure_time` | Categorical | Time period of departure |
| `stops` | Categorical | Number of stops (zero, one, two_or_more) |
| `arrival_time` | Categorical | Time period of arrival (dropped during preprocessing) |
| `destination_city` | Categorical | Arrival city |
| `class` | Categorical | Travel class (Economy/Business) |
| `duration` | Numeric | Flight duration in hours |
| `days_left` | Numeric | Days until departure |
| `price` | Numeric | **Target variable** - Ticket price |

### Data Characteristics

- **No missing values** in any column
- **No duplicate records**
- Price originally in INR (Indian Rupees), converted to USD (₹87.04 = $1)
- Multiple airlines: SpiceJet, AirAsia, Vistara, Air India, Indigo, GO_FIRST
- Six departure time periods: Early_Morning, Morning, Afternoon, Evening, Night, Late_Night

---

## Project Structure

```
final project/
├── final project[1].ipynb          # Main Jupyter notebook
├── airlines_flights_data.csv       # Dataset (300K+ records)
├── app.py                          # Streamlit web application
├── best_model.pkl                  # Trained Random Forest model (37MB)
├── scaler.pkl                      # StandardScaler for preprocessing
├── requirements.txt                # Python dependencies
├── save_models.py                  # Helper script to save models
├── run_app.bat                     # Windows batch file to run app
├── DEPLOYMENT.md                   # Deployment guide
├── README.md                       # This documentation
└── .streamlit/
    └── config.toml                 # Streamlit configuration
```

---

## Installation & Requirements

### Required Libraries

```python
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Installation

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## Data Processing Pipeline

### 1. Data Loading & Initial Exploration

```python
df = pd.read_csv("airlines_flights_data.csv")
df.info()
df.isna().sum()
df.duplicated().sum()
```

**Findings:**
- ✅ No missing values
- ✅ No duplicates
- Dataset ready for processing

### 2. Feature Engineering

#### Dropped Features
```python
df.drop(["arrival_time", 'flight', 'index'], axis=1, inplace=True)
```

**Rationale:**
- `index`: Redundant identifier
- `flight`: Flight code doesn't contribute to price prediction
- `arrival_time`: High correlation with departure_time + duration

#### Price Conversion
```python
df['price'] = (df['price']/87.04).round()
```

Converted prices from INR to USD for better interpretability.

### 3. Feature Categorization

**Categorical Features (6):**
- `airline`, `source_city`, `departure_time`, `destination_city`, `class`, `stops`

**Numerical Features (2):**
- `duration`, `days_left`

### 4. Outlier Treatment

Applied **IQR method** to `duration` feature:

```python
q1 = df['duration'].quantile(0.25)
q3 = df['duration'].quantile(0.75)
iqr = q3 - q1 
lower_bound = q1 - iqr * 1.5
upper_bound = q3 + iqr * 1.5

df['duration'] = df['duration'].clip(lower=lower_bound, upper=upper_bound)
```

### 5. Encoding

**Label Encoding** applied to all categorical features using `LabelEncoder()`.

### 6. Feature Scaling

**StandardScaler** applied to ensure all features have:
- Mean = 0
- Standard deviation = 1

```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

---

## Exploratory Data Analysis

### Visualizations Implemented

#### 1. Price Distribution
- **Type:** Histogram with KDE
- **Finding:** Price distribution shows right-skewed pattern

#### 2. Flights per Airline
- **Type:** Bar chart
- **Finding:** Distribution of flights across different airlines

#### 3. Departure Times Analysis
- **Type:** Bar chart
- **Finding:** Distribution of flights across different time periods

#### 4. City Analysis
- **Source Cities:** Distribution of departure cities
- **Destination Cities:** Distribution of arrival cities

#### 5. Price Analysis

##### Average Price by Airline
- Bar chart showing which airlines charge more on average

##### Price vs Departure Time
- Bar chart (descending order)
- **Insight:** Certain departure times command premium prices

##### Price vs Days Left
- Line plot showing price trend
- **Insight:** Price changes as departure date approaches

##### Economy vs Business
- **Finding:** Business class significantly more expensive than Economy

##### Price vs Number of Stops
- Bar chart showing impact of layovers
- **Insight:** Direct flights (zero stops) typically priced differently than multi-stop flights

#### 6. Outlier Detection
- Box plots for all numerical features (`duration`, `days_left`, `price`)

### Color Scheme

Custom color gradient (teal to salmon) for consistent and professional visualizations:

```python
import matplotlib.colors as mcolors
colors = mcolors.LinearSegmentedColormap.from_list("", ["teal", "salmon"])
```

---

## Model Development

### Train-Test Split

```python
x = df.drop('price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
```

- **Training Set:** 70%
- **Test Set:** 30%
- **Random State:** 101 (for reproducibility)

### Models Implemented

Four regression models were trained and compared:

#### 1. Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor()
RF_model.fit(x_train_scaled, y_train)
```

**Advantages:**
- Handles non-linear relationships
- Resistant to overfitting
- Feature importance insights

#### 2. Linear Regression

```python
from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()
reg_model.fit(x_train_scaled, y_train)
```

**Advantages:**
- Simple and interpretable
- Fast training
- Baseline model

#### 3. Support Vector Regression (SVM)

```python
from sklearn.svm import SVR

svm = SVR()
svm.fit(x_train_scaled, y_train)
```

**Advantages:**
- Effective in high-dimensional spaces
- Robust to outliers

#### 4. XGBoost Regressor

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
xgb_model.fit(x_train_scaled, y_train)
```

**Advantages:**
- State-of-the-art performance
- Handles missing values
- Built-in regularization

### Evaluation Metrics

For each model, we calculated:

1. **R² Score (Coefficient of Determination)**
   - Percentage of variance explained by the model
   - Range: 0-100% (higher is better)

2. **Mean Squared Error (MSE)**
   - Average squared difference between predictions and actual values
   - Lower is better

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(x_test_scaled)
print(f"R2: {r2_score(y_test, y_pred) * 100:.2f}%")
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## Results

### Model Performance Comparison

| Model | R² Score | Performance |
|-------|----------|-------------|
| **Random Forest** | ~98-99% | ⭐⭐⭐⭐⭐ Excellent |
| **XGBoost** | ~98-99% | ⭐⭐⭐⭐⭐ Excellent |
| **Linear Regression** | ~85-90% | ⭐⭐⭐ Good |
| **SVM** | ~80-85% | ⭐⭐ Fair |

### Best Performing Models

**Random Forest** and **XGBoost** emerged as the top performers with R² scores in the **98-99% range**, indicating excellent predictive capability.

### Visualization

A comparative bar plot displays all model performances:

```python
models = ["SVM", "Linear Regression", "Random Forest", "XGBooost"]
scores = [R2_for_svm, R2_for_Linear_Regression, R2_for_RF, R2_for_XGboost]

sns.barplot(x=models, y=scores, palette="coolwarm")
plt.ylim(80, 100)
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2)")
```

---

## Usage

### 🌐 Live Demo

**Try the deployed app:** [https://flight-predection-aks6zgm9ciwpyjnfdtv4w7.streamlit.app](https://flight-predection-aks6zgm9ciwpyjnfdtv4w7.streamlit.app)

### Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kar1myasser/Flight-predection.git
   cd Flight-predection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Or on Windows, double-click `run_app.bat`

4. **Open your browser:**
   Navigate to `http://localhost:8501`

### Running the Jupyter Notebook

1. **Open the notebook:**
   ```bash
   jupyter notebook "final project[1].ipynb"
   ```

2. **Execute cells sequentially:**
   - Import libraries
   - Load and explore data
   - Visualize data
   - Preprocess features
   - Train models
   - Evaluate and compare
   - Save optimized model

### Making Predictions via Web App

1. Select flight details (airline, source, destination, etc.)
2. Choose departure time and travel class
3. Enter duration and days until departure
4. Click "Predict Flight Price"
5. View estimated price with insights

### Making Predictions via Code

```python
import pickle
import numpy as np

# Load model and scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input (encoded values)
new_flight = np.array([[airline_encoded, source_encoded, departure_encoded,
                        stops_encoded, destination_encoded, class_encoded,
                        duration, days_left]])

# Scale and predict
new_flight_scaled = scaler.transform(new_flight)
predicted_price = model.predict(new_flight_scaled)[0]
print(f"Predicted Price: ${predicted_price:.2f} USD (₹{predicted_price * 87.04:.0f})")
```

---

## Key Insights

### From EDA

1. **Departure Time Impact:** Certain time slots command premium pricing
2. **Days Left Correlation:** Price varies significantly based on booking advance
3. **Class Premium:** Business class substantially more expensive than Economy
4. **Stops Effect:** Number of layovers directly impacts pricing
5. **Airline Variation:** Different airlines have different pricing strategies
6. **Duration Factor:** Longer flights generally cost more

### From Model Training

1. **Tree-based models** (Random Forest, XGBoost) outperform linear models
2. **Ensemble methods** provide superior accuracy for this problem
3. **Feature scaling** is critical for SVM and Linear Regression performance
4. **Non-linear relationships** exist between features and price

---

## Future Improvements

### 1. Model Enhancements

- **Hyperparameter tuning** using GridSearchCV or RandomizedSearchCV
- **Cross-validation** for more robust performance estimates
- **Feature importance analysis** to identify key price drivers
- **Ensemble stacking** combining multiple models

### 2. Feature Engineering

- **Time-based features:** Month, day of week, season
- **Route popularity:** Frequency of specific routes
- **Price history:** Historical pricing patterns
- **Demand indicators:** Holiday periods, events

### 3. Advanced Analysis

- **Temporal analysis:** Price trends over time
- **Route-specific models:** Specialized models for high-traffic routes
- **Competitor analysis:** Cross-airline price comparison
- **Dynamic pricing:** Real-time prediction updates

### 4. Deployment

- ✅ **Streamlit Cloud deployment** - Currently live!
- **REST API** for real-time predictions (FastAPI)
- **Docker containerization** for consistent environments
- **Model monitoring** for performance tracking
- **A/B testing** for model comparison in production
- **Custom domain** setup

### 5. Additional Features

- **Confidence intervals** for predictions
- **Feature contribution explanation** (SHAP values)
- **Alternative metrics:** MAE, RMSE, MAPE
- **Segmentation analysis:** Different models for different customer segments

---

## Technical Notes

### Reproducibility

- Random state set to **101** for consistent results
- All preprocessing steps documented
- Model parameters explicitly stated

### Performance Considerations

- Dataset size: 300K+ records
- Training time varies by model:
  - Linear Regression: Seconds
  - Random Forest: Minutes
  - XGBoost: Minutes
  - SVM: Longest (consider subset for testing)

### Limitations

1. **Geographic scope:** India-focused dataset
2. **Time period:** Snapshot data, not time-series
3. **External factors:** Doesn't account for fuel prices, economic conditions
4. **Seasonal variation:** May not capture full yearly patterns

---

## Deployment

### Live Application

**URL:** [https://flight-predection-aks6zgm9ciwpyjnfdtv4w7.streamlit.app](https://flight-predection-aks6zgm9ciwpyjnfdtv4w7.streamlit.app)

**Platform:** Streamlit Cloud (Free Tier)

**Features:**
- Interactive web interface
- Real-time price predictions
- User-friendly input forms
- Price insights and recommendations
- Responsive design

### Deployment Process

1. **Optimized Model:** Reduced from 795MB to 37MB
2. **Git Repository:** Pushed to GitHub
3. **Streamlit Cloud:** Connected and deployed
4. **Public Access:** Available worldwide

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

---

## Repository

**GitHub:** [https://github.com/kar1myasser/Flight-predection](https://github.com/kar1myasser/Flight-predection)

---

## Contributors

This project was developed as part of the DEPI (Digital Egypt Pioneers Initiative) program.

**Developer:** Karim Yasser ([@kar1myasser](https://github.com/kar1myasser))

---

## License

This project is for educational purposes.

---

## Contact

For questions or suggestions:
- GitHub Issues: [https://github.com/kar1myasser/Flight-predection/issues](https://github.com/kar1myasser/Flight-predection/issues)
- GitHub Profile: [@kar1myasser](https://github.com/kar1myasser)

---

## Acknowledgments

- **Dataset Source:** Airlines flight data
- **Libraries:** scikit-learn, XGBoost, pandas, matplotlib, seaborn
- **Program:** DEPI - Digital Egypt Pioneers Initiative

---

## Version History

- **v1.0** (December 11, 2025)
  - Initial release with complete pipeline and model comparison
  - Jupyter notebook with EDA and 4 ML models
  - Best R² score: 98-99% (Random Forest)
  
- **v1.1** (December 11, 2025)
  - Added Streamlit web application
  - Deployed to Streamlit Cloud
  - Optimized model size (795MB → 37MB)
  - Added interactive prediction interface
  - Complete documentation and deployment guide

---

## Screenshots

### Web Application
![Flight Price Predictor](https://img.shields.io/badge/Streamlit-Live-brightgreen)

The web app features:
- 🎨 Modern UI with custom theming
- 📊 Real-time predictions
- 💡 Price insights and recommendations
- 📱 Mobile-responsive design

---

*Last Updated: December 11, 2025*
