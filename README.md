# **Watt's Next? Predicting EV Trends and Charging Needs**

A comprehensive data science project analyzing Electric Vehicle (EV) trends, predicting performance metrics, and identifying optimal charging infrastructure locations using machine learning and geospatial analysis.

## **📊 Project Overview**

Electric vehicles represent a transformative shift in the transportation industry, addressing critical global challenges including climate change, air pollution, and fossil fuel dependency. This project leverages data science techniques to unlock insights from EV adoption patterns, performance characteristics, and infrastructure requirements.

### **Key Objectives**

* **Analyze EV Performance**: Explore electric range trends across manufacturers and model years  
* **Test Hypotheses**: Validate assumptions about BEV vs PHEV performance and regional adoption patterns  
* **Build Predictive Models**: Forecast electric range and classify vehicle eligibility for incentives  
* **Market Segmentation**: Categorize EVs using clustering techniques for strategic insights  
* **Infrastructure Planning**: Identify regions requiring additional charging infrastructure

## **👥 Authors**

* **Abhishek Rithik Origanti** (UID: 121305534\)  
* **Matheswara Annamalai Senthilkumar** (UID: 121281500\)

## **🚗 Dataset Information**

* **Size**: 216,772 rows × 17 columns  
* **Time Range**: Electric vehicles manufactured between 1999-2025  
* **Geographic Scope**: Primarily Washington State, USA  
* **Key Features**: Electric range, model year, manufacturer, vehicle type, location data

### **Dataset Columns**

| Column | Description | Usage |
| ----- | ----- | ----- |
| Electric Range | Maximum electric-only range | Primary target variable |
| Model Year | Vehicle manufacturing year | Trend analysis |
| Make/Model | Manufacturer and model | Performance comparison |
| Electric Vehicle Type | BEV vs PHEV classification | Classification tasks |
| CAFV Eligibility | Clean Air Vehicle incentive status | Prediction target |
| Geographic Data | County, City, State, Coordinates | Infrastructure planning |

## **🛠️ Technologies & Libraries**

### **Core Libraries**

```py
pandas              # Data manipulation and analysis
numpy               # Numerical computing
matplotlib/seaborn  # Data visualization
scikit-learn        # Machine learning algorithms
plotly              # Interactive visualizations
```

### **Specialized Tools**

```py
geopandas           # Geospatial data analysis
folium              # Interactive mapping
imblearn            # Handling class imbalance (SMOTE)
statsmodels         # Statistical analysis
```

## **📈 Key Findings & Results**

### **1\. Performance Analysis**

* **Electric Range Evolution**: Steady improvement in EV ranges over time, with 91.8% accuracy in range prediction  
* **Manufacturer Leadership**: Tesla, Jaguar, and Polestar lead with highest average electric ranges  
* **Market Diversity**: Growing variety from budget-friendly to premium long-range vehicles

### **2\. Predictive Modeling Results**

| Model Type | Accuracy | Key Insights |
| ----- | ----- | ----- |
| Electric Range Prediction | R² \= 0.918 | Model year and manufacturer are strong predictors |
| CAFV Eligibility | 99.85% | Electric range and model year determine eligibility |
| BEV vs PHEV Classification | 98.42% | Clear distinction based on range and year |

### **3\. Statistical Analysis**

* **Hypothesis Testing**: Significant difference in electric range between BEVs and PHEVs (p \< 0.001)  
* **Regional Patterns**: Strong association between geographic location and vehicle type preference  
* **CAFV Adoption**: 76% of vehicles eligible for clean air incentives

### **4\. Infrastructure Insights**

* **Urban Concentration**: High EV density in Seattle, Bellevue, and Tacoma  
* **Rural Gaps**: Central and eastern Washington require additional charging infrastructure  
* **Strategic Planning**: Heatmap analysis identifies optimal locations for new charging stations

## **🗂️ Project Structure**

```
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_predictive_modeling.ipynb
│   ├── 05_classification.ipynb
│   ├── 06_clustering.ipynb
│   └── 07_infrastructure_planning.ipynb
├── src/
│   ├── data_processing.py
│   ├── modeling.py
│   ├── visualization.py
│   └── utils.py
├── results/
│   ├── models/                 # Trained model files
│   ├── figures/               # Generated plots and maps
│   └── reports/               # Analysis summaries
├── requirements.txt
└── README.md
```

## **🚀 Getting Started**

### **Prerequisites**

* Python 3.8 or higher  
* Jupyter Notebook or JupyterLab  
* Git

### **Installation**

1. **Clone the repository**

```shell
git clone https://github.com/your-username/ev-trends-prediction.git
cd ev-trends-prediction
```

2. **Create virtual environment**

```shell
python -m venv ev_env
source ev_env/bin/activate  # On Windows: ev_env\Scripts\activate
```

3. **Install dependencies**

```shell
pip install -r requirements.txt
```

4. **Download dataset**

```shell
# Place your EV dataset in the data/raw/ directory
# Dataset: Electric_Vehicle_Population_Data-2.csv
```

### **Quick Start**

```py
# Load and explore the data
import pandas as pd
import numpy as np
from src.data_processing import clean_data, prepare_features

# Load dataset
df = pd.read_csv('data/raw/Electric_Vehicle_Population_Data-2.csv')

# Clean and preprocess
df_clean = clean_data(df)

# Run analysis
from src.modeling import predict_electric_range, classify_vehicle_type
from src.visualization import create_heatmap, plot_trends

# Generate predictions
range_predictions = predict_electric_range(df_clean)
vehicle_classifications = classify_vehicle_type(df_clean)

# Create visualizations
plot_trends(df_clean)
create_heatmap(df_clean)
```

## **📊 Analysis Workflow**

### **1\. Data Preparation**

* **Data Cleaning**: Handle missing values and outliers  
* **Feature Engineering**: Create scaled and encoded variables  
* **Data Validation**: Ensure data quality and consistency

### **2\. Exploratory Data Analysis**

* **Trend Analysis**: Electric range evolution over time  
* **Manufacturer Comparison**: Performance across different brands  
* **Geographic Distribution**: Regional adoption patterns

### **3\. Hypothesis Testing**

* **t-tests**: Compare BEV vs PHEV performance  
* **Chi-square Tests**: Analyze regional preferences  
* **ANOVA**: Manufacturer performance differences

### **4\. Predictive Modeling**

* **Linear Regression**: Electric range prediction  
* **Random Forest**: Enhanced prediction accuracy (R² \= 0.918)  
* **Cross-validation**: Model robustness validation

### **5\. Classification**

* **Logistic Regression**: BEV vs PHEV classification  
* **SMOTE**: Address class imbalance  
* **Performance Metrics**: Precision, recall, F1-score

### **6\. Infrastructure Planning**

* **Geospatial Analysis**: EV density mapping  
* **Heatmap Generation**: Identify infrastructure gaps  
* **Strategic Recommendations**: Optimal charging station locations

## **📸 Key Visualizations**

### **Electric Range Trends**

* **Box Plots**: Range distribution by model year and manufacturer  
* **Line Charts**: Average range evolution over time  
* **Bar Charts**: Top manufacturers by performance

### **Geographic Analysis**

* **Interactive Maps**: EV distribution across Washington State  
* **Heatmaps**: Density visualization for infrastructure planning  
* **Cluster Maps**: Market segmentation visualization

### **Model Performance**

* **Confusion Matrices**: Classification accuracy visualization  
* **Learning Curves**: Model training progression  
* **Feature Importance**: Key predictor identification

## **🎯 Business Applications**

### **For Manufacturers**

* **Product Strategy**: Identify market gaps and opportunities  
* **Performance Benchmarking**: Compare against competitors  
* **Technology Investment**: Focus areas for R\&D

### **For Policymakers**

* **Incentive Programs**: Optimize CAFV eligibility criteria  
* **Infrastructure Planning**: Strategic charging station deployment  
* **Adoption Forecasting**: Plan for future EV growth

### **For Consumers**

* **Purchase Decisions**: Compare vehicle performance and eligibility  
* **Range Planning**: Understand real-world capabilities  
* **Infrastructure Access**: Identify charging availability

## **📊 Model Performance Summary**

| Analysis Type | Method | Key Metric | Result |
| ----- | ----- | ----- | ----- |
| Range Prediction | Random Forest | R² Score | 0.918 |
| CAFV Eligibility | Logistic Regression | Accuracy | 99.85% |
| Vehicle Classification | Logistic Regression \+ SMOTE | Accuracy | 98.42% |
| Hypothesis Testing | t-test | Significance | p \< 0.001 |
| Cross Validation | 5-Fold CV | Mean Accuracy | 98.7% |

## **⚠️ Limitations & Future Work**

### **Current Limitations**

* **Geographic Scope**: Analysis primarily focused on Washington State  
* **Feature Limitations**: Limited to model year and electric range for some models  
* **Temporal Constraints**: Data snapshot doesn't capture rapid market changes  
* **Infrastructure Dynamics**: Static analysis of charging station locations

### **Future Enhancements**

* **Multi-State Analysis**: Expand to national and international datasets  
* **Real-time Data**: Incorporate dynamic pricing and availability information  
* **Advanced Models**: Implement deep learning for complex pattern recognition  
* **Economic Analysis**: Include cost-benefit analysis for infrastructure investments  
* **Environmental Impact**: Add carbon footprint and sustainability metrics

## **🤝 Contributing**

We welcome contributions to improve this project\! Please follow these steps:

1. **Fork the repository**  
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)  
3. **Commit changes** (`git commit -m 'Add some AmazingFeature'`)  
4. **Push to branch** (`git push origin feature/AmazingFeature`)  
5. **Open Pull Request**

### **Contribution Areas**

* Data preprocessing improvements  
* Additional machine learning models  
* Enhanced visualizations  
* Documentation updates  
* Bug fixes and optimizations

## **📝 License**

This project is licensed under the MIT License \- see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## **📚 References & Resources**

### **Academic & Industry Sources**

* [National Renewable Energy Laboratory (NREL) \- Electric Vehicle Analysis](https://www.nrel.gov/transportation/electric-vehicle-analysis.html)  
* [U.S. Department of Energy \- EV Infrastructure](https://www.energy.gov/eere/vehicles/electric-vehicles-infrastructure)  
* [International Energy Agency \- EV Outlook](https://www.iea.org/reports/global-ev-outlook-2023)

### **Technical Documentation**

* [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)  
* [Geopandas User Guide](https://geopandas.org/en/stable/getting_started.html)  
* [Folium Documentation](https://python-visualization.github.io/folium/)

### **Datasets**

* [Electric Vehicle Population Data](https://catalog.data.gov/dataset/electric-vehicle-population-data)  
* [Alternative Fuels Data Center](https://afdc.energy.gov/data_download)

## **📞 Contact & Support**

For questions, suggestions, or collaboration opportunities:

* **Project Repository**: [GitHub Repository]([https://github.com/your-username/ev-trends-prediction](https://github.com/abhioriganti/Predicting-EV-Trends-and-Charging-Needs))  


---

**⭐ Star this repository** if you find it useful, and feel free to **fork** and **contribute** to help advance EV adoption through data science\!

---

*This project demonstrates the power of data science in accelerating the transition to sustainable transportation. Together, we can build a cleaner, more efficient future.*

