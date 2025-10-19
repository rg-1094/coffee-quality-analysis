# ☕ Coffee Quality Analysis Project

A comprehensive data mining project analyzing coffee quality datasets to uncover hidden patterns in flavor profiles and quality factors using machine learning techniques.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This project implements data mining techniques to analyze coffee quality data, focusing on two main objectives:
1. **Clustering** coffee samples based on flavor characteristics
2. **Predicting** quality scores using regression analysis

The analysis helps stakeholders in the coffee industry make data-driven decisions about cultivation, purchasing, and quality improvement.

## 🎯 Objectives

### 1. Flavor Profile Clustering
- **Business Need**: Help coffee buyers identify flavor profiles for targeted purchasing
- **Technical**: Group coffees using K-Means clustering on sensory attributes
- **Outcome**: Natural groupings of similar-tasting coffees

### 2. Quality Rating Prediction  
- **Business Need**: Help farmers understand factors affecting coffee ratings
- **Technical**: Predict Total Cup Points using Linear Regression
- **Outcome**: Model to predict quality based on measurable factors

## 📊 Dataset

**Source**: [Kaggle Coffee Quality Dataset](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi)

**Key Features**:
- **Geographical**: Country of Origin, Region, Altitude
- **Production**: Variety, Processing Method
- **Quality Metrics**: Aroma, Flavor, Aftertaste, Acidity, Body, Balance (1-10 scale)
- **Defects**: Moisture, Category Defects, Quakers
- **Overall Rating**: Total Cup Points (0-100 scale)

**Records**: 1,000+ specialty coffee evaluations

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/coffee-quality-analysis.git
cd coffee-quality-analysis

# Install required packages
pip install -r requirements.txt
```

### Required Libraries
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Usage

### Running the Analysis
```python
# Execute the main analysis script
python coffee_quality_analysis.py
```

### Code Structure
```
coffee-quality-analysis/
│
├── coffee_quality_analysis.py  # Main analysis script
├── coffee_quality.csv          # Dataset (download from Kaggle)
├── processed_coffee_data.csv   # Processed dataset output
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## ⚙️ Methodology

### 1. Data Preprocessing
```python
# Handle missing values
numerical_cols: Median imputation
categorical_cols: Mode imputation

# Feature engineering
Label encoding for categorical variables
Standard scaling for clustering features
```

**Justification**: 
- Median imputation preserves numerical data distribution
- Mode imputation maintains categorical data integrity  
- Scaling ensures equal feature contribution in distance-based algorithms

### 2. Algorithms Implemented

#### K-Means Clustering
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_scaled)
```

**Justification**: Efficient for medium datasets, works well with numerical data, provides interpretable results

#### Linear Regression
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

**Justification**: Provides interpretable coefficients, fast computation, suitable for continuous targets

### 3. Evaluation Metrics

**Clustering**:
- WCSS (Within-Cluster Sum of Squares)
- Silhouette Score

**Regression**:
- R-squared Score
- Residual Analysis

## 📈 Results

### Clustering Findings
- **3 distinct flavor profiles** identified
- **Cluster characteristics**: Balanced, High-Acidity, Full-Bodied
- **Business impact**: Streamlined purchasing decisions

### Regression Insights
- **Key quality factors** identified through coefficients
- **Model performance**: R² = [Score from your analysis]
- **Business applications**: Quality improvement strategies

## 💡 Key Features

- **Comprehensive Analysis**: Both unsupervised (clustering) and supervised (regression) learning
- **Business Focus**: Practical applications for coffee industry stakeholders
- **Educational Value**: Well-documented code with detailed justifications
- **Reproducible Research**: Clear methodology and evaluation metrics

## 🔮 Future Enhancements

- [ ] Implement additional clustering algorithms (DBSCAN, Hierarchical)
- [ ] Try ensemble methods for improved prediction
- [ ] Add feature engineering for better model performance
- [ ] Include cross-validation for robust evaluation
- [ ] Develop interactive visualizations

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/)
- Faculty guidance for project structure and requirements
- Open-source community for Python data science libraries

---

**Note**: This project was developed as part of academic coursework in Data Mining. The focus is on methodological rigor, business relevance, and comprehensive documentation.
