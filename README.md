# Wine Classification Using Machine Learning

A comprehensive machine learning project that classifies wines into three different classes based on their chemical properties using various classification algorithms.

## 📊 Dataset Overview

The dataset contains results from chemical analysis of wines grown in the same region in Italy, derived from three different cultivars. The analysis determined quantities of 13 chemical constituents found in each wine type.

- **Total Samples**: 178 wine instances
- **Features**: 13 chemical attributes + 1 target variable
- **Classes**: 3 different wine types
- **Class Distribution**:
  - Class 1: 59 samples
  - Class 2: 71 samples
  - Class 3: 48 samples

## 🧪 Chemical Features

| Feature              | Description                        |
| -------------------- | ---------------------------------- |
| Alcohol              | Alcohol content                    |
| Malic Acid           | Malic acid concentration           |
| Ash                  | Ash content                        |
| Alcalinity of Ash    | Alkalinity of ash                  |
| Magnesium            | Magnesium content                  |
| Total Phenols        | Total phenolic compounds           |
| Flavanoids           | Flavanoid concentration            |
| Nonflavanoid Phenols | Non-flavanoid phenolic compounds   |
| Proanthocyanins      | Proanthocyanin content             |
| Color Intensity      | Wine color intensity               |
| Hue                  | Color hue                          |
| OD280/OD315          | Diluted wine optical density ratio |
| Proline              | Proline amino acid content         |

## 🎯 Project Objectives

- Develop robust machine learning models for wine classification
- Understand relationships between chemical variables and wine types
- Identify the most significant chemical features for wine classification
- Compare performance of different classification algorithms

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

- **Data Quality Assessment**: Checked for missing values (none found)
- **Statistical Analysis**: Calculated means, ranges, and distributions for each wine class
- **Correlation Analysis**: Generated correlation heatmaps to identify feature relationships
- **Feature Distribution**: Created histograms and box plots to visualize class differences
- **Outlier Detection**: Used IQR method to identify potential outliers

### 2. Feature Engineering

- **Feature Scaling**: Applied both StandardScaler and MinMaxScaler
- **Feature Importance**: Used Random Forest to identify most important features
- **Feature Selection**: Tested models with all features vs. top 5 important features

### 3. Machine Learning Models

The following classification algorithms were implemented and compared:

#### Models Tested:

1. **Logistic Regression** - Linear classifier with sigmoid function
2. **Support Vector Machine (SVM)** - With both linear and RBF kernels
3. **Decision Tree** - Tree-based classifier with interpretable structure
4. **Random Forest** - Ensemble method for feature importance analysis
5. **K-Nearest Neighbors (KNN)** - Distance-based classifier (optimized k=13)

## 📈 Results

### Model Performance (5-fold Cross-Validation)

| Model                                        | Accuracy   | F1-Score   |
| -------------------------------------------- | ---------- | ---------- |
| **Logistic Regression (Important Features)** | **97.91%** | **97.91%** |
| **Logistic Regression (All Features)**       | **97.91%** | **97.91%** |
| SVM (RBF Kernel)                             | 97.17%     | 97.17%     |
| KNN (k=13)                                   | 96.49%     | 96.49%     |
| Decision Tree                                | 92.18%     | 92.18%     |

### Most Important Features

Based on Random Forest feature importance analysis:

1. **Flavanoids** - Most discriminative feature
2. **Color Intensity** - Strong indicator of wine type
3. **Proline** - Significant amino acid marker
4. **Alcohol** - Important quality indicator
5. **OD280/OD315** - Optical density ratio

### Key Correlations Discovered

- **Strong Positive Correlations**:

  - Flavanoids & Total Phenols (0.85)
  - Ash & Alcalinity of Ash (0.80)
  - Malic Acid & Ash (0.78)
  - Color Intensity & Flavanoids (0.75)

- **Strong Negative Correlations**:
  - Flavanoids & Hue (-0.60)
  - Color Intensity & Malic Acid (-0.32)

## 📁 Project Structure

```
wine_class.ipynb          # Main Jupyter notebook with complete analysis
wine.csv                  # Dataset file (not included - add your own)
README.md                 # This file
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Analysis

1. Clone this repository
2. Add the wine dataset as `wine.csv` in the project directory
3. Open `wine_class.ipynb` in Jupyter Notebook
4. Run all cells to reproduce the analysis

### Dataset Format

The CSV file should have no headers with the following column order:

```
Target, Alcohol, Malic Acid, Ash, Alcalinity of Ash, Magnesium, Total Phenols,
Flavanoids, Nonflavanoid Phenols, Proanthocyanins, Color Intensity, Hue,
OD280/OD315 of Diluted Wines, Proline
```

## 🔍 Key Findings

### Statistical Insights by Wine Class

| Attribute       | Class 1 Mean | Class 2 Mean | Class 3 Mean |
| --------------- | ------------ | ------------ | ------------ |
| Alcohol         | 13.74        | 12.28        | 13.15        |
| Total Phenols   | 2.84         | 2.26         | 2.26         |
| Flavanoids      | 2.98         | 2.08         | 2.08         |
| Color Intensity | 5.53         | 3.09         | 3.09         |
| Proline         | 1116.59      | 519.51       | 519.51       |

### Model Insights

- **Logistic Regression** achieved the highest accuracy (97.91%) and proved most effective
- **Feature selection** showed that using only the top 5 features performed as well as using all features
- **SVM with RBF kernel** provided excellent generalization with 97.17% accuracy
- **Minimal misclassifications** occurred mainly between classes with overlapping chemical properties

## 📊 Visualizations Included

- Correlation heatmaps
- Feature distribution histograms by wine class
- Box plots for outlier detection
- Violin plots for scaled feature comparison
- Feature importance bar charts
- Confusion matrices for all models
- Model performance comparison charts


## 🔮 Applications & Implications

This classification system could be applied to:

- **Wine Quality Control** - Automated wine type verification
- **Authentication** - Detecting wine fraud or mislabeling
- **Production Optimization** - Understanding chemical factors affecting wine characteristics
- **Retail & Recommendations** - Chemical-based wine recommendation systems

## ⚠️ Limitations

- Limited dataset size (178 samples) may not capture all wine variations
- Regional specificity (Italian wines only)
- Some chemical overlap between classes leads to occasional misclassification
- Model performance may vary with wines from different regions or production methods

## 🔧 Future Improvements

- **Hyperparameter Tuning** - Optimize model parameters for better performance
- **Deep Learning** - Explore neural networks for complex pattern recognition
- **Feature Engineering** - Create ratio-based features from existing chemicals
- **Cross-Regional Validation** - Test on wines from different geographical regions
- **Ensemble Methods** - Combine multiple models for improved accuracy

## 📚 References

1. Xie, Q. (2023). Machine Learning on Wine Quality: Prediction and Feature Importance Analysis
2. Wine Phenolics Research (2002). PubMed: 12074959
3. Tanaka, T. et al. (2019). Potential Beneficial Effects of Wine Flavonoids on Allergic Diseases

## 👨‍💻 Author

**Ruslan Pavliichuk** 2025 

## 📄 License

This project is for educational purposes

---

_This analysis demonstrates the power of machine learning in understanding complex biological products through chemical analysis, achieving over 97% accuracy in wine classification._
