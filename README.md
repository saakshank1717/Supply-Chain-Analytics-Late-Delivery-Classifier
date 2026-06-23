# Supply Chain Analytics & Delay Prediction

Exploratory data analysis and machine learning pipeline to predict late deliveries and quantify financial losses in a global supply chain dataset.

---

## Business Problem

Over **57% of orders** in the dataset were delivered late, resulting in **$2.1M in losses**. This project identifies the key drivers of delivery delays and builds a classification model to predict late deliveries before they happen.

---

## Dataset

- **Source:** DataCo Supply Chain Dataset (Kaggle)
- **Size:** 172,765 orders after cleaning (original: 180,519)
- **Features used:** Shipping mode, customer segment, order type, scheduled vs real shipping days, order date/time features

---

## Project Structure

```
supply_chain/
│
├── supply_chain.ipynb       # Main notebook (EDA + ML)
├── README.md
└── DataCoSupplyChainDataset.csv
```

---

## Workflow

### 1. Data Cleaning
- Dropped PII columns (name, email, password, address)
- Removed cancelled orders
- Parsed datetime columns and handled missing values

### 2. Feature Engineering
- `Delay` = Real shipping days − Scheduled shipping days
- `Is_Delayed` = Binary flag (Delay > 0)
- `Profitability Flag` = Profit / Loss / Breakeven per order
- Temporal features: order month, day of week, hour

### 3. EDA
- 57.29% late delivery rate across 172K orders
- $2.1M in losses attributed to delayed shipments
- Delay patterns analyzed across shipping mode, customer segment, and time dimensions
- Profitability vs delay correlation mapped at order level

### 4. ML Modelling
- **Train/Test Split:** 80/20, stratified
- **Class Imbalance:** Handled via SMOTE (minority class upsampled to 79K samples)
- **Models trained:** Logistic Regression, Decision Tree, Random Forest

---

## Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.697    | 0.833     | 0.590  | 0.690    | 0.709   |
| Decision Tree       | 0.812    | 0.838     | 0.832  | **0.835**| 0.811   |
| Random Forest       | 0.790    | 0.838     | 0.784  | 0.811    | **0.880**|

> **Best F1:** Decision Tree (0.835) — best balance of precision and recall  
> **Best ROC AUC:** Random Forest (0.880) — best at ranking delay risk

---

## Key Findings

- **57.29%** of all orders were delivered late
- **$2.1M** in losses directly tied to delayed shipments
- Delay rate is relatively consistent across days of week and months — suggesting a systemic logistics issue rather than seasonal spikes
- Orders with longer scheduled shipping windows tend to have higher delay rates

---

## Tech Stack

- **Python** — pandas, numpy, matplotlib, seaborn
- **ML** — scikit-learn, imbalanced-learn (SMOTE)
- **Models** — Logistic Regression, Decision Tree, Random Forest

---

## Business KPIs

```
Total Orders        : 172,765
Late Deliveries     : 98,977
Late Delivery %     : 57.29%
On-Time Delivery %  : 42.71%
90th Pct Delay      : 3 days
Total Profit        : $7.5M
Loss due to Delays  : $2.1M
```
