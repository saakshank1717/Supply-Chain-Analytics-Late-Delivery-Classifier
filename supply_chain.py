

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

import matplotlib.ticker as ticker

import matplotlib.cm as cm

from warnings import filterwarnings

filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# generate viridis color palette
viridis_colors = cm.viridis(np.linspace(0, 1, 5))

primary_color = viridis_colors[0]
secondary_color = viridis_colors[1]
accent_color = viridis_colors[2]
danger_color = '#800000'
neutral_color = viridis_colors[4]

custom_palette = viridis_colors

df=pd.read_csv("/content/DataCoSupplyChainDataset.csv",encoding='latin1')
df.head()

df.shape

df.columns

"""# EDA"""

# Overview
print('rows, cols:', df.shape)

print('Columns:')
print(df.columns.tolist())

print('\nNum duplicates:', df.duplicated().sum())

print('\nMissing values (top 20):')
print(df.isna().sum().sort_values(ascending=False).head(20))

# Data Cleaning
columns_to_drop = [
    'Product Description',
    'Product Image',
    'Customer Email',
    'Customer Password',
    'Customer Fname',
    'Customer Lname',
    'Customer Street',
    'Customer Zipcode',
    'Order Zipcode',
    'Longitude',
    'Latitude',
    'Order Item Cardprod Id',
    'Order Item Id',
    'Order Item Discount',
    'Order Item Discount Rate',
    'Order Item Product Price',
    'Order Item Quantity',
    'Order Item Total',
    'Category Id',
    'Department Id',
    'Order Id',
    'Order Customer Id',
    'Customer Id',
    'Product Card Id',
    'Product Category Id',
    'Benefit per order',
    'Product Status',
    'Customer City',
    'Order City',
    'Order Country',
    'Order State',
    'Customer State',
    'Market'
]

df=df.drop(columns=columns_to_drop)

df.shape

# Remove canceled orders (not useful for delivery analysis)
df = df[df['Delivery Status'] != 'Shipping canceled']

# Standard date conversion
for c in ['order date (DateOrders)', 'shipping date (DateOrders)']:
    df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=False)

# After cleaning, check dataset again
print('rows, cols:', df.shape)

print('\nMissing values (top 5):')
print(df.isna().sum().sort_values(ascending=False).head(5))

# value counts for categorical columns with low cardinality
for col in df.columns:
    if df[col].nunique() < 10:
        print(f'\n{col} value counts:')
        print(df[col].value_counts())

# calculating order processing time and delay
'''
df['Order Processing Time'] = (
    df['shipping date (DateOrders)'] - df['order date (DateOrders)']
).dt.days

df['Delay'] = df['Order Processing Time'] - df['Days for shipment (scheduled)']'''
df['Delay'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']


df['Is_Delayed'] = df['Delay'] > 0

df['order_month'] = df['order date (DateOrders)'].dt.month

df['order_day'] = df['order date (DateOrders)'].dt.day_name()

df['order_hour'] = df['order date (DateOrders)'].dt.hour

df.describe()

df['Is_Delayed'].value_counts()

df['Profitability Flag'] = np.select(
    [
        df['Order Profit Per Order'] > 0,
        df['Order Profit Per Order'] < 0
    ],
    [
        'Profit',
        'Loss'
    ],
    default='Breakeven'
)

df['Profitability Flag'].value_counts()

df['Profitability Flag'].value_counts(normalize=True)

# visualization of profitability distribution

profit_counts = df['Profitability Flag'].value_counts(normalize=True) * 100

profit_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=[accent_color, danger_color, secondary_color]
)

plt.ylabel('')
plt.title('Profitability Distribution (%)')
plt.show()

def format_func(value):
    if value >= 1e6:
        return f'{value/1e6:.1f}M $'
    elif value >= 1e3:
        return f'{value/1e3:.1f}K $'
    else:
        return f'{value:.0f} $'


# Filter delayed orders
delayed_df = df[df['Delay'] > 0]

# Metrics dictionary
metrics = {}

metrics['Total Orders'] = len(df)
metrics['Late Deliveries'] = len(delayed_df)

metrics['90% Delay (days)'] = delayed_df['Delay'].quantile(0.90)

metrics['On time Delivery %'] = (
    (1 - metrics['Late Deliveries'] / metrics['Total Orders']) * 100
)

metrics['Late Delivery %'] = (
    metrics['Late Deliveries'] / metrics['Total Orders'] * 100
)

metrics['Total Profit'] = format_func(
    df.loc[df['Order Profit Per Order'] > 0, 'Order Profit Per Order'].sum()
)

metrics['Total Loss due to delays'] = format_func(
    df.loc[df['Delay'] > 0, 'Order Profit Per Order'].sum()
)

# Print KPIs
print('\n--- Business KPIs ---\n')

for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

"""# Profitability VS Delivery Analysis"""

profit_metrics = (
    df.groupby('Delay')['Order Profit Per Order']
      .agg(
          mean_profit='mean',
          total_profit='sum',
          order_count='count'
      )
      .reset_index()
)

profit_metrics

delay_distribution = (
    df['Delay']
      .value_counts(normalize=True)
      .sort_index() * 100
).reset_index()

delay_distribution.columns = ['Delay', 'Percentage']

delay_distribution

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

# Rename columns
delay_distribution.columns = ['Delay_Days', 'Percentage']

print("\nProfit Metrics by Delay Day:")
display(profit_metrics.round(1))

print("Delay Distribution (%):")
display(delay_distribution)


# Colors
primary_color = "#1f77b4"
accent_color = "#ff7f0e"


# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


# -----------------------------
# 1. Delay Distribution
# -----------------------------
sns.barplot(
    x='Delay_Days',
    y='Percentage',
    data=delay_distribution,
    color=accent_color,
    ax=ax1
)

ax1.set_title("Delay Distribution")
ax1.set_xlabel("Delay (Days)")
ax1.set_ylabel("Percentage of Orders (%)")


# Add percentage labels
for bar in ax1.patches:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.5,
        f"{height:.1f}%",
        ha='center',
        va='bottom'
    )


# -----------------------------
# 2. Profit Analysis
# -----------------------------

ax2.set_title("Profit Analysis by Delay Days")


# Total Profit Bar
ax2.bar(
    profit_metrics['Delay'],
    profit_metrics['total_profit'],
    color=primary_color,
    label="Total Profit"
)

ax2.set_xlabel("Delay Days")
ax2.set_ylabel("Total Profit", color=primary_color)

ax2.tick_params(
    axis='y',
    labelcolor=primary_color
)


# Mean Profit Line (Secondary Axis)
ax3 = ax2.twinx()

ax3.plot(
    profit_metrics['Delay'],
    profit_metrics['mean_profit'],
    marker='o',
    color=accent_color,
    label="Mean Profit"
)

ax3.set_ylabel(
    "Mean Profit",
    color=accent_color
)

ax3.tick_params(
    axis='y',
    labelcolor=accent_color
)


# Format money axis
def format_func(value, tick_number):
    if value >= 1e6:
        return f'{value/1e6:.1f}M $'
    elif value >= 1e3:
        return f'{value/1e3:.1f}K $'
    else:
        return f'{value:.0f} $'


ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(format_func)
)


# Combine legends
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax3.legend(
    lines + lines2,
    labels + labels2,
    loc='upper right'
)


ax3.grid(
    True,
    linestyle=':',
    alpha=0.5
)


plt.tight_layout()
plt.show()

def compute_delay_pct_by_category(category):

    cat_df = (
        df.groupby(category)
        .agg(
            total_orders=('Delay', 'count'),
            late_orders=('Is_Delayed', 'sum')
        )
        .reset_index()
    )

    cat_df['delay_pct'] = (
        cat_df['late_orders'] / cat_df['total_orders'] * 100
    )

    cat_df = (
        cat_df
        .sort_values('delay_pct', ascending=False)
        .head(10)
    )

    return cat_df


categories = [
    'Order Region',
    'Customer Segment',
    'Shipping Mode',
    'Order Status',
    'Type',
    'Department Name'
]


fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 7),
    constrained_layout=True
)

axes = axes.flatten()


for ax, category in zip(axes, categories):

    cat_df = compute_delay_pct_by_category(category)

    sns.barplot(
        data=cat_df,
        x='delay_pct',
        y=category,
        ax=ax,
        palette='viridis'
    )

    ax.set_title(f'Delay % by {category}')
    ax.set_xlabel('')
    ax.set_ylabel(category)


    # Add percentage labels
    for i, row in cat_df.reset_index(drop=True).iterrows():

        ax.text(
            row['delay_pct'] - 15,
            i,
            f"{row['delay_pct']:.1f}%",
            va='center',
            fontsize=10,
            color='white'
        )


plt.show()

"""# TIME BASED ANALYSIS"""

# Delay % by Month

delay_by_month = (
    df.groupby('order_month')['Is_Delayed']
      .mean()
      .reset_index()
)

delay_by_month['delay_pct'] = (
    delay_by_month['Is_Delayed'] * 100
)


# Delay % by Day of Week

delay_by_day = (
    df.groupby('order_day')['Is_Delayed']
      .mean()
      .reset_index()
)

delay_by_day['delay_pct'] = (
    delay_by_day['Is_Delayed'] * 100
)


# Delay % by Hour

delay_by_hour = (
    df.groupby('order_hour')['Is_Delayed']
      .mean()
      .reset_index()
)

delay_by_hour['delay_pct'] = (
    delay_by_hour['Is_Delayed'] * 100
)


# Display results

print("Delay % by Month")
display(delay_by_month)


print("Delay % by Day")
display(delay_by_day)


print("Delay % by Hour")
display(delay_by_hour)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# -----------------------------
# Prepare ordering
# -----------------------------

# Month ordering
if delay_by_month['order_month'].dtype != 'object':
    delay_by_month = delay_by_month.sort_values('order_month')


# Day ordering (if names exist)
day_order = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]

if delay_by_day['order_day'].dtype == 'object':
    delay_by_day['order_day'] = pd.Categorical(
        delay_by_day['order_day'],
        categories=day_order,
        ordered=True
    )

    delay_by_day = delay_by_day.sort_values('order_day')


# Hour ordering
delay_by_hour = delay_by_hour.sort_values('order_hour')



# -----------------------------
# Plot
# -----------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(20,6)
)


# =============================
# 1. MONTH ANALYSIS
# =============================

sns.lineplot(
    data=delay_by_month,
    x='order_month',
    y='delay_pct',
    marker='o',
    linewidth=3,
    ax=axes[0]
)


axes[0].set_title(
    "Monthly Delay Trend",
    fontsize=14,
    fontweight='bold'
)

axes[0].set_xlabel("Month")
axes[0].set_ylabel("Delayed Orders (%)")


axes[0].grid(
    linestyle='--',
    alpha=0.4
)


# labels + peak highlight

max_month = delay_by_month.loc[
    delay_by_month['delay_pct'].idxmax()
]


for x,y in zip(
    delay_by_month['order_month'],
    delay_by_month['delay_pct']
):

    axes[0].text(
        x,
        y+0.5,
        f"{y:.1f}%",
        ha='center',
        fontsize=9
    )


axes[0].annotate(
    f"Highest: {max_month['delay_pct']:.1f}%",
    xy=(
        max_month['order_month'],
        max_month['delay_pct']
    ),
    xytext=(0,30),
    textcoords='offset points',
    arrowprops=dict(
        arrowstyle='->'
    )
)



# =============================
# 2. DAY ANALYSIS
# =============================


sns.barplot(
    data=delay_by_day,
    x='order_day',
    y='delay_pct',
    ax=axes[1]
)


axes[1].set_title(
    "Delay Percentage by Day",
    fontsize=14,
    fontweight='bold'
)

axes[1].set_xlabel("")
axes[1].set_ylabel("Delayed Orders (%)")


axes[1].tick_params(
    axis='x',
    rotation=45
)



for bar in axes[1].patches:

    axes[1].text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.5,
        f"{bar.get_height():.1f}%",
        ha='center',
        fontsize=9
    )



# =============================
# 3. HOUR ANALYSIS
# =============================


sns.lineplot(
    data=delay_by_hour,
    x='order_hour',
    y='delay_pct',
    marker='o',
    linewidth=3,
    ax=axes[2]
)



axes[2].set_title(
    "Delay Trend by Order Hour",
    fontsize=14,
    fontweight='bold'
)


axes[2].set_xlabel(
    "Order Hour"
)

axes[2].set_ylabel(
    "Delayed Orders (%)"
)


axes[2].grid(
    linestyle='--',
    alpha=0.4
)



# peak hour

peak_hour = delay_by_hour.loc[
    delay_by_hour['delay_pct'].idxmax()
]


axes[2].annotate(
    f"Peak: {peak_hour['delay_pct']:.1f}%",
    xy=(
        peak_hour['order_hour'],
        peak_hour['delay_pct']
    ),
    xytext=(0,30),
    textcoords='offset points',
    arrowprops=dict(
        arrowstyle='->'
    )
)



# -----------------------------
# Final formatting
# -----------------------------

plt.suptitle(
    "Temporal Delay Pattern Analysis",
    fontsize=18,
    fontweight='bold'
)


plt.tight_layout()

plt.show()

"""# MACHINE LEARNING MODELLING"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

X = df[
[
'Type',
'Days for shipment (scheduled)',
'Category Name',
'Customer Segment',
'Department Name',
'Order Region',
'Shipping Mode',
'order_hour',
'order_day',
'order_month'
]
]

y = df['Late_delivery_risk']

cat_cols = X.select_dtypes(
    include=['object','category']
).columns.tolist()


print("Categorical Columns:")
print(cat_cols)


for col in cat_cols:

    freq = X[col].value_counts(normalize=True)

    X[col+'_freq'] = X[col].map(freq)

X_encoded = X.drop(
    columns=cat_cols
)


print(
"Shape after encoding:",
X_encoded.shape
)


X = X_encoded

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



print(
"Before SMOTE:",
Counter(y_train)
)

smote = SMOTE(
    random_state=42
)


X_train_bal, y_train_bal = smote.fit_resample(
    X_train,
    y_train
)


print(
"After SMOTE:",
Counter(y_train_bal)
)

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_bal
)


X_test_scaled = scaler.transform(
    X_test
)

models = {

"Logistic Regression":
LogisticRegression(
    max_iter=1000,
    random_state=42
),


"Decision Tree":
DecisionTreeClassifier(
    random_state=42
),


"Random Forest":
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

}

results = {}

roc_results = {}


def evaluate_model(
    name,
    model,
    Xtr,
    Xte
):

    model.fit(
        Xtr,
        y_train_bal
    )


    y_pred = model.predict(
        Xte
    )


    y_prob = model.predict_proba(
        Xte
    )[:,1]


    print("\n====================")
    print(name)
    print("====================")


    print(
    "Accuracy:",
    round(
        accuracy_score(y_test,y_pred),
        3
    )
    )


    print(
    "Precision:",
    round(
        precision_score(y_test,y_pred),
        3
    )
    )


    print(
    "Recall:",
    round(
        recall_score(y_test,y_pred),
        3
    )
    )


    print(
    "F1:",
    round(
        f1_score(y_test,y_pred),
        3
    )
    )


    print(
    classification_report(
        y_test,
        y_pred
    )
    )



    auc = roc_auc_score(
        y_test,
        y_prob
    )


    results[name] = [
        accuracy_score(y_test,y_pred),
        precision_score(y_test,y_pred),
        recall_score(y_test,y_pred),
        f1_score(y_test,y_pred),
        auc
    ]


    roc_results[name] = (
        y_test,
        y_prob
    )


    # confusion matrix

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )


    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.show()

for name, model in models.items():


    if name == "Logistic Regression":

        evaluate_model(
            name,
            model,
            X_train_scaled,
            X_test_scaled
        )

    else:

        evaluate_model(
            name,
            model,
            X_train_bal,
            X_test
        )

comparison = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ]
).T


comparison.round(3)

# ===========================
# ROC CURVE
# ===========================


plt.figure(figsize=(8,6))


for name,(y_true,y_prob) in roc_results.items():

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )


    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )


plt.plot(
    [0,1],
    [0,1],
    '--'
)


plt.title(
    "ROC-AUC Curve Comparison"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)


plt.legend()

plt.grid(
    linestyle=':',
    alpha=0.5
)

plt.show()
