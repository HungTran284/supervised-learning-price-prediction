#%% Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy.stats import f_oneway, chi2_contingency
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#%% Định nghĩa hàm
# Tổng quan dữ liệu
def overview_data(df):
    print(f"{'DataFrame Overview':^50}")
    print("=====" * 10)
    print(f"Shape: {df.shape}")
    print("-----" * 10)
    print("Info:")
    print(df.info())
    print("-----" * 10)
    print("Missing Values:")
    print(df.isnull().sum())
    print("-----" * 10)
    print(f"Duplicate Rows: {df.duplicated().sum()}")

# Biểu đồ tỷ lệ biến phân loại
def plot_category_distribution(df, col, figsize, labelsize):
    data_counts = df[col].value_counts()  # Đếm số lượng
    total_count = data_counts.sum()  # Tổng số lượng
    data_counts = data_counts.reset_index()  # Đưa về DataFrame
    data_counts.columns = [col, "count"]  # Đặt tên cột
    data_counts["percentage"] = (data_counts["count"] / total_count) * 100  # Tính %

    fig, ax = plt.subplots(figsize=figsize)
    norm = plt.Normalize(data_counts["count"].min(), data_counts["count"].max())
    colors = sns.color_palette("Reds", as_cmap=True)(norm(data_counts["count"]))
    sns.barplot(data=data_counts, x=col, y="count", palette=colors, ax=ax)

    for p, perc in zip(ax.patches, data_counts["percentage"]):
        ax.annotate(f'{perc:.1f}%',  # Hiển thị giá trị %
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                    ha='center', va='bottom', color='black', fontsize=labelsize)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=labelsize)
    ax.set_yticklabels(ax.get_yticks(), fontsize=labelsize)
    ax.set_ylabel("Count", fontsize=labelsize + 5)  # Đổi nhãn trục y thành Count
    ax.set_xlabel(col, fontsize=labelsize + 5)
    plt.show()

# Vẽ biểu đồ tỷ lệ lớp 0, 1 trong bài toán phân loại
def plot_pie_chart(df, column):
    value_counts = df[column].value_counts()

    plt.rcParams['figure.figsize'] = (4, 4)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title(f"Distribution of {column}")
    plt.legend()
    plt.show()

# Trả về df chứa outlier
def find_outliers(df, columns):
    df_outliers = pd.DataFrame()
    for col in columns:
        Q1 = df[col].quantile(0.25)  # Q1 (25th percentile)
        Q3 = df[col].quantile(0.75)  # Q3 (75th percentile)
        IQR = Q3 - Q1  # Tính IQR
        # Ngưỡng xác định outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Lọc các dòng có outlier trong cột hiện tại
        outlier_rows = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        # Thêm vào df_outliers nhưng loại bỏ trùng lặp dòng
        df_outliers = pd.concat([df_outliers, outlier_rows]).drop_duplicates()
    return df_outliers

# Vẽ biểu đồ boxplots
def boxplots(df, columns, cols_per_row):
    plt.rcParams["figure.figsize"] = (16, 8)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 10
    num_rows = math.ceil(len(columns)/cols_per_row)
    fig, axes = plt.subplots(nrows=num_rows, ncols=cols_per_row)
    axes = axes.flatten()
    for i, col in enumerate(df[columns].columns):
        sns.boxplot(y=df[columns][col], ax=axes[i])
        axes[i].set_title(col)
    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()

# Vẽ biểu đồ box và histogram cho các biến liên tục
def box_hist(df, column):
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 7
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Vẽ Boxplot
    sns.boxplot(y=df[column], ax=axes[0])
    axes[0].set_title(f'Boxplot of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Value')

    # Vẽ Histogram + KDE
    axes[1].hist(df[column], bins=15, edgecolor='black', alpha=1, density=True)
    sns.kdeplot(df[column], color='red', linewidth=2, ax=axes[1])
    axes[1].set_title(f'Histogram of {column}')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Trả về df outlier và khoảng outlier của cột
def count_outliers_col(df, column):
    quantiles = df[column].quantile([0.25, 0.75])
    Q1, Q3 = float(quantiles[0.25]), float(quantiles[0.75])
    IQR = Q3 - Q1
    lower_bound = float(Q1 - 1.5 * IQR)
    upper_bound = float(Q3 + 1.5 * IQR)

    col_values = df[column]
    min_val, max_val = float(col_values.min()), float(col_values.max())

    round_digits = 2
    min_val = round(min_val, round_digits)
    max_val = round(max_val, round_digits)
    lower_bound = round(lower_bound, round_digits)
    upper_bound = round(upper_bound, round_digits)

    outliers_below = min_val < lower_bound
    outliers_above = max_val > upper_bound
    if outliers_below and outliers_above:
        range_outliers = f"[{min_val}, {lower_bound}) ∪ ({upper_bound}, {max_val}]"
    elif outliers_below:
        range_outliers = f"[{min_val}, {lower_bound})"
    elif outliers_above:
        range_outliers = f"({upper_bound}, {max_val}]"
    else:
        range_outliers = None

    outlier_df = df[[column]][(col_values < lower_bound) | (col_values > upper_bound)]
    return outlier_df, range_outliers

# Hàm tổng quan về outlier trong cột
def overview_outliers(df, col_outlier, column, range_outlier):
    print(f"Tổng quan về outlier của cột {column}")
    print("====" * 10)
    print(f"Số lượng outlier: {col_outlier.shape[0]}")
    print(f"Khoảng outlier: {range_outlier}")
    print(f"Tỷ lệ % của outlier so với tập dữ liệu: {round(len(col_outlier) / len(df) * 100, 2)}%")

# Kiểm định ANOVA biến phân loại với biến liên tục
def anova_test(df, target):
    categorical_vars = df.select_dtypes(include=["object"]).columns
    anova_results = []

    for var in categorical_vars:
        groups = [df[df[var] == cat][target].dropna() for cat in df[var].unique()]
        if len(groups) > 1:
            f_stat, p_value = f_oneway(*groups)
            f_stat = round(f_stat, 3)
            p_value = round(p_value, 3)
            impact = "Yes" if p_value < 0.05 else "No"
            anova_results.append({"Variable": var, "F-statistic": f_stat, "p-value": p_value, "Impact": impact})

    return pd.DataFrame(anova_results)

# Kiểm định chi bình phương biến phân loại với biến phân loại (nhị phân)
def chi_square_test(df, target):
    categorical_vars = df.select_dtypes(include="object").columns.drop(target)
    results = []
    for var in categorical_vars:
        contingency_table = pd.crosstab(df[var], df[target])
        chi2, p, dof, _ = chi2_contingency(contingency_table)
        results.append({
            'Variable': var,
            'Chi2': round(chi2, 2),
            'p-value': round(p, 4),
            'dof': dof
        })
    return pd.DataFrame(results).set_index('Variable')

# Kiểm tra đa cộng tuyến
def calculate_vif(df, independent_vars):
    X = df[independent_vars].copy()
    X = sm.add_constant(X)  # Thêm hằng số (const) vào mô hình
    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data

# Trực quan tỷ lệ lớp 0, 1 trong từng biến phân loại
def catcols_distribution_with_ratio(df, col, target, status, figsize, labelsize):
    # Thiết lập tham số chung cho biểu đồ
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = labelsize

    # Tạo figure với subplot: 2 hàng, 1 cột
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Biểu đồ 1: Countplot (hàng trên)
    sns.countplot(data=df, x=col, hue=target, ax=ax1)
    ax1.set_title(f'Distribution of {col} vs {target}', fontsize=labelsize + 5)
    ax1.set_xlabel(col, fontsize=labelsize)
    ax1.set_ylabel("Count", fontsize=labelsize)
    ax1.tick_params(axis='x', rotation=90, labelsize=labelsize)
    ax1.tick_params(axis='y', labelsize=labelsize)
    ax1.legend(fontsize=labelsize)

    # Biểu đồ 2: Bar plot tỷ lệ (hàng dưới)
    agree_count = df[df[target] == status].groupby(col).size()
    total_count = df.groupby(col).size()
    agree_rate = (agree_count / total_count).fillna(0) * 100
    bars = agree_rate.sort_values().plot(kind='bar', edgecolor='black', alpha=0.5, ax=ax2)
    for bar in bars.patches:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{bar.get_height():.2f}", ha='center', fontsize=labelsize, color='black')
    ax2.set_title(f"{status} ratio by {col}", fontsize=labelsize + 5)
    ax2.set_xlabel(col, fontsize=labelsize)
    ax2.set_ylabel(f"{status} ratio (%)", fontsize=labelsize)
    ax2.tick_params(axis='x', rotation=90, labelsize=labelsize)
    ax2.tick_params(axis='y', labelsize=labelsize)

    plt.tight_layout()
    plt.show()

# Vẽ confusion matrix
def plot_confusion_matrices(confusion_matrices, fontsize):
    k_folds = len(confusion_matrices)
    fig, axes = plt.subplots(1, k_folds, figsize=(20, 4))  # 1 hàng, k_folds cột

    if k_folds == 1:
        axes = [axes]
    plt.rcParams.update({'font.size': fontsize})

    for i, (fold, cm) in enumerate(confusion_matrices):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": fontsize + 2})
        ax.set_title(f"Fold {fold}", fontsize=fontsize + 4, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=fontsize + 2)
        ax.set_ylabel("Actual", fontsize=fontsize + 2)

    plt.tight_layout()
    plt.show()

# Trực quan feature importance
def plot_feature_importance(feature_importance_df, row_index=5):
    row = feature_importance_df.iloc[row_index]
    plt.figure(figsize=(10, 5))
    plt.barh(row.index, row.values)
    plt.ylabel("Features")
    plt.xlabel("Values")
    plt.title(f"Feature Importance - {row.name}")
    plt.gca().invert_yaxis()
    plt.show()

# # Lưu mô hình đã huấn luyện
# import pickle
# import os
# def save_model(model_name, saved_models):
#     os.makedirs(f"../SavedModel/{model_name}_SavedModel", exist_ok=True)
#     for fold, model in saved_models:
#         filename = f"../SavedModel/{model_name}_SavedModel/{model_name}_model_fold_{fold}.pkl"
#         with open(filename, "wb") as f:
#             pickle.dump(model, f)