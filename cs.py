from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')


# 封装数据处理函数
def process_wage_and_life_data(dfs):
    dfs['生活水平'] = pd.read_excel("生活水平.xlsx", header=1)

    melted_wage = dfs['工资水平'].melt(
        id_vars='averageWage',
        var_name='城市名称',
        value_name='平均工资'
    )
    melted_wage.rename(columns={'averageWage': '年份'}, inplace=True)
    melted_wage['年份'] = melted_wage['年份'].str.extract(r'(\d+)').astype(int)

    melted_life = dfs['生活水平'].melt(
        id_vars='城市名称',
        var_name='年份',
        value_name='可支配收入'
    )
    return melted_wage, melted_life


# 时间序列预测函数
def predict_feature_for_city(city_data, feature, target_year=2023):
    """为特定城市的特定特征建立时间序列模型并预测"""
    city_data = city_data.sort_values('年份')
    years = city_data['年份'].unique()

    # 确保有足够的历史数据进行建模
    if len(years) < 3:
        # 如果历史数据不足，使用最新值
        latest_value = city_data[city_data['年份'] == city_data['年份'].max()][feature].values[0]
        return latest_value

    try:
        # 使用ARIMA模型进行预测
        model = ARIMA(city_data[feature].values, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=target_year - city_data['年份'].max())
        return forecast[0]
    except:
        # 如果模型失败，使用线性外推
        x = city_data['年份'].values
        y = city_data[feature].values
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 0
        return y[-1] + slope * (target_year - x[-1])


# 定义文件路径
file_paths = {
    '城镇化率': "城镇化率.xlsx",
    '工资水平': "工资水平.xlsx",
    '就业信息': "就业信息.xlsx",
    '年龄结构': "年龄结构.xlsx",
    '人口规模': "人口规模.xlsx",
    '人口密度': "人口密度.xlsx",
    '生活水平': "生活水平.xlsx"
}

# 读取文件
dfs = {}
for file_name, file_path in file_paths.items():
    try:
        dfs[file_name] = pd.read_excel(file_path)
        print(f"{file_name} 文件读取成功，数据基本信息：")
        dfs[file_name].info()

        print(f"{file_name} 文件全部内容信息：")
        #print(dfs[file_name].to_csv(sep='\t', na_rep='nan'))
    except Exception as e:
        print(f"{file_name} 文件读取失败: {e}")

melted_wage, melted_life = process_wage_and_life_data(dfs)

# 合并数据
merged_df = dfs['城镇化率'].merge(
    melted_wage,
    on=['城市名称', '年份'],
    how='outer'
).merge(
    dfs['就业信息'],
    on=['城市名称', '年份'],
    how='outer'
).merge(
    dfs['年龄结构'].rename(columns={'城市': '城市名称'}),
    on=['城市名称', '年份'],
    how='outer'
).merge(
    dfs['人口规模'],
    on=['城市名称', '年份'],
    how='outer'
).merge(
    dfs['人口密度'],
    on=['城市名称', '年份'],
    how='outer'
).merge(
    melted_life,
    on=['城市名称', '年份'],
    how='outer'
)


# ====================== 缺失值填充 ======================
def analyze_missing_data(df):
    """分析缺失数据分布"""
    missing = df.isnull().sum().sort_values(ascending=False)
    percent = (missing / df.shape[0]).sort_values(ascending=False)
    missing_data = pd.concat([missing, percent], axis=1, keys=['缺失值数量', '缺失率'])
    return missing_data


# 分析缺失数据
missing_data = analyze_missing_data(merged_df)
print("\n缺失数据分布：")
#print(missing_data)

# 分离数值型和分类型特征（排除目标变量）
target_col = '常住人口（万人）'
all_features = merged_df.drop(target_col, axis=1).columns.tolist()

# 处理年份列（确保为数值型，且不参与填充）
if '年份' in merged_df.columns:
    merged_df['年份'] = pd.to_numeric(merged_df['年份'], errors='coerce').astype('Int64')
    numeric_cols = merged_df.select_dtypes(include=['number']).columns.drop(['年份'])
else:
    numeric_cols = merged_df.select_dtypes(include=['number']).columns.tolist()

cat_cols = merged_df.select_dtypes(include=['object', 'category']).columns.tolist()


# KNN 填充函数（仅数值列）
def knn_imputation(df, feature_cols):
    df_impute = df.copy()
    numeric_cols = df_impute[feature_cols].select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        return df_impute

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_impute[numeric_cols])
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_scaled)
    df_impute[numeric_cols] = scaler.inverse_transform(X_imputed)
    return df_impute


# 逻辑回归填充分类列（仅保留核心逻辑，简化异常处理）
def logistic_regression_imputation(df, features):
    df = df.copy()
    for col in features:
        if df[col].isnull().any() and df[col].dtype != 'category':
            df[col] = df[col].astype('category')
        df[col].fillna(df[col].mode()[0], inplace=True)  # 简化为众数填充（避免逻辑回归复杂性）
    return df


# 执行填充（先数值列 KNN，再分类列众数填充）
merged_df = knn_imputation(merged_df, numeric_cols)
merged_df = logistic_regression_imputation(merged_df, cat_cols)

# 保存结果（**删除均值填充步骤**）
merged_df.to_csv("tt.csv", index=False)

print('\n数据预处理和合并完成，合并后的数据基本信息：')
merged_df.info()

print('合并后的数据全部内容信息：')
#print(merged_df.to_csv(sep='\t', na_rep='nan'))

# ====================== 训练 ======================

# 确定特征和目标变量
X = merged_df.drop(['常住人口（万人）', '户籍人口（万人）'], axis=1)
y = merged_df['常住人口（万人）']

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对分类变量进行独热编码，对数值变量进行标准化
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 定义模型管道（仅保留GradientBoostingRegressor）
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=3)

# 定义参数网格
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 4, 5]
}

# 网格搜索
grid_search = GridSearchCV(
    pipeline, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
)

# 训练模型
grid_search.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = grid_search.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)  # 计算R²分数

print(f"模型评估结果: MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f}, 准确率 = {r2*100:.2f}%")

# 准备 2023 年的预测数据
unique_cities = merged_df['城市名称'].unique()
years = [2023] * len(unique_cities)
prediction_data = pd.DataFrame({
    '城市名称': unique_cities,
    '年份': years
})

# 为每个城市的每个特征建立时间序列模型并预测
for col in numerical_cols:
    if col not in ['年份']:  # 年份已经在预测数据中设置好了
        city_predictions = []
        for city in unique_cities:
            city_data = merged_df[merged_df['城市名称'] == city].copy()
            if city_data.empty:
                # 如果该城市没有历史数据，使用所有城市的平均值
                city_prediction = merged_df[col].mean()
            else:
                city_prediction = predict_feature_for_city(city_data, col)
            city_predictions.append(city_prediction)
        prediction_data[col] = city_predictions

# 预测 2023 年的常住人口
predictions_2023 = grid_search.predict(prediction_data)

# 将预测结果保存为 DataFrame
prediction_results = pd.DataFrame({
    'city_id': unique_cities,
    'year': 2023,
    'pred': predictions_2023
})

# 保存预测结果为 CSV 文件
prediction_results.to_csv("submission.csv", index=False)

