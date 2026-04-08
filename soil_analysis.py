"""
土壤实验数据分析脚本
功能：分组建模，识别不同地区影响土壤质量指数(SQI)的关键因素
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    print("=" * 60)
    print("步骤 1: 数据加载与预处理")
    print("=" * 60)
    
    df = pd.read_excel(file_path)
    print(f"\n数据形状: {df.shape}")
    print(f"\n数据列名: {df.columns.tolist()}")
    print("\n前5行数据预览:")
    print(df.head())
    
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    le = LabelEncoder()
    df['treatment_encoded'] = le.fit_transform(df['treatment'])
    print(f"\ntreatment 编码映射:")
    for i, treatment in enumerate(le.classes_):
        print(f"  {treatment} -> {i}")
    
    feature_cols = ['treatment_encoded', 'pH', 'CEC', 'Ex.Ca', 'Ex.Mg', 'AP', 'AK', 'pHBC']
    target_col = 'SQI'
    
    missing_cols = set(feature_cols + [target_col, 'region']) - set(df.columns)
    if missing_cols:
        print(f"\n警告: 以下列缺失: {missing_cols}")
        print("可用的列:", df.columns.tolist())
    
    return df, feature_cols, target_col, le

def split_by_region(df):
    """按地区拆分数据"""
    print("\n" + "=" * 60)
    print("步骤 2: 按地区拆分数据")
    print("=" * 60)
    
    print("\n地区分布:")
    print(df['region'].value_counts())
    
    df_guangxi = df[df['region'] == '广西'].copy()
    df_jiangxi = df[df['region'] == '江西'].copy()
    
    print(f"\n广西组样本数: {len(df_guangxi)}")
    print(f"江西组样本数: {len(df_jiangxi)}")
    
    return df_guangxi, df_jiangxi

def train_random_forest(df, features, target, region_name):
    """训练随机森林回归模型"""
    print(f"\n训练 {region_name} 地区的模型...")
    
    X = df[features]
    y = df[target]
    
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X, y)
    r2_score = rf.score(X, y)
    print(f"  R² 得分: {r2_score:.4f}")
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return rf, feature_importance

def analyze_treatment_ranking(df):
    """按处理方式分组分析SQI，获取排名"""
    print("\n" + "=" * 60)
    print("步骤 5: 不同处理方式对SQI的影响分析")
    print("=" * 60)
    
    treatment_stats = df.groupby('treatment')['SQI'].agg(['mean', 'std', 'count']).reset_index()
    treatment_stats = treatment_stats.sort_values('mean', ascending=False)
    
    print("\n【处理方式排名结果】")
    print(f"{'排名':<6}{'处理方式':<20}{'平均SQI':<12}{'标准差':<12}{'样本数':<10}")
    print("-" * 60)
    for i, row in treatment_stats.iterrows():
        print(f"{i+1:<6}{row['treatment']:<20}{row['mean']:.4f}{'':<7}{row['std']:.4f}{'':<7}{row['count']:<10}")
    
    best_treatment = treatment_stats.iloc[0]
    print(f"\n最佳处理: {best_treatment['treatment']}")
    print(f"平均 SQI: {best_treatment['mean']:.4f}")
    print(f"标准差: {best_treatment['std']:.4f}")
    print(f"样本数: {int(best_treatment['count'])}")
    
    print("\n" + "=" * 60)
    print("【结论】")
    print(f"效果最好的处理是 {best_treatment['treatment']}，平均 SQI 为 {best_treatment['mean']:.2f}")
    print("=" * 60)
    
    return treatment_stats

def analyze_feature_importance(importance_gx, importance_jx):
    """特征重要性分析"""
    print("\n" + "=" * 60)
    print("步骤 4: 特征重要性分析")
    print("=" * 60)
    print("\n特征重要性数据已保存")

def generate_summary_report(importance_gx, importance_jx):
    """生成分析摘要报告"""
    print("\n" + "=" * 60)
    print("分析摘要报告")
    print("=" * 60)
    
    print("\n【广西地区】")
    print("Top 3 关键影响因素:")
    for i, row in importance_gx.head(3).iterrows():
        feature_name = row['Feature'].replace('_encoded', '')
        print(f"  {i+1}. {feature_name}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
    
    print("\n【江西地区】")
    print("Top 3 关键影响因素:")
    for i, row in importance_jx.head(3).iterrows():
        feature_name = row['Feature'].replace('_encoded', '')
        print(f"  {i+1}. {feature_name}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
    
    print("\n【对比分析】")
    print("  - pH 在两个地区的重要性: 广西 {:.2f}% vs 江西 {:.2f}%".format(
        importance_gx[importance_gx['Feature'] == 'pH']['Importance'].values[0] * 100 if len(importance_gx[importance_gx['Feature'] == 'pH']) > 0 else 0,
        importance_jx[importance_jx['Feature'] == 'pH']['Importance'].values[0] * 100 if len(importance_jx[importance_jx['Feature'] == 'pH']) > 0 else 0
    ))
    print("  - CEC 在两个地区的重要性: 广西 {:.2f}% vs 江西 {:.2f}%".format(
        importance_gx[importance_gx['Feature'] == 'CEC']['Importance'].values[0] * 100 if len(importance_gx[importance_gx['Feature'] == 'CEC']) > 0 else 0,
        importance_jx[importance_jx['Feature'] == 'CEC']['Importance'].values[0] * 100 if len(importance_jx[importance_jx['Feature'] == 'CEC']) > 0 else 0
    ))
    
    report_file = r'C:\Users\Greal1sh\Desktop\analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("土壤质量指数(SQI)影响因素分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【广西地区】\n")
        f.write("特征重要性排序:\n")
        for i, row in importance_gx.iterrows():
            feature_name = row['Feature'].replace('_encoded', '')
            f.write(f"  {i+1}. {feature_name}: {row['Importance']:.4f}\n")
        
        f.write("\n【江西地区】\n")
        f.write("特征重要性排序:\n")
        for i, row in importance_jx.iterrows():
            feature_name = row['Feature'].replace('_encoded', '')
            f.write(f"  {i+1}. {feature_name}: {row['Importance']:.4f}\n")
    
    print(f"\n详细报告已保存至: {report_file}")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("土壤质量指数(SQI)影响因素分析")
    print("=" * 60)
    
    file_path = 'C:/Users/Greal1sh/Desktop/cleaned_data.xlsx'
    
    try:
        df, feature_cols, target_col, le = load_and_preprocess_data(file_path)
        df_guangxi, df_jiangxi = split_by_region(df)
        
        print("\n" + "=" * 60)
        print("步骤 3: 训练随机森林回归模型")
        print("=" * 60)
        
        rf_gx, importance_gx = train_random_forest(df_guangxi, feature_cols, target_col, "广西")
        rf_jx, importance_jx = train_random_forest(df_jiangxi, feature_cols, target_col, "江西")
        
        print("\n广西地区特征重要性:")
        print(importance_gx)
        
        print("\n江西地区特征重要性:")
        print(importance_jx)
        
        importance_gx.to_csv('C:/Users/Greal1sh/Desktop/importance_guangxi.csv', index=False, encoding='utf-8-sig')
        importance_jx.to_csv('C:/Users/Greal1sh/Desktop/importance_jiangxi.csv', index=False, encoding='utf-8-sig')
        print("\n特征重要性已保存至 CSV 文件")
        
        analyze_feature_importance(importance_gx, importance_jx)
        treatment_stats = analyze_treatment_ranking(df)
        generate_summary_report(importance_gx, importance_jx)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"\n错误: 找不到文件 {file_path}")
        print("请确认文件路径是否正确")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    main()
