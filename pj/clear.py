import pandas as pd

# 输入路径
in_path = r"D:\Psychological Resilience\neuroproject\Completed_Documents\master_behavior_age19.csv"
out_path = r"D:\Psychological Resilience\neuroproject\Completed_Documents\master_behavior_age19_cleaned.csv"

# 读取数据
df = pd.read_csv(in_path)

# 定义关键列（存在缺失就删除该行）
key_vars = [
    'id', 'semotion19', 'emotionabuse_x', 'physicalabuse_x', 'sexualabuse_x',
    'ctqabuse_x', 'sex_x', 'ses'
]

# 删除缺失行
df_clean = df.dropna(subset=key_vars)

# 输出结果
df_clean.to_csv(out_path, index=False)
print(f"✅ 清理完成！从 {len(df)} 行中保留 {len(df_clean)} 行。")
print(f"清理后文件已保存至：{out_path}")

print("\n--- 缺失比例 ---")
print(df_clean.isna().mean().sort_values(ascending=False).head(10))

print("\n--- 关键变量描述统计 ---")
print(df_clean[['semotion19', 'ctqabuse_x', 'ses']].describe())
