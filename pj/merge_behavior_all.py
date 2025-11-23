import pandas as pd
import os

# 主路径
base_main = r"D:\Psychological Resilience\neuroproject"
base_beh = os.path.join(base_main, "behavior")

files = {
    "data_age19": os.path.join(base_main, "data_age19_1137.csv"),
    "ctq": os.path.join(base_beh, "CTQ_19.csv"),
    "ctqabuse": os.path.join(base_beh, "CTQabuse_19.csv"),
    "sdq": os.path.join(base_beh, "SDQ_19.csv"),
    "prs": os.path.join(base_beh, "prs_mdd.csv"),
    "sex_site": os.path.join(base_beh, "IMAGEN_sex_site.csv"),
    "ses": os.path.join(base_beh, "SES.csv"),
}

dfs = {}
for key, path in files.items():
    if not os.path.exists(path):
        print(f"⚠️ 未找到文件: {path}")
        continue
    df = pd.read_csv(path)
    print(f"\n[{key}] {os.path.basename(path)} 共有 {df.shape[0]} 行, {df.shape[1]} 列")
    print("列名示例:", list(df.columns)[:8])
    dfs[key] = df

# 统一列名为小写
for k in dfs:
    dfs[k].columns = [c.lower() for c in dfs[k].columns]

# 统一 id 名称
for k, df in dfs.items():
    id_col = [c for c in df.columns if "id" in c or "subject" in c]
    if not id_col:
        print(f"⚠️ 文件 {k} 没检测到 id 列，请手动检查！")
    else:
        df.rename(columns={id_col[0]: "id"}, inplace=True)

# 合并，以 data_age19 为主表
merged = dfs["data_age19"]
for name in ["ctq", "ctqabuse", "sdq", "prs", "sex_site", "ses"]:
    if name in dfs:
        merged = pd.merge(merged, dfs[name], on="id", how="left")
        print(f"合并 {name} 完成 → 当前形状：{merged.shape}")

# 输出文件
out_path = os.path.join(base_main, "master_behavior_age19.csv")
merged.to_csv(out_path, index=False)

print("\n✅ 合并完成！输出文件：", out_path)
print("最终行列：", merged.shape)
