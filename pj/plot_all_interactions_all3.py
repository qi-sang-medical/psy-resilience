import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#  文件路径 
base_dir = r"D:\Psychological_Resilience\neuroproject\Completed_Documents"
beh_path = os.path.join(base_dir, "master_behavior_age19_cleaned.csv")
# 三种条件对应的脑区数据
roi_paths = {
    "angry": os.path.join(base_dir, "roi_mean_angry.csv"),
    "happy": os.path.join(base_dir, "roi_mean_happy.csv"),
    "diff":  os.path.join(base_dir, "roi_mean_merged_ah.csv")
}
# 输出路径（分任务）
out_dirs = {
    name: os.path.join(base_dir, f"interaction_plots_{name}")
    for name in roi_paths.keys()
}
for d in out_dirs.values():
    os.makedirs(d, exist_ok=True)
#  读取行为数据 
df_beh = pd.read_csv(beh_path)
# 主循环：3 个任务条件 
for task, roi_path in roi_paths.items():
    print(f"\n====== 正在处理 {task.upper()} 条件 ======")
    df_roi = pd.read_csv(roi_path)
    df = pd.merge(df_beh, df_roi, on="id", how="inner")
    roi_cols = [c for c in df_roi.columns if c != "id"]
    results = []
    for roi in roi_cols:
        formula = f"semotion19 ~ emotionabuse_x * {roi} + sex_x + ses"
        try:
            model = smf.ols(formula=formula, data=df).fit()
            p_val = model.pvalues.get(f"emotionabuse_x:{roi}", np.nan)
            results.append((roi, p_val))
            # 绘制显著或接近显著的交互图
            if p_val < 0.1:
                x = np.linspace(df["emotionabuse_x"].min(), df["emotionabuse_x"].max(), 50)
                roi_mean = df[roi].mean()
                roi_high = roi_mean + df[roi].std()
                roi_low = roi_mean - df[roi].std()
                pred_high = model.predict(pd.DataFrame({
                    "emotionabuse_x": x,
                    roi: roi_high,
                    "sex_x": df["sex_x"].mean(),
                    "ses": df["ses"].mean()
                }))
                pred_low = model.predict(pd.DataFrame({
                    "emotionabuse_x": x,
                    roi: roi_low,
                    "sex_x": df["sex_x"].mean(),
                    "ses": df["ses"].mean()
                }))
                plt.figure(figsize=(8, 6))
                plt.plot(x, pred_high, color="red", label=f"{roi} 高 (+1SD)")
                plt.plot(x, pred_low, color="blue", label=f"{roi} 低 (-1SD)")
                plt.xlabel("情感虐待得分 (emotionabuse_x)", fontsize=12)
                plt.ylabel("情绪症状得分 (semotion19)", fontsize=12)
                plt.title(f"{roi} × 情感虐待 交互效应 (p={p_val:.3f}) [{task}]", fontsize=14)
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                out_path = os.path.join(out_dirs[task], f"{roi}_interaction.png")
                plt.savefig(out_path, dpi=300)
                plt.close()
        except Exception as e:
            print(f" {roi} 出错: {e}")
    #  输出结果表 
    res_df = pd.DataFrame(results, columns=["ROI", "p_value"]).sort_values("p_value")
    res_path = os.path.join(out_dirs[task], "interaction_summary.csv")
    res_df.to_csv(res_path, index=False, encoding="utf-8-sig")
    sig_count = (res_df["p_value"] < 0.05).sum()
    print(f" {task.upper()} 条件绘图完成，共生成 {sig_count} 张显著交互图（p<0.05）。")
    print(f" 结果保存于: {out_dirs[task]}")
print("\n 所有条件均已完成。")
