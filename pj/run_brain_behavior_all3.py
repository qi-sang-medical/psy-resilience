# 分析显著性
import os, sys
import pandas as pd, numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
# 配置区 
brain_dir = r"D:\Psychological_Resilience\neuroproject\brain"
roi_angry_csv = os.path.join(brain_dir, "roi_mean_angry.csv")
roi_happy_csv = os.path.join(brain_dir, "roi_mean_happy.csv")
beh_clean = r"D:\Psychological_Resilience\neuroproject\Completed_Documents\master_behavior_age19_cleaned.csv"
out_dir = r"D:\Psychological_Resilience\neuroproject\Completed_Documents"
os.makedirs(out_dir, exist_ok=True)
# 分析设置
outcome = "semotion19"
abuse = "emotionabuse_x"
covs = ["sex_x", "ses"]
min_n = 30
# 加载文件 
for p in (roi_angry_csv, roi_happy_csv, beh_clean):
    if not os.path.exists(p):
        print(" 找不到文件：", p)
        sys.exit(1)
df_ang = pd.read_csv(roi_angry_csv)
df_hap = pd.read_csv(roi_happy_csv)
df_beh = pd.read_csv(beh_clean)
# 确保 id 列名称正确
for df in [df_ang, df_hap, df_beh]:
    if df.columns[0].lower() != "id":
        df.rename(columns={df.columns[0]: "id"}, inplace=True)
# 统一 ROI 名称 
angry_roi_cols = [c for c in df_ang.columns if c != "id"]
happy_roi_cols = [c for c in df_hap.columns if c != "id"]
roi_common = [r for r in angry_roi_cols if r in happy_roi_cols]
if len(roi_common) == 0:
    print(" 没有在 angry/happy 两个 CSV 中找到相同的 ROI 列名，请检查。")
    sys.exit(1)
# 合并 Angry / Happy 
ang2 = df_ang[["id"] + roi_common].rename(columns={r: f"{r}_angry" for r in roi_common})
hap2 = df_hap[["id"] + roi_common].rename(columns={r: f"{r}_happy" for r in roi_common})
roi_merged = pd.merge(ang2, hap2, on="id", how="inner")
# 计算差值列 
for r in roi_common:
    a, h, d = f"{r}_angry", f"{r}_happy", f"{r}_angry_minus_happy"
    roi_merged[d] = roi_merged[a] - roi_merged[h]
roi_merged.to_csv(os.path.join(out_dir, "roi_mean_merged_ah.csv"), index=False)
# 合并行为数据 
merged_all = pd.merge(roi_merged, df_beh, on="id", how="inner")
print(f" 合并后样本数: {len(merged_all)}")
# 定义函数 
def run_moderation(df, roi_list, suffix):
    results = []
    for roi in roi_list:
        use_cols = [outcome, abuse] + [c for c in covs if c in df.columns] + [roi]
        sub = df[use_cols].dropna()
        n = len(sub)
        if n < min_n:
            results.append([roi, n, np.nan, np.nan, np.nan])
            continue
        cov_str = " + ".join([c for c in covs if c in df.columns])
        formula = f"{outcome} ~ {abuse} * {roi}"
        if cov_str:
            formula += " + " + cov_str
        try:
            mod = ols(formula, data=sub).fit()
            inter_param = [p for p in mod.params.index if (abuse in p and roi in p and ":" in p)]
            inter_param = inter_param[0] if inter_param else None
            coef = mod.params.get(inter_param, np.nan)
            tval = mod.tvalues.get(inter_param, np.nan)
            pval = mod.pvalues.get(inter_param, np.nan)
            results.append([roi, n, coef, tval, pval])
        except Exception as e:
            print(f" 跳过 {roi}：{e}")
            results.append([roi, n, np.nan, np.nan, np.nan])
    res_df = pd.DataFrame(results, columns=["ROI", "n", "coef_inter", "t_inter", "p_inter"])
    mask = ~res_df["p_inter"].isna()
    if mask.sum() > 0:
        rej, p_adj, _, _ = multipletests(res_df.loc[mask, "p_inter"], method="fdr_bh")
        res_df.loc[mask, "p_fdr"] = p_adj
        res_df.loc[mask, "sig_fdr"] = rej
    else:
        res_df["p_fdr"] = np.nan
        res_df["sig_fdr"] = False
    outpath = os.path.join(out_dir, f"mod_{suffix}_results.csv")
    res_df.to_csv(outpath, index=False)
    print(f" 输出: {outpath}   显著(FDR): {int(res_df['sig_fdr'].sum())}")
    return res_df
# 三个分析 
res_diff = run_moderation(merged_all, [f"{r}_angry_minus_happy" for r in roi_common], "diff")
res_angry = run_moderation(merged_all, [f"{r}_angry" for r in roi_common], "angry")
res_happy = run_moderation(merged_all, [f"{r}_happy" for r in roi_common], "happy")

print(f" 全部完成，结果保存在: {out_dir}")
