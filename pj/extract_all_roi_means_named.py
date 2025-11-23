#批量提取所有 ROI 的均值
import nibabel as nib
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
#  路径设置 
base_dir = r"D:\Psychological_Resilience\neuroproject\brain"
aal_path = os.path.join(base_dir, "AAL3.nii")
label_path = os.path.join(base_dir, "AAL3v1_1mm.nii.txt")  
angry_dir = os.path.join(base_dir, "19angry")
happy_dir = os.path.join(base_dir, "19happy")
# 输出路径 
out_angry = os.path.join(base_dir, "roi_mean_angry.csv")
out_happy = os.path.join(base_dir, "roi_mean_happy.csv")
print(" 正在加载 AAL3 模板...")
aal_img = nib.load(aal_path)
aal_data = aal_img.get_fdata().astype(int)
roi_ids = sorted(list(set(aal_data.flatten())))
if 0 in roi_ids:
    roi_ids.remove(0)
print(f" 模板加载成功，共检测到 {len(roi_ids)} 个脑区编号。")
# 读取标签表 
label_dict = {}
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            idx = int(parts[0])
            name = parts[1]
            label_dict[idx] = name
print(f" 标签表读取成功，共 {len(label_dict)} 个脑区名称。")
# 定义提取函数 
def extract_mean_from_folder(folder_path, condition_name):
    results = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".nii") or f.endswith(".nii.gz")]
    print(f" 开始处理 {condition_name}，共 {len(files)} 个文件。")
    for file in tqdm(files):
        file_path = os.path.join(folder_path, file)
        nii = nib.load(file_path)
        data = nii.get_fdata()
        subj_id = file.split("_")[0]  # 提取参与者ID
        roi_means = {"id": subj_id}
        for roi in roi_ids:
            mask = aal_data == roi
            roi_values = data[mask]
            roi_values = roi_values[np.isfinite(roi_values)]
            label = label_dict.get(roi, f"ROI_{roi}")
            roi_means[label] = np.mean(roi_values) if roi_values.size > 0 else np.nan
        results.append(roi_means)
    df = pd.DataFrame(results)
    print(f" {condition_name} 提取完成，共 {len(df)} 个参与者。")
    return df
#  执行提取 
df_angry = extract_mean_from_folder(angry_dir, "angry")
df_angry.to_csv(out_angry, index=False)
print(f" 已保存至: {out_angry}")
df_happy = extract_mean_from_folder(happy_dir, "happy")
df_happy.to_csv(out_happy, index=False)
print(f" 已保存至: {out_happy}")
print("\n 提取完成：ROI 平均激活值已保存！")
