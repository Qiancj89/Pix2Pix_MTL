# 论文实验代码公开整理版

本目录是供 GitHub 发布前审核的代码副本；原实验脚本未改动。目录按投稿版图片级实验、数据拆分与患者级敏感性分析、病灶 ROI、可解释性与校准、临床专家盲评、补充数值表分组。

## 分类

- `01_primary_image_level`：投稿版 Enhanced Pix2PixGAN、Basic Pix2Pix、6 个损失消融以及可用的多骨干网络分类代码。`legacy_diffusion_candidates` 和 `legacy_preprocessing` 保存父目录中的早期代码，但无法确认哪一份对应另一台电脑上的投稿版比较结果，不应据此声称精确复现。
- `02_split_and_patient_sensitivity`：原始图片级拆分核查、患者映射与冻结拆分、条件扩散及 U-Net 比较模型、患者级 7:3 敏感性分析。`04`、`06`、`07`、`10` 四个历史模块保持相邻，以保留它们之间的相对路径关系。
- `03_lesion_roi`：与原 CEUS 几何对齐的病灶 ROI 定量评价。
- `04_interpretability_and_calibration`：Grad-CAM 审计与描述性校准图。
- `05_clinical_reader_assessment`：测试集盲评配对制作及两名读者的评分分析。发布副本改用环境变量指定私有输入，不包含医生姓名或真实数据路径。