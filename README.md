# Ovarian ultrasound / synthetic CEUS experiment code

## Directory map

| Directory | Scope |
|---|---|
| `01_primary_image_level/synthesis` | Submitted Enhanced Pix2PixGAN, basic Pix2PixGAN, and six loss-ablation implementations. |
| `01_primary_image_level/classification` | Available multi-backbone classifier implementation and original evaluation script. |
| `01_primary_image_level/legacy_diffusion_candidates` | Available earlier diffusion implementations from the parent `DiffGen` project. Their exact correspondence to reported presubmission comparisons has not been verified. |
| `01_primary_image_level/legacy_preprocessing` | Earlier augmentation and dataset-construction utilities from the parent project. |
| `02_split_and_patient_sensitivity/04_...` | Patient mapping, frozen split, common model/data code, classifier training and evaluation, diffusion comparator, and integrity tests. |
| `02_split_and_patient_sensitivity/06_...` | Reconstruction and audit of the submitted image-level development split. |
| `02_split_and_patient_sensitivity/07_...` | ResNet18 U-Net and Swin-Tiny U-Net comparison models and evaluation. |
| `02_split_and_patient_sensitivity/10_...` | Patient-level 7:3 rerun of the original Pix2Pix family and downstream analyses. |
| `03_lesion_roi` | Geometrically aligned lesion-ROI synthesis metrics. |
| `04_interpretability_and_calibration` | Grad-CAM audits and descriptive calibration figure. |
| `05_clinical_reader_assessment` | Blinded test-pair preparation and two-reader analysis. |
