# Credit Risk Default Prediction

## Business Problem
[Lembaga pembiayaan perlu memprediksi probabilitas gagal bayar sebelum 
pencairan pinjaman, dan menentukan threshold approval yang optimal secara finansial]

## Dataset
Home Credit Default Risk (Kaggle), [isi jumlah baris] aplikasi, target rate 
default [isi]%, fitur digabung dari application_train, bureau, dan 
previous_application.

## Approach
1. EDA, pembersihan anomali (DAYS_EMPLOYED), agregasi multi-tabel
2. Feature engineering (rasio finansial, binning)
3. Baseline Logistic Regression -> LightGBM -> tuning Optuna (30 trial, 3-fold CV)
4. Evaluasi dengan AUC, PR-AUC, KS statistic
5. Interpretasi SHAP (global & individual)
6. Kalibrasi probabilitas
7. Threshold optimal berbasis simulasi cost bisnis
8. Experiment tracking dengan MLflow

## Key Findings
- Model comparison: [tabel AUC/PR-AUC/KS dari Hari 9]
- Fitur paling berpengaruh: [isi dari SHAP, biasanya EXT_SOURCE_1/2/3]
- Threshold optimal: [isi]% (vs default 50%), menghemat estimasi [isi]% 
  total cost dibanding threshold default

## Business Recommendation
[Contoh: "Gunakan threshold X% alih-alih 50% default, mengurangi estimasi 
kerugian sebesar Y per [periode]. Fokus review manual pada aplikasi dengan 
EXT_SOURCE rendah dan rasio cicilan-pendapatan tinggi."]

## Limitations
- Asumsi biaya (loss given default, lost margin) adalah estimasi, perlu 
  divalidasi dengan data finansial riil perusahaan
- Kalibrasi dilakukan di test set yang sama dengan evaluasi akhir, idealnya 
  pakai validation set terpisah
- Model tidak memperhitungkan perubahan kondisi makroekonomi dari waktu ke waktu

## Tech Stack
Python, LightGBM, Optuna, SHAP, MLflow, scikit-learn

## How to Reproduce

```bash
# 1. Clone and enter the repo
git clone https://github.com/NatanaelAlbert22/feature-ab-test-funnel-analysis.git
cd feature-ab-test-funnel-analysis

# 2. Create environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. Download the dataset from Kaggle into data/raw/
#    https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing
```

4. Run `notebooks/01_eda_funnel.ipynb` to clean the data and build the funnel.
5. Run `notebooks/02_hypothesis_testing.ipynb` to reproduce the statistical tests.

## Author

**NATANAEL ALBERT** · [LinkedIn](https://www.linkedin.com/in/natanael-albert/) · [GitHub](https://github.com/NatanaelAlbert22)