"""
automate_NamaSiswa.py
=====================
Script otomatisasi preprocessing dataset Wine Quality.
Mengikuti seluruh workflow notebook eksperimen.

Cara menjalankan:
    python automate_NamaSiswa.py
    python automate_NamaSiswa.py --input dataset_raw/winequality.csv --output preprocessing/dataset_preprocessing

Author  : NamaSiswa
Dataset : Wine Quality (UCI)
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
def load_data(input_path: str) -> pd.DataFrame:
    """Memuat dataset dari path yang diberikan."""
    log.info(f"Loading dataset dari: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    # Support file CSV dengan separator titik koma (format UCI Wine)
    try:
        df = pd.read_csv(input_path, sep=";")
        if df.shape[1] == 1:          # coba koma jika hanya 1 kolom terbaca
            df = pd.read_csv(input_path, sep=",")
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file: {e}")

    log.info(f"Dataset berhasil dimuat. Shape: {df.shape}")
    log.info(f"Kolom: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────
# 2. DATA UNDERSTANDING
# ──────────────────────────────────────────────
def data_understanding(df: pd.DataFrame) -> None:
    """Menampilkan ringkasan informasi dataset."""
    log.info("=" * 50)
    log.info("DATA UNDERSTANDING")
    log.info("=" * 50)
    log.info(f"Shape          : {df.shape}")
    log.info(f"Tipe data      :\n{df.dtypes}")
    log.info(f"Missing values :\n{df.isnull().sum()}")
    log.info(f"Duplikat       : {df.duplicated().sum()}")
    log.info(f"Statistik deskriptif:\n{df.describe()}")
    log.info(f"Distribusi target 'quality':\n{df['quality'].value_counts().sort_index()}")


# ──────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ──────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing value dengan median untuk kolom numerik."""
    missing_before = df.isnull().sum().sum()
    log.info(f"Total missing values sebelum penanganan: {missing_before}")

    if missing_before > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.info(f"  Kolom '{col}': diisi dengan median = {median_val:.4f}")

    missing_after = df.isnull().sum().sum()
    log.info(f"Total missing values setelah penanganan: {missing_after}")
    return df


# ──────────────────────────────────────────────
# 4. HANDLE DUPLICATES
# ──────────────────────────────────────────────
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    log.info(f"Duplikat dihapus: {n_before - n_after} baris "
             f"({n_before} → {n_after})")
    return df


# ──────────────────────────────────────────────
# 5. HANDLE OUTLIERS  (IQR Capping)
# ──────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame,
                    exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Capping (Winsorization).
    Outlier di-*cap* pada batas bawah/atas, bukan dihapus,
    agar jumlah data tetap terjaga.
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    log.info("Menangani outlier dengan IQR Capping ...")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outlier > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            log.info(f"  '{col}': {n_outlier} outlier di-cap "
                     f"[{lower:.4f}, {upper:.4f}]")

    return df


# ──────────────────────────────────────────────
# 6. FEATURE ENGINEERING  — buat target biner
# ──────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat kolom target biner 'quality_label':
      1 = wine berkualitas baik  (quality >= 7)
      0 = wine berkualitas biasa (quality <  7)
    """
    df["quality_label"] = (df["quality"] >= 7).astype(int)
    log.info(
        f"Distribusi quality_label:\n"
        f"{df['quality_label'].value_counts().to_string()}"
    )
    return df


# ──────────────────────────────────────────────
# 7. FEATURE SELECTION
# ──────────────────────────────────────────────
def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Memisahkan fitur (X) dan target (y).
    Kolom 'quality' dihapus karena sudah di-encode ke 'quality_label'.
    """
    drop_cols = ["quality", "quality_label"]
    X = df.drop(columns=drop_cols)
    y = df["quality_label"]
    log.info(f"Fitur yang digunakan ({len(X.columns)}): {list(X.columns)}")
    log.info(f"Target                               : quality_label")
    return X, y


# ──────────────────────────────────────────────
# 8. TRAIN-TEST SPLIT
# ──────────────────────────────────────────────
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Membagi data menjadi train set dan test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(
        f"Train set: {X_train.shape[0]} baris | "
        f"Test set : {X_test.shape[0]} baris"
    )
    log.info(
        f"Distribusi y_train:\n{y_train.value_counts().to_string()}"
    )
    log.info(
        f"Distribusi y_test :\n{y_test.value_counts().to_string()}"
    )
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 9. SCALING  (StandardScaler, fit hanya di train)
# ──────────────────────────────────────────────
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Melakukan StandardScaler.
    Scaler di-fit HANYA pada X_train untuk menghindari data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    log.info("StandardScaler diterapkan (fit on train, transform on test).")
    return X_train_scaled, X_test_scaled


# ──────────────────────────────────────────────
# 10. SAVE PREPROCESSED DATASET
# ──────────────────────────────────────────────
def save_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """Menyimpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"),  index=False)

    log.info(f"Dataset preprocessing disimpan ke: {output_dir}")
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        fpath = os.path.join(output_dir, fname)
        log.info(f"  ✔ {fname} ({os.path.getsize(fpath):,} bytes)")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def run_pipeline(input_path: str, output_dir: str) -> None:
    """Menjalankan seluruh pipeline preprocessing secara berurutan."""
    log.info("=" * 50)
    log.info("MULAI PIPELINE PREPROCESSING")
    log.info("=" * 50)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Understanding
    data_understanding(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Handle duplicates
    df = handle_duplicates(df)

    # Step 5: Handle outliers (kecuali kolom target)
    df = handle_outliers(df, exclude_cols=["quality"])

    # Step 6: Feature engineering (buat target biner)
    df = feature_engineering(df)

    # Step 7: Pilih fitur & target
    X, y = select_features(df)

    # Step 8: Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 9: Scaling
    X_train, X_test = scale_features(X_train, X_test)

    # Step 10: Simpan
    save_datasets(X_train, X_test, y_train, y_test, output_dir)

    log.info("=" * 50)
    log.info("PIPELINE PREPROCESSING SELESAI ✔")
    log.info("=" * 50)


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated preprocessing pipeline - Wine Quality Dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset_raw/winequality.csv",
        help="Path ke dataset mentah (default: dataset_raw/winequality.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="preprocessing/dataset_preprocessing",
        help="Folder output dataset preprocessing "
             "(default: preprocessing/dataset_preprocessing)",
    )
    args = parser.parse_args()
    run_pipeline(input_path=args.input, output_dir=args.output)
