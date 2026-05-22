import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset CSV dari filepath yang diberikan."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# ──────────────────────────────────────────────
# 2. HAPUS KOLOM TIDAK RELEVAN
# ──────────────────────────────────────────────
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hapus kolom identifier (User_ID) dan kolom raw skor yang sudah
    direpresentasikan oleh kolom severity (menghindari data leakage).
    """
    cols_to_drop = ['User_ID', 'PHQ_9_Score', 'GAD_7_Score']
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[drop_irrelevant_columns] Dropped: {existing} → shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 3. TANGANI MISSING VALUES
# ──────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isi missing values:
    - Kolom numerik  → median
    - Kolom kategorikal → modus
    """
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    total_missing = df.isnull().sum().sum()
    print(f"[handle_missing_values] Total missing after handling: {total_missing}")
    return df


# ──────────────────────────────────────────────
# 4. HAPUS DUPLIKAT
# ──────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[remove_duplicates] Removed {before - after} duplicate rows → shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
# 5. ENCODING KOLOM KATEGORIKAL
# ──────────────────────────────────────────────
def encode_categorical(df: pd.DataFrame, output_dir: str) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode semua kolom kategorikal.
    Menyimpan mapping encoder untuk inference nanti.
    
    Returns:
        df: DataFrame dengan kolom kategorikal ter-encoded
        encoders_dict: Dictionary mapping {col_name: {value: encoded_value}}
    """
    encoders = {}
    encoders_dict = {}
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
        # Simpan mapping untuk inference — convert numpy int64 ke Python int
        mapping = {}
        for cls, encoded in zip(le.classes_, le.transform(le.classes_)):
            mapping[str(cls)] = int(encoded)
        
        encoders[col] = le
        encoders_dict[col] = mapping
    
    # Simpan encoder objects ke file
    joblib.dump(encoders, os.path.join(output_dir, 'label_encoders.pkl'))
    
    # Simpan mapping JSON untuk referensi
    with open(os.path.join(output_dir, 'encoding_mapping.json'), 'w') as f:
        json.dump(encoders_dict, f, indent=2)
    
    print(f"[encode_categorical] Encoded columns: {cat_cols}")
    return df, encoders_dict


# ──────────────────────────────────────────────
# 6. DETEKSI & TANGANI OUTLIER (IQR CLIPPING)
# ──────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame,
                    cols: list = None) -> pd.DataFrame:
    """
    Tangani outlier dengan IQR clipping pada kolom numerik yang ditentukan.
    Default: Age, Daily_Screen_Time_Hours, Sleep_Duration_Hours.
    """
    if cols is None:
        cols = ['Age', 'Daily_Screen_Time_Hours', 'Sleep_Duration_Hours']
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    print(f"[handle_outliers] Outlier clipped for: {cols}")
    return df


# ──────────────────────────────────────────────
# 7. SPLIT & NORMALISASI
# ──────────────────────────────────────────────
def split_and_scale(df: pd.DataFrame,
                    output_dir: str,
                    target_col: str = 'PHQ_9_Severity',
                    test_size: float = 0.2,
                    random_state: int = 42):
    """
    Pisahkan fitur dan target, lakukan train-test split, lalu
    normalisasi fitur numerik menggunakan StandardScaler.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_cols = ['Age', 'Daily_Screen_Time_Hours', 'Sleep_Duration_Hours']
    num_cols = [c for c in num_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    
    # Simpan scaler untuk inference nanti
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"[split_and_scale] Scaler saved to {output_dir}/scaler.pkl")

    print(f"[split_and_scale] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 8. SIMPAN OUTPUT
# ──────────────────────────────────────────────
def save_preprocessed(X_train, X_test, y_train, y_test,
                       output_dir: str = 'social_media_mental_health_preprocessing'):
    """Simpan dataset train dan test ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.csv')
    test_path  = os.path.join(output_dir, 'test.csv')

    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test,  y_test],  axis=1).to_csv(test_path,  index=False)

    print(f"[save_preprocessed] Saved:\n  {train_path}\n  {test_path}")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def preprocess_pipeline(input_filepath: str,
                         output_dir: str = 'social_media_mental_health_preprocessing'):
    """
    Jalankan seluruh pipeline preprocessing dari raw CSV ke data siap latih.

    Args:
        input_filepath: Path ke file CSV raw.
        output_dir: Folder tujuan output.

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("=" * 50)
    print("  Automate Preprocessing - Social Media & Mental Health")
    print("=" * 50)

    # Pastikan output_dir sudah ada sebelum encode_categorical dan split_and_scale
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_filepath)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df, encoders_dict = encode_categorical(df, output_dir)
    df = handle_outliers(df)

    X_train, X_test, y_train, y_test = split_and_scale(df, output_dir)
    save_preprocessed(X_train, X_test, y_train, y_test, output_dir)

    # Simpan metadata preprocessing
    metadata = {
        "dataset": "social_media_mental_health",
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "target_column": "PHQ_9_Severity",
        "encoding_mapping": encoders_dict,
        "numeric_columns": ['Age', 'Daily_Screen_Time_Hours', 'Sleep_Duration_Hours']
    }
    with open(os.path.join(output_dir, 'preprocessing_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n[DONE] Preprocessing selesai. Data siap dilatih.")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automate preprocessing Social Media & Mental Health dataset.')
    parser.add_argument('--input',  type=str, default='social_media_mental_health.csv',
                        help='Path ke file CSV raw')
    parser.add_argument('--output', type=str, default='social_media_mental_health_preprocessing',
                        help='Folder output dataset preprocessing')
    args = parser.parse_args()

    preprocess_pipeline(args.input, args.output)
