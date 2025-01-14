from scipy import stats
import numpy as np

import numpy as np
import pandas as pd

def detect_outlier(df_train):
    """IQRを用いて外れ値を検出

    Args:
        df_train (pd.DataFrame): 検出対象のデータ

    Returns:
        list: 外れ値が検出された列のリスト
    """
    outlier_columns = []
    for column in df_train.select_dtypes(include=[np.number]).columns:
        Q1 = df_train[column].quantile(0.25)
        Q3 = df_train[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 外れ値が存在する場合、そのカラム名をリストに追加
        if df_train[(df_train[column] < lower_bound) | (df_train[column] > upper_bound)].any(axis=None):
            outlier_columns.append(column)
            
    return outlier_columns

def delete_outlier(df_train, columns=None):
    """IQRを用いて外れ値を削除する

    Args:
        df_train (pd.DataFrame): 削除対象のデータ
        columns (list[str], optional): 外れ値を削除するカラムを指定

    Returns:
        pd.DataFrame: 外れ値が削除されたデータフレーム
    """
    df = df_train.copy()
    # 外れ値を検出するカラムが指定されていない場合、数値型のカラムすべてで検出を行う
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 四分位範囲外のデータを保持
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
    return df


def detect_nan(df_train, columns=None):
    """データフレーム内の欠損値を検出する。

    Args:
        df_train (pd.DataFrame): 操作対象のデータフレーム。
        columns (list, optional): 欠損値検出を行う列のリスト。指定しない場合はデータフレーム全体で欠損値を検出。

    Returns:
        pd.Series: 指定された列またはデータフレーム全体の各列における欠損値の合計数。
    """
    # columnsが指定されていない場合、データフレーム全体の欠損値を検出
    if columns is None:
        return df_train.isnull().sum()
    else:
        # 指定されたカラムの欠損値を検出
        return df_train[columns].isnull().sum()

def delete_nan(df_train, columns=None):
    """データフレームから欠損値を削除する。

    Args:
        df_train (pd.DataFrame): 操作対象のデータフレーム。
        columns (list, optional): 欠損値削除を行う列のリスト。指定しない場合はデータフレーム全体で欠損値を削除。

    Returns:
        pd.DataFrame: 欠損値を削除したデータフレーム。
    """
    df = df_train.copy()
    # columnsが指定されていない場合、データフレーム全体から欠損値を削除
    if columns is None:
        df = df.dropna()
    else:
        # 指定されたカラムから欠損値を削除
        df = df.dropna(subset=columns)
    return df

def detect_non_gauss(df_train):
    """正規分布でないカラムを検出する

    Args:
        df_train (pd.DataFrame): 検出対象のデータ

    Returns:
        list: 正規分布でないカラムをリストで返す
    """
    non_gauss_columns = []  # 正規分布でないカラムを保持するリスト
    for column in df_train.columns:
        # 対象カラムが数値型の場合のみ検証
        if df_train[column].dtype in ['float64', 'int64']:
            # Anderson-Darling正規性検定を実行
            k2, p = stats.normaltest(df_train[column])
            # p値が0.05未満なら正規分布ではないと判断
            if p < 0.05:
                non_gauss_columns.append(column)
    return non_gauss_columns

def transform_log(df_train, columns=None):
    """対数変換を行う

    Args:
        df_train (pd.DataFrame): 変換対象のデータ
        columns (list[str]): 変換するカラムを指定できる

    Returns:
        pd.DataFrame: 変換済みのデータ
    """
    df = df_train.copy()
    # columnsが指定されていない場合、データフレーム全体に対して対数変換を実行
    if columns is None:
        df = np.log(df + 1)  # 0値を考慮して+1
    else:
        # 指定されたカラムのみ対数変換を実行
        df[columns] = np.log(df[columns] + 1)
    return df

def transform_std(df_train, columns=None):
    """標準化を行う

    Args:
        df_train (pd.DataFrame): 標準化対象のデータ
        columns (list[str]): 標準化するカラムを指定できる

    Returns:
        pd.DataFrame: 標準化済みのデータ
    """
    df = df_train.copy()
    # columnsが指定されていない場合、データフレーム全体に対して標準化を実行
    if columns is None:
        df = (df - df.mean()) / df.std()
    else:
        # 指定されたカラムのみ標準化を実行
        df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()
    return df

def classify_dtypes(df_train):
    """データフレームのデータ型を分類する

    Args:
        df_train (pd.DataFrame): 操作対象のデータフレーム。

    Returns:
        dict: データ型ごとのカラムをリストで返す
    """
    dtypes_dict = {}  # データ型ごとのカラムリストを保持する辞書
    for dtype in df_train.dtypes.unique():
        # 各データ型に対応するカラムを分類
        dtypes_dict[dtype] = df_train.columns[df_train.dtypes == dtype].tolist()
    return dtypes_dict

def transform_label(data, cols, label_encoder):
    """ラベルエンコーダーを適用する

    Args:
        data (pd.DataFrame): 操作対象のデータ
        cols (list[str]): ラベルエンコーダーを適用するカラム
        label_encoder (LabelEncoder): ラベルエンコーダー

    Returns:
        pd.DataFrame: ラベルエンコーダーを適用したデータ
    """
    df = data.copy()
    for col in cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def transform_onehot(data, cols, onehot_encoder):
    """ワンホットエンコーダーを適用する

    Args:
        data (pd.DataFrame): 操作対象のデータ
        cols (list[str]): ワンホットエンコーダーを適用するカラム
        onehot_encoder (OneHotEncoder): ワンホットエンコーダー

    Returns:
        pd.DataFrame: ワンホットエンコーダーを適用したデータ
    """
    df = data.copy()
    for col in cols:
        df[col] = onehot_encoder.fit_transform(df[col])
    return df

def transform_to_gray(df_train, columns=None):
    """グレースケール変換を行う

    Args:
        df_train (pd.DataFrame): 変換対象のデータ
        columns (list[str]): 変換するカラムを指定できる

    Returns:
        pd.DataFrame: 変換済みのデータ
    """
    return df_train.copy()

#TODO

#正規性の検定と外れ値の検出はもう少し考える（データセットによっては検出しすぎる）
