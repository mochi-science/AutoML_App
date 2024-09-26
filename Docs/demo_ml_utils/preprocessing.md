### `preprocessing.py` の機能まとめ

#### 1. 概要
`preprocessing.py` は、データの前処理に関する関数を提供するモジュールです。このモジュールには、外れ値や欠損値の検出・削除、正規分布の検出、対数変換、標準化、データ型の分類、ラベルエンコーディング、ワンホットエンコーディングなどの機能が含まれています。

#### 2. インポートしているライブラリ
- **`scipy`**
  - `stats`: 統計関数
- **`numpy`**
  - `np`: 数値計算
- **`pandas`**
  - `pd`: データフレーム操作用

#### 3. 関数の説明

##### 3.1. `detect_outlier(df_train)`
- **概要**: IQRを用いて外れ値を検出します。
- **引数**: `df_train` - 外れ値を検出するデータフレーム
- **戻り値**: 外れ値が検出された列のリスト

```python
def detect_outlier(df_train):
    outlier_columns = []
    for column in df_train.select_dtypes(include=[np.number]).columns:
        Q1 = df_train[column].quantile(0.25)
        Q3 = df_train[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if df_train[(df_train[column] < lower_bound) | (df_train[column] > upper_bound)].any(axis=None):
            outlier_columns.append(column)
            
    return outlier_columns
```

##### 3.2. `delete_outlier(df_train, columns=None)`
- **概要**: IQRを用いて外れ値を削除します。
- **引数**: `df_train` - 外れ値を削除するデータフレーム
  - `columns` - 外れ値を削除するカラムのリスト（指定しない場合は全ての数値カラム）
- **戻り値**: 外れ値が削除されたデータフレーム

```python
def delete_outlier(df_train, columns=None):
    if columns is None:
        columns = df_train.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        Q1 = df_train[column].quantile(0.25)
        Q3 = df_train[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_train = df_train[(df_train[column] >= lower_bound) & (df_train[column] <= upper_bound)]
        
    return df_train
```

##### 3.3. `detect_nan(df_train, columns=None)`
- **概要**: データフレーム内の欠損値を検出します。
- **引数**: `df_train` - 操作対象のデータフレーム
  - `columns` - 欠損値検出を行う列のリスト（指定しない場合は全てのカラム）
- **戻り値**: 欠損値の合計数

```python
def detect_nan(df_train, columns=None):
    if columns is None:
        return df_train.isnull().sum()
    else:
        return df_train[columns].isnull().sum()
```

##### 3.4. `delete_nan(df_train, columns=None)`
- **概要**: データフレームから欠損値を削除します。
- **引数**: `df_train` - 操作対象のデータフレーム
  - `columns` - 欠損値削除を行う列のリスト（指定しない場合は全てのカラム）
- **戻り値**: 欠損値が削除されたデータフレーム

```python
def delete_nan(df_train, columns=None):
    if columns is None:
        df_train = df_train.dropna()
    else:
        df_train = df_train.dropna(subset=columns)
    return df_train
```

##### 3.5. `detect_non_gauss(df_train)`
- **概要**: 正規分布でないカラムを検出します。
- **引数**: `df_train` - 検出対象のデータフレーム
- **戻り値**: 正規分布でないカラムのリスト

```python
def detect_non_gauss(df_train):
    non_gauss_columns = []
    for column in df_train.columns:
        if df_train[column].dtype in ['float64', 'int64']:
            k2, p = stats.normaltest(df_train[column])
            if p < 0.05:
                non_gauss_columns.append(column)
    return non_gauss_columns
```

##### 3.6. `transform_log(df_train, columns=None)`
- **概要**: 対数変換を行います。
- **引数**: `df_train` - 変換対象のデータフレーム
  - `columns` - 変換するカラムのリスト（指定しない場合は全てのカラム）
- **戻り値**: 変換済みのデータフレーム

```python
def transform_log(df_train, columns=None):
    if columns is None:
        df_train = np.log(df_train + 1)
    else:
        df_train[columns] = np.log(df_train[columns] + 1)
    return df_train
```

##### 3.7. `transform_std(df_train, columns=None)`
- **概要**: 標準化を行います。
- **引数**: `df_train` - 標準化対象のデータフレーム
  - `columns` - 標準化するカラムのリスト（指定しない場合は全てのカラム）
- **戻り値**: 標準化済みのデータフレーム

```python
def transform_std(df_train, columns=None):
    if columns is None:
        df_train = (df_train - df_train.mean()) / df_train.std()
    else:
        df_train[columns] = (df_train[columns] - df_train[columns].mean()) / df_train[columns].std()
    return df_train
```

##### 3.8. `classify_dtypes(df_train)`
- **概要**: データフレームのデータ型を分類します。
- **引数**: `df_train` - 操作対象のデータフレーム
- **戻り値**: データ型ごとのカラムをリストで返す辞書

```python
def classify_dtypes(df_train):
    dtypes_dict = {}
    for dtype in df_train.dtypes.unique():
        dtypes_dict[dtype] = df_train.columns[df_train.dtypes == dtype].tolist()
    return dtypes_dict
```

##### 3.9. `transform_label(data, cols, label_encoder)`
- **概要**: ラベルエンコーダーを適用します。
- **引数**: `data` - 操作対象のデータフレーム
  - `cols` - ラベルエンコーダーを適用するカラムのリスト
  - `label_encoder` - ラベルエンコーダー
- **戻り値**: ラベルエンコーダーを適用したデータフレーム

```python
def transform_label(data, cols, label_encoder):
    for col in cols:
        data[col] = label_encoder.fit_transform(data[col])
    return data
```

##### 3.10. `transform_onehot(data, cols, onehot_encoder)`
- **概要**: ワンホットエンコーダーを適用します。
- **引数**: `data` - 操作対象のデータフレーム
  - `cols` - ワンホットエンコーダーを適用するカラムのリスト
  - `onehot_encoder` - ワンホットエンコーダー
- **戻り値**: ワンホットエンコーダーを適用したデータフレーム

```python
def transform_onehot(data, cols, onehot_encoder):
    for col in cols:
        data[col] = onehot_encoder.fit_transform(data[col])
    return data
```

### まとめ
`preprocessing.py` は、データの前処理に関する関数を提供するモジュールです。このモジュールを使用することで、外れ値や欠損値の検出・削除、正規分布の検出、対数変換、標準化、データ型の分類、ラベルエンコーディング、ワンホットエンコーディングなどの前処理を簡単に行うことができます。