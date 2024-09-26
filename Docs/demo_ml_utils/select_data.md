### `select_data.py` の機能まとめ

#### 1. 概要
`select_data.py` は、Streamlit を使用してデータセットを選択し、選択されたデータセットを読み込むための関数を提供するモジュールです。このモジュールを使用することで、ユーザーは利用可能なデータセットから選択し、データをデータフレームとしてロードできます。

#### 2. インポートしているライブラリ
- **`os`**
  - ファイルパスの操作に使用
- **`pandas`**
  - `pd`: データフレーム操作用
- **`streamlit`**
  - `st`: StreamlitのUIコンポーネントを提供

#### 3. 関数の説明

##### 3.1. `selected_data(data_names)`
- **概要**: Streamlitのselectboxを使用してデータセット名を選択します。
- **引数**: `data_names` - データ名のリスト
- **戻り値**: 選択されたデータ名（文字列）

```python
def selected_data(data_names):
    """Streamlitのselectboxを使ってデータセット名を選択する

    Args:
        data_names (list[str]): データ名一覧 

    Returns:
        str: データ名
    """
    selected_data_name = st.selectbox('データセットを選択してください', options=data_names)
    return selected_data_name
```

##### 3.2. `load_data(selected_data_name)`
- **概要**: 指定されたデータセット名に基づいてデータを読み込み、データフレームを返します。
- **引数**: `selected_data_name` - データ名
- **戻り値**: 読み込んだデータフレーム（`pd.DataFrame`）

```python
def load_data(selected_data_name):
    """指定されたデータセット名に基づいてデータを読み込み、データフレームを返す

    Args:
        selected_data_name (str): データ名

    Returns:
        pd.DataFrame: データ
    """
    if selected_data_name:
        data_path = os.path.join("./jp_datas", f"{selected_data_name}.csv")
        data = pd.read_csv(data_path, index_col=0)
        return data
    else:
        return None
```

### まとめ
`select_data.py` は、Streamlit を使用してデータセットを選択し、選択されたデータセットを読み込むための関数を提供するモジュールです。このモジュールを使用することで、ユーザーは利用可能なデータセットから選択し、データをデータフレームとしてロードできます。