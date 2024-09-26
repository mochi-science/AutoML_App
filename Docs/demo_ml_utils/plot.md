### `plot.py` の機能まとめ

#### 1. 概要
`plot.py` は、データの可視化に関する関数を提供するモジュールです。このモジュールには、データフレームをヒストグラムや箱ひげ図としてプロットする関数や、画像データのプロット、外れ値や欠損値を可視化する関数が含まれています。

#### 2. インポートしているライブラリ
- **`streamlit`**
  - `st`: StreamlitのUIコンポーネントを提供
- **`pandas`**
  - `pd`: データフレーム操作用
- **`seaborn`**
  - `sns`: データ可視化用
- **`numpy`**
  - `np`: 数値計算用
- **`matplotlib.pyplot`**
  - `plt`: プロット用
- **`sklearn.utils`**
  - `shuffle`: データシャッフル用
- **`sklearn.preprocessing`**
  - `MinMaxScaler`: データ標準化用
- **`altair`**
  - `alt`: インタラクティブな可視化用

#### 3. 関数の説明

##### 3.1. `plot_table_data(df)`
- **概要**: データフレームの各列をヒストグラムまたはカウントプロットとして表示する。
- **引数**: `df` - プロットするデータフレーム
- **戻り値**: 作成したAltairチャート

```python
def plot_table_data(df):
    # Altairで描画するためのチャートリストを初期化
    charts = []

    for column in df.columns:
        # 数値データの場合の処理
        if pd.api.types.is_numeric_dtype(df[column]):
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(column, bin=True),
                y='count()',
                tooltip=[column, 'count()']
            ).properties(
                title=f"{column} - ヒストグラム"
            )
            charts.append(chart)
        
        # カテゴリカルデータの場合の処理
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            chart = alt.Chart(df).mark_bar().encode(
                x=column,
                y='count()',
                tooltip=[column, 'count()']
            ).properties(
                title=f"{column} - カウントプロット"
            )
            charts.append(chart)
    
    # 全てのチャートを縦に並べる
    final_chart = alt.vconcat(*charts).configure_view(
        strokeWidth=0
    )
    
    st.altair_chart(final_chart, use_container_width=True)
    return final_chart
```

##### 3.2. `plot_boxplot(df)`
- **概要**: データフレームの各列を箱ひげ図として表示する。
- **引数**: `df` - プロットするデータフレーム
- **戻り値**: 作成したAltairチャート

```python
def plot_boxplot(df):
    mm = MinMaxScaler()
    df_normalized = pd.DataFrame(mm.fit_transform(df), columns=df.columns)
    
    # DataFrameを長い形式に変換する
    df_long = df_normalized.melt(var_name='Column', value_name='Value')

    # 数値カラムのみをフィルタリングする
    df_long_numeric = df_long[df_long['Value'].apply(lambda x: np.issubdtype(type(x), np.number))]

    # Altairで箱ひげ図を作成する
    chart = alt.Chart(df_long_numeric).mark_boxplot().encode(
        x='Column:N',  # カテゴリデータとしてカラム名をx軸に設定
        y='Value:Q',   # 量的データとして値をy軸に設定
        tooltip=['Column:N', 'Value:Q']  # ツールチップに情報を追加
    ).properties(
        title='箱ひげ図',
        height=400
    )

    return chart
```

##### 3.3. `plot_img_data(df_example)`
- **概要**: 画像データを表示する。
- **引数**: `df_example` - 画像データが含まれるデータフレーム
- **戻り値**: 作成したMatplotlibのFigureオブジェクト

```python
def plot_img_data(df_exmaple):
    # ターゲットのユニークな値を取得
    unique_targets = df_exmaple['目的変数'].unique()
    selected_target = st.selectbox('ターゲットを選択してください', options=unique_targets)

    # 選択されたターゲットに対応する行のみをフィルタリング
    filtered_df = df_exmaple[df_exmaple['目的変数'] == selected_target]

    # ランダムに10行を選択
    sampled_df = shuffle(filtered_df).head(10)

    # 画像として表示
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    axs = axs.flatten()
    for i, row in enumerate(sampled_df.itertuples(index=False)):
        # ピクセルデータを8x8の画像として変換
        image = np.array(row[:-1]).reshape(8, 8)  # 最後の列がターゲットなので、それ以外を画像データとして扱う
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')
    
    return fig
```

##### 3.4. `plot_outlier(df_train)`
- **概要**: 外れ値が含まれる列を箱ひげ図として表示する。
- **引数**: `df_train` - 外れ値を検出するデータフレーム

```python
def plot_outlier(df_train):
    # 外れ値がある列をグラフで表示
    outlier_columns = detect_outlier(df_train)
    for column in outlier_columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=st.session_state["df_train"][column])
        st.pyplot(fig)
```

##### 3.5. `plot_nan(df_train, show=True)`
- **概要**: 欠損値をヒートマップとして表示する。
- **引数**: `df_train` - 欠損値を検出するデータフレーム
- **戻り値**: 作成したMatplotlibのFigureオブジェクト

```python
def plot_nan(df_train, show=True):
    fig, ax = plt.subplots()
    # 欠損値の数を計算し、DataFrameに変換
    missing_values = df_train.isnull().sum().to_frame()

    # ヒートマップの表示
    sns.heatmap(missing_values, annot=True, fmt="d", cmap="Reds")
    if show:
        st.pyplot(fig)
        
    return fig
```

### まとめ
`plot.py` は、データフレームや画像データの可視化に関する関数を提供するモジュールです。このモジュールを使用することで、データのヒストグラム、箱ひげ図、欠損値のヒートマップ、画像データのプロットなどを簡単に行うことができます。