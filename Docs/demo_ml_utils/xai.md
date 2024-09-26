### `xai.py` の機能まとめ

#### 1. 概要
`xai.py` は、機械学習モデルの説明可能性を提供するためのクラス `XAI_Manager` を提供するモジュールです。このモジュールは、SHAP（SHapley Additive exPlanations）値を計算し、特徴量の重要度や依存関係を可視化するための機能を提供します。

#### 2. インポートしているライブラリ
- **`shap`**
  - SHAP値の計算と可視化に使用
- **`streamlit`**
  - `st`: StreamlitのUIコンポーネントを提供
- **`matplotlib.pyplot`**
  - `plt`: プロット用

#### 3. クラスの説明

##### 3.1. `XAI_Manager`
SHAP値の計算と可視化を管理するクラスです。

###### プロパティ
- **`model`**: 学習済みのモデル
- **`model_type`**: モデルのタイプ（分類/回帰）
- **`X`**: モデルの予測に使用される特徴量のデータセット
- **`shap_values`**: SHAP値
- **`expected_value`**: モデルの期待値

###### メソッド
- **`__init__(self)`**
  - クラスの初期化

```python
class XAI_Manager:
    def __init__(self):
        """
        XAI_Managerクラスの初期化。
        
        Args:
            model: 学習済みのモデル。
            X: モデルの予測に使用される特徴量のデータセット。
        """
        self.model = None
        self.X = None
        self.shap_values = None
```

- **`set_model(self, model, model_type)`**
  - モデルとそのタイプを設定

```python
def set_model(self, model, model_type):
    self.model = model
    self.model_type = model_type
```

- **`set_data(self, X)`**
  - 特徴量データを設定

```python
def set_data(self, X):
    self.X = X
```

- **`calc_shap_values(self)`**
  - SHAP値を計算し、進行状況をStreamlitのプログレスバーで表示

```python
def calc_shap_values(self):
    with st.spinner('SHAP値を計算中...'):
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(self.X)
        self.expected_value = explainer.expected_value
    return self.shap_values
```

- **`plot_bar(self)`**
  - SHAP値に基づいたサマリープロットをStreamlit上で表示

```python
def plot_bar(self):
    fig, ax = plt.subplots()
    shap.summary_plot(self.shap_values, self.X, plot_type="bar", show=False)
    st.pyplot(fig)
    return fig
```

- **`plot_dot(self, dot_index=0)`**
  - 特徴量の重要度を示すSHAP値に基づいたプロットをStreamlit上で表示

```python
def plot_dot(self, dot_index=0):
    fig, ax = plt.subplots()
    self.dot_index = dot_index
    if self.model_type == "回帰":
        shap.summary_plot(self.shap_values, self.X, plot_type="dot", show=False)
    else:
        shap.summary_plot(self.shap_values[dot_index], self.X, plot_type="dot", show=False)
    st.pyplot(fig)
    return fig
```

- **`selected_feature(self)`**
  - Streamlit上で依存度プロットに使用する特徴量を選択

```python
def selected_feature(self):
    self.feature_base = st.selectbox("主要な特徴量の選択：依存度プロットの主軸となる特徴量を選択してください。", self.X.columns.tolist())
    self.feature_color = st.selectbox("色分けに使用する特徴量の選択", self.X.columns.tolist())
    st.write("※各データポイントの色を決定するための特徴量を選択してください。この特徴量により、主要な特徴量の影響を異なる角度から分析できます。")
    return self.feature_base, self.feature_color
```

- **`plot_dependence(self, dot_index=0)`**
  - 特定の特徴量に対する依存度プロットをStreamlit上で表示

```python
def plot_dependence(self, dot_index=0):
    fig, ax = plt.subplots()
    if self.model_type == "回帰":
        shap_values_class = self.shap_values
    else:
        shap_values_class = self.shap_values[dot_index]
    shap.dependence_plot(self.feature_base, shap_values_class, self.X, ax=ax,
                         interaction_index=self.feature_color, show=False)
    st.pyplot(fig)
```

### まとめ
`xai.py` は、機械学習モデルの説明可能性を提供するためのクラス `XAI_Manager` を提供するモジュールです。このモジュールを使用することで、SHAP値を計算し、特徴量の重要度や依存関係を可視化することができます。