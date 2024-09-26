### `manager.py` の機能まとめ

#### 1. 概要
`manager.py` は、データの管理と機械学習モデルの管理を行うためのクラス `DataManager` と `ML_Manager` を提供します。これにより、データの前処理、可視化、機械学習モデルの選択、ハイパーパラメータの最適化、モデルの評価と予測が容易になります。

#### 2. インポートしているライブラリ
- **内部モジュール**
  - `select_data`: データの選択と読み込み
  - `plot`: データの可視化
  - `preprocessing`: データの前処理
  - `ml_utils`: 機械学習モデルのリストとハイパーパラメータ

- **外部ライブラリ**
  - `streamlit`: ユーザーインターフェースの構築
  - `optuna`: ハイパーパラメータの最適化
  - `sklearn`: 機械学習モデルと評価指標
  - `numpy`: 数値計算
  - `re`: 正規表現

#### 3. クラスの説明

##### 3.1. `DataManager`
データの選択、読み込み、前処理、可視化を管理するクラスです。

###### プロパティ
- **`data`**: 現在のデータ
- **`data_name`**: 選択されたデータセットの名前
- **`is_selected`**: データが選択されたかどうか
- **`is_loaded`**: データが読み込まれたかどうか
- **`target_name`**: 目的変数の名前
- **`dataset_descriptions`**: データセットの説明
- **`dataset_types`**: データセットの種類（表形式/画像形式）

###### メソッド
- **`select_data()`**: データセットの選択
- **`load_data()`**: データセットの読み込み
- **`plot_data(show=True, box=False)`**: データのプロット
- **`detect_outlier()`**: 外れ値の検出
- **`delete_outlier(columns=None)`**: 外れ値の削除
- **`detect_nan(show=True)`**: 欠損値の検出
- **`plot_nan(show=True)`**: 欠損値のプロット
- **`delete_nan(columns=None)`**: 欠損値の削除
- **`transform_log(columns=None)`**: 対数変換
- **`transform_std(columns=None)`**: 標準化
- **`classify_dtypes()`**: データ型の分類
- **`transform_label(columns=None)`**: ラベルエンコーディング
- **`transform_onehot(columns=None)`**: ワンホットエンコーディング
- **`detect_non_gauss()`**: 非正規分布の検出

##### 3.2. `ML_Manager`
機械学習モデルの選択、ハイパーパラメータの最適化、モデルの評価と予測を管理するクラスです。

###### プロパティ
- **`model`**: 現在のモデル
- **`model_name`**: モデルの名前
- **`model_type`**: モデルのタイプ（分類/回帰）
- **`model_hyperparameters`**: モデルのハイパーパラメータ
- **`random_state`**: ランダムステート
- **`n_trials`**: Optunaの試行回数
- **`model_descriptions`**: モデルの説明

###### メソッド
- **`set_data(data, target_name)`**: データのセット
- **`info_model()`**: モデルの情報
- **`split_data(train_siz, test_siz)`**: データの分割
- **`select_model()`**: モデルの選択
- **`objective(trial)`**: Optunaの目的関数
- **`optimize(n_trials=None)`**: ハイパーパラメータの最適化
- **`calc_fit_index(y_true, y_pred)`**: 評価指標の計算
- **`calc_score(y_true, y_pred)`**: Optuna用の評価指標の計算
- **`train()`**: モデルの学習
- **`predict(input_data, proba=False)`**: 予測
- **`get_trials_dataframe()`**: Optunaの試行結果をデータフレームとして取得

#### 4. `DataManager` クラスの詳細

##### 4.1. データの選択と読み込み
- **`select_data()`**: データセットを選択し、選択状態を更新します。
- **`load_data()`**: 選択されたデータセットを読み込み、データを更新します。

##### 4.2. データの可視化
- **`plot_data(show=True, box=False)`**: 表形式または画像形式でデータをプロットします。
- **`plot_nan(show=True)`**: 欠損値を可視化します。

##### 4.3. データの前処理
- **`detect_outlier()`**: 外れ値を検出し、含まれる列を返します。
- **`delete_outlier(columns=None)`**: 指定された列の外れ値を削除します。
- **`detect_nan(show=True)`**: 欠損値を検出し、含まれる列を返します。
- **`delete_nan(columns=None)`**: 指定された列の欠損値を削除します。
- **`transform_log(columns=None)`**: 指定された列を対数変換します。
- **`transform_std(columns=None)`**: 指定された列を標準化します。
- **`classify_dtypes()`**: データ型を分類します。
- **`transform_label(columns=None)`**: 指定された列をラベルエンコーディングします。
- **`transform_onehot(columns=None)`**: 指定された列をワンホットエンコーディングします。
- **`detect_non_gauss()`**: 非正規分布の列を検出します。

#### 5. `ML_Manager` クラスの詳細

##### 5.1. データのセットアップ
- **`set_data(data, target_name)`**: データとターゲット変数をセットします。
- **`split_data(train_siz, test_siz)`**: データをトレーニングデータとテストデータに分割します。

##### 5.2. モデルの選択と情報表示
- **`select_model()`**: モデルとそのハイパーパラメータを選択します。
- **`info_model()`**: モデルの情報を表示します。

##### 5.3. ハイパーパラメータの最適化
- **`objective(trial)`**: Optunaの目的関数を定義します。
- **`optimize(n_trials=None)`**: ハイパーパラメータを最適化します。

##### 5.4. モデルの学習と予測
- **`train()`**: 最適なハイパーパラメータでモデルを学習します。
- **`predict(input_data, proba=False)`**: 予測を行います。

##### 5.5. モデルの評価
- **`calc_fit_index(y_true, y_pred)`**: 評価指標を計算します。
- **`calc_score(y_true, y_pred)`**: Optuna用の評価指標を計算します。
- **`get_trials_dataframe()`**: Optunaの試行結果をデータフレームとして取得します。