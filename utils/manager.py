from .select_data import load_data, selected_data
from .plot import plot_table_data, plot_img_data, plot_nan, plot_boxplot
from .preprocessin import detect_outlier, delete_outlier, detect_nan, delete_nan, \
                            detect_non_gauss, transform_log, transform_std, \
                            transform_label, transform_onehot, classify_dtypes
from .ml_utils import classification_model_list, regression_model_list, \
                    classification_hyperparameters, regression_hyperparameters
from .ml_utils import model_description
import streamlit as st
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error,\
    log_loss
from sklearn.model_selection import train_test_split
import numpy as np
import re
import time


class DataManager:
    """データ管理を行うクラス
    """
    def __init__(self):
        # データ関係
        self.data = None        # データセット
        self.data_name = None   # データセット名
        
        # データロード関係
        self.is_selected = False        # データが選択されたかどうか
        self.is_loaded = False          # データが読み込まれたかどうか
        self.target_name = "目的変数"    # 目的変数の列名
        
        # データセットの説明
        self.dataset_descriptions = {
            'アヤメデータ': 'アヤメデータセットは、アヤメ花の3種類（ヒオウギアヤメ、ブルーフラッグ、バージニカ）の各50サンプルについて、がくと花弁の長さと幅を含んでいます。',
            '数学の成績データ': '数学の成績データセットは、ポルトガルの2つの学校の学生の数学の成績に関する情報を含んでいます。',
            'ワインデータ': 'ワインデータセットは、イタリアの同じ地域から来た3つの異なる栽培ワインの化学分析結果を含んでいます。各成分の量に基づいてワインの種類を推測することが目的です。',
            '乳がんデータ': '乳がんデータセットは、乳がんの診断結果（良性/悪性）と、細胞核の特徴を示す数値データを含んでいます。',
            '糖尿病データ': '糖尿病データセットは、Pimaインディアンの女性の医療記録から収集された特徴と、5年以内に糖尿病の診断を受けたかどうかの情報を含んでいます。',
            # '赤ワインの品質データ': '赤ワインの品質データセットは、ポルトガルのヴィーニョ・ヴェルデ地域で生産された赤ワインの化学分析結果を含んでいます。',
            # '白ワインの品質データ': '白ワインの品質データセットは、ポルトガルのヴィーニョ・ヴェルデ地域で生産された白ワインの化学分析結果を含んでいます。',
            # '手書き数字データ': '手書き数字データセットは、0から9までの手書き数字の8x8ピクセルの画像データを含んでいます。各画像は、数字を識別するために使用できる64の特徴量を持っています。'
        }
        
        self.TYPE_TABLE = "table"   # 表形式データ
        self.TYPE_IMAGE = "image"   # 画像形式データ
        
        # データセットの種類
        self.dataset_types = {
            '乳がんデータ': self.TYPE_TABLE,
            '糖尿病データ': self.TYPE_TABLE,
            'アヤメデータ': self.TYPE_TABLE,
            'ワインデータ': self.TYPE_TABLE,
            '赤ワインの品質データ': self.TYPE_TABLE,
            '白ワインの品質データ': self.TYPE_TABLE,
            '数学の成績データ': self.TYPE_TABLE,
            # '手書き数字データ': self.TYPE_IMAGE
        }
    
    def select_data(self):
        """データセットの選択
        
        Returns:
            str: 選択されたデータセット名
        """
        # 利用可能なデータセットから選択
        self.data_name = selected_data(list(self.dataset_descriptions.keys()))
        self.is_selected = True
        return self.data_name

    def load_data(self):
        """選択されたデータセットの読み込み
        
        Returns:
            pd.DataFrame: 選択されたデータセット
        """
        if self.is_selected:
            # 選択されたデータセットを読み込む
            self.data = load_data(self.data_name)
            self.data_type = self.dataset_types[self.data_name]
            self.is_loaded = True
            return self.data
        else:
            # データが選択されていない場合はNoneを返す
            return None

    def plot_data(self, show=True, box=False):
        """データのプロット（表形式または画像形式）
        
        Args:
            show (bool): プロットを表示するかどうか
            box (bool): ボックスプロットを表示するかどうか

        Returns:
            altair.vegalite.v4.api.Chart: プロットした図
        """
        if self.data_type == self.TYPE_TABLE:
            # 表形式データのプロット
            if box:
                fig = plot_boxplot(self.data)
            else:
                fig = plot_table_data(self.data)
        elif self.data_type == self.TYPE_IMAGE:
            # 画像形式データのプロット
            fig = plot_img_data(self.data)

        if show:
            st.altair_chart(fig)
        return fig

    def detect_outlier(self):
        """外れ値の検出
        
        Returns:
            list: 外れ値が含まれる列名のリスト
        """
        outlier_columns = detect_outlier(self.data)
        # 検出された外れ値が含まれる列のみをプロット
        if len(outlier_columns) > 0:
            fig = self.plot_data(show=True, box=True)
        return outlier_columns

    def delete_outlier(self, columns=None):
        """外れ値の削除
        
        Args:
            columns (list): 外れ値を削除する列名のリスト

        Returns:
            pd.DataFrame: 外れ値を削除したデータ
        """
        # 外れ値を削除したデータで現在のデータを更新
        self.data = delete_outlier(self.data, columns=columns)
        return self.data
    
    def detect_nan(self, show=True):
        """欠損値の検出
        
        Args:
            show (bool): プロットを表示するかどうか

        Returns:
            list: 欠損値が含まれる列名のリスト
        """
        nan_columns = detect_nan(self.data)
        # 検出された欠損値が含まれる列のみをプロット
        if len(nan_columns) > 0:
            fig = self.plot_nan(show=show)
        return nan_columns
    
    def plot_nan(self, show=True):
        """欠損値のプロット
        
        Args:
            show (bool): プロットを表示するかどうか

        Returns:
            altair.vegalite.v4.api.Chart: プロットした図
        """
        fig = plot_nan(self.data, show=show)
        return fig

    def delete_nan(self, columns=None):
        """欠損値の削除
        
        Args:
            columns (list): 欠損値を削除する列名のリスト
        
        Returns:
            pd.DataFrame: 欠損値を削除したデータ
        """
        # 欠損値を削除したデータで現在のデータを更新
        self.data = delete_nan(self.data, columns=columns)
        return self.data

    def transform_log(self, columns=None):
        """対数変換
        
        Args:
            columns (list): 対数変換する列名のリスト

        Returns:
            pd.DataFrame: 対数変換したデータ
        """
        # 対数変換したデータで現在のデータを更新
        self.data = transform_log(self.data, columns=columns)
        return self.data

    def transform_std(self, columns=None):
        """標準化
        
        Args:
            columns (list): 標準化する列名のリスト

        Returns:
            pd.DataFrame: 標準化したデータ
        """
        # 標準化したデータで現在のデータを更新
        target_data = self.data[self.target_name]  # Keep the target variable unchanged
        self.data = transform_std(self.data.drop(self.target_name, axis=1), columns=columns)
        self.data[self.target_name] = target_data
        return self.data

    def classify_dtypes(self):
        """データ型の分類
        
        Args:
            data (pd.DataFrame): データセット

        Returns:
            dict: データ型ごとの列名のリスト
        """
        dtypes_dict = classify_dtypes(self.data)
        return dtypes_dict

    def transform_label(self, columns=None):
        """ラベルエンコーディング
        
        Args:
            columns (list): ラベルエンコーディングする列名のリスト

        Returns:
            pd.DataFrame: ラベルエンコーディングしたデータ
        """
        # ラベルエンコーディングしたデータで現在のデータを更新
        self.data = transform_label(self.data, columns=columns)
        return self.data

    def transform_onehot(self, columns=None):
        """ワンホットエンコーディング
        
        Args:
            columns (list): ワンホットエンコーディングする列名のリスト

        Returns:
            pd.DataFrame: ワンホットエンコーディングしたデータ
        """
        # ワンホットエンコーディングしたデータで現在のデータを更新
        self.data = transform_onehot(self.data, columns=columns)
        return self.data
    
    def detect_non_gauss(self):
        """正規分布でない列の検出
        
        Returns:
            list: 正規分布でない列名のリスト
        """
        # non_gauss_columns = detect_non_gauss(self.data)
        non_gauss_columns = "処理未実装"
        return non_gauss_columns


class ML_Manager:
    """機械学習モデルの管理を行うクラス
    """
    def __init__(self, n_trials=10):
        self.model = None                   # モデル
        self.model_name = None              # モデル名
        self.model_type = None              # モデルのタイプ（分類 or 回帰）
        self.model_hyperparameters = None   # モデルのハイパーパラメータを格納する変数
        
        self.random_state = 42              # 乱数シード
        
        self.n_trials = n_trials            # Optunaの試行回数
        
        # モデルの説明
        self.model_descriptions = model_description
        
    
    def set_data(self, data, target_name):
        """データのセット
        
        Args:
            data (pd.DataFrame): データセット
            target_name (str): 目的変数
        """
        self.data = data                    # データセット
        self.target_name = target_name      # 目的変数

    def info_model(self):
        """モデルの情報を表示
        """
        st.info(self.model_descriptions[self.model_name])
        
    def split_data(self, test_siz):
        """データの分割

        Args:
            test_siz (float): テストデータの割合
        """
        # データの分割
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.data.drop(self.target_name, axis=1),      # 目的変数を除く
                             self.data[self.target_name],                   # 目的変数
                             test_size=test_siz, random_state=self.random_state)
    
    def select_model(self):
        """モデルの選択
        
        Returns:
            str: 選択されたモデル名
        """
        # 全モデルを辞書型で展開
        all_model_list = {**classification_model_list, **regression_model_list}
        cols = st.columns(2)
        
        # モデルの選択（StreamlitのUIを使用）
        with cols[0]:
            self.model_name = st.selectbox("モデルの選択", list(all_model_list.keys()))
        with cols[1]:
            self.model_type = st.selectbox("モデルの選択", ["分類", "回帰"])
        
        # 選択されたモデル名をセットする
        if self.model_type == "分類":
            self.model = classification_model_list[self.model_name]
            self.model_hyperparameters = classification_hyperparameters[self.model_name]
        elif self.model_type == "回帰":
            self.model = regression_model_list[self.model_name]
            self.model_hyperparameters = regression_hyperparameters[self.model_name]
        else:
            raise ValueError("モデルのタイプは回帰か分類かにしてください")
        
    def objective(self, trial):
        """Optunaの目的関数
        
        Returns:
            float: 評価指標の値（スコアは使用するモデルによって変わる）
        """
        # ハイパーパラメータの提案
        params = {}
        # 辞書型として定義したハイパーパラメータをOptunaの提案関数を使ってサンプリング
        for param_name, param_info in self.model_hyperparameters.items():
            param_type = param_info[0]
            if param_type == "int":
                params[param_name] = trial.suggest_int(param_name, param_info[1], param_info[2])
            elif param_type == "float":
                if param_info[3]:  # logスケールの場合
                    params[param_name] = trial.suggest_loguniform(param_name, param_info[1], param_info[2])
                else:
                    params[param_name] = trial.suggest_uniform(param_name, param_info[1], param_info[2])
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_info[1])
            elif param_type == "fixed":
                params[param_name] = param_info[1]

        # モデルの初期化と学習
        model = self.model(**params)
        
        # 学習と評価
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # 進捗バーの更新（StreamlitのUIを使用）
        self.optimize_bar.progress(trial.number / self.n_trials, text=f"Optunaでの最適化中...({trial.number}/{self.n_trials})")
        return self.calc_score(self.y_test, y_pred)  # 最適化の目的はスコアの最大化
        
    def optimize(self, n_trials=None):
        """ハイパーパラメータの最適化
        
        Args:
            n_trials (int): 試行回数
        
        Returns:
            dict: 最適化されたハイパーパラメータ
        """
        start = time.time()
        
        # 試行回数の設定
        if n_trials is not None:
            self.n_trials = n_trials
        
        # Streamlitの進捗バーを初期化
        self.optimize_bar = st.progress(0, text=f"Optunaでの最適化中...(0/{self.n_trials})")

        # Optunaでの最適化（分類にはAccracy、回帰にはR2を使っているのでmaximizeのみでOK）
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=self.n_trials)

        # 最適化にかかった時間を表示
        st.write(f"Optunaでの最適化にかかった時間: {time.time() - start:.2f}秒")

        # 進捗バーの更新 100/100（StreamlitのUIを使用）
        self.optimize_bar.progress(100, text=f"Optunaでの最適化中...({self.n_trials}/{self.n_trials})")
        
        # 最適化されたハイパーパラメータを取得
        best_params = self.study.best_params
        self.best_params = best_params
        print("Best parameters:", best_params)
        
    def calc_fit_index(self, y_true, y_pred):
        """評価指標の計算
        
        Args:
            y_true (np.ndarray): 正解ラベル
            y_pred (np.ndarray): 予測ラベル

        Returns:
            dict: 評価指標の辞書
        """
        # モデルタイプに応じて評価指標を計算
        if self.model_type == "分類":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                # "precision": precision_score(y_true, y_pred),
                # "recall": recall_score(y_true, y_pred),
                # "f1": f1_score(y_true, y_pred),
                # "roc_auc": roc_auc_score(y_true, y_pred, multi_class="ovr")
                # "logloss": log_loss(y_true, y_pred)
            }
        elif self.model_type == "回帰":
            return {
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred)
            }
        else:
            raise ValueError("モデルのタイプは回帰か分類かにしてください")
    
    def calc_score(self, y_true, y_pred):
        """Optuna用の評価指標の計算（どちらも最大化できる指標にする）
        
        Args:
            y_true (np.ndarray): 正解ラベル
            y_pred (np.ndarray): 予測ラベル

        Returns:
            float: 評価指標の値
        """
        # モデルタイプに応じて評価指標を計算
        if self.model_type == "分類":
            # Accuracyを返す
            return self.calc_fit_index(y_true, y_pred)["accuracy"]
        elif self.model_type == "回帰":
            # R2を返す
            return self.calc_fit_index(y_true, y_pred)["r2"]
        else:
            raise ValueError("モデルのタイプは回帰か分類かにしてください")

    def train(self):
        """モデルの学習
        """
        # 最適化していない場合はエラーを出す
        if self.best_params is None:
            raise ValueError("ハイパーパラメータの最適化を先に行ってください")

        # 最適化されたハイパーパラメータでモデルを初期化して学習
        self.best_model = self.model(**self.best_params)
        self.best_model.fit(self.X_train, self.y_train)
    
    def predict(self, input_data, proba=False):
        """予測

        Args:
            input_data (np.ndarray): 予測に使用するデータ
            proba (bool): 確率を返すかどうか

        Returns:
            np.ndarray: 予測ラベルまたは確率
        """
        # 確率を返すかどうかで分岐
        if proba:
            return self.best_model.predict_proba(input_data)
        else:
            return self.best_model.predict(input_data)
        
    def get_trials_dataframe(self):
        """Optunaの試行結果をDataFrameで取得

        Returns:
            pd.DataFrame: Optunaの試行結果
        """
        # Optunaの試行結果をDataFrameで取得
        opt_result = self.study.trials_dataframe()
        # 不要な列を削除
        opt_result = opt_result.drop(["number", "datetime_start",
                                        "datetime_complete"], axis=1)
        
        # ステータスを見やすく
        opt_result['state'] = opt_result['state'].map(lambda x: '✅' if x == 'COMPLETE' else '❌')
        
        # duration列を変換する（秒単位から分または時間単位に）
        def convert_duration(timedelta):
            seconds = timedelta.total_seconds()  # Timedeltaから秒数を取得
            if isinstance(seconds, str):
                return seconds
            elif seconds >= 3600:
                return f"{seconds / 3600:.2f}時間"
            elif seconds >= 60:
                return f"{seconds / 60:.2f}分"
            else:
                return f"{seconds}秒"
        opt_result['duration'] = opt_result['duration'].apply(convert_duration)
        
        # 順番を入れ替える
        columns_order = ['state', 'value', 'duration'] + [col for col in opt_result.columns if col.startswith('params_')]
        opt_result = opt_result[columns_order]
        
        # 日本語化
        column_name_map = {
            "value": "評価値",
            "duration": "所要時間",
            "state": "実行ステータス",
        }

        # パラメータのカラム名を特定するための正規表現パターン
        params_pattern = re.compile(r'^params_')

        # パラメータ以外の列名を日本語に変換
        new_columns = {}
        for col in opt_result.columns:
            if not params_pattern.match(col):
                # パラメータ以外の列名を変更
                new_columns[col] = column_name_map.get(col, col)
            else:
                # パラメータの列名はそのままにするか、適宜変更する
                # 例: 'params_learning_rate' -> '学習率'
                # new_columns[col] = '日本語の列名'  # 必要に応じて
                pass

        # 列名の変更を適用
        opt_result.rename(columns=new_columns, inplace=True)
        
        return opt_result
        

# TODO
# 考慮すべき点:
# コメントアウトされた評価指標: 今回は簡潔さのために一部の評価指標をコメントアウトしていますが、将来的には
# モデルの性能を多面的に評価するためにこれらを活用することを検討してください。

# set_dataメソッドの汎用性: 現在の実装では、target_nameを使ってターゲット変数を指定していますが、
# データセットが多様である場合、さらに柔軟なデータセットの処理方法を考慮する必要があるかもしれません。

# モデルの保存と再利用: 最適化されたモデルのパラメータやモデル自体を保存し、後で再利用できる機能を
# 追加することで、モデルの実用性が高まります。

# エラーハンドリング: 現在の実装では基本的なエラーハンドリングが含まれていますが、データやモデルに関する
# 問題が発生した際により詳細な情報を提供することで、ユーザーが問題を解決しやすくなります。

# 全体として、ML_Managerクラスは機械学習モデルの管理に関して堅固な基盤を提供しており、小さな改善を
# 加えることで、さらに実用的なツールへと発展させることができるでしょう。

# エラーハンドリングはStreamlit用のエラーハンドリング関数を定義しておく（もしくはログ、UI上どちらにも出すようにする）