from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb

classification_model_list = {
    "LightGBM": lgb.LGBMClassifier,
    "XGBoost": xgb.XGBClassifier,
    "ランダムフォレスト": RandomForestClassifier,
    "決定木": DecisionTreeClassifier,
    "MLP": MLPClassifier,
    "線形モデル": LogisticRegression,
}

regression_model_list = {
    "LightGBM": lgb.LGBMRegressor,
    "XGBoost": xgb.XGBRegressor,
    "ランダムフォレスト": RandomForestRegressor,
    "決定木": DecisionTreeRegressor,
    "MLP": MLPRegressor,
    "線形モデル": LinearRegression,
}

model_description = {
    "LightGBM": 'LightGBMはMicrosoftの開発した、高速で高性能な勾配ブースティングフレームワークです。様々なビジネスやコンペティションで広く使用されています。',
    "XGBoost": "XGBoostは、勾配ブースティングを用いて分類や回帰を行う機械学習手法です。",
    "ランダムフォレスト": 'ランダムフォレストは、アンサンブル学習法の一種であり、複数の決定木を組み合わせて予測を行います。',
    "決定木": "決定木は、データを分割していくことで分類や回帰を行う手法です。",
    "MLP": "MLPは、多層のニューラルネットワークを使って分類や回帰を行う手法です。",
    "線形モデル": "線形モデルは、入力変数と出力変数の間の線形な関係を仮定してモデル化する手法です。",
}

classification_hyperparameters = {
    
    "LightGBM": {
        "num_leaves": ["int", 2, 64, False],  # 葉の数
        "learning_rate": ["float", 0.001, 0.2, False],  # 学習率
        "lambda_l1": ["float", 1e-8, 10.0, True],
        "lambda_l2": ["float", 1e-8, 10.0, True],
        "feature_fraction": ["float", 0.4, 1, False],
        "bagging_fraction": ["float", 0.4, 1, False],
        "min_child_samples": ["int", 5, 100, False]
    },
    "ランダムフォレスト": {
        "max_features": ["float", 0.01, 1.0, True],
        "max_leaf_nodes": ["int", 1, 1000, False],
        "min_samples_split": ["int", 2, 5, False],
        "min_samples_leaf": ["int", 1, 10, False],
        "max_depth": ["int", 1, 32, False]  # 木の深さ
    },
    "XGBoost": {
        "max_depth": ["int", 1, 9, False],
        "learning_rate": ["float", 1e-3, 1.0, True],
        "n_estimators": ["int", 100, 1000, False],
        "min_child_weight": ["int", 1, 10, False],
        "subsample": ["float", 0.6, 1.0, False],
        "colsample_bytree": ["float", 0.6, 1.0, False],
        "gamma": ["float", 1e-8, 1.0, True]
    },
    "決定木": {
        "max_depth": ["int", 1, 20, False],
        "min_samples_split": ["int", 2, 20, False],
        "min_samples_leaf": ["int", 1, 20, False],
        "max_features": ["categorical", ["auto", "sqrt", "log2"], False],
        "criterion": ["categorical", ["gini", "entropy"], False]
    },
    "MLP": {
        "hidden_layer_sizes": ["categorical", [(50,), (100,), (50, 50), (100, 50)]],  # 隠れ層のサイズ
        "alpha": ["float", 0.0001, 0.01, True]  # L2正則化パラメータ
    },
    "線形モデル": {
        "C": ["loguniform", 1e-5, 100, False],
        "solver": ["categorical", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], False],
        "max_iter": ["int", 100, 1000, False],
        "penalty": ["categorical", ["l1", "l2", "elasticnet", "none"], False]
    },
}

regression_hyperparameters = {
    "LightGBM": {
        "num_leaves": ["int", 2, 64, False],  # 葉の数
        "learning_rate": ["float", 0.001, 0.2, False],  # 学習率
        "lambda_l1": ["float", 1e-8, 10.0, True],
        "lambda_l2": ["float", 1e-8, 10.0, True],
        "feature_fraction": ["float", 0.4, 1, False],
        "bagging_fraction": ["float", 0.4, 1, False],
        "min_child_samples": ["int", 5, 100, False]
    },
    "ランダムフォレスト": {
        "max_features": ["float", 0.01, 1.0, True],
        "max_leaf_nodes": ["int", 1, 1000, False],
        "min_samples_split": ["int", 2, 5, False],
        "min_samples_leaf": ["int", 1, 10, False],
        "max_depth": ["int", 1, 32, False]  # 木の深さ
    },
    "XGBoost": {
        "max_depth": ["int", 1, 9, False],
        "learning_rate": ["float", 1e-3, 1.0, True],
        "n_estimators": ["int", 100, 1000, False],
        "min_child_weight": ["int", 1, 10, False],
        "subsample": ["float", 0.6, 1.0, False],
        "colsample_bytree": ["float", 0.6, 1.0, False],
        "gamma": ["float", 1e-8, 1.0, True]
    },
    "決定木": {
        "max_depth": ["int", 1, 20, False],
        "min_samples_split": ["int", 2, 20, False],
        "min_samples_leaf": ["int", 1, 20, False],
        "max_features": ["categorical", ["auto", "sqrt", "log2"], False],
    },
    "MLP": {
        "hidden_layer_sizes": ["categorical", [(50,), (100,), (50, 50), (100, 50)]],  # 隠れ層のサイズ
        "alpha": ["float", 0.0001, 0.01, True]  # L2正則化パラメータ
    },
    "線形モデル": {
        "fit_intercept": ["categorical", [True, False], False],
        "normalize": ["categorical", [True, False], False],
        "positive": ["categorical", [True, False], False]
    }   ,
}

# def input_hyperparameters(model_name, hyperparameters):

# 結果の確認セクションで出力のデータフレームに正誤と予測値を二つ入れる
# 入力用のデータと出力のデータフレームを一緒にしておく
# SHAPの可視化でdependance plotが出ない