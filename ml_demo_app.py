import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import japanize_matplotlib
import re
import altair as alt
import time
from utils.manager import DataManager, ML_Manager
from utils.xai import XAI_Manager


def mat_data_editor(data_df):
    """カテゴリデータを自由に編集するためのStreamlit用UIを提供する関数
    Args:
        data_df (pd.DataFrame): 数学データセット（名前を変換していない状態）

    Returns:
        pd.DataFrame: 編集後のデータセット
    """
    # ラベル名を日本語に変換
    data_df = categori_to_name_by_matdata(data_df)
    # データ編集用のUIを提供（データフレーム内で選択肢が使える特殊記法）
    edited_data = st.data_editor(
        data_df,
        column_config={
            "学校": st.column_config.SelectboxColumn(
                "学校",
                help="The category of school",
                width="medium",
                options=["GP", "MS"],
                required=True,
            ),
            "性別": st.column_config.SelectboxColumn(
                "性別",
                help="The gender of the student",
                width="medium",
                options=["F", "M"],
                required=True,
            ),
            "住所タイプ": st.column_config.SelectboxColumn(
                "住所タイプ",
                help="The type of address",
                width="medium",
                options=["R", "U"],
                required=True,
            ),
            "家族のサイズ": st.column_config.SelectboxColumn(
                "家族のサイズ",
                help="The size of the family",
                width="medium",
                options=["GT3", "LE3"],
                required=True,
            ),
            "両親との同居状況": st.column_config.SelectboxColumn(
                "両親との同居状況",
                help="The living situation with parents",
                width="medium",
                options=["A", "T"],
                required=True,
            ),
            "母親の仕事": st.column_config.SelectboxColumn(
                "母親の仕事",
                help="The mother's occupation",
                width="medium",
                options=["at_home", "health", "other", "services", "teacher"],
                required=True,
            ),
            "父親の仕事": st.column_config.SelectboxColumn(
                "父親の仕事",
                help="The father's occupation",
                width="medium",
                options=["at_home", "health", "other", "services", "teacher"],
                required=True,
            ),
            "学校を選んだ理由": st.column_config.SelectboxColumn(
                "学校を選んだ理由",
                help="The reason for choosing the school",
                width="medium",
                options=["course", "home", "other", "reputation"],
                required=True,
            ),
            "生徒の保護者": st.column_config.SelectboxColumn(
                "生徒の保護者",
                help="The student's guardian",
                width="medium",
                options=["father", "mother", "other"],
                required=True,
            ),
            "追加の教育支援": st.column_config.SelectboxColumn(
                "追加の教育支援",
                help="Additional educational support",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
            "家族からの学習支援": st.column_config.SelectboxColumn(
                "家族からの学習支援",
                help="Family support for learning",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
            "追加の有料授業（数学）": st.column_config.SelectboxColumn(
                "追加の有料授業（数学）",
                help="Additional paid lessons (mathematics)",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
            "学校外の活動": st.column_config.SelectboxColumn(
                "学校外の活動",
                help="Activities outside of school",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
            "幼稚園への通園経験": st.column_config.SelectboxColumn(
                "幼稚園への通園経験",
                help="Experience of attending kindergarten",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
            "高等教育への意欲": st.column_config.SelectboxColumn(
                "高等教育への意欲",
                help="Motivation for higher education",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
            "家でのインターネットのアクセス": st.column_config.SelectboxColumn(
                "家でのインターネットのアクセス",
                help="Internet access at home",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
            "恋愛関係": st.column_config.SelectboxColumn(
                "恋愛関係",
                help="Romantic relationship",
                width="medium",
                options=["no", "yes"],
                required=True,
            ),
        },
        hide_index=True,
    )
    # ラベル名をカテゴリ名に変換
    data_df = name_to_categori_by_matdata(edited_data)
    
    return edited_data

# 数学の成績データのラベル名
mat_data_label_name = {
    "学校": {
        "0": "GP",
        "1": "MS"
    },
    "性別": {
        "0": "F",
        "1": "M"
    },
    "住所タイプ": {
        "0": "R",
        "1": "U"
    },
    "家族のサイズ": {
        "0": "GT3",
        "1": "LE3"
    },
    "両親との同居状況": {
        "0": "A",
        "1": "T"
    },
    "母親の仕事": {
        "0": "at_home",
        "1": "health",
        "2": "other",
        "3": "services",
        "4": "teacher"
    },
    "父親の仕事": {
        "0": "at_home",
        "1": "health",
        "2": "other",
        "3": "services",
        "4": "teacher"
    },
    "学校を選んだ理由": {
        "0": "course",
        "1": "home",
        "2": "other",
        "3": "reputation"
    },
    "生徒の保護者": {
        "0": "father",
        "1": "mother",
        "2": "other"
    },
    "追加の教育支援": {
        "0": "no",
        "1": "yes"
    },
    "家族からの学習支援": {
        "0": "no",
        "1": "yes"
    },
    "追加の有料授業（数学）": {
        "0": "no",
        "1": "yes"
    },
    "学校外の活動": {
        "0": "no",
        "1": "yes"
    },
    "幼稚園への通園経験": {
        "0": "no",
        "1": "yes"
    },
    "高等教育への意欲": {
        "0": "no",
        "1": "yes"
    },
    "家でのインターネットのアクセス": {
        "0": "no",
        "1": "yes"
    },
    "恋愛関係": {
        "0": "no",
        "1": "yes"
    }
}

def categori_to_name_by_matdata(df):
    """数学の成績データのラベル名を日本語に変換する関数
    Args:
        df (pd.DataFrame): 数学データセット（名前を変換していない状態）

    Returns:
        pd.DataFrame: 日本語に変換したデータセット
    """
    # 順番にmat_data_label_nameのラベル名を日本語に変換
    for col in df.columns:
        if col in mat_data_label_name:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(mat_data_label_name[col])
    return df

def name_to_categori_by_matdata(df):
    """数学の成績データのラベル名をカテゴリ名に変換する関数
    Args:
        df (pd.DataFrame): 数学データセット（名前を変換していない状態）
    
    Returns:
        pd.DataFrame: カテゴリ名に変換したデータセット
    """
    # 順番にmat_data_label_nameのラベル名をカテゴリ名に変換
    for col in df.columns:
        if col in mat_data_label_name:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace({v: k for k, v in mat_data_label_name[col].items()})
    return df.astype(np.int64)

# 全メニュータブの名前
tabs_name = [
    "1.データを選択&見る",
    "2.データを加工",
    "3.機械学習(AI)",
    "4.未知データを予測",
    "5.AIの予測根拠を分析"
]

# データセットのラベル名
dataset_label_names = {
    '乳がんデータ': ['陰性', '陽性'],
    '糖尿病データ': ['糖尿病進行度'],
    'アヤメデータ': ['ヒオウギアヤメ', 'ブルーフラッグ', 'バージニカ'],
    'ワインデータ': ['ワインA', 'ワインB', 'ワインC'],
    '手書き数字データ': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    '数学の成績データ': ['成績の数字']
}

# データセットの説明
dataset_descriptions = {
    '乳がんデータ': '乳がんデータセットは、乳がんの診断結果（良性/悪性）と、細胞核の特徴を示す数値データを含んでいます。',
    '糖尿病データ': '糖尿病データセットは、Pimaインディアンの女性の医療記録から収集された特徴と、5年以内に糖尿病の診断を受けたかどうかの情報を含んでいます。',
    'アヤメデータ': 'アヤメデータセットは、アヤメ花の3種類（Setosa、Versicolour、Virginica）の各50サンプルについて、がくと花弁の長さと幅を含んでいます。',
    'ワインデータ': 'ワインデータセットは、イタリアの同じ地域から来た3つの異なる栽培ワインの化学分析結果を含んでいます。各成分の量に基づいてワインの種類を推測することが目的です。',
    '手書き数字データ': '手書き数字データセットは、0から9までの手書き数字の8x8ピクセルの画像データを含んでいます。各画像は、数字を識別するために使用できる64の特徴量を持っています。',
    '数学の成績データ': '数学の成績データセットは、ポルトガルの2つの学校の学生の数学の成績に関する情報を含んでいます。',
}

# マネージャーがない場合は作成
if "dm" not in st.session_state:
    st.session_state["dm"] = DataManager()
if "mm" not in st.session_state:
    st.session_state["mm"] = ML_Manager()
if "xm" not in st.session_state:
    st.session_state["xm"] = XAI_Manager()

# マネージャーを使いやすいようにする
dm = st.session_state["dm"]
mm = st.session_state["mm"]
xm = st.session_state["xm"]

# メモがない場合は初期化
if "memo" not in st.session_state:
    st.session_state["memo"] = ""

# タブを作成
tabs = st.tabs(tabs_name)

# アクション後に更新を入れるので，サイドバーは一番下で読み込む
def load_sidebar():
    
    with st.sidebar:
        st.warning("現在、β板のため、エラーやバグが発生する可能性があります。何かしら異常がございましたら、スタッフまでお声がけください。")
        # やる順番と、タスクを進めている実感を持たせるために、ToDoリスト（自動更新）を配置
        st.write("# 手順")
        st.write()
        
        # ステップを設定
        steps = [
            ['step1', 'Step1. データを選択する。'],
            ['step2', 'Step2. データを加工する。'],
            ['step3', 'Step3. 機械学習を行う。'],
            ['step4', 'Step4. 未知データを予測する。'],
            ['step5', 'Step5. XAIによる分析を行う。'],
        ]
        
        # ステップの状態を表示
        desc_list = []
        for step_key, step_desc in steps: 
            # ステップが完了している場合はチェックマークを表示
            icon = '✅' if st.session_state['steps_completed'][step_key] else '🔲'
            desc_list.append(f'{icon} {step_desc}')
        
        # ステップの説明を表示
        st.write(desc_list[0])
        if dm.data is not None:
            st.write(f"- 現在読み込んでいるデータは、「**{dm.data_name}**」です。")
        st.write(desc_list[1])
        st.write(desc_list[2])
        st.write(desc_list[3])
        st.write(desc_list[4])
            
        # 全てのステップが完了している場合は、おめでとうメッセージを表示
        if st.session_state["steps_completed"]["step5"]:
            st.write("# Congratulations!")
            st.write("### 全てのステップが完了しました！")
            
        # エラー発生時のメッセージ&リセットボタン
        st.success("グラフが変化しないなどの場合は、下のボタンを押してください。")
        st.button("データ更新")
        
        # メモを残す
        st.session_state["memo"] = st.text_area("メモ", value=st.session_state["memo"], placeholder="メモを残すことができます。")
        
        # バグの確認
        st.write("### 現在確認中のバグ（エラー）")
        st.write("- 一度機械学習まで行った後に違うデータを読み込んだ場合、ページをリロードして初期化してください。")

# ステップの状態を初期化
def init_steps():
    steps = {
        'step1': False,
        'step2': False,
        'step3': False,
        'step4': False,
        'step5': False,
    }
    
    return steps

def init_session_state():
    """セッションステートを初期化する関数
    """
    st.session_state["steps_completed"] = init_steps()          # ステップの状態を初期化
    st.session_state["is_push_detect_outlier_button"] = False   # 外れ値検出ボタンが押されたかどうか
    st.session_state["is_push_detect_nan_button"] = False       # 欠損値検出ボタンが押されたかどうか
    st.session_state["is_optimized"] = False                    # モデルの最適化が終わったかどうか
    st.session_state["is_calc_shap_values"] = False             # SHAP値の計算が終わったかどうか

# ステップが初期化されていない場合は初期化
if 'steps_completed' not in st.session_state:
    st.session_state['steps_completed'] = init_steps()
        
def completed_step(step_key):
    """ステップが完了したことを示す関数
    Args:
        step_key (str): ステップのキー
    """
    st.session_state['steps_completed'][step_key] = True
    

# データの選択&可視化
with tabs[0]:
    box_selecet_data = st.columns(2)
    # データ選択を行う
    with box_selecet_data[0]:
        data_name = dm.select_data()
    with box_selecet_data[1]:
        # 順番を工夫するためにボタンを押したかどうかをフラグにする
        is_loading = st.button("データを読み込む（初期化する）")
    # データの説明を表示
    st.info(dataset_descriptions[data_name])
    if is_loading:
        with st.spinner("データ読み込み中 ..."):
            # データのプロットを行う
            data = dm.load_data()
            # init_session_state()
            completed_step("step1")
            
    # データ読み込み後はデータをプロットし続ける
    if dm.data is not None:
        fig = dm.plot_data()
        
# データを加工
with tabs[1]:
    # 外れ値の処理
    if st.session_state["steps_completed"]["step1"]:
        completed_step("step2")
        with st.expander("外れ値の検出"):
            # セッションにフラグがない場合はボタンを表示して、押されたらフラグを立てる
            if "is_push_detect_outlier_button" not in st.session_state and st.button("外れ値の検出"):
                st.session_state["is_push_detect_outlier_button"] = True
            # フラグが立っている場合は外れ値を検出
            if "is_push_detect_outlier_button" in st.session_state and st.session_state["is_push_detect_outlier_button"]:
                st.session_state["outlier_columns"] = dm.detect_outlier()
                st.session_state["is_push_detect_outlier_button"] = False
            # フラグが立っていない場合はプロットを行い、再検出用のボタンを表示
            elif "outlier_columns" in st.session_state:
                st.session_state["is_push_detect_outlier_button"] = st.button("外れ値の検出をし直す")
                dm.plot_data(box=True)
            
            # 外れ値の削除を行う処理
            if "outlier_columns" in st.session_state:
                # 外れ値を削除するカラムを選択
                # st.session_state["outlier_columns"]
                delete_columns = st.multiselect("外れ値を削除するカラムを選択", st.session_state["outlier_columns"])
                # ボタンが押されたら外れ値を削除
                if st.button("外れ値の削除"):
                    dm.delete_outlier()
                    
            
        # 欠損値の処理
        with st.expander("欠損値の検出"):
            # セッションにフラグがない場合はボタンを表示して、押されたらフラグを立てる
            if "is_push_detect_nan_button" not in st.session_state and st.button("欠損値の検出"):
                st.session_state["is_push_detect_nan_button"] = True
            # フラグが立っている場合は欠損値を検出
            if "is_push_detect_nan_button" in st.session_state and st.session_state["is_push_detect_nan_button"]:
                st.session_state["nan_columns"] = dm.detect_nan()
                st.session_state["is_push_detect_nan_button"] = False
            # フラグが立っていない場合はプロットを行い、再検出用のボタンを表示
            elif "nan_columns" in st.session_state:
                st.session_state["is_push_detect_nan_button"] = st.button("欠損値の検出をし直す")
                nan_data = dm.data[dm.data.isnull().any(axis=1)]
                dm.plot_nan()
                
            # 欠損値の削除を行う処理
            if "nan_columns" in st.session_state:
                # ボタンが押されたら欠損値を削除
                if st.button("欠損値の削除"):
                    dm.delete_nan()
                    
            
        # 正規性の検出
        with st.expander("正規分布の確認"):
            # 全てのカラムに対して処理を行う
            for col in dm.data.columns:
                # 一つのカラム分のエリアを作成
                with st.container():
                    # グラフとボタン用のエリアを作成(大きさ1:3で分割)
                    normalize_area = st.columns([1, 3])
                    # ボタン用エリア
                    with normalize_area[0]:
                        if st.button(f"対数変換: {col}"):
                            # 対数変換を行う
                            dm.transform_log(col)
                            
                    # グラフ用エリア
                    with normalize_area[1]:
                        # ヒストグラムを作成
                        histogram = alt.Chart(dm.data).mark_bar().encode(
                            alt.X(col, bin=True),
                            y='count()'
                        )

                        # ヒストグラムを表示
                        st.altair_chart(histogram, use_container_width=True)
                        
        # 標準化の処理
        with st.expander("標準化の処理"):
            if st.button("標準化"):
                dm.transform_std(columns=dm.data.columns.tolist().remove(dm.target_name))
                
            
            # カラムごとに平均値を取得
            mean_values = dm.data.mean().round(2)
            mean_values = pd.DataFrame(mean_values.rename("平均値")).reset_index()
            # 棒グラフを作成
            bar_chart = alt.Chart(mean_values).mark_bar().encode(
                x='index',
                y='平均値'
            )
            
            # 棒グラフを表示
            st.altair_chart(bar_chart, use_container_width=True)
            
            # カラムごとに標準偏差を取得
            std_values = dm.data.std().round(2)
            std_values = pd.DataFrame(std_values.rename("標準偏差")).reset_index()
            # 棒グラフを作成
            bar_chart_std = alt.Chart(std_values).mark_bar().encode(
                x='index',
                y='標準偏差'
            )

            # 標準偏差の棒グラフを表示
            st.altair_chart(bar_chart_std, use_container_width=True)
            
        # 標準化の処理
        with st.expander("カテゴリ変数の数値化"):
            # カテゴリ変数のみを抽出
            categorical_columns = dm.data.select_dtypes(include=['object']).columns.tolist()

            # カテゴリ変数の棒グラフを表示
            for col in categorical_columns:
                st.write(f"## {col}")
                value_counts = dm.data[col].value_counts()
                bar_chart = alt.Chart(value_counts.reset_index()).mark_bar().encode(
                    x='index',
                    y='value'
                )
                st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.error("先にStep1を完了させましょう。")
    
# 機械学習（AI）
with tabs[2]:
    if st.session_state["steps_completed"]["step2"]:
        # データをセットする
        mm.set_data(dm.data, target_name="目的変数")
        # データを分割する
        split_area = st.columns(2)
        # 学習データとテストデータの割合を設定
        with split_area[0]:
            train_per = st.number_input("トレーニングデータの割合(%)", min_value=10, max_value=90, value=70, step=5)
        with split_area[1]:
            test_per = st.number_input("テストデータの割合(%)", min_value=10, max_value=90, value=100-train_per, step=5, disabled=True)
        is_data_split = st.button("データを分割する")
        
        # 押したときの処理
        if is_data_split:
            mm.split_data(test_per/100)
            st.session_state["is_data_split"] = True
            
        # データ分割後に常に行う処理
        if "is_data_split" in st.session_state and st.session_state["is_data_split"]:
            # 現在使っているデータセットは全部同じtarget_name
            mm.select_model()
            if mm.model is not None:
                mm.info_model()
                n_trials = st.number_input("最適化の試行回数", min_value=1, value=100)
                if st.button("学習開始"):
                    with st.spinner("モデル学習中...."):
                        mm.optimize(n_trials)
                    mm.train()
                    st.session_state["is_optimized"] = True
                
                # モデルの最適化が終わっている場合は最適なパラメータを表示
                if "is_optimized" in st.session_state and st.session_state["is_optimized"]: 
                    st.success("学習完了！")
                    st.write(f"最適なパラメータ: {mm.best_params}")
                    st.write(mm.get_trials_dataframe())
                    completed_step("step3")
            
    else:
        st.error("先にStep2を完了させましょう。")

# 未知データの予測をする。
with tabs[3]:
    if st.session_state["steps_completed"]["step3"]:
        with st.expander("テストデータの予測結果"):
            # テストデータの予測部分
            st.write("## テストデータの予測結果")
            # 既に分割済みのデータを使う
            score = mm.calc_score(mm.y_test, mm.predict(mm.X_test))
            # 予測モデルによってスコアの表示を変える
            if mm.model_type == "分類":
                st.write(f"### 正解率: {round(score, 2)*100}%")
            elif mm.model_type == "回帰":
                st.write(f"### 決定係数: {round(score, 2)*100}")
                    
            # テストデータの予測結果を表示する
            # ラベルを日本語化
            dataset_label_names[dm.data_name]
            if mm.model_type == "分類":
                # 置き換える
                test_predictions = pd.DataFrame([dataset_label_names[dm.data_name][int(i)] for i in mm.predict(mm.X_test)])
                test_targets = pd.DataFrame([dataset_label_names[dm.data_name][int(i)] for i in mm.y_test])
                # ○×のDataFrameを作成
                tf = pd.DataFrame(np.where(test_predictions == test_targets, '🟢', '❌'))
                # テストデータの予測結果と正解値を結合
                test_results = pd.concat([tf, test_predictions, test_targets, 
                                mm.X_test.reset_index(drop=True)], axis=1)
                # 列名を設定
                test_results.columns = ['正誤', '予測値', '正解値'] + mm.X_test.columns.tolist()
                
            elif mm.model_type == "回帰":
                # 回帰の場合は目的変数の意味のみを出力
                st.write(dataset_label_names[dm.data_name][0])
                test_predictions = pd.DataFrame(mm.predict(mm.X_test))
                test_targets = pd.DataFrame(mm.y_test.reset_index(drop=True))
                # テストデータの予測結果と正解値を結合
                test_results = pd.concat([test_predictions, test_targets, 
                                mm.X_test.reset_index(drop=True)], axis=1)
                # 列名を設定
                test_results.columns = ['予測値', '正解値'] + mm.X_test.columns.tolist()
                
            # テストデータの予測結果を表示
            st.dataframe(test_results)
            # 制度指標の計算と表示
            # if prediction is not None:
            #     st.write("## 予測結果")
            #     st.write(f"予測結果: {round(mm.calc_score(prediction))}%")
            # else:
            #     st.write("## 予測結果")
            #     st.write("予測を行ってください。")
            completed_step("step4")
        
        with st.expander("データを編集して予測"):
            # データを好きに編集して予測してみる
            st.write("## データを編集して予測")
            # 編集する行数を入力
            num_rows = st.number_input("入力したいデータ数", min_value=1, value=1, 
                                       max_value=len(mm.X_test))
            edit_area = st.columns([1,4])
            
            with edit_area[0]:
                st.write("実際の正解値")
                output_correct_label = mm.y_test.reset_index(drop=True).rename("正解値")[:num_rows]
                # 分類の場合はラベルを日本語化
                if mm.model_type == "分類":
                    output_correct_label = [dataset_label_names[dm.data_name][int(i)] for i in output_correct_label]
                st.dataframe(output_correct_label, hide_index=True)
            
            with edit_area[1]:
                st.write("↓入力するデータを編集できます↓")
                if dm.data_name == "数学の成績データ":
                    edited_data = mat_data_editor(mm.X_test.reset_index(drop=True).iloc[:num_rows,:])
                    edited_data = edited_data.astype(np.int64)
                else:
                    edited_data = mm.X_test.reset_index(drop=True).iloc[:num_rows,:]
                    edited_data = st.data_editor(edited_data, hide_index=True)
            
            prediction = None
            # 予測ボタンが押されたら
            if st.button("編集したデータで予測する"):
                # 入力データをDataFrameに変換
                input_df = pd.DataFrame(edited_data)
                # 予測実行
                prediction = mm.predict(input_df)
                # 分類の場合はラベルを日本語化
                if mm.model_type == "分類":
                    prediction = [dataset_label_names[dm.data_name][int(i)] for i in prediction]
                # 予測結果の表示
                # 予測結果は上のdata_areaに出力
                st.session_state["df_edited_result"] = pd.concat([pd.DataFrame(prediction, columns=['予測']),
                    edited_data], axis=1)
            # 一度予測したらその結果は常に表示
            if "df_edited_result" in st.session_state:
                # 予測の列の背景色を変えて，整数値固定，それ以外は小数点以下2桁にする
                # if mm.model_type == "分類":
                #     format_dict = {**{"予測": "{:.0f}"}, **{col: "{:.2f}" for col in mm.X_test.columns}}
                # else:
                #     format_dict = {**{"予測": "{:.2f}"}, **{col: "{:.2f}" for col in mm.X_test.columns}}
                    
                # test_results_styled = st.session_state["df_edited_result"].style.applymap(
                #     lambda x: 'background-color: #FFCCCC', subset=['予測']).format(
                #         format_dict
                #         )
                # st.dataframe(test_results_styled, hide_index=True)
                st.dataframe(st.session_state["df_edited_result"], hide_index=True)
    else:
        st.error("先にStep3を完了させましょう。")

# AIの予測根拠を分析
with tabs[4]:
    if not (mm.model_name == "XGBoost" or mm.model_name == "LightGBM" or \
            mm.model_name == "ランダムフォレスト" or mm.model_name == "決定木"):
        st.error("XAIはXGBoost, LightGBM, ランダムフォレスト, 決定木のみ対応しています。")
    elif st.session_state["steps_completed"]["step4"]:
        # XAIの初期化
        xm.set_model(mm.best_model, mm.model_type)
        xm.set_data(mm.X_test)
        
        st.write("# SHAP値の計算")
        # SHAP値の計算開始ボタンを押したかどうか
        st.session_state["is_calc_shap_values"] = st.button("SHAP値計算")
        
        # ボタンを押した場合はSHAP値を計算
        if st.session_state["is_calc_shap_values"]:
            st.session_state["shap_values"] = xm.calc_shap_values()
            completed_step("step5")
        
        # SHAP値の計算が終わっている場合はプロットを行う
        if "shap_values" in st.session_state:
            st.write("# SHAP値の可視化")
            # ドットプロットのインデックスを入力受付
            dot_input = st.number_input("ドットプロットのインデックス", 
                                        min_value=0, max_value=len(xm.shap_values)-1)
            # サマリープロットを表示するエリアを作成
            summary_plot_area = st.columns(2)
            if xm.shap_values is not None:
                with summary_plot_area[0]:
                    xm.plot_bar()
                with summary_plot_area[1]:
                    xm.plot_dot(dot_input)

                # 依存度プロットの表示
                st.write("# 依存度プロット")
                xm.selected_feature()
                xm.plot_dependence()
    else:
        st.error("先にStep4を完了させましょう。")
        
        

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)
load_sidebar()


# TODO
# - ifで分岐していると、どこでどのような処理をしているのかわかりにくい