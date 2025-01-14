import streamlit as st
import pandas as pd
import numpy as np

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
