import os
import pandas as pd
import streamlit as st

DATA_PATH = "./data_japanese"

def selected_data(data_names):
    """Streamlitのselectboxを使ってデータセット名を選択する

    Args:
        data_names (list[str]): データ名一覧 

    Returns:
        str: データ名
    """
    selected_data_name = st.selectbox('データセットを選択してください', options=data_names)
    return selected_data_name


def load_data(selected_data_name):
    """指定されたデータセット名に基づいてデータを読み込み、データフレームを返す

    Args:
        selected_data_name (str): データ名

    Returns:
        pd.DataFrame: データ
    """
    if selected_data_name:
        data_path = os.path.join(DATA_PATH, f"{selected_data_name}.csv")
        data = pd.read_csv(data_path, index_col=0)
        return data
    else:
        return None
