import shap
import streamlit as st
import matplotlib.pyplot as plt

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
        
    def set_model(self, model, model_type):
        self.model = model
        self.model_type = model_type
        
    def set_data(self, X):
        self.X = X

    def calc_shap_values(self):
        """
        SHAP値の計算。計算中はStreamlitのプログレスバーを表示。
        """
        with st.spinner('SHAP値を計算中...'):
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(self.X)
            self.expected_value = explainer.expected_value
        return self.shap_values

    def plot_bar(self):
        """
        SHAP値に基づいたサマリープロットをStreamlit上で表示。
        """
        fig, ax = plt.subplots()
        # shap.summary_plot(self.shap_values, self.X, plot_type="dot", show=False)
        shap.summary_plot(self.shap_values, self.X, plot_type="bar", show=False)
        st.pyplot(fig)
        return fig
        
    def plot_dot(self, dot_index=0):
        """
        特徴量の重要度を示すSHAP値に基づいたプロットをStreamlit上で表示。
        """
        fig, ax = plt.subplots()
        self.dot_index = dot_index
        if self.model_type == "回帰":
            shap.summary_plot(self.shap_values, self.X, plot_type="dot", show=False)
        else:
            # st.write(dot_index, self.shap_values[:,:,dot_index].shape, self.X.shape)
            shap.summary_plot(self.shap_values[:,:,dot_index], self.X, plot_type="dot", show=False)
        st.pyplot(fig)
        return fig

    def selected_feature(self):
        """
        Streamlit上で依存度プロットに使用する特徴量を選択。
        """
        self.feature_base = st.selectbox("主要な特徴量の選択：依存度プロットの主軸となる特徴量を選択してください。", self.X.columns.tolist())
        self.feature_color = st.selectbox("色分けに使用する特徴量の選択", self.X.columns.tolist())
        st.write("※各データポイントの色を決定するための特徴量を選択してください。この特徴量により、主要な特徴量の影響を異なる角度から分析できます。")
        return self.feature_base, self.feature_color
    
    def plot_dependence(self, dot_index=0):
        """
        特定の特徴量に対する依存度プロットをStreamlit上で表示。
        """
        fig, ax = plt.subplots()
        if self.model_type == "回帰":
            shap_values_class = self.shap_values
        else:
            shap_values_class = self.shap_values[:,:,dot_index]
        # st.write(self.feature_base, shap_values_class)
        shap.dependence_plot(self.feature_base, shap_values_class, self.X, ax=ax,
                             interaction_index=self.feature_color, show=False)
        st.pyplot(fig)
        
#TODO

#エラーハンドリングとドキュメンテーション: エラーハンドリングの追加と、各メソッドやクラスのドキュメンテーションを
# 充実させることで、さらにユーザーフレンドリーな設計とすることが可能です。特にSHAP値の計算は、
# サポートされているモデルの種類やデータセットのサイズによってはエラーが発生する可能性があるため、
# その点をユーザーに明示することが重要です。

# summary_plotのdotは次元数が多いため難しい．自動で次元数推定位を行うなどをする．
# dependanceplotも１次元しか扱えないので，複数扱えるようにする
# ユーザからの入力を受け付けるようにするのはどうか？

