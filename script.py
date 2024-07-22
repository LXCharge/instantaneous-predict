import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import matlab.engine
import streamlit as st

st.set_page_config(
    page_title="页面标题",  # 页面标题
    page_icon=":rainbow:",  # icon
    # layout="wide",  # 页面布局
    initial_sidebar_state="auto"  # 侧边栏
)

if 'eng' not in st.session_state:
    st.session_state.eng = matlab.engine.start_matlab()
    # st.set_option('deprecation.showPyplotGlobalUse', False)


def cals(a, b, c, d):
    X = matlab.double([a, b, c, d])
    eng = st.session_state.eng
    # 调用matlab自带函数
    load = eng.load('test0720.mat')
    # print(load.keys())
    n_X1 = eng.rdivide(eng.minus(X, load['mu']), load['sigma'])
    n_X2 = eng.rdivide(eng.minus(X, load['mu2']), load['sigma2'])
    # 模型预测
    Y1 = eng.predict(load['regression'], n_X1)
    Y2 = eng.predict(load['regression2'], n_X2)
    # 计算SHAP值
    explainer = eng.shapley(load['regression'], load['n_trainingPredictors'])
    # eng.eval('explainer = shapley(regression, n_trainingPredictors);', nargout=0)
    # eng.eval('explainer = fit(regression, n_X1);')
    explainer = eng.fit(explainer, n_X1)
    eng.workspace['explainer'] = explainer
    S1 = eng.eval('explainer.ShapleyValues.ShapleyValue;')
    # S1 = eng.workspace['explainer.ShapleyValues.ShapleyValues']
    # S1 = explainer
    S1 = eng.ctranspose(S1)
    S1 = eng.mtimes(S1, 100.0)

    explainer2 = eng.shapley(
        load['regression2'], load['n_trainingPredictors2'])
    explainer2 = eng.fit(explainer2, n_X2)
    eng.workspace['explainer2'] = explainer2
    S2 = eng.eval('explainer2.ShapleyValues.ShapleyValue;')
    S2 = eng.ctranspose(S2)
    S2 = eng.mtimes(S2, 100.0)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)
    S1 = np.array(S1)
    S2 = np.array(S2)
    # stop
    # eng.quit()

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置默认的中英文字体
    plt.rcParams['font.weight'] = 'bold'  # 设置字体加粗
    plt.rcParams['axes.labelweight'] = 'bold'  # 设置字体加粗
    plt.rcParams['font.size'] = 30

    # 将MATLAB得到的S向量 转换为python里的数组变量
    shap_values1 = S1
    shap_values2 = S2

    # 将输入参数X向量 转换为python里的DataFrame变量
    # 将MATLAB里的向量转成python里的DataFrame;列名分别为Rc Rd MC Temp（4个输入参数）
    X_train = pd.DataFrame(np.array(X).reshape(
        1, -1), columns=['Rc', 'Rd', 'MC', 'Temp'])
    print(X_train)

    # 画图
    plt.rcParams['font.size'] = 15
    fig1 = shap.force_plot(2.15,
                           shap_values1[0],
                           X_train.iloc[0],
                           matplotlib=True, show=False)

    fig2 = shap.force_plot(3.63,
                           shap_values2[0],
                           X_train.iloc[0],
                           matplotlib=True, show=False)
    return fig1, fig2, Y1, Y2


step = 1e-4
a = st.number_input('Rc(%):', step=step, format='%.4f')
b = st.number_input('Rd(mm):', step=step, format='%.4f')
c = st.number_input('MC(%):', step=step, format='%.4f')
d = st.number_input('Temp(℃):', step=step, format='%.4f')
X_train = pd.DataFrame({
    'Rc': [a], 'Rd': [b], 'MC': [c], 'Temp': [d]
})
# st.table(X_train)
if st.button("predict"):
    fig1, fig2, Y1, Y2 = cals(a, b, c, d)
    st.subheader(
        '_Based on feature values, predicted instantaneous strain is $' + str(round(Y1*100, 2)) + '/10^{-2}\%$_')
    st.pyplot(fig1)
    st.subheader(
        f'_Based on feature values, predicted instantaneous strain is $' + str(round(Y2*100, 2)) + '/10^{-2}\%$_')
    st.pyplot(fig2)
