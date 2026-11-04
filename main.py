import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🏙️ 경기도 시군별 인구 구조 기반 세대당 인구수 예측 모델")
st.write("공공데이터포털(KOSIS) 데이터를 바탕으로, 시군별 세대수 및 성비에 따른 **1세대당 거주 인구수**를 예측하고 시각화하는 스트림릿 웹 앱입니다.")

# ==========================================
# 1. 내장된 CSV 데이터 로드 및 전처리
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    csv_data = """시군구별(1),시군구별(2),2026.09,2026.09.1,2026.09.2,2026.09.3,2026.09.4,2026.09.5,2026.09.6,2026.09.7,2026.09.8,2026.09.9
시군구별(1),시군구별(2),합계 (명),합계 (명),합계 (명),세대수 (세대),내국인 (명),내국인 (명),내국인 (명),등록외국인 (명),등록외국인 (명),등록외국인 (명)
시군구별(1),시군구별(2),계,남,여,소계,계,남,여,계,남,여
합계,소계,14265640,7209788,7055852,6191729,13777186,6915350,6861836,488454,294438,194016
수원시,소계,1225702,614369,611333,546855,1186308,595064,591244,39394,19305,20089
용인시,소계,1111855,552450,559405,450991,1090103,540111,549992,21752,12339,9413
성남시,소계,916838,451923,464915,407918,899160,443816,455344,17678,8107,9571
화성시,소계,1057339,558985,498354,434633,1001559,518632,482927,55780,40353,15427
부천시,소계,785831,387152,398679,344196,753731,370689,383042,32100,16463,15637
안산시,소계,657179,338811,318368,289639,608343,311466,296877,48836,27345,21491
평택시,소계,662188,350956,311232,302078,624536,328844,295692,37652,22112,15540
안양시,소계,552109,271031,281078,247783,537877,264101,273776,14232,6930,7302
시흥시,소계,582498,304481,278017,250325,537655,279093,258562,44843,25388,19455
김포시,소계,514489,255850,258639,214717,506863,252118,254745,7626,3732,3894
파주시,소계,529367,267490,261877,222927,516422,260840,255582,12945,6650,6295
의정부시,소계,470081,230489,239592,207361,463388,227289,236099,6693,3200,3493
광주시,소계,409395,208398,200997,173007,400587,203875,196712,8808,4523,4285
하남시,소계,332900,162624,170276,138356,328409,160288,168121,4491,2336,2155
광명시,소계,284305,138139,146166,121773,280629,136287,144342,3676,1852,1824
군포시,소계,266679,131015,135664,115160,261775,128525,133250,4904,2490,2414
오산시,소계,244671,126309,118362,106517,230919,118227,112692,13752,8082,5670
이천시,소계,225333,117621,107712,98748,208154,107297,100857,17179,10324,6855
양주시,소계,279261,142751,136510,116631,271957,138883,133074,7304,3868,3436
구리시,소계,190472,93375,97097,84566,186950,91632,95318,3522,1743,1779
안성시,소계,199273,103772,95501,88029,183921,94132,89789,15352,9640,5712
포천시,소계,159670,84587,75083,73341,147326,76662,70664,12344,7925,4419
의왕시,소계,168239,82455,85784,72076,166336,81504,84832,1903,951,952
여주시,소계,114210,58053,56157,51152,108782,54964,53818,5428,3089,2339
양평군,소계,125203,63045,62158,57467,123281,61989,61292,1922,1056,866
동두천시,소계,92569,46627,45942,42918,90317,45371,44946,2252,1256,996
가평군,소계,63191,32080,31111,30467,61453,31057,30396,1738,1023,715
연천군,소계,41595,22301,19294,20464,39498,20977,18521,2097,1324,773"""
    
    df_raw = pd.read_csv(StringIO(csv_data))
    df = df_raw.iloc[3:].copy()
    
    df_clean = pd.DataFrame({
        'Region': df.iloc[:, 0],
        'Total_Population': pd.to_numeric(df.iloc[:, 2], errors='coerce'),
        'Male_Population': pd.to_numeric(df.iloc[:, 3], errors='coerce'),
        'Female_Population': pd.to_numeric(df.iloc[:, 4], errors='coerce'),
        'Total_Households': pd.to_numeric(df.iloc[:, 5], errors='coerce')
    }).dropna()

    df_clean['Persons_Per_Household'] = df_clean['Total_Population'] / df_clean['Total_Households']
    df_clean['Sex_Ratio'] = df_clean['Male_Population'] / df_clean['Female_Population']

    return df_clean

df = load_and_preprocess_data()

with st.expander("📊 정제된 시군별 인구 데이터 미리보기"):
    st.dataframe(df)

# ==========================================
# 2. 탐색적 데이터 분석 (EDA)
# ==========================================
st.header("1. 탐색적 데이터 분석 (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("총 세대수 vs 세대당 인구수")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='Total_Households', y='Persons_Per_Household', ax=ax, color='indigo', s=70)
    ax.set_xlabel("Total Households")
    ax.set_ylabel("Persons Per Household")
    st.pyplot(fig)

with col2:
    st.subheader("변수 간 상관관계 히트맵")
    fig, ax = plt.subplots(figsize=(6, 4))
    numeric_df = df[['Total_Population', 'Total_Households', 'Sex_Ratio', 'Persons_Per_Household']]
    sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax)
    st.pyplot(fig)

# ==========================================
# 3. 머신러닝 모델 학습
# ==========================================
st.header("2. 머신러닝 예측 모델 (Random Forest)")

features = ['Total_Households', 'Sex_Ratio', 'Total_Population']
target = 'Persons_Per_Household'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

m1, m2 = st.columns(2)
m1.metric("결정계수 ($R^2$ Score)", f"{r2:.4f}")
m2.metric("평균 절대 오차 (MAE)", f"{mae:.4f}")

# ==========================================
# 4. 실시간 시뮬레이션
# ==========================================
st.header("3. 시군구 가구 구조 예측 시뮬레이터")

st.sidebar.header("🔍 지역 인구 조건 설정")
input_pop = st.sidebar.number_input("총 인구수 (명)", min_value=10000, max_value=1500000, value=500000, step=10000)
input_households = st.sidebar.number_input("총 세대수 (세대)", min_value=5000, max_value=700000, value=200000, step=5000)
input_sex_ratio = st.sidebar.slider("성비 (남성인구 / 여성인구)", min_value=0.80, max_value=1.30, value=1.01, step=0.01)

input_data = pd.DataFrame({
    'Total_Households': [input_households],
    'Sex_Ratio': [input_sex_ratio],
    'Total_Population': [input_pop]
})

predicted_val = model.predict(input_data)[0]

st.success(f"💡 **예측 결과:** 해당 조건 시군의 **1세대당 평균 거주 인구수**는 약 **{predicted_val:.2f}명**으로 추정됩니다.")
