import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 스트림릿 페이지 기본 설정 (넓은 화면 레이아웃)
st.set_page_config(
    page_title="Gyeonggi Population Dashboard",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ 경기도 시군별 인구 및 세대 구조 분석 대시보드")
st.caption("Data Science Project: Linear Regression Model for Household Size Prediction")
st.markdown("---")

# ==========================================
# 1. 데이터 로드 및 전처리
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
        'Total_Households': pd.to_numeric(df.iloc[:, 5], errors='coerce'),
        'Foreigners': pd.to_numeric(df.iloc[:, 9], errors='coerce')
    }).dropna()

    df_clean['Persons_Per_Household'] = df_clean['Total_Population'] / df_clean['Total_Households']
    df_clean['Sex_Ratio'] = df_clean['Male_Population'] / df_clean['Female_Population']
    df_clean['Foreigner_Ratio'] = (df_clean['Foreigners'] / df_clean['Total_Population']) * 100

    return df_clean

df = load_and_preprocess_data()

# ==========================================
# 2. 상단 요약 카드 (KPI Metrics)
# ==========================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("분석 대상 시군 수", f"{len(df)}개 지역")
m2.metric("경기도 총인구수", f"{df['Total_Population'].sum():,}명")
m3.metric("평균 세대당 인구수", f"{df['Persons_Per_Household'].mean():.2f}명")
m4.metric("최고 세대당 인구 지역", f"{df.loc[df['Persons_Per_Household'].idxmax(), 'Region']} ({df['Persons_Per_Household'].max():.2f}명)")

st.markdown("---")

# ==========================================
# 3. 탭(Tab) 구성
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. 데이터 탐색 & 시각화", 
    "🤖 2. 선형 회귀 모델 학습", 
    "🎛️ 3. 실시간 예측 시뮬레이터",
    "📋 4. 데이터 원본 확인"
])

# ------------------------------------------
# TAB 1: 데이터 탐색
# ------------------------------------------
with tab1:
    st.subheader("경기도 주요 시군구 인구 구조 분석")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**1세대당 거주 인구수 상위/하위 5개 지역**")
        top_bottom = pd.concat([
            df.sort_values(by='Persons_Per_Household', ascending=False).head(5),
            df.sort_values(by='Persons_Per_Household', ascending=True).head(5)
        ])
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ['#1f77b4' if x > df['Persons_Per_Household'].mean() else '#ff7f0e' for x in top_bottom['Persons_Per_Household']]
        sns.barplot(data=top_bottom, x='Persons_Per_Household', y='Region', palette=colors, ax=ax)
        ax.axvline(df['Persons_Per_Household'].mean(), color='red', linestyle='--', label=f"Avg: {df['Persons_Per_Household'].mean():.2f}")
        ax.set_xlabel("Persons Per Household")
        ax.legend()
        st.pyplot(fig)
        
    with c2:
        st.markdown("**변수간 상관관계 (Correlation Heatmap)**")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        corr_data = df[['Total_Population', 'Total_Households', 'Sex_Ratio', 'Foreigner_Ratio', 'Persons_Per_Household']]
        sns.heatmap(corr_data.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax, linewidths=0.5)
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("**총 세대수와 세대당 인구수의 분포 (Scatter Plot)**")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.scatterplot(data=df, x='Total_Households', y='Persons_Per_Household', hue='Sex_Ratio', size='Total_Population', sizes=(50, 400), palette='viridis', ax=ax)
    ax.set_xlabel("Total Households")
    ax.set_ylabel("Persons Per Household")
    st.pyplot(fig)

# ------------------------------------------
# TAB 2: 선형 회귀 모델 (Linear Regression)
# ------------------------------------------
with tab2:
    st.subheader("선형 회귀 (Linear Regression) 모델 성능 평가")
    st.write("독립변수들과 종속변수 간의 **선형 관계(일직선 방정식)**를 학습하여 인구수와 세대수에 따른 세대당 인구수를 예측합니다.")
    
    features = ['Total_Households', 'Sex_Ratio', 'Total_Population', 'Foreigner_Ratio']
    target = 'Persons_Per_Household'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("결정계수 ($R^2$ Score)", f"{r2:.4f}", help="1에 가까울수록 선형 모델의 설명력이 높습니다.")
    col_m2.metric("평균 절대 오차 (MAE)", f"{mae:.4f}명")
    col_m3.metric("평균 제곱근 오차 (RMSE)", f"{rmse:.4f}명")

    st.markdown("---")
    
    st.markdown("**선형 회귀 모델의 가중치 (Coefficients / 각 변수의 영향력)**")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    })
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm', ax=ax)
    ax.set_xlabel("Coefficient Value")
    st.pyplot(fig)

# ------------------------------------------
# TAB 3: 실시간 시뮬레이터 + 실제 지역 비교 기능 추가!
# ------------------------------------------
with tab3:
    st.subheader("🎛️ 지역별 조건 설정 및 실시간 예측 (선형 회귀)")
    st.write("아래 슬라이더와 입력창을 조절하면 학습된 **선형 회귀 방정식**에 따라 세대당 평균 인구수가 실시간으로 계산됩니다.")

    s_col1, s_col2 = st.columns(2)
    
    with s_col1:
        in_pop = st.number_input("총 인구수 (명)", min_value=10000, max_value=1500000, value=500000, step=10000)
        in_house = st.number_input("총 세대수 (세대)", min_value=5000, max_value=700000, value=200000, step=5000)
    
    with s_col2:
        in_sex = st.slider("성비 (남성/여성)", min_value=0.80, max_value=1.30, value=1.01, step=0.01)
        in_for = st.slider("외국인 비율 (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)

    input_data = pd.DataFrame({
        'Total_Households': [in_house],
        'Sex_Ratio': [in_sex],
        'Total_Population': [in_pop],
        'Foreigner_Ratio': [in_for]
    })

    pred_res = model.predict(input_data)[0]

    st.markdown("---")
    st.success(f"🎉 **예측 분석 결과:** 해당 인구 구조 조건을 가진 지역의 1세대당 거주 인구수는 **약 {pred_res:.2f}명**으로 예상됩니다.")

    # [추가된 기능] 실제 시군별 데이터와 모델 예측값 비교 섹션
    st.markdown("---")
    st.subheader("📍 경기도 실제 시군별 데이터와 모델 예측값 비교")
    st.write("실제 경기도 시군구 데이터를 선택하여, 우리 모델이 예측한 값과 실제 값이 얼마나 일치하는지 비교해 보세요.")
    
    selected_region = st.selectbox("비교할 지역을 선택하세요", df['Region'].values)
    region_data = df[df['Region'] == selected_region].iloc[0]
    
    actual_val = region_data['Persons_Per_Household']
    pred_input = pd.DataFrame({
        'Total_Households': [region_data['Total_Households']],
        'Sex_Ratio': [region_data['Sex_Ratio']],
        'Total_Population': [region_data['Total_Population']],
        'Foreigner_Ratio': [region_data['Foreigner_Ratio']]
    })
    model_pred = model.predict(pred_input)[0]
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("선택 지역", selected_region)
    col_b.metric("실제 세대당 인구수", f"{actual_val:.2f}명")
    col_c.metric("모델 예측값", f"{model_pred:.2f}명", delta=f"{model_pred - actual_val:.2f}명 (오차)")

# ------------------------------------------
# TAB 4: 원본 데이터
# ------------------------------------------
with tab4:
    st.subheader("수집 및 정제된 원본 데이터셋")
    st.dataframe(df, use_container_width=True)
    
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 정제된 CSV 데이터 다운로드",
        data=csv_bytes,
        file_name="Gyeonggi_Population_Processed.csv",
        mime="text/csv"
    )
