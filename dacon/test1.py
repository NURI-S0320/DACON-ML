import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 데이터 로드
df = pd.read_csv('open/train.csv')

# 기본 통계량 확인
print("=== 기본 통계량 ===")
print(df['가격(백만원)'].describe())

# 1. 제조사별 평균 가격 분석
manufacturer_price = df.groupby('제조사')['가격(백만원)'].agg(['mean', 'count', 'std']).round(2)
manufacturer_price = manufacturer_price.sort_values('mean', ascending=False)
print("\n=== 제조사별 평균 가격 ===")
print(manufacturer_price)

# 2. 차량상태별 평균 가격
condition_price = df.groupby('차량상태')['가격(백만원)'].agg(['mean', 'count', 'std']).round(2)
print("\n=== 차량상태별 평균 가격 ===")
print(condition_price)

# 3. 구동방식별 평균 가격
drive_price = df.groupby('구동방식')['가격(백만원)'].agg(['mean', 'count', 'std']).round(2)
print("\n=== 구동방식별 평균 가격 ===")
print(drive_price)

# 4. 사고이력에 따른 가격 차이
accident_price = df.groupby('사고이력')['가격(백만원)'].agg(['mean', 'count', 'std']).round(2)
print("\n=== 사고이력별 평균 가격 ===")
print(accident_price)

# 5. 연속형 변수와 가격의 상관관계
numeric_cols = ['배터리용량', '주행거리(km)', '보증기간(년)', '연식(년)']
correlations = df[numeric_cols + ['가격(백만원)']].corr()['가격(백만원)'].sort_values(ascending=False)
print("\n=== 연속형 변수와 가격의 상관관계 ===")
print(correlations)

# 6. 모델별 평균 가격 (상위 10개)
model_price = df.groupby('모델')['가격(백만원)'].agg(['mean', 'count', 'std']).round(2)
model_price = model_price.sort_values('mean', ascending=False)
print("\n=== 모델별 평균 가격 (상위 10개) ===")
print(model_price.head(10))

# 7. 주행거리와 가격의 관계 분석
distance_corr = stats.pearsonr(df['주행거리(km)'], df['가격(백만원)'])
print("\n=== 주행거리와 가격의 상관관계 ===")
print(f"Pearson correlation: {distance_corr[0]:.4f}")
print(f"P-value: {distance_corr[1]:.4f}")

# 8. 배터리 용량과 가격의 관계 분석
# 결측치 제외
battery_price = df[df['배터리용량'].notna()]
battery_corr = stats.pearsonr(battery_price['배터리용량'], battery_price['가격(백만원)'])
print("\n=== 배터리용량과 가격의 상관관계 ===")
print(f"Pearson correlation: {battery_corr[0]:.4f}")
print(f"P-value: {battery_corr[1]:.4f}")

# 9. 보증기간과 가격의 관계
warranty_corr = stats.pearsonr(df['보증기간(년)'], df['가격(백만원)'])
print("\n=== 보증기간과 가격의 상관관계 ===")
print(f"Pearson correlation: {warranty_corr[0]:.4f}")
print(f"P-value: {warranty_corr[1]:.4f}")

# 10. 결측치 현황 파악
missing_values = df.isnull().sum()
print("\n=== 결측치 현황 ===")
print(missing_values[missing_values > 0])