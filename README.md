# 차량 가격 예측 프로젝트

## 프로젝트 개요

이 프로젝트는 주어진 차량 데이터를 이용하여 차량 가격을 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다. 데이터 분석을 통해 가격에 영향을 미치는 다양한 요인을 파악하고, 여러 머신러닝 모델을 적용하여 예측 성능을 높이는 데 중점을 두었습니다.

## 사용 기술 스택

* Python
* pandas, numpy, matplotlib, seaborn, scipy
* scikit-learn, lightgbm, optuna

## 데이터 설명

사용된 데이터는 차량의 다양한 정보 (제조사, 모델, 가격, 연식, 주행거리, 배터리 용량 등)를 포함하고 있습니다. `train.csv` 파일을 사용하여 모델을 학습하고, 별도의 테스트 데이터를 사용하여 모델의 성능을 평가했습니다.

## 데이터 분석

`data.ipynb` 파일에서 수행된 주요 데이터 분석 내용은 다음과 같습니다.

* **기본 통계량 확인:** 데이터의 전반적인 분포 및 통계적 특성을 파악했습니다.

    ```python
    import pandas as pd

    df = pd.read_csv('./train.csv')
    print(df['가격(백만원)'].describe())
    ```

* **제조사별 평균 가격 분석:** 제조사별 평균 가격, 차량 수, 가격 표준 편차 등을 분석하여 제조사가 가격에 미치는 영향을 확인했습니다.

    ```python
    manufacturer_price = df.groupby('제조사')['가격(백만원)'].agg(['mean', 'count', 'std'])
    print(manufacturer_price)
    ```

* **연속형 변수와 가격의 상관관계 분석:** 배터리 용량, 주행 거리, 연식 등 연속형 변수와 가격 간의 상관관계를 분석했습니다.

    ```python
    numeric_cols = ['배터리용량', '주행거리(km)', '보증기간(년)', '연식(년)']
    correlations = df[numeric_cols + ['가격(백만원)']].corr()['가격(백만원)']
    print(correlations)
    ```

* **결측치 확인:** 데이터에 존재하는 결측치를 확인하고, 처리 전략을 수립했습니다.

    ```python
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    ```

* **모델별 가격 통계 분석:** 모델별 가격의 평균, 표준 편차 등을 분석하여 모델 가격의 특성을 파악했습니다.

    ```python
    model_price = df.groupby('모델')['가격(백만원)'].agg(['mean', 'count', 'std'])
    print(model_price.head(10))
    ```

## 모델링

`test_1.ipynb` 파일에서 구현된 주요 모델링 과정은 다음과 같습니다.

* **데이터 전처리:** 범주형 변수 인코딩, 결측치 처리, 이상치 처리 등을 수행하여 모델 학습에 적합한 형태로 데이터를 변환했습니다.

    ```python
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    def preprocess_data(df):
        # ... (전처리 코드)
        return df
    ```

* **모델 학습:** Random Forest, Gradient Boosting, LightGBM 등 다양한 회귀 모델을 학습하고, 앙상블 기법을 적용하여 예측 성능을 높였습니다.

    ```python
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import lightgbm as lgb

    def train_model(X, y):
        rf = RandomForestRegressor()
        hgb = GradientBoostingRegressor()
        lgb_model = lgb.LGBMRegressor()

        rf.fit(X, y)
        hgb.fit(X, y)
        lgb_model.fit(X, y)

        return {'rf': rf, 'hgb': hgb, 'lgb': lgb_model}
    ```

* **하이퍼파라미터 튜닝:** Optuna 라이브러리를 사용하여 모델의 하이퍼파라미터를 최적화했습니다.

    ```python
    import optuna

    def objective(trial):
        # ... (Optuna 최적화 코드)
        return rmse
    ```

* **모델 평가:** 교차 검증, RMSE, R2 Score 등 다양한 지표를 사용하여 모델의 성능을 평가했습니다.

    ```python
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score

    def evaluate_model(model, X, y):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return rmse, r2
    ```

* **Ioniq5 특화 모델 개발:** 특정 모델(Ioniq5)의 예측 성능을 높이기 위해 별도의 모델을 개발하고, 가중치 조정, 마이크로 세그멘테이션 등의 기법을 적용했습니다.

    ```python
    class EnhancedSegmentModel:
        def __init__(self):
            # ...

        def fit(self, X, y):
            # ...

        def predict(self, X):
            # ...
    ```

## 코드 설명

* `data.ipynb`: 데이터 분석 및 전처리 코드 (Jupyter Notebook)
* `test_1.ipynb`: 모델 학습, 평가 및 예측 코드 (Jupyter Notebook)

## 실행 방법

1.  Python 3.x가 설치되어 있어야 합니다.
2.  필요한 라이브러리를 설치합니다.

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn scipy lightgbm optuna
    ```

3.  `data.ipynb` 및 `test_1.ipynb` 파일을 실행하여 데이터 분석 및 모델링 과정을 수행합니다.

## 결과

최종 모델의 예측 성능은 다음과 같습니다.

* RMSE: 1.3849
* R2 Score: 0.9986

* === 성능 분석 리포트 ===

* 브랜드별 평균 RMSE:
* P          1.8709
* Premium    1.6034
* Performance 0.7191
* T          0.4397
* Entry      0.5137
* Domestic   1.8793
* Import     0.4644

* 데이터 크기별 성능:
* 중(200-300)      0.4930
* 소(200미만)        1.2092

* 최고 성능 Top 3 세그먼트:
* Premium-P            RMSE: 0.3156 (데이터 수: 80)
* High-Premium-Import  RMSE: 0.3245 (데이터 수: 70)
* Basic-Import         RMSE: 0.3383 (데이터 수: 80)

* 최저 성능 Bottom 3 세그먼트:
* High-Premium-P       RMSE: 2.8823 (데이터 수: 68)
* Basic-Domestic       RMSE: 3.1323 (데이터 수: 117)
* Luxury-P             RMSE: 3.4263 (데이터 수: 69)

