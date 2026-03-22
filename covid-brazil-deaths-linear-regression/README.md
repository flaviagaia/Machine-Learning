# COVID-19 Brazil Deaths Linear Regression

## PT-BR

Projeto de `machine learning` tradicional para prever a tendência de óbitos por Covid-19 no Brasil usando `Linear Regression` e `R²`, com base em dados abertos oficiais do governo federal.

O projeto utiliza microdados do `SIVEP-Gripe / SRAG` disponibilizados pelo Ministério da Saúde no Portal de Dados Abertos do SUS e transforma essa base granular em uma série temporal nacional diária de óbitos por Covid-19.

### Fonte oficial dos dados

Fonte utilizada:

- Portal de Dados Abertos do SUS
- dataset: `SRAG 2019 a 2026`
- recurso oficial usado para este projeto: base congelada de `2024`

Links oficiais:

- [Dataset SRAG 2019 a 2026](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026)
- [Arquivo CSV oficial 2024](https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-26-06-2025.csv)

### Objetivo do case

O objetivo é modelar a dinâmica temporal dos óbitos por Covid-19 no Brasil a partir da própria série histórica agregada.

Em vez de trabalhar com a contagem bruta ruidosa de notificações individuais, o projeto prevê a `média móvel de 7 dias` dos óbitos diários, o que torna a série mais estável e mais adequada para regressão linear.

### Como os dados foram construídos

Os microdados oficiais são agregados em [scripts/build_official_daily_dataset.py](./scripts/build_official_daily_dataset.py) e também pela lógica equivalente em [src/data_pipeline.py](./src/data_pipeline.py).

O pipeline de agregação:

1. lê o CSV oficial do `SIVEP-Gripe / SRAG`;
2. filtra registros com `CLASSI_FIN = 5`, correspondentes a SRAG por Covid-19;
3. filtra registros com `EVOLUCAO = 2`, correspondentes a óbito;
4. usa `DT_EVOLUCA` como data do desfecho;
5. restringe a série ao ano de `2024`;
6. agrega os óbitos por dia em nível nacional.

O resultado final é salvo em:

- [data/covid_brazil_daily_deaths_2024.csv](./data/covid_brazil_daily_deaths_2024.csv)

### O que é regressão linear

`Linear Regression` é um modelo supervisionado usado para estimar uma variável contínua a partir de relações lineares entre features explicativas e um alvo.

No formato mais simples, a equação é:

```text
y = beta_0 + beta_1 x_1 + beta_2 x_2 + ... + beta_n x_n + epsilon
```

Onde:

- `y` é o valor a ser previsto;
- `x_1 ... x_n` são as variáveis explicativas;
- `beta_0` é o intercepto;
- `beta_1 ... beta_n` são os coeficientes aprendidos;
- `epsilon` representa o erro residual.

Neste projeto:

- `y` é a média móvel de 7 dias dos óbitos diários;
- as `features` são lags, médias móveis e variáveis temporais;
- o modelo tenta aproximar a tendência da série ao longo do tempo.

### Por que usar regressão linear aqui

Este projeto foi desenhado para mostrar um baseline clássico e interpretável.

A regressão linear faz sentido aqui porque:

- é simples de explicar;
- funciona bem como baseline para séries temporais com engenharia de lags;
- permite interpretar diretamente a ideia de combinação linear de tendências recentes;
- se conecta bem com `R²`, `MAE` e `RMSE`.

Ela não pretende substituir modelos temporais mais sofisticados, mas serve muito bem como ponto de partida reproduzível e didático.

### Engenharia de features

As features são criadas em [src/modeling.py](./src/modeling.py).

As principais incluem:

- `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`
- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_std_7`
- `trend_1_7`
- `day_of_week`
- `week_of_year`
- `month`

Essas variáveis ajudam o modelo a capturar:

- memória curta da série;
- tendência recente;
- padrão semanal;
- nível médio recente de óbitos.

### Alvo do modelo

O alvo não é a contagem bruta diária, e sim:

- `target_deaths_7d_avg`: média móvel de 7 dias dos óbitos diários

Essa escolha é importante porque reduz ruído de notificação e melhora a capacidade do modelo de capturar a tendência epidemiológica.

### Métricas de avaliação

#### `R²`

É a métrica principal do projeto.

O `R²` mede quanto da variância do alvo é explicada pelo modelo:

```text
R² = 1 - (SS_res / SS_tot)
```

Interpretação prática:

- `1.0` significa ajuste perfeito;
- valores próximos de `0` indicam baixo poder explicativo;
- valores negativos indicam desempenho pior do que prever simplesmente a média.

#### `MAE`

Mede o erro absoluto médio das previsões.

É útil para interpretar o erro em unidades mais diretas do problema.

#### `RMSE`

Penaliza mais fortemente erros grandes.

É útil para entender a estabilidade do ajuste quando existem oscilações mais fortes na série.

### Resultado atual

Na configuração atual do projeto:

- `Observations`: `346`
- `Train size`: `276`
- `Test size`: `70`
- `R²`: `0.9420`
- `MAE`: `0.4576`
- `RMSE`: `0.5639`

Esses resultados mostram que a regressão linear, com engenharia adequada de lags e suavização do alvo, consegue explicar bem a dinâmica recente da série agregada.

### Visualização

A comparação entre série real e previsão do modelo está nesta imagem:

![Actual vs predicted COVID-19 deaths](./assets/actual_vs_predicted.png)

O projeto também gera um heatmap de correlação entre as features e o alvo, o que ajuda a inspecionar dependências lineares e redundâncias entre variáveis temporais:

![Correlation heatmap](./assets/correlation_heatmap.png)

### Arquitetura do projeto

- [src/data_pipeline.py](./src/data_pipeline.py): leitura e agregação da base oficial
- [src/modeling.py](./src/modeling.py): engenharia de features, treino e avaliação
- [main.py](./main.py): execução principal
- [tests/test_pipeline.py](./tests/test_pipeline.py): teste automatizado
- [scripts/build_official_daily_dataset.py](./scripts/build_official_daily_dataset.py): script para reconstruir a série diária a partir do CSV oficial

### Como executar

```bash
cd "covid-brazil-deaths-linear-regression"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
python3 -m unittest discover -s tests -v
```

### Como reconstruir a base a partir da fonte oficial

```bash
curl -L -s "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-26-06-2025.csv" | python3 scripts/build_official_daily_dataset.py
```

### Como adaptar para produção ou estudo mais avançado

Possíveis extensões:

1. incluir múltiplos anos de dados oficiais;
2. prever série semanal em vez de diária;
3. comparar com `Ridge`, `Lasso` e modelos autoregressivos;
4. incluir decomposição temporal e variáveis sazonais mais fortes;
5. modelar casos por UF além do agregado nacional.

---

## EN

Traditional `machine learning` project that predicts the trend of COVID-19 deaths in Brazil using `Linear Regression` and `R²`, based on official Brazilian federal open data.

The project uses `SIVEP-Gripe / SRAG` microdata published by the Ministry of Health through the SUS open data portal and transforms that granular dataset into a national daily time series of COVID-19 deaths.

### Official data source

Source used:

- SUS Open Data Portal
- dataset: `SRAG 2019 to 2026`
- official resource used in this project: frozen `2024` file

Official links:

- [SRAG 2019 to 2026 dataset](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026)
- [Official 2024 CSV file](https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-26-06-2025.csv)

### Use case

The goal is to model the temporal dynamics of COVID-19 deaths in Brazil from the aggregated historical series itself.

Instead of using the noisy raw count of individual notifications, the project predicts the `7-day moving average` of daily deaths, which makes the series more stable and more suitable for linear regression.

### How the data was built

The official microdata is aggregated in [scripts/build_official_daily_dataset.py](./scripts/build_official_daily_dataset.py) and by equivalent logic in [src/data_pipeline.py](./src/data_pipeline.py).

Aggregation flow:

1. read the official `SIVEP-Gripe / SRAG` CSV;
2. filter records where `CLASSI_FIN = 5`, corresponding to COVID-19 SRAG;
3. filter records where `EVOLUCAO = 2`, corresponding to death;
4. use `DT_EVOLUCA` as the outcome date;
5. restrict the series to the year `2024`;
6. aggregate deaths by day at the national level.

The final series is saved to:

- [data/covid_brazil_daily_deaths_2024.csv](./data/covid_brazil_daily_deaths_2024.csv)

### What linear regression is

`Linear Regression` is a supervised model used to estimate a continuous target through linear relationships between explanatory features and an outcome.

In its simplest form, the equation is:

```text
y = beta_0 + beta_1 x_1 + beta_2 x_2 + ... + beta_n x_n + epsilon
```

Where:

- `y` is the value to predict;
- `x_1 ... x_n` are explanatory variables;
- `beta_0` is the intercept;
- `beta_1 ... beta_n` are learned coefficients;
- `epsilon` is the residual error.

In this project:

- `y` is the 7-day moving average of daily deaths;
- the `features` are lags, rolling statistics, and calendar variables;
- the model tries to approximate the recent trend of the series over time.

### Why linear regression was used here

This project was designed to show a classical and interpretable baseline.

Linear regression makes sense here because:

- it is simple to explain;
- it works well as a baseline for time series with lag engineering;
- it makes the idea of combining recent trends very transparent;
- it connects naturally to `R²`, `MAE`, and `RMSE`.

It is not meant to replace more sophisticated time-series models, but it works very well as a reproducible and educational starting point.

### Feature engineering

Features are created in [src/modeling.py](./src/modeling.py).

Main variables include:

- `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`
- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_std_7`
- `trend_1_7`
- `day_of_week`
- `week_of_year`
- `month`

These variables help the model capture:

- short-term memory;
- recent trend;
- weekly pattern;
- recent average level of deaths.

### Model target

The target is not the raw daily death count, but:

- `target_deaths_7d_avg`: the 7-day moving average of daily deaths

This choice is important because it reduces notification noise and improves the model's ability to capture the epidemiological trend.

### Evaluation metrics

#### `R²`

This is the main metric of the project.

`R²` measures how much of the target variance is explained by the model:

```text
R² = 1 - (SS_res / SS_tot)
```

Practical interpretation:

- `1.0` means perfect fit;
- values near `0` indicate weak explanatory power;
- negative values mean the model is worse than simply predicting the mean.

#### `MAE`

Measures the average absolute prediction error.

It is useful for interpreting the error scale in intuitive units.

#### `RMSE`

Penalizes larger errors more strongly.

It helps evaluate the stability of the fit when the series has sharper oscillations.

### Current result

In the current project configuration:

- `Observations`: `346`
- `Train size`: `276`
- `Test size`: `70`
- `R²`: `0.9420`
- `MAE`: `0.4576`
- `RMSE`: `0.5639`

These results show that linear regression, when combined with lag engineering and target smoothing, can explain the recent dynamics of the aggregated series quite well.

### Visualization

The comparison between actual and predicted values is shown here:

![Actual vs predicted COVID-19 deaths](./assets/actual_vs_predicted.png)

The project also generates a correlation heatmap across features and the target, which helps inspect linear dependencies and redundancy among temporal variables:

![Correlation heatmap](./assets/correlation_heatmap.png)

### Project structure

- [src/data_pipeline.py](./src/data_pipeline.py): official data reading and aggregation
- [src/modeling.py](./src/modeling.py): feature engineering, training, and evaluation
- [main.py](./main.py): main execution entry point
- [tests/test_pipeline.py](./tests/test_pipeline.py): automated test
- [scripts/build_official_daily_dataset.py](./scripts/build_official_daily_dataset.py): script to rebuild the daily series from the official CSV

### Run

```bash
cd "covid-brazil-deaths-linear-regression"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
python3 -m unittest discover -s tests -v
```

### Rebuild the dataset from the official source

```bash
curl -L -s "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-26-06-2025.csv" | python3 scripts/build_official_daily_dataset.py
```

### How to extend it

Possible next steps:

1. include multiple years of official data;
2. predict weekly instead of daily series;
3. compare against `Ridge`, `Lasso`, and autoregressive models;
4. add stronger seasonal features;
5. model deaths by state in addition to the national aggregate.
