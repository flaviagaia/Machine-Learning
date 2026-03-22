# Credit Risk Scoring

## PT-BR

Projeto de `machine learning` tradicional para prever inadimplência em crédito a partir de dados tabulares. O foco aqui é mostrar um pipeline clássico e bem estruturado de risco: geração ou ingestão de dados, preprocessamento, comparação entre modelos, ajuste de threshold e avaliação com métricas apropriadas para classificação desbalanceada.

### Objetivo do case

Em concessão de crédito, o problema central é estimar a probabilidade de um cliente entrar em default. Esse tipo de caso costuma envolver:

- variáveis numéricas e categóricas;
- classes desbalanceadas;
- necessidade de interpretabilidade;
- decisão baseada em probabilidade, não apenas em classe final.

Este projeto foi construído para reproduzir esse fluxo com `scikit-learn`, usando um conjunto sintético mas realista, com variáveis que normalmente aparecem em problemas de risco.

### Dados utilizados

O dataset é gerado de forma reproduzível em [src/data.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/src/data.py) e salvo em [data/credit_risk_dataset.csv](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/data/credit_risk_dataset.csv).

As features simuladas incluem:

- `age`
- `annual_income`
- `employment_years`
- `credit_history_years`
- `loan_amount`
- `loan_term_months`
- `interest_rate`
- `monthly_debt`
- `debt_to_income`
- `credit_utilization`
- `late_payments_12m`
- `recent_credit_inquiries`
- `existing_loans`
- `savings_balance`
- `home_ownership`
- `loan_purpose`
- `employment_type`
- `region`

A variável alvo é:

- `defaulted`: `1` para inadimplência e `0` para cliente adimplente

Embora o conjunto seja sintético, a lógica de geração foi desenhada para refletir relações plausíveis de negócio. Por exemplo:

- `debt_to_income` e `credit_utilization` aumentam o risco;
- renda, saldo em poupança e histórico de crédito tendem a reduzir o risco;
- muitos atrasos recentes e muitas consultas de crédito aumentam a chance de default;
- tipo de vínculo empregatício e finalidade do empréstimo afetam o score de risco.

### Pipeline técnico

O pipeline principal está em [src/train.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/src/train.py) e a modelagem em [src/modeling.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/src/modeling.py).

Fluxo implementado:

1. geração ou atualização do dataset sintético;
2. separação estratificada entre treino e validação;
3. preprocessamento com `ColumnTransformer`;
4. comparação entre múltiplos modelos clássicos;
5. seleção do melhor modelo por desempenho em validação;
6. ajuste de threshold para maximizar `F1-score`;
7. export de artefatos e métricas.

### Bibliotecas e frameworks usados

- `pandas`
  Para manipulação tabular e leitura/escrita do dataset.
- `numpy`
  Para geração probabilística do conjunto sintético e operações numéricas.
- `scikit-learn`
  Para preprocessamento, treino, split estratificado, métricas e pipelines.
- `joblib`
  Para persistir o melhor modelo treinado.
- `unittest`
  Para validar a execução end-to-end do pipeline.

### Por que essas técnicas foram escolhidas

#### `ColumnTransformer`

Foi usado para tratar numéricos e categóricos de forma separada. Isso é importante em dados tabulares reais, onde:

- variáveis numéricas precisam de imputação e normalização;
- variáveis categóricas precisam de codificação segura e consistente.

#### `SimpleImputer`

Mesmo que a base sintética esteja completa, a pipeline já foi preparada como se fosse um cenário real de produção, em que valores faltantes podem existir.

#### `StandardScaler`

Foi aplicado sobre variáveis numéricas para estabilizar modelos lineares, especialmente a regressão logística.

#### `OneHotEncoder`

Foi usado para transformar variáveis categóricas em representação numérica sem impor ordinalidade artificial.

### Modelos comparados

O projeto compara três abordagens tradicionais:

#### `Logistic Regression`

Escolhida por ser um baseline clássico de risco de crédito. É simples, rápida, interpretável e muito usada em cenários regulados.

#### `Random Forest`

Incluída para capturar relações não lineares e interações entre variáveis sem exigir engenharia manual muito sofisticada.

#### `Gradient Boosting`

Incluída como um baseline tabular mais forte, capaz de capturar padrões mais complexos do que a regressão logística.

### Estratégia de seleção do melhor modelo

O melhor modelo é escolhido a partir do conjunto de validação, considerando:

- `F1-score`
- `PR-AUC`
- `ROC-AUC`

O critério principal é `F1-score`, porque o problema é binário e tem assimetria entre classes. Em risco, olhar apenas para acurácia costuma ser uma armadilha.

### Threshold tuning

O projeto não assume que `0.5` é o melhor cutoff para transformar probabilidade em classe final.

Em vez disso, ele calcula a curva de `precision-recall` e escolhe o threshold que maximiza `F1-score` no conjunto de validação. Essa é uma decisão importante porque:

- em crédito, probabilidade é mais útil do que classe bruta;
- o threshold deve refletir o objetivo do negócio;
- modelos com bom ranking podem performar mal se o cutoff for mal escolhido.

O threshold selecionado fica salvo junto com as métricas.

### Métricas usadas e por quê

#### `F1-score`

É a métrica principal porque equilibra `precision` e `recall`, o que faz sentido quando falsos positivos e falsos negativos têm custo relevante.

#### `Precision`

Ajuda a medir quantos clientes sinalizados como risco realmente apresentam risco.

#### `Recall`

Importante para saber quanto do risco real o modelo consegue capturar.

#### `ROC-AUC`

Mostra a capacidade global de separação entre bons e maus clientes ao longo de vários thresholds.

#### `PR-AUC`

É especialmente útil quando o problema tende ao desbalanceamento, porque enfatiza melhor o comportamento da classe positiva.

### Artefatos gerados

Ao rodar [main.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/main.py), o projeto gera:

- [artifacts/best_credit_risk_model.joblib](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/artifacts/best_credit_risk_model.joblib)
- [artifacts/metrics.json](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/artifacts/metrics.json)
- [artifacts/test_predictions.csv](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/artifacts/test_predictions.csv)

### Resultado atual

Na execução atual, o melhor modelo foi `Logistic Regression`, com:

- `F1-score`: aproximadamente `0.695`
- `ROC-AUC`: aproximadamente `0.788`
- `PR-AUC`: aproximadamente `0.734`

Isso é coerente com um baseline clássico de risco em um conjunto tabular sintético de porte médio.

### Como executar

```bash
cd "/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit Risk Scoring"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
python3 -m unittest discover -s tests -v
```

### Como adaptar para dados reais

Para usar esse projeto em um contexto real, os principais passos seriam:

1. substituir a geração sintética pela leitura de uma base real;
2. revisar as variáveis para incluir informações disponíveis no processo real de crédito;
3. ajustar a estratégia de split para refletir tempo, safra ou política de originação;
4. calibrar o threshold com base em custo de negócio e política de aprovação;
5. incluir explicabilidade com `feature importance`, `SHAP` ou análise de coeficientes;
6. monitorar drift e performance por coortes.

---

## EN

Traditional `machine learning` project for credit default prediction using tabular data. The goal is to show a strong classical risk pipeline: data generation or ingestion, preprocessing, model comparison, threshold tuning, and evaluation with metrics that make sense for imbalanced classification.

### Use case

In credit underwriting, the core problem is estimating the probability that a borrower will default. This usually involves:

- numerical and categorical variables;
- class imbalance;
- interpretability requirements;
- decision-making based on probability, not only the final class.

This project reproduces that workflow with `scikit-learn`, using a synthetic but realistic dataset built around variables commonly found in risk problems.

### Data

The dataset is generated reproducibly in [src/data.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/src/data.py) and saved to [data/credit_risk_dataset.csv](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/data/credit_risk_dataset.csv).

The simulated features include:

- `age`
- `annual_income`
- `employment_years`
- `credit_history_years`
- `loan_amount`
- `loan_term_months`
- `interest_rate`
- `monthly_debt`
- `debt_to_income`
- `credit_utilization`
- `late_payments_12m`
- `recent_credit_inquiries`
- `existing_loans`
- `savings_balance`
- `home_ownership`
- `loan_purpose`
- `employment_type`
- `region`

Target:

- `defaulted`: `1` for default and `0` for non-default

Even though the dataset is synthetic, its generation logic was designed to reflect plausible business relationships:

- higher debt-to-income and utilization increase risk;
- income, savings, and longer credit history reduce risk;
- more late payments and more credit inquiries increase default probability;
- employment type and loan purpose also shift the risk score.

### Technical pipeline

The training logic lives in [src/train.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/src/train.py) and the model definitions in [src/modeling.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/src/modeling.py).

Implemented flow:

1. generate or refresh the synthetic dataset;
2. perform a stratified train/validation split;
3. preprocess features with `ColumnTransformer`;
4. compare multiple classical models;
5. select the best candidate on validation;
6. tune the classification threshold to maximize `F1-score`;
7. export model artifacts and metrics.

### Libraries and frameworks used

- `pandas`
  For tabular manipulation and dataset persistence.
- `numpy`
  For probabilistic synthetic data generation and numeric operations.
- `scikit-learn`
  For preprocessing, training, stratified splitting, metrics, and pipelines.
- `joblib`
  For saving the best trained model.
- `unittest`
  For validating the end-to-end pipeline.

### Why these techniques were used

#### `ColumnTransformer`

Used to process numeric and categorical features separately. This mirrors real tabular ML work, where:

- numeric features need imputation and scaling;
- categorical features need safe and consistent encoding.

#### `SimpleImputer`

Even though the synthetic dataset is complete, the pipeline was intentionally designed as if it were dealing with a real production dataset where missing values may appear.

#### `StandardScaler`

Applied to numeric variables to stabilize linear models, especially logistic regression.

#### `OneHotEncoder`

Used to convert categorical fields into numeric representations without imposing artificial ordinal structure.

### Compared models

The project compares three classical approaches:

#### `Logistic Regression`

Chosen as the core credit-risk baseline. It is simple, fast, interpretable, and widely used in regulated environments.

#### `Random Forest`

Included to capture non-linear interactions without requiring complex manual feature engineering.

#### `Gradient Boosting`

Included as a stronger tabular baseline capable of learning more complex relationships than a purely linear model.

### Model selection strategy

The best model is selected on the validation set using:

- `F1-score`
- `PR-AUC`
- `ROC-AUC`

The primary criterion is `F1-score`, because this is a binary classification problem with asymmetric business costs. Accuracy alone would be misleading here.

### Threshold tuning

The project does not assume that `0.5` is the best cutoff for converting probabilities into final labels.

Instead, it computes the `precision-recall` curve and selects the threshold that maximizes `F1-score` on validation. This is important because:

- in credit risk, probabilities are often more useful than hard classes;
- the threshold should reflect business objectives;
- a good ranking model can still underperform if the cutoff is poorly chosen.

The selected threshold is stored with the final metrics.

### Metrics and why they matter

#### `F1-score`

Main metric, because it balances `precision` and `recall`, which is useful when both false positives and false negatives matter.

#### `Precision`

Measures how many applicants flagged as risky are actually risky.

#### `Recall`

Measures how much of the true risk the model is able to capture.

#### `ROC-AUC`

Shows the model's ranking quality across multiple thresholds.

#### `PR-AUC`

Especially useful under class imbalance because it emphasizes performance on the positive class.

### Generated artifacts

Running [main.py](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/main.py) creates:

- [artifacts/best_credit_risk_model.joblib](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/artifacts/best_credit_risk_model.joblib)
- [artifacts/metrics.json](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/artifacts/metrics.json)
- [artifacts/test_predictions.csv](/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit%20Risk%20Scoring/artifacts/test_predictions.csv)

### Current result

In the current run, the best model is `Logistic Regression`, with:

- `F1-score`: approximately `0.695`
- `ROC-AUC`: approximately `0.788`
- `PR-AUC`: approximately `0.734`

This is a reasonable outcome for a classical credit-risk baseline over a medium-sized synthetic tabular dataset.

### Run

```bash
cd "/Users/flaviagaia/Documents/CV_FLAVIA_CODEX/Credit Risk Scoring"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
python3 -m unittest discover -s tests -v
```

### How to adapt it to real data

To move this project into a real-world credit setting, the main steps would be:

1. replace synthetic generation with a real data source;
2. revise features based on the actual underwriting process;
3. adjust the split strategy to reflect time, cohort, or origination policy;
4. calibrate the threshold according to business cost and approval policy;
5. add explainability through feature importance, SHAP, or coefficient analysis;
6. monitor drift and performance by cohort.
