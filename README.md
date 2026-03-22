# Machine Learning

## PT-BR

Este repositório reúne projetos práticos de `machine learning` com foco em modelos tradicionais, problemas tabulares e casos de uso de negócio. A proposta é organizar experimentos e aplicações que mostrem não apenas treino de modelo, mas também boas práticas de preparação de dados, avaliação, interpretação e uso em cenários reais.

## O que é machine learning

`Machine learning` é uma área da inteligência artificial voltada para construir sistemas que aprendem padrões a partir de dados.

Em vez de programar todas as regras manualmente, o modelo observa exemplos históricos e aprende relações estatísticas entre variáveis de entrada e resultados esperados. Depois disso, ele pode:

- classificar novos casos;
- prever valores numéricos;
- agrupar perfis semelhantes;
- ranquear itens;
- detectar padrões ou anomalias.

Na prática, `machine learning` é muito usado quando o problema é complexo demais para ser resolvido com regras fixas ou quando há grande volume de dados e histórico suficiente para aprender com exemplos.

## Para que machine learning serve

Modelos de `machine learning` podem ser usados em diferentes contextos de negócio, por exemplo:

- previsão de churn;
- risco de crédito;
- detecção de fraude;
- segmentação de clientes;
- previsão de demanda;
- recomendação de produtos;
- classificação de documentos;
- scoring comercial;
- manutenção preditiva;
- análise de comportamento de usuários.

Ou seja, o objetivo do `machine learning` não é apenas “prever”, mas apoiar tomada de decisão com base em padrões extraídos dos dados.

## Tipos principais de machine learning

### Aprendizado supervisionado

É usado quando existe uma variável alvo conhecida.

Exemplos:

- classificar se um cliente vai dar default ou não;
- prever o preço de um imóvel;
- estimar a chance de cancelamento de um serviço.

Problemas mais comuns:

- classificação
- regressão

### Aprendizado não supervisionado

É usado quando não existe uma variável alvo explícita e o objetivo é descobrir estrutura nos dados.

Exemplos:

- segmentar clientes;
- encontrar grupos com comportamento parecido;
- detectar padrões incomuns.

Problemas mais comuns:

- clusterização
- redução de dimensionalidade
- detecção de anomalias

### Aprendizado por reforço

É usado quando um agente aprende por tentativa e erro, recebendo recompensas ou penalidades ao longo do tempo.

Exemplos:

- otimização de decisão sequencial;
- controle de sistemas;
- recomendação adaptativa;
- ambientes de simulação.

Embora seja uma área importante, este repositório tende a focar mais em aprendizado supervisionado e não supervisionado aplicado a dados tabulares e problemas de negócio.

## Modelos tradicionais de machine learning

Quando se fala em “modelos tradicionais”, normalmente estamos nos referindo a algoritmos clássicos que continuam extremamente relevantes em produção, especialmente para dados estruturados.

Alguns exemplos importantes:

- `Linear Regression`
- `Logistic Regression`
- `Decision Tree`
- `Random Forest`
- `Gradient Boosting`
- `XGBoost`
- `LightGBM`
- `CatBoost`
- `Support Vector Machine`
- `K-Nearest Neighbors`
- `Naive Bayes`
- `KMeans`
- `DBSCAN`
- `PCA`

Esses modelos continuam muito úteis porque:

- funcionam muito bem em dados tabulares;
- treinam rápido;
- são mais simples de explicar;
- exigem menos infraestrutura do que redes neurais profundas;
- muitas vezes entregam excelente performance em cenários reais.

## Como pensar um projeto de machine learning

Um bom projeto de `machine learning` não é só “treinar um algoritmo”. Em geral, o pipeline correto envolve:

1. entender o problema de negócio;
2. definir a variável alvo;
3. preparar e limpar os dados;
4. criar features úteis;
5. dividir treino, validação e teste;
6. comparar múltiplos modelos;
7. escolher métricas coerentes com o problema;
8. calibrar thresholds ou hiperparâmetros quando necessário;
9. interpretar resultados;
10. pensar em deploy, monitoramento e reuso.

Em muitos casos, o maior ganho não vem do modelo “mais sofisticado”, mas da qualidade da base, da feature engineering e da avaliação correta.

## Métricas comuns

As métricas dependem do tipo de problema.

Para classificação:

- `Accuracy`
- `Precision`
- `Recall`
- `F1-score`
- `ROC-AUC`
- `PR-AUC`

Para regressão:

- `MAE`
- `RMSE`
- `R²`

Para clusterização:

- `Silhouette Score`
- avaliação qualitativa dos grupos

Escolher a métrica certa é parte central do projeto. Em problemas desbalanceados, por exemplo, usar apenas `accuracy` costuma levar a conclusões erradas.

## Quando usar modelos tradicionais

Modelos tradicionais são especialmente indicados quando:

- o problema usa dados tabulares;
- a base não é gigantesca;
- a interpretabilidade importa;
- o tempo de desenvolvimento precisa ser rápido;
- a solução precisa ser robusta e simples de manter.

Eles são muito comuns em:

- bancos;
- fintechs;
- seguros;
- varejo;
- telecom;
- marketing analytics;
- crédito e risco;
- operações e supply chain.

## Estrutura esperada deste repositório

Este repositório foi criado para armazenar projetos de `machine learning` em uma estrutura organizada, por exemplo:

- `credit-risk-scoring`
- `churn-prediction-lab`
- `customer-segmentation`
- `house-price-regression`
- `fraud-detection-baseline`

Cada projeto idealmente terá:

- `README` próprio;
- código reproduzível;
- dados de exemplo ou instrução de carga;
- métricas e artefatos;
- explicação técnica das escolhas de modelagem.

## Como usar este repositório

Se você está começando:

1. leia esta introdução;
2. escolha um problema supervisionado ou não supervisionado;
3. abra o projeto correspondente;
4. rode o pipeline localmente;
5. compare modelos e entenda as métricas;
6. depois adapte a estrutura para seus próprios dados.

## Objetivo deste repositório

O foco aqui é construir uma coleção de projetos que mostrem:

- fundamentos sólidos de `machine learning`;
- capacidade de modelar problemas reais;
- boas práticas de avaliação;
- clareza técnica na explicação;
- organização de portfólio voltada para recrutadores e entrevistas técnicas.

---

## EN

This repository groups practical `machine learning` projects focused on traditional models, tabular problems, and business-oriented use cases. The goal is to organize experiments and applications that show not only model training, but also good practices in data preparation, evaluation, interpretation, and real-world usage.

## What machine learning is

`Machine learning` is a field of artificial intelligence focused on building systems that learn patterns from data.

Instead of manually coding every rule, the model observes historical examples and learns statistical relationships between input variables and expected outcomes. After that, it can:

- classify new cases;
- predict numerical values;
- group similar profiles;
- rank items;
- detect patterns or anomalies.

In practice, `machine learning` becomes valuable when the problem is too complex for fixed rules or when there is enough historical data to learn from examples.

## What machine learning is used for

`Machine learning` models can be applied to many business contexts, for example:

- churn prediction;
- credit risk;
- fraud detection;
- customer segmentation;
- demand forecasting;
- product recommendation;
- document classification;
- lead scoring;
- predictive maintenance;
- user behavior analysis.

In other words, the goal of `machine learning` is not only to “predict”, but to support decision-making based on patterns extracted from data.

## Main types of machine learning

### Supervised learning

Used when there is a known target variable.

Examples:

- classify whether a borrower will default;
- predict house prices;
- estimate the probability of customer churn.

Most common problems:

- classification
- regression

### Unsupervised learning

Used when there is no explicit target variable and the goal is to discover structure in the data.

Examples:

- segment customers;
- find behavior-based groups;
- detect unusual patterns.

Most common problems:

- clustering
- dimensionality reduction
- anomaly detection

### Reinforcement learning

Used when an agent learns through trial and error by receiving rewards or penalties over time.

Examples:

- sequential decision optimization;
- control systems;
- adaptive recommendation;
- simulation environments.

While reinforcement learning is important, this repository is mainly focused on supervised and unsupervised learning for tabular and business data.

## Traditional machine learning models

When people refer to “traditional models”, they usually mean classical algorithms that are still extremely relevant in production, especially for structured data.

Important examples include:

- `Linear Regression`
- `Logistic Regression`
- `Decision Tree`
- `Random Forest`
- `Gradient Boosting`
- `XGBoost`
- `LightGBM`
- `CatBoost`
- `Support Vector Machine`
- `K-Nearest Neighbors`
- `Naive Bayes`
- `KMeans`
- `DBSCAN`
- `PCA`

These models remain very useful because:

- they work very well on tabular data;
- they train quickly;
- they are easier to explain;
- they require less infrastructure than deep neural networks;
- they often deliver excellent performance in real business scenarios.

## How to think about a machine learning project

A good `machine learning` project is not just about “training an algorithm”. In most cases, the correct pipeline includes:

1. understanding the business problem;
2. defining the target variable;
3. preparing and cleaning the data;
4. creating useful features;
5. splitting data into train, validation, and test;
6. comparing multiple models;
7. choosing metrics that match the problem;
8. calibrating thresholds or hyperparameters when needed;
9. interpreting the results;
10. thinking about deployment, monitoring, and reuse.

In many cases, the biggest gains come not from the “most sophisticated” model, but from data quality, feature engineering, and correct evaluation.

## Common metrics

Metrics depend on the problem type.

For classification:

- `Accuracy`
- `Precision`
- `Recall`
- `F1-score`
- `ROC-AUC`
- `PR-AUC`

For regression:

- `MAE`
- `RMSE`
- `R²`

For clustering:

- `Silhouette Score`
- qualitative group analysis

Choosing the right metric is a central part of the project. In imbalanced problems, for example, relying only on `accuracy` usually leads to misleading conclusions.

## When to use traditional models

Traditional models are especially suitable when:

- the problem uses tabular data;
- the dataset is not extremely large;
- interpretability matters;
- development time must be fast;
- the solution needs to be robust and easy to maintain.

They are widely used in:

- banking;
- fintech;
- insurance;
- retail;
- telecom;
- marketing analytics;
- credit and risk;
- operations and supply chain.

## Expected structure of this repository

This repository was created to store `machine learning` projects in an organized way, for example:

- `credit-risk-scoring`
- `churn-prediction-lab`
- `customer-segmentation`
- `house-price-regression`
- `fraud-detection-baseline`

Each project should ideally include:

- its own `README`;
- reproducible code;
- sample data or loading instructions;
- metrics and artifacts;
- technical explanation of the modeling choices.

## How to use this repository

If you are just getting started:

1. read this introduction;
2. choose a supervised or unsupervised problem;
3. open the corresponding project;
4. run the local pipeline;
5. compare models and understand the metrics;
6. then adapt the structure to your own data.

## Goal of this repository

The goal here is to build a collection of projects that show:

- strong `machine learning` fundamentals;
- ability to model real problems;
- solid evaluation practices;
- technical clarity in explanations;
- portfolio organization that works well for recruiters and technical interviews.
