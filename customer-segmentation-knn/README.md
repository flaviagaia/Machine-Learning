# Customer Segmentation with KNN Profiles

## PT-BR

Projeto de `machine learning` tradicional focado em encontrar clientes semelhantes usando `K-Nearest Neighbors (KNN)`. Em vez de prever uma variável alvo diretamente, este case usa distância em espaço vetorial para identificar perfis próximos, apoiar segmentação e criar uma camada simples de recomendação orientada por similaridade.

### Objetivo do case

A ideia central é responder perguntas como:

- quais clientes são mais parecidos com este cliente?
- que perfil comportamental esse cliente tem?
- quais segmentos aparecem ao redor dele?
- como usar proximidade para apoiar marketing, CRM ou personalização?

Esse tipo de abordagem é muito útil quando o negócio quer trabalhar com `lookalike profiles`, recuperação de clientes comparáveis ou agrupamento operacional baseado em comportamento.

### Por que usar KNN aqui

`KNN` é uma técnica simples e poderosa quando a noção de semelhança entre registros faz sentido.

Neste projeto, ele foi escolhido porque:

- é intuitivo para explicar;
- funciona bem em dados tabulares normalizados;
- permite recuperar perfis reais, não apenas scores abstratos;
- é útil para CRM, recomendação e segmentação assistida.

Em vez de dizer apenas “este cliente pertence ao segmento X”, o projeto permite dizer:

“estes são os clientes historicamente mais parecidos com ele”.

### Dados utilizados

O dataset é gerado de forma reproduzível em [src/data.py](./src/data.py) e salvo em [data/customer_profiles.csv](./data/customer_profiles.csv).

Cada registro representa um cliente e inclui variáveis como:

- `age`
- `annual_income`
- `purchase_frequency_monthly`
- `average_ticket`
- `digital_engagement_score`
- `return_rate_pct`
- `support_tickets_quarter`
- `loyalty_months`
- `discount_sensitivity_score`
- `web_visits_monthly`
- `app_sessions_weekly`
- `tenure_months`

Além disso, a base contém uma coluna de referência:

- `segment_label`

Ela não é usada pelo KNN como alvo supervisionado. Aqui ela funciona como uma forma de avaliar qualitativamente se os vizinhos encontrados fazem sentido em termos de proximidade de comportamento.

### Segmentos simulados

A base foi desenhada com quatro perfis sintéticos:

- `value_seekers`
- `loyal_midmarket`
- `premium_repeaters`
- `high_potential`

Esses segmentos diferem em renda, frequência de compra, ticket médio, engajamento digital, sensibilidade a desconto e tempo de relacionamento.

### Pipeline técnico

O pipeline principal está em [src/knn_profiles.py](./src/knn_profiles.py).

Fluxo implementado:

1. geração ou carregamento do dataset tabular;
2. seleção das variáveis numéricas que definem o perfil do cliente;
3. normalização com `StandardScaler`;
4. ajuste de `NearestNeighbors` com distância euclidiana;
5. recuperação dos `k` clientes mais parecidos com um cliente-alvo;
6. export de artefatos com resumo e exemplos de vizinhança.

### Bibliotecas e frameworks usados

- `pandas`
  Para manipulação dos perfis de clientes.
- `numpy`
  Para geração da base sintética.
- `scikit-learn`
  Para normalização, KNN e métrica de `silhouette`.
- `joblib`
  Para persistir o pipeline treinado.
- `unittest`
  Para validar o fluxo principal.

### Técnicas utilizadas e por quê

#### `StandardScaler`

Como KNN depende de distância, escalas diferentes entre variáveis podem distorcer a noção de proximidade.

Por exemplo:

- renda anual pode estar na casa de dezenas de milhares;
- taxa de retorno pode variar entre `0` e `18`;
- frequência de compra pode variar entre `1` e `10`.

Sem normalização, variáveis com valores maiores dominariam a distância.

#### `NearestNeighbors`

Foi usado em vez de um classificador supervisionado porque o foco aqui é:

- recuperar exemplos próximos;
- medir similaridade;
- explorar perfis;
- apoiar decisão orientada por vizinhança.

Ou seja, este projeto trata KNN como mecanismo de `profile retrieval`, não apenas como algoritmo de classificação.

#### `Silhouette Score`

O `silhouette score` foi incluído como uma leitura auxiliar da separação estrutural entre os grupos simulados no espaço padronizado.

Ele não substitui análise de negócio, mas ajuda a indicar se os perfis gerados estão razoavelmente organizados em regiões diferentes do espaço de features.

### Saída do projeto

Ao rodar [main.py](./main.py), o projeto gera:

- [artifacts/knn_profiles.joblib](./artifacts/knn_profiles.joblib)
- [artifacts/profile_summary.json](./artifacts/profile_summary.json)
- [artifacts/neighbor_examples.csv](./artifacts/neighbor_examples.csv)

Esses artefatos guardam:

- o pipeline com `StandardScaler + NearestNeighbors`
- um resumo do dataset
- um cliente de exemplo e seus vizinhos mais próximos

### Resultado atual

Na execução atual, o projeto mostra:

- `1200` clientes
- `4` segmentos simulados
- `silhouette score` em torno de `0.198`
- vizinhos próximos coerentes com o segmento do cliente consultado

No exemplo atual, o cliente `C0001`, do grupo `premium_repeaters`, recupera vizinhos do mesmo perfil, o que indica que a estrutura sintética e a normalização estão produzindo uma noção razoável de semelhança.

### Como executar

```bash
cd "customer-segmentation-knn"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
python3 -m unittest discover -s tests -v
```

### Como adaptar para dados reais

Para usar esse projeto em um cenário real, o caminho natural seria:

1. substituir a base sintética por dados reais de CRM ou e-commerce;
2. revisar quais features realmente representam comportamento de cliente;
3. testar diferentes métricas de distância, como `cosine` ou `manhattan`;
4. incluir filtros por canal, país ou categoria de cliente;
5. usar os vizinhos para recomendação, targeting ou recuperação de campanhas parecidas.

---

## EN

Traditional `machine learning` project focused on finding similar customers with `K-Nearest Neighbors (KNN)`. Instead of directly predicting a target variable, this case uses distance in feature space to identify close profiles, support segmentation, and build a lightweight similarity-driven recommendation layer.

### Use case

The core idea is to answer questions such as:

- which customers are most similar to this one?
- what behavioral profile does this customer have?
- which segments appear around this profile?
- how can proximity be used to support CRM, personalization, or marketing?

This kind of approach is useful when the business wants to work with `lookalike profiles`, comparable customer retrieval, or behavior-based operational segmentation.

### Why KNN was used here

`KNN` is a simple and powerful technique whenever similarity between records is meaningful.

It was chosen here because:

- it is intuitive to explain;
- it works well on normalized tabular data;
- it retrieves real profiles instead of only abstract scores;
- it is useful for CRM, recommendation, and assisted segmentation.

Instead of saying only “this customer belongs to segment X”, the project makes it possible to say:

“these are the historically most similar customers to this profile”.

### Data

The dataset is generated reproducibly in [src/data.py](./src/data.py) and saved to [data/customer_profiles.csv](./data/customer_profiles.csv).

Each row represents a customer and includes variables such as:

- `age`
- `annual_income`
- `purchase_frequency_monthly`
- `average_ticket`
- `digital_engagement_score`
- `return_rate_pct`
- `support_tickets_quarter`
- `loyalty_months`
- `discount_sensitivity_score`
- `web_visits_monthly`
- `app_sessions_weekly`
- `tenure_months`

The dataset also includes a reference label:

- `segment_label`

It is not used as a supervised target. Here it serves as a qualitative check to verify whether the retrieved neighbors make sense in terms of behavioral similarity.

### Simulated segments

The synthetic dataset was designed around four customer profiles:

- `value_seekers`
- `loyal_midmarket`
- `premium_repeaters`
- `high_potential`

These segments differ in income, purchase frequency, average ticket, digital engagement, discount sensitivity, and relationship length.

### Technical pipeline

The main pipeline lives in [src/knn_profiles.py](./src/knn_profiles.py).

Implemented flow:

1. generate or load the tabular dataset;
2. select the numeric features that define the customer profile;
3. normalize them with `StandardScaler`;
4. fit `NearestNeighbors` using Euclidean distance;
5. retrieve the `k` most similar customers for a target customer;
6. export artifacts with summary and neighbor examples.

### Libraries and frameworks used

- `pandas`
  For customer profile manipulation.
- `numpy`
  For synthetic data generation.
- `scikit-learn`
  For scaling, KNN, and `silhouette` evaluation.
- `joblib`
  For saving the fitted pipeline.
- `unittest`
  For validating the main workflow.

### Techniques used and why

#### `StandardScaler`

Because KNN is distance-based, different feature scales can distort similarity.

For example:

- annual income may be in the tens of thousands;
- return rate may range from `0` to `18`;
- purchase frequency may range from `1` to `10`.

Without scaling, larger-valued variables would dominate the distance computation.

#### `NearestNeighbors`

It was used instead of a supervised classifier because the main goal here is to:

- retrieve similar examples;
- measure similarity;
- explore profiles;
- support neighborhood-based decisions.

In other words, this project treats KNN as a `profile retrieval` mechanism, not only as a classification algorithm.

#### `Silhouette Score`

The `silhouette score` was added as an auxiliary signal of structural separation between the simulated groups in scaled feature space.

It does not replace business validation, but it helps indicate whether the generated profiles are reasonably organized into distinct regions of the feature space.

### Project outputs

Running [main.py](./main.py) generates:

- [artifacts/knn_profiles.joblib](./artifacts/knn_profiles.joblib)
- [artifacts/profile_summary.json](./artifacts/profile_summary.json)
- [artifacts/neighbor_examples.csv](./artifacts/neighbor_examples.csv)

These artifacts store:

- the fitted `StandardScaler + NearestNeighbors` pipeline
- a summary of the dataset
- one sample customer and its nearest neighbors

### Current result

In the current run, the project shows:

- `1200` customers
- `4` simulated segments
- a `silhouette score` around `0.198`
- coherent nearest neighbors for the sample customer

In the current example, customer `C0001`, from the `premium_repeaters` group, retrieves neighbors from the same profile, which indicates that the synthetic structure and normalization are producing a reasonable notion of similarity.

### Run

```bash
cd "customer-segmentation-knn"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
python3 -m unittest discover -s tests -v
```

### How to adapt it to real data

To use this project in a real-world setting, the natural path would be:

1. replace the synthetic dataset with real CRM or e-commerce data;
2. review which features truly represent customer behavior;
3. test other distance metrics, such as `cosine` or `manhattan`;
4. include filters by channel, country, or customer type;
5. use neighbors for recommendation, targeting, or campaign retrieval.
