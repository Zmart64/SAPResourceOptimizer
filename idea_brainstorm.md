## 1  Cluster-Level Load Forecasting

### Goal
Predict the total CPU utilisation and total memory utilisation of the whole cluster 5–60 minutes into the future.

* Make day-ahead decisions about power-capping and cooling
* Decide when to buy spot instances or shift batch workloads
* Perform what-if analysis for fail-over scenarios


### What we could need
1. Load the **Node** table.  It holds one record per node per minute.  Aggregate across nodes so that every timestamp becomes a single row with `cpu_sum`, `cpu_mean`, `mem_sum`, `mem_mean`.
2. Optionally enrich each row with cluster-wide traffic indicators such as mean `providerRPC_MCR` taken from **MSRTMCR**.

### Modelling Options
* Baselines: last-value persistence; seasonal naive (same minute previous day).
* Classical: SARIMA, Facebook Prophet or ETS with daily seasonality.
* Deep learning: Temporal Convolutional Network, N-BEATS or Temporal Fusion Transformer with quantile output heads for uncertainty.

**I think relatively low effort?**

## 2  Hot-Node Early Warning

### Goal
Identify, every minute, which physical nodes will exceed 80 % CPU utilisation within the next 10 minutes so that orchestrators can migrate or throttle workloads before saturation.

A reliable early-warning system directly improves service reliability and user experience.  Furthermore, the prediction granularity (tens of thousands of nodes) shows that the system scales beyond toy scenarios.

### Data and Pre-processing
1. **Sampling strategy**: randomly select about 3 000 nodes (≈8 %) to reduce data volume while preserving diversity.  All rows for these nodes stay in the dataset.
2. **Join traffic context**: use the `nodeid` field to join **MSResource** and **MSRTMCR** so that for each node-minute you know how much call traffic its resident containers handle.  Aggregate call rate to obtain `rpc_mcr_sum`, `http_mcr_sum`, etc.
3. **Feature construction**: create lag features for CPU, memory and traffic (values from the previous 5 minutes) plus cyclic encodings for hour-of-day and day-of-week.
4. **Labeling**: assign label 1 if `cpu_utilisation` ten minutes ahead exceeds 0.8, else 0.  Positive class frequency is on the order of 5 % so class imbalance handling is needed.

### Modelling Options
* Gradient-boosted decision trees (XGBoost, LightGBM) with class-weighted loss.
* Sequence models such as GRU or LSTM that ingest the past 10 minutes of features.
* Post-processing: choose probability threshold to maximise Precision@N, where N is the number of nodes an operator can reasonably inspect.

**medium effort?**

## 3  Service-Graph GNN Forecasting

### Goal
Leverage the dynamic micro-service call graph to predict the CPU utilisation of every stateless service 5 minutes into the future, using for example a Temporal Graph Neural Network (GNN).

In micro-service architectures, load waves propagate along dependency edges: a spike at an upstream service often foreshadows a spike downstream.  Classic univariate forecasts ignore this structure, whereas a GNN can learn it.  The project demonstrates modern graph-learning techniques on a real, massive production graph, providing research-grade novelty.

### Data and Pre-processing
1. **Temporal slicing**: restrict to one representative hour per day (e.g., 12:00-13:00) to limit the number of snapshots to ~780.
2. **Graph construction**: for each minute build a directed graph where vertices are `msname` (collapse multiple pods) and edge weight equals the minute’s call rate from **MSRTMCR** or **MSCallGraph**.
3. **Node features**: previous CPU value, incoming/outgoing call rate, static degrees, betweenness centrality.
4. **Normalisation**: z-score features within each snapshot to stabilise training.
5. **Storage**: save snapshots as PyTorch Geometric `Data` objects or in DGL format for fast loading.

### Modelling Options
* Diffusion Convolutional Recurrent Neural Network (DCRNN).
* Temporal Graph Attention Network with positional encodings.
* Loss: Mean Absolute Error plus an edge-weighted smoothness regulariser to encourage similar predictions across strongly connected services.

### Evaluation and Demonstration
* Metrics: MAE and RMSE averaged over services (micro) and averaged over snapshots (macro).
* Deliverable: interactive Graph-viz plot where hovering a node shows actual vs predicted CPU trajectory; highlight paths where error exceeded a threshold to illustrate missed cascades.

**Probably high effort**


## 4  Replica-Count Recommender (Autoscaling Advisor)

### Goal
Recommend, for each micro-service, the number of pod replicas required in the next 10 minutes so that its 95th-percentile CPU stays below 70 % utilisation.

Horizontal Pod Autoscalers (HPA) in Kubernetes typically use simple rule-based logic (average CPU over the last 1–5 minutes).  A data-driven recommender could reduce over-provisioning while still guarding against overload, directly translating prediction quality into cost savings.

### Data and Pre-processing
1. **Service selection**: focus on the 500 most active services to keep label generation controllable.
2. **Label engineering**: for each service-minute compute `required_replicas = ceil(cpu_util * current_replicas / 0.7)` assuming linear scaling efficiency.
3. **Features**: past CPU values, current replica count, aggregated call-rate, hour-of-day, day-of-week, rolling variance of load.
4. **Offline simulator**: build a simple simulator that applies a recommended replica count, caps CPU accordingly, and records wasted core-seconds vs overload events.  This forms your evaluation harness.

### Modelling Options
* Regression trees (XGBoost) predicting an integer, with monotonicity constraints so recommendations never decrease when load increases.
* Reinforcement learning agent (Deep Q-Network) where the action is an integer change in replica count and reward balances cost of replicas against SLO violation penalty.

### Evaluation and Demonstration
* Offline simulation curves showing actual vs recommended replicas for selected spike periods.
* Aggregate metrics: total core-seconds saved, number and severity of overload violations.
* Optional “what-if” slider in the dashboard allowing users to tune the cost vs risk trade-off.

**medium-high. effort**


## 5  Workload Archetype Discovery and Archetype-level Forecasting

### Goal
Cluster micro-services into behavioural archetypes (e.g., periodic, bursty, stable) and train a single lightweight forecast model per archetype instead of thousands of individual models.

Operating teams reason better about classes of services than individual IDs.  Archetype-level modelling reduces maintenance overhead and sheds light on fundamental workload patterns present in the organisation.

### Data and Pre-processing
1. Down-sample each service’s CPU series to 5-minute intervals and extract a 48-point daily vector.
2. Feature extraction with TS-Fresh, FFT magnitude spectrum, or Symbolic Aggregate approXimation (SAX) bag-of-words.
3. Dimensionality reduction via PCA or UMAP to 20 dimensions.
4. Density-based clustering such as HDBSCAN (automatically picks cluster count) or classic k-Medoids.
5. For every cluster, fit a small ARIMA or single-layer LSTM using all member series.

### Modelling Options
Fairly light-weight; main modelling lies in the clustering choice.  Forecast models can be the same ones as Idea 1 but trained on normalised data per archetype.


**Effort: low to medium**

---

## Summary of Effort versus Potential Impact

| Idea | Estimated Effort (★ low – ★★★★ high) | Potential Impact / Novelty |
|------|---------------------------------------|----------------------------|
|1 Cluster Load Forecast | ★★ | ★★ |
|2 Hot-Node Warning     | ★★ | ★★½ |
|3 Service GNN          | ★★★★ | ★★★★ |
|4 Replica Recommender  | ★★★ | ★★★ |
|5 Archetype Discovery  | ★★ | ★★ |

