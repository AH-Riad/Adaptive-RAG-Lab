# D²RAG Publication Protocol v2

## Purpose

This protocol freezes the experimental procedure needed for a defensible conference result and a later journal extension.

## V7 changes

V7 makes four methodological corrections.

1. Diagnosis-first feedback: query/strategy mismatches are diagnosed before generic ranking and score heuristics. This prevents a generic ranking label from hiding a structural strategy failure.

2. Hierarchical policy backoff: the action policy first uses the full diagnosis-aware state, then backs off to progressively less specific states when the exact state has insufficient support.

3. Learned-only post-retrieval adaptation: the primary D²RAG controller no longer executes an unpublished diagnostic heuristic when no supported learned policy exists. The default is KEEP. This makes the measured adaptation attributable to the frozen dev policy.

4. Metric separation: strategy adaptation is evaluated with ranking quality at a fixed K, while Top-K adaptation is evaluated with retrieval breadth/cost metrics. Increasing K is not credited with improving nDCG@5 merely because more documents were retrieved.

## Data separation

- Train/calibration: existing FiQA training material and development split where already defined by the project.
- Policy construction: FiQA dev only.
- All policy thresholds, backoff support levels, and cost settings are frozen before held-out test evaluation.
- Held-out FiQA test is not used for policy construction, threshold search, or design selection.

## Primary baselines

- Dense
- BM25S
- Hybrid
- Single-stage adaptive
- D²RAG V7

## Primary retrieval metrics

- Recall@5
- MRR@5
- nDCG@5

## Adaptive metrics

- adaptation rate
- strategy-switch rate
- Top-K expansion rate
- average attempts
- average retrieval cost proxy
- adaptation nDCG@5 delta for strategy-changing actions
- Recall@EvaluatedK delta for Top-K expansion
- nDCG@EvaluatedK delta for Top-K expansion
- successful adaptation rate
- harmful adaptation rate
- policy coverage
- exact-policy vs backoff-policy usage
- action distribution by diagnosis
- KEEP rate

## Evidence metrics

- Brier score
- ECE
- acceptance precision
- acceptance recall
- false acceptance rate
- confidence separation

## Policy state

The runtime state is:

`(query_type, current_strategy, confidence_bucket, diagnosis, current_top_k)`

The diagnosis is generated without qrels at runtime.

## Policy selection rule

An action can be selected from development data only when:

- support is sufficient for the current policy level;
- mean counterfactual gain is at least 0.02;
- positive-gain rate is at least 0.55;
- harmful-gain rate is at most 0.25.

Otherwise the controller backs off to a less specific state. If no supported state exists, D²RAG keeps the current plan.

## Top-K interpretation

Top-K expansion is a breadth/cost action. It should be interpreted using Recall@K, nDCG@K, and retrieval-cost overhead. It is not expected to improve nDCG@5 when the top-ranked order is unchanged.

## Required ablations

1. Dense baseline
2. BM25S baseline
3. Hybrid baseline
4. Single-stage adaptive
5. D²RAG pre-retrieval only
6. D²RAG post-retrieval only
7. D²RAG dual-stage
8. D²RAG policy without diagnosis backoff
9. D²RAG without learned feedback actions
10. D²RAG cost-aware policy vs quality-only policy

## Required statistical analysis

Report paired per-query differences against the strongest fixed baseline and use paired bootstrap confidence intervals for primary metrics. Report the number of queries, random seed where applicable, and the exact frozen configuration.

## Conference paper structure

1. Introduction
2. Related Work
3. Problem Definition
4. D²RAG Architecture
5. Diagnosis-Conditioned Feedback Policy
6. Experimental Setup
7. Results
8. Ablation Study
9. Error Analysis
10. Limitations
11. Conclusion

## Journal extension

The journal version should add at least one additional benchmark, a broader ablation matrix, multi-seed analysis where stochastic components exist, latency/cost measurements, deeper error taxonomy, calibration analysis, and a component-level sensitivity study.

## Freeze checklist

Before held-out test:

- [ ] V7 unit tests pass.
- [ ] Full pytest collection is verified.
- [ ] Top-K integrity has zero mismatches.
- [ ] FiQA dev policy is generated once and copied to a frozen artifact.
- [ ] Evidence gate and calibrator are frozen.
- [ ] Query analyzer and initial decision policy are frozen.
- [ ] Retrieval implementations and embedding model are frozen.
- [ ] Evaluation code is frozen.
- [ ] Held-out test qrels have not been inspected for tuning decisions.

## Reproducibility artifacts

Save:

- policy JSON
- calibrator JSON
- benchmark result CSV/JSON
- per-query trajectories
- ablation results
- environment/package versions
- repository commit hash
