# D²RAG Publication Protocol v1

## 1. Research objective

D²RAG is evaluated as a dual-decision retrieval system with two adaptive stages:

1. Pre-retrieval decision: query analysis determines an initial retrieval strategy and Top-K.
2. Post-retrieval decision: evidence assessment determines whether the retrieved state is adequate; if not, a failure-conditioned action policy selects KEEP, a retrieval-strategy switch, or a Top-K expansion.

The central experimental question is whether this closed-loop design can improve retrieval quality while controlling additional retrieval cost relative to fixed and single-stage baselines.

## 2. Experimental separation

Use the FiQA train/dev/test split with strict separation.

### Development phase

Use the development split to:

- train or calibrate the evidence confidence model;
- build the failure-conditioned action policy;
- inspect failure modes;
- select the final controller configuration;
- run smoke tests and ablations for debugging.

The development split is not the final evidence for the reported test-set claim.

### Freeze point

After the final development experiment:

- freeze the evidence calibrator;
- freeze the evidence acceptance configuration;
- freeze the D²RAG action policy;
- freeze supported Top-K values;
- freeze retrieval backends and embedding model;
- freeze the evaluation code and metric definitions.

Record the frozen artifact paths and commit hash in the research log.

### Final test phase

Run the frozen configuration once on the held-out FiQA test split for the primary comparison.

Do not retune thresholds, policy actions, costs, diagnosis rules, or retrieval parameters after inspecting test results.

## 3. Required baselines

Report the same benchmark conditions for:

- Dense retrieval;
- BM25S retrieval;
- Hybrid retrieval;
- single-stage adaptive retrieval;
- D²RAG.

Keep corpus, query set, embeddings, document units, relevance judgments, and evaluation metrics identical across systems.

## 4. Required D²RAG ablations

The conference paper should contain at least these ablations:

### A. Pre-retrieval only

Query Analyzer + Decision Engine, without post-retrieval feedback.

### B. Post-retrieval only

Fixed initial retrieval configuration + EvidenceAssessor + FeedbackController.

### C. Dual-stage D²RAG

Pre-retrieval decision + EvidenceAssessor + failure-conditioned FeedbackController.

### D. Policy ablation

Compare the learned failure-conditioned policy against a diagnostic-only controller.

### E. Cost ablation

Evaluate the effect of the retrieval-cost weight used by the action evaluator.

## 5. Primary retrieval metrics

Report:

- Recall@5;
- MRR@5;
- nDCG@5.

For adaptive runs also report final evaluated-K metrics when the final K differs from 5, clearly labeled with the actual final K.

Do not mix `Recall@5` with `Recall@EvaluatedK` in a single metric column.

## 6. Adaptation metrics

The main paper should report:

- adaptation rate;
- strategy-switch rate;
- Top-K expansion rate;
- average number of retrieval attempts;
- average confidence delta;
- successful adaptation rate;
- harmful adaptation rate;
- action success by diagnosis;
- KEEP rate among rejected retrieval states;
- oracle-action match rate where counterfactual actions are available;
- average action regret relative to the best available action;
- retrieval-cost overhead.

An adaptation is not counted as successful merely because a retry happened. It is successful when the final retrieval quality improves relative to the state before adaptation under the predefined evaluation metric.

## 7. Evidence-assessment reporting

Report calibration separately from retrieval quality.

Use:

- Brier score;
- expected calibration error (ECE);
- acceptance precision;
- acceptance recall;
- false-accept rate;
- false-reject rate.

The paper must distinguish score-based evidence features from benchmark qrels relevance. Terms such as `score_relevant_count` must never be presented as ground-truth relevance.

## 8. Failure-conditioned policy v6

The v6 policy is trained only on retrieval states rejected by the evidence gate.

Its state representation is:

`(query_type, current_strategy, confidence_bucket, diagnosis, current_top_k)`

The policy considers:

- KEEP;
- strategy switches among dense, BM25S, and hybrid;
- supported Top-K expansions.

A non-KEEP action is selected only when its mean cost-adjusted utility gain over KEEP reaches the frozen minimum-gain threshold and has sufficient development support.

This rule prevents global success states from dominating the post-retrieval policy and makes the learned controller specifically conditional on failed retrieval states.

## 9. Top-K integrity

A requested Top-K must equal the number of returned retrieval units for the benchmark retrievers in the final experiments.

Supported values are:

`3, 5, 8, 10, 15`

The final experiment must record both requested K and actual retrieved count for every attempt.

Any integrity mismatch is a system error and must be fixed before publishing results that depend on that K.

## 10. Error analysis

Build a qualitative error taxonomy from the frozen development run, including:

- false accept: evidence accepted but benchmark retrieval quality is zero or poor;
- false reject: evidence rejected despite strong benchmark retrieval quality;
- strategy mismatch;
- coverage failure;
- ranking uncertainty;
- retrieval disagreement;
- unproductive adaptation;
- successful recovery after adaptation.

For each category, report counts and representative query IDs in an appendix or supplementary material.

## 11. Statistical reporting

For the main test comparison, report the paired metric values per query whenever possible.

For two-system comparisons, use a paired statistical test appropriate to the metric distribution and report the effect size together with the p-value. The test choice and significance criterion must be fixed before inspecting the test outcomes.

Avoid claiming superiority from a single small smoke sample.

## 12. Conference-paper structure

A practical first paper can use this structure:

1. Introduction and motivation for adaptive retrieval.
2. Related work: adaptive RAG, hybrid retrieval, confidence-aware retrieval, retrieval control.
3. D²RAG architecture.
4. Failure-conditioned post-retrieval policy.
5. Experimental setup on FiQA.
6. Main results against fixed and single-stage baselines.
7. Ablations and adaptation-cost analysis.
8. Error analysis and limitations.
9. Conclusion.

The key contribution should be presented as a retrieval-control mechanism and experimentally supported with ablations, adaptation metrics, and cost analysis rather than as a generic RAG implementation.

## 13. Journal extension plan

A journal extension should not be only a longer version of the conference paper.

Add substantive scope such as:

- a second benchmark such as NQ or another BEIR retrieval dataset;
- a broader failure taxonomy;
- additional ablations;
- robustness across query types;
- sensitivity to retrieval cost;
- stronger calibration analysis;
- repeated runs or multiple evaluation seeds where meaningful;
- cross-dataset analysis of the learned action policy;
- deeper analysis of when Top-K expansion is useful versus strategy switching.

## 14. Freeze checklist

Before touching the held-out test split, verify:

- all unit/integration tests pass;
- Top-K integrity passes for every supported K;
- the v6 policy artifact exists and loads successfully;
- the D²RAG smoke run completes all 50 development queries;
- no query is silently skipped;
- no retrieval backend mismatch remains;
- the controller logs diagnosis, action, source, confidence, expected improvement, and final state;
- experiment configuration is recorded;
- the git commit is recorded.

Only after this checklist passes should the final test evaluation begin.

## 15. Current research status

The project should now treat the following as engineering milestones rather than final scientific claims:

- Top-K propagation and integrity: stabilized;
- evidence calibration: implemented;
- failure-conditioned action policy: being introduced in v6;
- D²RAG final test evaluation: not yet frozen;
- final conference result: not yet established;
- journal extension: planned after the conference-scale experiment is complete.
