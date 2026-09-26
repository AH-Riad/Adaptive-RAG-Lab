import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.action_evaluator import ActionEvaluator
from src.evaluation.topk_action_evaluator import TopKActionEvaluator
from src.evaluation.action_policy_builder import FailureConditionedActionPolicyBuilder


class DiagnosisFirstActionPolicyBuilder(FailureConditionedActionPolicyBuilder):
    """Build a diagnosis-conditioned post-retrieval action policy from FiQA dev."""

    VERSION = "v7"
    SUPPORTED_TOP_K = (3, 5, 8, 10, 15)

    EXACT_MIN_SUPPORT = 8
    QUERY_TYPE_BACKOFF_MIN_SUPPORT = 20
    STRATEGY_BACKOFF_MIN_SUPPORT = 30
    DIAGNOSIS_BACKOFF_MIN_SUPPORT = 40

    MIN_GAIN = 0.02
    MIN_WIN_RATE = 0.55
    MAX_HARM_RATE = 0.25

    @classmethod
    def _diagnose(
        cls,
        query_type,
        strategy,
        top_k,
        evidence,
        features,
        confidence,
    ):
        if evidence.retrieved_count == 0:
            return "no_evidence", "No evidence was retrieved."

        if query_type == "lexical" and strategy == "dense":
            return (
                "lexical_strategy_mismatch",
                "The query is lexical but dense retrieval is currently being used.",
            )

        if query_type == "semantic" and strategy == "bm25":
            return (
                "semantic_strategy_mismatch",
                "The query is semantic but lexical retrieval is currently being used.",
            )

        if query_type == "ambiguous" and strategy == "dense":
            return (
                "ambiguous_strategy_risk",
                "The query is ambiguous and dense retrieval may benefit from lexical support.",
            )

        if query_type == "comparison" and strategy == "hybrid" and top_k < 10:
            return (
                "comparison_coverage_risk",
                "Comparison queries may require broader evidence coverage.",
            )

        disagreement = features.get("dense_bm25_agreement")
        if disagreement is not None and 0.0 < disagreement < 0.50:
            return (
                "retrieval_disagreement",
                "Dense and lexical retrieval signals disagree strongly.",
            )

        score_coverage = getattr(evidence, "coverage", 0.0)
        if score_coverage < 0.50 and evidence.retrieved_count >= 3:
            return (
                "coverage_gap",
                "Retrieved evidence has insufficient score-based coverage.",
            )

        top1 = features.get("top1_score", 0.0)
        top1_top2_gap = features.get("top1_top2_gap", 0.0)
        if top1 >= 0.45 and top1_top2_gap < 0.05:
            return (
                "ranking_uncertainty",
                "Top-ranked evidence is weakly separated from the next result.",
            )

        if confidence < 0.25:
            return (
                "very_weak_evidence",
                "Calibrated evidence confidence is very low.",
            )

        if confidence < 0.50:
            return (
                "weak_evidence",
                "Evidence confidence is below the medium-confidence range.",
            )

        return (
            "uncertain_failure",
            "Evidence was rejected but no dominant failure pattern was detected.",
        )

    @staticmethod
    def _append_row(groups, state, action, utility, baseline_utility):
        groups[state].append(
            {
                "action": action,
                "utility": float(utility),
                "utility_gain": float(utility - baseline_utility),
            }
        )

    def _aggregate(self, groups, min_support):
        policy = {}

        for state, rows in groups.items():
            action_groups = defaultdict(list)
            for row in rows:
                action_groups[row["action"]].append(row)

            candidates = {}
            for action, action_rows in action_groups.items():
                gains = [row["utility_gain"] for row in action_rows]
                utilities = [row["utility"] for row in action_rows]
                positive = [gain for gain in gains if gain > 0.0]
                harmful = [gain for gain in gains if gain < 0.0]
                candidates[action] = {
                    "count": len(action_rows),
                    "average_gain": sum(gains) / len(gains),
                    "average_utility": sum(utilities) / len(utilities),
                    "win_rate": len(positive) / len(gains),
                    "harm_rate": len(harmful) / len(gains),
                }

            if "keep" not in candidates:
                candidates["keep"] = {
                    "count": len(rows),
                    "average_gain": 0.0,
                    "average_utility": 0.0,
                    "win_rate": 0.0,
                    "harm_rate": 0.0,
                }

            eligible = {}
            for action, record in candidates.items():
                if action == "keep":
                    eligible[action] = record
                    continue

                eligible[action] = record

            selected_action = "keep"
            selected_record = candidates["keep"]

            non_keep = []
            for action, record in candidates.items():
                if action == "keep":
                    continue
                if record["count"] < min_support:
                    continue
                if record["average_gain"] < self.MIN_GAIN:
                    continue
                if record["win_rate"] < self.MIN_WIN_RATE:
                    continue
                if record["harm_rate"] > self.MAX_HARM_RATE:
                    continue
                non_keep.append((action, record))

            if non_keep:
                selected_action, selected_record = max(
                    non_keep,
                    key=lambda item: (
                        item[1]["average_gain"],
                        item[1]["win_rate"],
                        -item[1]["harm_rate"],
                        item[1]["count"],
                        item[0],
                    ),
                )

            policy[str(state)] = {
                "selected_action": selected_action,
                "selected_record": selected_record,
                "candidates": eligible,
                "samples": len(rows),
                "selection_rule": {
                    "minimum_support": min_support,
                    "minimum_gain": self.MIN_GAIN,
                    "minimum_win_rate": self.MIN_WIN_RATE,
                    "maximum_harm_rate": self.MAX_HARM_RATE,
                    "objective": "counterfactual utility gain relative to the actual current retrieval state",
                },
                "policy_eligible": selected_action != "keep",
            }

        return policy

    def build(self, queries, qrels, query_types, retrievers):
        strategy_evaluator = ActionEvaluator(retrievers)
        topk_evaluator = TopKActionEvaluator(
            retrievers,
            candidate_top_k=self.SUPPORTED_TOP_K,
            cost_weight=self.cost_weight,
        )

        strategy_groups = defaultdict(list)
        topk_groups = defaultdict(list)
        combined_groups = defaultdict(list)

        strategy_backoff_query_type = defaultdict(list)
        strategy_backoff_strategy = defaultdict(list)
        strategy_backoff_diagnosis = defaultdict(list)

        topk_backoff_query_type = defaultdict(list)
        topk_backoff_strategy = defaultdict(list)
        topk_backoff_diagnosis = defaultdict(list)

        combined_backoff_query_type = defaultdict(list)
        combined_backoff_strategy = defaultdict(list)
        combined_backoff_diagnosis = defaultdict(list)

        state_records = []
        rejected_states = 0
        accepted_states = 0

        for query_id, query in queries.items():
            query_type = query_types[query_id]
            relevant_scores = qrels.get(query_id, {})
            initial_plan = self._initial_plan(query, query_type)
            current_strategy = initial_plan.strategy.value
            current_retriever = retrievers[current_strategy]

            for state_top_k in self.SUPPORTED_TOP_K:
                state = self._build_evidence_state(
                    query=query,
                    query_type=query_type,
                    retriever=current_retriever,
                    strategy=current_strategy,
                    top_k=state_top_k,
                )

                if state["evidence"].accepted:
                    accepted_states += 1
                    continue

                rejected_states += 1
                diagnosis = state["diagnosis"]
                confidence_bucket = state["confidence_bucket"]

                exact_state = (
                    query_type,
                    current_strategy,
                    confidence_bucket,
                    diagnosis,
                    state_top_k,
                )
                backoff_qt = (
                    query_type,
                    current_strategy,
                    diagnosis,
                    state_top_k,
                )
                backoff_strategy = (
                    current_strategy,
                    diagnosis,
                    state_top_k,
                )
                backoff_diagnosis = (
                    diagnosis,
                    state_top_k,
                )

                strategy_evaluations = strategy_evaluator.evaluate_strategy_actions(
                    query=query,
                    relevant_scores=relevant_scores,
                    query_type=query_type,
                    current_strategy=current_strategy,
                    top_k=state_top_k,
                )
                current_utility = max(
                    item["utility"]
                    for item in strategy_evaluations
                    if item["candidate_strategy"] == current_strategy
                )

                topk_evaluations = topk_evaluator.evaluate_query(
                    query=query,
                    relevant_scores=relevant_scores,
                    current_strategy=current_strategy,
                    current_top_k=state_top_k,
                )

                for evaluation in strategy_evaluations:
                    action = evaluation["action"]
                    utility = evaluation["utility"]
                    self._append_row(
                        strategy_groups,
                        exact_state,
                        action,
                        utility,
                        current_utility,
                    )
                    for groups, key in (
                        (strategy_backoff_query_type, backoff_qt),
                        (strategy_backoff_strategy, backoff_strategy),
                        (strategy_backoff_diagnosis, backoff_diagnosis),
                    ):
                        self._append_row(groups, key, action, utility, current_utility)

                    if action != "keep":
                        self._append_row(
                            combined_groups,
                            exact_state,
                            action,
                            utility,
                            current_utility,
                        )
                        for groups, key in (
                            (combined_backoff_query_type, backoff_qt),
                            (combined_backoff_strategy, backoff_strategy),
                            (combined_backoff_diagnosis, backoff_diagnosis),
                        ):
                            self._append_row(groups, key, action, utility, current_utility)

                for evaluation in topk_evaluations:
                    action = evaluation["action"]
                    utility = evaluation["utility"]
                    self._append_row(
                        topk_groups,
                        exact_state,
                        action,
                        utility,
                        current_utility,
                    )
                    for groups, key in (
                        (topk_backoff_query_type, backoff_qt),
                        (topk_backoff_strategy, backoff_strategy),
                        (topk_backoff_diagnosis, backoff_diagnosis),
                    ):
                        self._append_row(groups, key, action, utility, current_utility)

                    if action != "keep":
                        self._append_row(
                            combined_groups,
                            exact_state,
                            action,
                            utility,
                            current_utility,
                        )
                        for groups, key in (
                            (combined_backoff_query_type, backoff_qt),
                            (combined_backoff_strategy, backoff_strategy),
                            (combined_backoff_diagnosis, backoff_diagnosis),
                        ):
                            self._append_row(groups, key, action, utility, current_utility)

                self._append_row(
                    combined_groups,
                    exact_state,
                    "keep",
                    current_utility,
                    current_utility,
                )
                for groups, key in (
                    (combined_backoff_query_type, backoff_qt),
                    (combined_backoff_strategy, backoff_strategy),
                    (combined_backoff_diagnosis, backoff_diagnosis),
                ):
                    self._append_row(groups, key, "keep", current_utility, current_utility)

                state_records.append(
                    {
                        "query_id": query_id,
                        "query_type": query_type,
                        "current_strategy": current_strategy,
                        "current_top_k": state_top_k,
                        "evidence_confidence": state["confidence"],
                        "confidence_bucket": confidence_bucket,
                        "diagnosis": diagnosis,
                        "diagnosis_reason": state["diagnosis_reason"],
                        "accepted": False,
                    }
                )

        strategy_policy = self._aggregate(strategy_groups, self.EXACT_MIN_SUPPORT)
        topk_policy = self._aggregate(topk_groups, self.EXACT_MIN_SUPPORT)
        combined_policy = self._aggregate(combined_groups, self.EXACT_MIN_SUPPORT)

        artifact = {
            "dataset": "fiqa",
            "split": "dev",
            "version": self.VERSION,
            "policy_type": "diagnosis_first_hierarchical_failure_conditioned_policy",
            "training_filter": "evidence_rejected_only",
            "supported_top_k": list(self.SUPPORTED_TOP_K),
            "objective": {
                "strategy": "counterfactual nDCG@K gain relative to current strategy at the same K",
                "topk": "counterfactual nDCG@K minus retrieval-cost penalty",
                "combined": "same current-state baseline for strategy and Top-K actions",
                "cost_weight": self.cost_weight,
                "minimum_gain": self.MIN_GAIN,
                "minimum_win_rate": self.MIN_WIN_RATE,
                "maximum_harm_rate": self.MAX_HARM_RATE,
            },
            "state_definition": [
                "query_type",
                "current_strategy",
                "confidence_bucket",
                "diagnosis",
                "current_top_k",
            ],
            "diagnosis_priority": [
                "no_evidence",
                "strategy_mismatch",
                "retrieval_disagreement",
                "coverage_gap",
                "ranking_uncertainty",
                "confidence_weakness",
                "uncertain_failure",
            ],
            "backoff_levels": [
                {
                    "name": "exact",
                    "key": "query_type,current_strategy,confidence_bucket,diagnosis,current_top_k",
                    "minimum_support": self.EXACT_MIN_SUPPORT,
                },
                {
                    "name": "query_type",
                    "key": "query_type,current_strategy,diagnosis,current_top_k",
                    "minimum_support": self.QUERY_TYPE_BACKOFF_MIN_SUPPORT,
                },
                {
                    "name": "strategy",
                    "key": "current_strategy,diagnosis,current_top_k",
                    "minimum_support": self.STRATEGY_BACKOFF_MIN_SUPPORT,
                },
                {
                    "name": "diagnosis",
                    "key": "diagnosis,current_top_k",
                    "minimum_support": self.DIAGNOSIS_BACKOFF_MIN_SUPPORT,
                },
            ],
            "strategy_policy": strategy_policy,
            "topk_policy": topk_policy,
            "combined_policy": combined_policy,
            "strategy_backoff_query_type": self._aggregate(
                strategy_backoff_query_type,
                self.QUERY_TYPE_BACKOFF_MIN_SUPPORT,
            ),
            "strategy_backoff_strategy": self._aggregate(
                strategy_backoff_strategy,
                self.STRATEGY_BACKOFF_MIN_SUPPORT,
            ),
            "strategy_backoff_diagnosis": self._aggregate(
                strategy_backoff_diagnosis,
                self.DIAGNOSIS_BACKOFF_MIN_SUPPORT,
            ),
            "topk_backoff_query_type": self._aggregate(
                topk_backoff_query_type,
                self.QUERY_TYPE_BACKOFF_MIN_SUPPORT,
            ),
            "topk_backoff_strategy": self._aggregate(
                topk_backoff_strategy,
                self.STRATEGY_BACKOFF_MIN_SUPPORT,
            ),
            "topk_backoff_diagnosis": self._aggregate(
                topk_backoff_diagnosis,
                self.DIAGNOSIS_BACKOFF_MIN_SUPPORT,
            ),
            "combined_backoff_query_type": self._aggregate(
                combined_backoff_query_type,
                self.QUERY_TYPE_BACKOFF_MIN_SUPPORT,
            ),
            "combined_backoff_strategy": self._aggregate(
                combined_backoff_strategy,
                self.STRATEGY_BACKOFF_MIN_SUPPORT,
            ),
            "combined_backoff_diagnosis": self._aggregate(
                combined_backoff_diagnosis,
                self.DIAGNOSIS_BACKOFF_MIN_SUPPORT,
            ),
            "state_records": state_records,
            "training_summary": {
                "accepted_states_skipped": accepted_states,
                "rejected_states_used": rejected_states,
                "policy_states_strategy_exact": len(strategy_groups),
                "policy_states_topk_exact": len(topk_groups),
                "policy_states_combined_exact": len(combined_groups),
                "strategy_backoff_states_query_type": len(strategy_backoff_query_type),
                "strategy_backoff_states_strategy": len(strategy_backoff_strategy),
                "strategy_backoff_states_diagnosis": len(strategy_backoff_diagnosis),
                "topk_backoff_states_query_type": len(topk_backoff_query_type),
                "topk_backoff_states_strategy": len(topk_backoff_strategy),
                "topk_backoff_states_diagnosis": len(topk_backoff_diagnosis),
            },
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(artifact, file, indent=2)

        return artifact
