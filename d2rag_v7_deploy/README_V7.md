# D²RAG V7 Deployment

V7 is a concrete methodological correction, not a threshold-only tweak.

## What changes

- diagnosis-first failure classification
- diagnosis-aware hierarchical policy backoff
- learned-only post-retrieval adaptation
- counterfactual action quality checks using gain, win rate, and harm rate
- unified current-state baseline for strategy and Top-K counterfactual records
- separate interpretation of strategy quality and Top-K breadth/cost
- corrected smoke-test adaptation delta using nDCG@5 at the same K
- pytest publication gate files

## Safety

The apply script creates a timestamped backup under `.d2rag_backups` before replacement.

## Application

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\d2rag_v7_deploy\apply_d2rag_v7.ps1
```

The script copies the V7 files into the repository and does not delete the old V1-V6 artifacts.

## Validation order

```powershell
py -m src.tests.test_feedback_controller
py -m src.tests.test_failure_conditioned_policy_v7
py -m pytest --collect-only -q
py -m pytest -q
py -m src.tests.test_topk_integrity_v7
py -m src.tests.build_fiqa_action_policy_v7
py -m src.tests.inspect_fiqa_action_policy_v7
py -m src.tests.test_d2rag_fiqa_dev_smoke_v7
```

Do not evaluate the held-out FiQA test until V7 is frozen after development validation.
