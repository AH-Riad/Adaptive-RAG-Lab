$ErrorActionPreference = "Stop"

$repo = (Get-Location).Path
$package = Join-Path $repo "d2rag_v7_deploy"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = Join-Path $repo ".d2rag_backups\v7_$stamp"

if (-not (Test-Path $package)) {
    throw "V7 package folder was not found: $package"
}

New-Item -ItemType Directory -Force -Path $backup | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $backup "src\adaptation") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $backup "src\evaluation") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $backup "src\tests") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $backup "docs") | Out-Null

$targets = @(
    "src\adaptation\feedback_controller.py",
    "src\adaptation\adaptive_retrieval_orchestrator.py"
)

foreach ($target in $targets) {
    $source = Join-Path $repo $target
    if (Test-Path $source) {
        Copy-Item $source (Join-Path $backup $target) -Force
    }
}

$files = @{
    "src\adaptation\feedback_controller.py" = "src\adaptation\feedback_controller.py"
    "src\adaptation\adaptive_retrieval_orchestrator.py" = "src\adaptation\adaptive_retrieval_orchestrator.py"
    "src\evaluation\action_policy_builder_v7.py" = "src\evaluation\action_policy_builder_v7.py"
    "src\tests\test_feedback_controller.py" = "src\tests\test_feedback_controller.py"
    "src\tests\test_failure_conditioned_policy_v7.py" = "src\tests\test_failure_conditioned_policy_v7.py"
    "src\tests\test_topk_integrity_v7.py" = "src\tests\test_topk_integrity_v7.py"
    "src\tests\test_d2rag_fiqa_dev_smoke_v7.py" = "src\tests\test_d2rag_fiqa_dev_smoke_v7.py"
    "src\tests\build_fiqa_action_policy_v7.py" = "src\tests\build_fiqa_action_policy_v7.py"
    "src\tests\inspect_fiqa_action_policy_v7.py" = "src\tests\inspect_fiqa_action_policy_v7.py"
    "pytest.ini" = "pytest.ini"
    "docs\D2RAG_Publication_Protocol_v2.md" = "docs\D2RAG_Publication_Protocol_v2.md"
}

foreach ($target in $files.Keys) {
    $destination = Join-Path $repo $target
    $source = Join-Path $package $files[$target]
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item $source $destination -Force
}

Write-Host "D2RAG V7 applied."
Write-Host "Backup: $backup"
