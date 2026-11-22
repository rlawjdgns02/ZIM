# Windows / single-GPU: evaluate ZIM ViT-B vs SAM ViT-B on P3M-500-NP (original) and its
# face-blurred subset. Same settings as script/run_eval_dataset.sh, but runs evaluation.py
# directly (no torchrun; with one GPU DDP is disabled anyway) and keeps data/outputs on D:.
#
# prerequisites:
#   python script/prepare_p3m.py      -> D:/zim_data/P3M500NP
#   python script/blur_faces.py       -> D:/zim_data/P3M500NP_blur
# usage (from the repo root, after `conda activate D:\conda_envs\zim`):
#   powershell -ExecutionPolicy Bypass -File script/run_eval_p3m.ps1
# results: D:/zim_data/outputs/<valset>_vit_b/{summary.txt, per_image.csv, log.txt}

param(
    [string]$DataRoot = "D:/zim_data",
    [string]$OutRoot = "D:/zim_data/outputs",
    [string[]]$Valsets = @("P3M500NP", "P3M500NP_blur")
)

# not "Stop": in Windows PowerShell 5.1 any stderr line (tqdm progress) would abort the run;
# failure is detected from the exit code instead.
$ErrorActionPreference = "Continue"
foreach ($valset in $Valsets) {
    $save = "$OutRoot/${valset}_vit_b"
    New-Item -ItemType Directory -Force $save | Out-Null
    Write-Host "=== $valset -> $save"
    python script/evaluation.py `
        --amp True `
        --data-root $DataRoot `
        --network-encoder vit_b `
        --network-decoder zim `
        --eval-workers 4 `
        --eval-image-size 1024 `
        --eval-prompt-type point,bbox `
        --eval-model-list zim,sam `
        --eval-zim-weights results/zim_vit_b_2043 `
        --eval-sam-weights results/sam_vit_b_01ec64.pth `
        --eval-save-dir $save `
        --dataset-valset $valset `
        --dataset-data-type p3mnp `
        --dataset-data-list-txt data_list.txt 2>&1 | Tee-Object -FilePath "$save/log.txt"
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed on $valset (exit $LASTEXITCODE)" }
}
