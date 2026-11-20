# Evaluate ZIM vs SAM on a dataset prepared by script/prepare_dataset.py
# usage: bash script/run_eval_dataset.sh <valset> <data_type[,data_type...]> [encoder: vit_b|vit_l]
#   e.g. bash script/run_eval_dataset.sh AM2K am2k
#        bash script/run_eval_dataset.sh AIM500 aimcat-transparent,aimcat-animal vit_b
# results: outputs/<valset>_<encoder>/{summary.txt, per_image.csv}

valset=$1
data_type=$2
encoder=${3:-vit_b}

if [ "${encoder}" = "vit_l" ]; then
    zim_weights="results/zim_vit_l_2092"
    sam_weights="results/sam_vit_l_0b3195.pth"
else
    zim_weights="results/zim_vit_b_2043"
    sam_weights="results/sam_vit_b_01ec64.pth"
fi

ngpus=$(nvidia-smi --list-gpus | wc -l)
torchrun --standalone --nnodes=1 --nproc_per_node=${ngpus} script/evaluation.py \
--amp True \
--data-root data \
--network-encoder ${encoder} \
--network-decoder zim \
--eval-workers 4 \
--eval-image-size 1024 \
--eval-prompt-type point,bbox \
--eval-model-list zim,sam \
--eval-zim-weights ${zim_weights} \
--eval-sam-weights ${sam_weights} \
--eval-save-dir outputs/${valset}_${encoder} \
--dataset-valset ${valset} \
--dataset-data-type ${data_type} \
--dataset-data-list-txt data_list.txt
