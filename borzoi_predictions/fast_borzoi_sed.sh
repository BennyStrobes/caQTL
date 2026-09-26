#!/bin/bash
#SBATCH --gpus 1                             # Request one core
#SBATCH -t 0-120:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-gpu                           # Partition to run in
#SBATCH --mem=21GB  



source ~/.bashrc
conda activate borzoi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

output_file="${1}"
vcf_input_file="${2}"
borzoi_training_dir="${3}"
model_num="${4}"
variant_peak_pair_file="${5}"
peak_gtf_file="${6}"
borzoi_targets_file="${7}"


echo $output_file

python "fast_borzoi_sed.py" -o ${output_file} -v ${variant_peak_pair_file} -g ${peak_gtf_file} --batch_size 4 --rc --stats logSED,refLog,altLog -t ${borzoi_targets_file} ${borzoi_training_dir}"params_pred.json" ${borzoi_training_dir}"model0_best_f3c"${model_num}".h5" $vcf_input_file
