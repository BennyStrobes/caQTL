#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 5-09:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=40G                        # Memory total in MiB (for all cores): every gene's low-rank Q and Gram matrix Q'Q are held in memory


fm_input_summary_file=${1}
cell_type=${2}
cofine_results_dir=${3}
model_version=${4}  # "cofine" (eQTL, caQTL and peak-gene link data), "cofine_link_se_inflation" (CoFine with a learned inflation of the link standard errors) or "cofine_no_chromatin" (the same sampler without caQTL and peak-gene link data)

source ~/.bash_profile
conda activate plink_env
mkdir -p ${cofine_results_dir}

date

if [ "${model_version}" == "cofine" ]; then
	python run_cofine.py --sumstat_summary_file ${fm_input_summary_file} --output_stem ${cofine_results_dir}${cell_type}_cofine
elif [ "${model_version}" == "cofine_link_se_inflation" ]; then
	python run_cofine.py --sumstat_summary_file ${fm_input_summary_file} --output_stem ${cofine_results_dir}${cell_type}_cofine_link_se_inflation --learn_link_se_inflation
elif [ "${model_version}" == "cofine_no_chromatin" ]; then
	python run_cofine.py --sumstat_summary_file ${fm_input_summary_file} --output_stem ${cofine_results_dir}${cell_type}_cofine_no_chromatin --no_chromatin
else
	echo "unknown model_version: ${model_version}"
	exit 1
fi

date
