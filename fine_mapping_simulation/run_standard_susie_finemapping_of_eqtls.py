import numpy as np
import os
import sys
import pdb
from susie_rss import SUSIE_RSS














########################
# Command line args
########################
sumstat_summary_file = sys.argv[1] # one line per gene
fine_mapping_output_file = sys.argv[2]
eqtl_sample_size = int(sys.argv[3])

t = open(fine_mapping_output_file, 'w')
t.write('gene_id\tvariant_ids_file\tpips\n')

# Loop through genes and run SuSiE fine-mapping for each gene
f = open(sumstat_summary_file, 'r')
head_count = 0
for line in f:
    line = line.strip()
    data = line.split('\t')
    # Skip the header line
    if head_count == 0:
        head_count += 1
        header = np.copy(data)
        continue

    # Extract relevant information for the gene
    gene_id = data[0]
    variant_ids_file = data[1]
    ld_file = data[2]
    eqtl_effects_file = data[3]
    eqtl_se_file = data[4]
    #caqtl_effects_file = data[5]
    #caqtl_se_file = data[6]
    #peak_gene_effects_file = data[7]
    #peak_gene_se_file = data[8]
    print(gene_id)

    # Load in data for the gene
    ld_matrix = np.load(ld_file)
    eqtl_effects = np.load(eqtl_effects_file)
    eqtl_se = np.load(eqtl_se_file)

    # Run SuSiE fine-mapping on the eQTL sumstats
    susie_fit = SUSIE_RSS(L=10).fit(eqtl_effects, eqtl_se, ld_matrix, eqtl_sample_size)
    pips = susie_fit.pip
    credible_sets = susie_fit.credible_sets

    # Write results to output file
    t.write(gene_id + '\t' + variant_ids_file + '\t' + ','.join([str(x) for x in pips]) + '\n')


f.close()
t.close()