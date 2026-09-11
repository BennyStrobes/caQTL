import numpy as np
import os
import sys
import pdb
import pgenlib as pg




def randomly_select_genes(gene_annotation_file, chrom_num, n_genes):
    f = open(gene_annotation_file, 'r')
    gene_names = []
    gene_positions = []
    for line in f:
        line = line.rstrip()
        data = line.split('\t')
        if data[0] != 'chr' + str(chrom_num):
            continue
        gene_names.append(data[3])
        gene_positions.append(int(data[1]))

    f.close()

    gene_names = np.array(gene_names)
    gene_positions = np.array(gene_positions)

    # randomly select genes
    selected_gene_indices = np.random.choice(len(gene_names), n_genes, replace=False)
    selected_gene_names = gene_names[selected_gene_indices]
    selected_gene_positions = gene_positions[selected_gene_indices]

    return selected_gene_names, selected_gene_positions


def load_in_genotype(gtex_genotype_data_dir, chrom_num):
    stem = gtex_genotype_data_dir + 'gtex_v9_eqtl_chr' + str(chrom_num)
    pgen = pg.PgenReader((stem + '.pgen').encode())
    n_samples = pgen.get_raw_sample_ct()
    n_variants = pgen.get_variant_ct()

    # np.loadtxt skips the '#'-prefixed header lines by default
    pvar = np.loadtxt(stem + '.pvar', dtype=str, delimiter='\t')
    psam = np.loadtxt(stem + '.psam', dtype=str, delimiter='\t')

    if pvar.shape[0] != n_variants or psam.shape[0] != n_samples:
        print('assumption error: pvar/psam dimensions do not match pgen')
        pdb.set_trace()

    return pgen, pvar, psam


def extract_dosages_from_reader(pgen, snp_mask, dtype=np.float32):
    # Returns dosage matrix of shape (n_snps_in_mask, n_samples)
    n_samples = pgen.get_raw_sample_ct()
    idx = np.flatnonzero(snp_mask)
    G = np.empty((len(idx), n_samples), dtype=dtype)
    buf = np.empty(n_samples, dtype=dtype)
    for j, v_idx in enumerate(idx):
        pgen.read_dosages(int(v_idx), buf)
        G[j] = buf
    return G





#####################
# Command line args
######################
gtex_genotype_data_dir = sys.argv[1]
gene_annotation_file = sys.argv[2]
eqtl_sample_size = int(sys.argv[3])
ld_directory = sys.argv[4]
n_genes = int(sys.argv[5])

chrom_num=5
cis_window_size = 50000
np.random.seed(1)

# Output stem
output_stem = ld_directory + 'n_genes_' + str(n_genes) + '_eqtl_sample_size_' + str(eqtl_sample_size)

gene_names, gene_positions = randomly_select_genes(gene_annotation_file, chrom_num, n_genes)

# Load genotype data for this chromosome once
pgen, pvar, psam = load_in_genotype(gtex_genotype_data_dir, chrom_num)
variant_positions = pvar[:, 1].astype(float)
variant_ids = pvar[:, 2]
sample_ids = psam[:, 0]

# Randomly select indices of samples to use for the simulation
selected_sample_indices = np.sort(np.random.choice(len(sample_ids), eqtl_sample_size, replace=False))

# Generate output file for sample indices
sample_indices_output_file = output_stem + '_sample_indices.txt'
t = open(sample_indices_output_file, 'w')
t.write('Sample_Index\tSample_ID\n')
for i in range(len(selected_sample_indices)):
    t.write(str(selected_sample_indices[i]) + '\t' + sample_ids[selected_sample_indices[i]] + '\n')
t.close()


# Open cross-gene summary output file
cross_gene_summary_output_file = output_stem + '_cross_gene_summary.txt'
t = open(cross_gene_summary_output_file, 'w')
t.write('Gene_Name\tGene_Chromosome\tGene_Position\tNum_Cis_SNPs\tLD_Matrix_File\tGenotype_File\tVariant_IDs_File\n')

for i in range(len(gene_names)):
    gene_name = gene_names[i]
    gene_position = gene_positions[i]
    gene_position_lb = gene_position - cis_window_size
    gene_position_ub = gene_position + cis_window_size

    # get the genotype data for the gene (n_snps x n_samples)
    cis_snp_mask = (variant_positions >= gene_position_lb) & (variant_positions < gene_position_ub)
    cis_variant_ids = variant_ids[cis_snp_mask]
    G = extract_dosages_from_reader(pgen, cis_snp_mask)

    # pgenlib encodes missing dosages as -9; drop SNPs with any missing calls
    observed = np.all(G >= 0, axis=1)
    G = G[observed, :]
    cis_variant_ids = cis_variant_ids[observed]

    # subset to the selected samples
    G = G[:, selected_sample_indices]

    # Remove SNPs that are monomorphic in the selected samples
    keep = G.std(axis=1) > 0
    G = G[keep, :]
    cis_variant_ids = cis_variant_ids[keep]

    # Generate LD
    ld_mat = np.corrcoef(G)

    # Save LD matrix to file
    ld_output_file = output_stem + '_gene_' + gene_name + '_ld.npy'
    np.save(ld_output_file, ld_mat)

    # Save Genotype to output file
    genotype_output_file = output_stem + '_gene_' + gene_name + '_genotype.npy'
    np.save(genotype_output_file, G)

    # Save variant IDs to output file
    variant_ids_output_file = output_stem + '_gene_' + gene_name + '_variant_ids.txt'
    np.savetxt(variant_ids_output_file, cis_variant_ids, fmt='%s')

    t.write(gene_name + '\t' + str(chrom_num) + '\t' + str(gene_position) + '\t' + str(len(cis_variant_ids)) + '\t' + ld_output_file + '\t' + genotype_output_file + '\t' + variant_ids_output_file + '\n')

t.close()

print(cross_gene_summary_output_file)