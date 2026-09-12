import numpy as np
import os
import sys
import pdb


def standardize_geno(geno_mat):
    # geno_mat: (n_snps, n_samples). Center and scale each SNP to mean 0, sd 1
    geno_mean = np.mean(geno_mat, axis=1, keepdims=True)
    geno_sd = np.std(geno_mat, axis=1, ddof=1, keepdims=True)
    if np.any(geno_sd == 0.0):
        print('assumption error: zero-variance SNP in genotype matrix')
        pdb.set_trace()
    return (geno_mat - geno_mean) / geno_sd


def simulate_phenotype(std_geno_mat, causal_effects):
    # Genetic component of the phenotype
    genetic_pheno = np.dot(np.transpose(std_geno_mat), causal_effects)
    # Residual variance set so total phenotypic variance is ~1
    genetic_var = np.var(genetic_pheno, ddof=1)
    if genetic_var >= 1.0:
        raise ValueError('genetic variance >= 1 (' + str(genetic_var) + '); should have been rejected upstream')
    resid_var = 1.0 - genetic_var
    pheno = genetic_pheno + np.random.normal(0, np.sqrt(resid_var), len(genetic_pheno))
    return pheno


def genetic_variance(std_geno_mat, causal_effects):
    # Realized variance of the genetic component of a phenotype
    return np.var(np.dot(np.transpose(std_geno_mat), causal_effects), ddof=1)


def marginal_effects_and_se(pheno, std_geno_mat):
    # Univariate OLS (with intercept) of pheno on each SNP
    n = len(pheno)
    y_c = pheno - np.mean(pheno)
    G_c = std_geno_mat - np.mean(std_geno_mat, axis=1, keepdims=True)
    sxx = np.sum(G_c ** 2, axis=1)
    betas = np.dot(G_c, y_c) / sxx
    rss = np.maximum(np.sum(y_c ** 2) - (betas ** 2) * sxx, 0.0)
    ses = np.sqrt((rss / (n - 2)) / sxx)
    return betas, ses


def generate_sumstats(std_geno_mat, causal_effects):
    # Simulate phenotype from standardized genotype and return marginal effects / SEs
    pheno = simulate_phenotype(std_geno_mat, causal_effects)
    betas, ses = marginal_effects_and_se(pheno, std_geno_mat)
    return betas, ses














######################
# Command line args
######################
simulation_number = int(sys.argv[1])
gene_summary_file = sys.argv[2]
sumstats_output_dir = sys.argv[3]
simulation_stem = sys.argv[4]

# Set random seed
np.random.seed(simulation_number)

# Simulation parameters
mediated_probability = 0.2
n_nearby_peaks_per_gene = 5
constrained_gene_probability = 0.25
n_causal_variants_per_pheno = 2

non_constrained_eqtl_h2 = 0.05
constrained_eqtl_h2 = 0.0075
caqtl_h2 = 0.1

# Redraw a gene's causal effects if any phenotype's realized genetic variance reaches this cap
max_genetic_var = 0.9


simulated_causal_effect_summary_output_file = sumstats_output_dir + simulation_stem + '_simulated_causal_effect_summary.txt'
t_causal = open(simulated_causal_effect_summary_output_file, 'w')
t_causal.write('Gene_Name\tIs_Constrained\tCausal_eQTL_Effects_File\tCausal_caQTL_Effects_File\tCausal_Peak_Gene_Effects_File\n')

sumstats_summary_output_file = sumstats_output_dir + simulation_stem + '_sumstats_summary.txt'
t_sumstats = open(sumstats_summary_output_file, 'w')
t_sumstats.write('Gene_Name\tVariant_IDs_File\tLD_File\teQTL_Effects_File\teQTL_SE_File\tcaQTL_Effects_File\tcaQTL_SE_File\tEstimated_Peak_Gene_Effects_File\tEstimated_Peak_Gene_SE_File\n')


# Loop through genes
f = open(gene_summary_file, 'r')
head_count = 0
for line in f:
    if head_count == 0:
        head_count += 1
        continue
    line = line.rstrip()
    data = line.split('\t')
    gene_name = data[0]
    chrom_num = data[1]
    gene_position = int(data[2])
    n_cis_variants = int(data[3])   
    ld_file = data[4]
    genotype_file = data[5]
    variant_ids_file = data[6]

    # Load genotype data for this gene and standardize each SNP
    geno_mat = np.load(genotype_file)
    std_geno_mat = standardize_geno(geno_mat.astype(float))

    #######################################
    # 1. SIMULATE CAUSAL EFFECTS FOR GENE
    #######################################

    # First simulate whether the gene is constrained or not (drawn once; not part of the rejection loop)
    is_constrained = np.random.binomial(1, constrained_gene_probability)

    if is_constrained:
        tmp_eqtl_h2 = np.copy(constrained_eqtl_h2)
    else:
        tmp_eqtl_h2 = np.copy(non_constrained_eqtl_h2)

    # Rejection sampling: redraw all causal effects if any phenotype's genetic variance reaches the cap
    n_rejections = 0
    while True:
        # Initialize causal QTL effects
        causal_eqtl_effects = np.zeros(n_cis_variants)
        causal_caqtl_effects = np.zeros((n_nearby_peaks_per_gene, n_cis_variants))
        causal_peak_gene_effects = np.zeros(n_nearby_peaks_per_gene)

        # Simulate eQTL only effects
        causal_eqtl_effects_indices = np.random.choice(n_cis_variants, n_causal_variants_per_pheno, replace=False)
        causal_eqtl_effects[causal_eqtl_effects_indices] = np.random.normal(0, np.sqrt(tmp_eqtl_h2), n_causal_variants_per_pheno)

        # Simulate caQTL effects
        for peak_idx in range(n_nearby_peaks_per_gene):
            causal_caqtl_effects_indices = np.random.choice(n_cis_variants, n_causal_variants_per_pheno, replace=False)
            causal_caqtl_effects[peak_idx, causal_caqtl_effects_indices] = np.random.normal(0, np.sqrt(caqtl_h2), n_causal_variants_per_pheno)

            # Simulate peak-gene effects
            peak_is_mediated = np.random.binomial(1, mediated_probability)
            if peak_is_mediated:
                causal_peak_gene_effects[peak_idx] = np.random.normal(0, np.sqrt(tmp_eqtl_h2/caqtl_h2))

            causal_eqtl_effects += causal_peak_gene_effects[peak_idx] * causal_caqtl_effects[peak_idx,:]

        # Check realized genetic variance of the gene and of each peak
        genetic_vars = [genetic_variance(std_geno_mat, causal_eqtl_effects)]
        for peak_idx in range(n_nearby_peaks_per_gene):
            genetic_vars.append(genetic_variance(std_geno_mat, causal_caqtl_effects[peak_idx,:]))
        if np.max(genetic_vars) < max_genetic_var:
            break
        n_rejections += 1

    if n_rejections > 0:
        print('  redrew causal effects ' + str(n_rejections) + ' time(s) for ' + gene_name)

    # Save the simulated causal effects to npy files
    gene_output_stem = sumstats_output_dir + simulation_stem + '_gene_' + gene_name
    causal_eqtl_effects_file = gene_output_stem + '_causal_eqtl_effects.npy'
    causal_caqtl_effects_file = gene_output_stem + '_causal_caqtl_effects.npy'
    causal_peak_gene_effects_file = gene_output_stem + '_causal_peak_gene_effects.npy'
    np.save(causal_eqtl_effects_file, causal_eqtl_effects)
    np.save(causal_caqtl_effects_file, causal_caqtl_effects)
    np.save(causal_peak_gene_effects_file, causal_peak_gene_effects)

    # Write the file names to the causal effect summary file
    t_causal.write('\t'.join([gene_name, str(is_constrained), causal_eqtl_effects_file, causal_caqtl_effects_file, causal_peak_gene_effects_file]) + '\n')


    #######################################
    # 2. Generate eQTL and caQTL summary statistics
    #######################################
    # eQTL sumstats (effects on standardized-genotype scale)
    standardized_eqtl_effects, standardized_eqtl_se = generate_sumstats(std_geno_mat, causal_eqtl_effects)

    # caQTL sumstats (effects on standardized-genotype scale)
    standardized_caqtl_effects = np.zeros((n_nearby_peaks_per_gene, n_cis_variants))
    standardized_caqtl_se = np.zeros((n_nearby_peaks_per_gene, n_cis_variants))
    for peak_idx in range(n_nearby_peaks_per_gene):
        standardized_caqtl_effects[peak_idx,:], standardized_caqtl_se[peak_idx,:] = generate_sumstats(std_geno_mat, causal_caqtl_effects[peak_idx,:])

    # Save QTL sumstats to npy files
    eqtl_effects_file = gene_output_stem + '_eqtl_effects.npy'
    eqtl_se_file = gene_output_stem + '_eqtl_se.npy'
    caqtl_effects_file = gene_output_stem + '_caqtl_effects.npy'
    caqtl_se_file = gene_output_stem + '_caqtl_se.npy'
    np.save(eqtl_effects_file, standardized_eqtl_effects)
    np.save(eqtl_se_file, standardized_eqtl_se)
    np.save(caqtl_effects_file, standardized_caqtl_effects)
    np.save(caqtl_se_file, standardized_caqtl_se)

    # Simulate estimated peak-gene links and standard errors
    # Estimate = truth + gaussian noise, with SE = 1/sqrt(n) (OLS SE of one variance-1 trait on another under the null)
    # Link sample size assumed equal to the eQTL sample size; noise independent of eQTL sumstats
    n_link_samples = std_geno_mat.shape[1]
    estimated_peak_gene_se = np.ones(n_nearby_peaks_per_gene) / np.sqrt(n_link_samples)
    estimated_peak_gene_effects = causal_peak_gene_effects + np.random.normal(0, estimated_peak_gene_se)

    # Save estimated peak-gene links to npy files
    estimated_peak_gene_effects_file = gene_output_stem + '_estimated_peak_gene_effects.npy'
    estimated_peak_gene_se_file = gene_output_stem + '_estimated_peak_gene_se.npy'
    np.save(estimated_peak_gene_effects_file, estimated_peak_gene_effects)
    np.save(estimated_peak_gene_se_file, estimated_peak_gene_se)

    # Write the file names to the sumstats summary file
    t_sumstats.write('\t'.join([gene_name, variant_ids_file, ld_file, eqtl_effects_file, eqtl_se_file, caqtl_effects_file, caqtl_se_file, estimated_peak_gene_effects_file, estimated_peak_gene_se_file]) + '\n')



t_causal.close()
t_sumstats.close()
f.close()


