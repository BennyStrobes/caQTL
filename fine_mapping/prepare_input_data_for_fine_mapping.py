import sys
import argparse
import pdb
import numpy as np
import gzip
import os
import subprocess

def create_ensamble_id_to_chrom_and_tss_mapping(gene_annotation_file):
    # Returns dictionary mapping ensamble id (gene_id in gtf with version removed, e.g. ENSG00000223972) to (chrom_num, tss)
    # chrom_num is chromosome string with 'chr' prefix removed (e.g. '1', 'X')
    # tss is gene start if on + strand, gene end if on - strand
    mapping = {}
    f = open(gene_annotation_file)
    for line in f:
        if line.startswith('#'):
            continue
        data = line.rstrip().split('\t')
        if data[2] != 'gene':
            continue
        chrom_num = data[0]
        if chrom_num.startswith('chr'):
            chrom_num = chrom_num[3:]
        start = int(data[3])
        end = int(data[4])
        strand = data[6]
        if strand == '+':
            tss = start
        elif strand == '-':
            tss = end
        else:
            print('assumption error: unexpected strand ' + strand)
            pdb.set_trace()
        # Extract ensamble id from info field
        ensamble_id = None
        for field in data[8].split(';'):
            field = field.strip()
            if field.startswith('gene_id '):
                ensamble_id = field.split(' ')[1].strip('"')
                break
        # Skip chrY copies of pseudoautosomal region genes (would duplicate chrX copy after version removal)
        if ensamble_id is not None and ensamble_id.endswith('_PAR_Y'):
            continue
        # Remove version suffix (e.g. ENSG00000223972.5 -> ENSG00000223972) to match eqtl file
        if ensamble_id is not None:
            ensamble_id = ensamble_id.split('.')[0]
        if ensamble_id is None:
            print('assumption error: no gene_id found')
            pdb.set_trace()
        if ensamble_id in mapping:
            print('assumption error: duplicate ensamble id ' + ensamble_id)
            pdb.set_trace()
        mapping[ensamble_id] = (chrom_num, tss)
    f.close()
    return mapping



def generate_eqtl_sumstats(eqtl_file, ens_id_to_chrom_num_and_tss, sum_stats_fm_input_dir, cell_type, distance_window, eqtl_summary_file):
    t = open(eqtl_summary_file, 'w')
    t.write('Gene_Name\tGene_Chromosome\tGene_TSS\tNum_Cis_SNPs\tVariant_IDs_File\teQTL_Effects_File\teQTL_SE_File\n')

    valid_chroms = {}
    for chrom_num in range(1, 23):
        valid_chroms['chr' + str(chrom_num)] = True

    # Generate mapping from gene id to (1) vector of eqtl sumstats, ses, and variant ids
    gene_id_to_eqtl_sumstats = {}
    f = gzip.open(eqtl_file, 'rt')
    head_count = 0
    for line in f:
        line = line.rstrip()
        data = line.split('\t')
        if head_count == 0:
            head_count += 1
            continue
        # Extract relevent fields from line
        chrom_num = data[0]
        if chrom_num not in valid_chroms:
            continue
        variant_position = int(data[1])
        ens_id = data[3]
        variant_id = data[4]
        allele_frequency = float(data[5])
        beta = float(data[6])
        se = float(data[7])
        p_value = float(data[8])
        # Standardize effect sizes to per-genotype-SD scale (genotype variance = 2p(1-p) under HWE)
        genotype_sd = np.sqrt(2.0 * allele_frequency * (1.0 - allele_frequency))
        beta_std = beta * genotype_sd
        se_std = se * genotype_sd

        if ens_id not in ens_id_to_chrom_num_and_tss:
            continue
        gene_chrom_num, gene_tss = ens_id_to_chrom_num_and_tss[ens_id]
        if 'chr' + gene_chrom_num != chrom_num:
            print('assumption error: gene chrom ' + gene_chrom_num + ' does not match variant chrom ' + chrom_num)
            pdb.set_trace()
        if abs(variant_position - gene_tss) > distance_window:
            continue
        if ens_id not in gene_id_to_eqtl_sumstats:
            gene_id_to_eqtl_sumstats[ens_id] = {'betas': [], 'ses': [], 'betas_std': [], 'ses_std': [], 'variant_ids': []}
        gene_id_to_eqtl_sumstats[ens_id]['betas'].append(beta)
        gene_id_to_eqtl_sumstats[ens_id]['ses'].append(se)
        gene_id_to_eqtl_sumstats[ens_id]['betas_std'].append(beta_std)
        gene_id_to_eqtl_sumstats[ens_id]['ses_std'].append(se_std)
        gene_id_to_eqtl_sumstats[ens_id]['variant_ids'].append(variant_id)
    f.close()

    for ens_id in gene_id_to_eqtl_sumstats:
        betas = np.array(gene_id_to_eqtl_sumstats[ens_id]['betas'])
        ses = np.array(gene_id_to_eqtl_sumstats[ens_id]['ses'])
        betas_std = np.array(gene_id_to_eqtl_sumstats[ens_id]['betas_std'])
        ses_std = np.array(gene_id_to_eqtl_sumstats[ens_id]['ses_std'])
        variant_ids = np.array(gene_id_to_eqtl_sumstats[ens_id]['variant_ids'])
        gene_chrom_num, gene_tss = ens_id_to_chrom_num_and_tss[ens_id]

        # Save variant ids and standardized eqtl sumstats to files (same formats as fine-mapping simulation)
        gene_output_stem = sum_stats_fm_input_dir + cell_type + '_gene_' + ens_id
        variant_ids_file = gene_output_stem + '_variant_ids.txt'
        eqtl_effects_file = gene_output_stem + '_eqtl_effects.npy'
        eqtl_se_file = gene_output_stem + '_eqtl_se.npy'
        np.savetxt(variant_ids_file, variant_ids, fmt='%s')
        np.save(eqtl_effects_file, betas_std)
        np.save(eqtl_se_file, ses_std)

        # Write the file names to the summary file
        t.write('\t'.join([ens_id, gene_chrom_num, str(gene_tss), str(len(variant_ids)), variant_ids_file, eqtl_effects_file, eqtl_se_file]) + '\n')

    t.close()
    return


def load_peak_gene_links(peak_gene_links_file, gene_ids):
    # Returns dictionary mapping gene id to list of (peak_id, beta_link, se_link) for genes in gene_ids
    # beta_link / se_link are from the negative binomial count component of the hurdle model (as in exploratory analysis)
    gene_id_to_links = {}
    seen_links = {}
    genes_in_file = {}  # all gene ids in links file (including NA rows), for diagnostics
    f = gzip.open(peak_gene_links_file, 'rt')
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            print('peak-gene links file header: ' + ' '.join(header))
            beta_link_col = header.index('hurdle_count_beta')
            se_link_col = header.index('hurdle_count_se')
            continue
        peak_id = data[0]
        gene_id = data[1]
        genes_in_file[gene_id] = True
        if data[beta_link_col] == 'NA' or data[se_link_col] == 'NA':
            continue
        beta_link = float(data[beta_link_col])
        se_link = float(data[se_link_col])
        if (peak_id, gene_id) in seen_links:
            print('assumption error: duplicate peak-gene pair ' + peak_id + ' ' + gene_id)
            pdb.set_trace()
        seen_links[(peak_id, gene_id)] = True
        if gene_id not in gene_ids:
            continue
        if gene_id not in gene_id_to_links:
            gene_id_to_links[gene_id] = []
        gene_id_to_links[gene_id].append((peak_id, beta_link, se_link))
    f.close()
    # Diagnostics: where do eqtl genes without usable links go?
    n_eqtl_genes_not_in_file = 0
    n_eqtl_genes_na_only = 0
    for gene_id in gene_ids:
        if gene_id not in genes_in_file:
            n_eqtl_genes_not_in_file += 1
        elif gene_id not in gene_id_to_links:
            n_eqtl_genes_na_only += 1
    n_file_genes_not_in_eqtl = 0
    for gene_id in genes_in_file:
        if gene_id not in gene_ids:
            n_file_genes_not_in_eqtl += 1
    print('peak-gene links file: ' + str(len(genes_in_file)) + ' genes total, ' + str(n_file_genes_not_in_eqtl) + ' not among eqtl genes')
    print('eqtl genes: ' + str(len(gene_ids)) + ' total, ' + str(n_eqtl_genes_not_in_file) + ' absent from links file, ' + str(n_eqtl_genes_na_only) + ' present but only NA links')
    return gene_id_to_links


def load_caqtl_sumstats(caqtl_file, needed_peaks, needed_variants):
    # Returns dictionary mapping peak id to dictionary mapping variant id to (beta_std, se_std)
    # Only peaks in needed_peaks and variants in needed_variants are kept (to limit memory)
    # Column indices follow exploratory/generate_beta_combined.py
    peak_id_to_caqtl_sumstats = {}
    f = gzip.open(caqtl_file, 'rt')
    head_count = 0
    line_num = 0
    for line in f:
        line_num += 1
        if line_num % 10000000 == 0:
            print('caQTL line ' + str(line_num), flush=True)
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            print('caQTL file header: ' + ' '.join(header))
            continue
        peak_id = data[3]
        if peak_id not in needed_peaks:
            continue
        variant_id = data[4]
        if variant_id not in needed_variants:
            continue
        if data[11] == 'NA' or data[12] == 'NA':
            continue
        allele_frequency = float(data[7])
        beta = float(data[11])
        se = float(data[12])
        # Standardize effect sizes to per-genotype-SD scale (genotype variance = 2p(1-p) under HWE)
        genotype_sd = np.sqrt(2.0 * allele_frequency * (1.0 - allele_frequency))
        beta_std = beta * genotype_sd
        se_std = se * genotype_sd
        if peak_id not in peak_id_to_caqtl_sumstats:
            peak_id_to_caqtl_sumstats[peak_id] = {}
        if variant_id in peak_id_to_caqtl_sumstats[peak_id]:
            print('assumption error: duplicate peak-variant pair ' + peak_id + ' ' + variant_id)
            pdb.set_trace()
        peak_id_to_caqtl_sumstats[peak_id][variant_id] = (beta_std, se_std)
    f.close()
    return peak_id_to_caqtl_sumstats


def load_eqtl_summary_file(eqtl_summary_file):
    # Returns dictionary mapping gene id to (gene_chrom_num, gene_tss, variant_ids, variant_ids_file, eqtl_effects_file, eqtl_se_file) from eqtl summary file
    gene_id_to_eqtl_sumstats = {}
    f = open(eqtl_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_id = data[0]
        gene_chrom_num = data[1]
        gene_tss = int(data[2])
        variant_ids_file = data[4]
        eqtl_effects_file = data[5]
        eqtl_se_file = data[6]
        variant_ids = np.loadtxt(variant_ids_file, dtype=str, ndmin=1)
        gene_id_to_eqtl_sumstats[gene_id] = (gene_chrom_num, gene_tss, variant_ids, variant_ids_file, eqtl_effects_file, eqtl_se_file)
    f.close()
    return gene_id_to_eqtl_sumstats


def generate_caqtl_and_peak_gene_link_sumstats(eqtl_summary_file, caqtl_file, peak_gene_links_file, sum_stats_fm_input_dir, cell_type, min_peak_variant_coverage, caqtl_mediated_summary_file):
    # For each gene in eqtl summary file, build fine-mapping simulation-formatted objects:
    #   caqtl effects / ses: (K peaks x p variants) matrices on standardized-genotype scale
    #   peak-gene effects / ses: (K,) vectors
    # Peaks linked to the gene are kept if they have caqtl sumstats for at least min_peak_variant_coverage of the gene's variants
    # Variant set is the gene's eqtl variant set; variants not tested in a kept peak's caqtl scan are set to nan in that peak's row
    t = open(caqtl_mediated_summary_file, 'w')
    t.write('Gene_Name\tGene_Chromosome\tGene_TSS\tNum_Cis_SNPs\tNum_Peaks\tVariant_IDs_File\teQTL_Effects_File\teQTL_SE_File\tPeak_IDs_File\tcaQTL_Effects_File\tcaQTL_SE_File\tPeak_Gene_Effects_File\tPeak_Gene_SE_File\n')

    # Load eqtl sumstats (variant ids, effects, ses) for each gene from eqtl summary file
    gene_id_to_eqtl_sumstats = load_eqtl_summary_file(eqtl_summary_file)

    # Load peak-gene links for genes with eqtl sumstats
    gene_id_to_links = load_peak_gene_links(peak_gene_links_file, gene_id_to_eqtl_sumstats)
    print(str(len(gene_id_to_links)) + ' of ' + str(len(gene_id_to_eqtl_sumstats)) + ' eqtl genes have peak-gene links')

    # Determine which peaks and variants we need caqtl sumstats for
    needed_peaks = {}
    needed_variants = {}
    for gene_id in gene_id_to_links:
        for peak_id, beta_link, se_link in gene_id_to_links[gene_id]:
            needed_peaks[peak_id] = True
        for variant_id in gene_id_to_eqtl_sumstats[gene_id][2]:
            needed_variants[variant_id] = True

    # Load caqtl sumstats
    peak_id_to_caqtl_sumstats = load_caqtl_sumstats(caqtl_file, needed_peaks, needed_variants)
    print(str(len(peak_id_to_caqtl_sumstats)) + ' of ' + str(len(needed_peaks)) + ' linked peaks have caqtl sumstats')

    n_genes_written = 0
    n_genes_no_peaks = 0
    n_links_kept = 0
    n_links_no_caqtl = 0
    n_links_low_coverage = 0
    for gene_id in gene_id_to_eqtl_sumstats:
        gene_chrom_num, gene_tss, variant_ids, variant_ids_file, eqtl_effects_file, eqtl_se_file = gene_id_to_eqtl_sumstats[gene_id]
        n_variants = len(variant_ids)

        # Keep peaks with caqtl sumstats covering enough of the gene's variants
        # Genes with no linked peaks are still written (with empty caqtl and peak-gene arrays)
        kept_peaks = []
        for peak_id, beta_link, se_link in gene_id_to_links.get(gene_id, []):
            if peak_id not in peak_id_to_caqtl_sumstats:
                n_links_no_caqtl += 1
                continue
            n_covered = 0
            for variant_id in variant_ids:
                if variant_id in peak_id_to_caqtl_sumstats[peak_id]:
                    n_covered += 1
            if n_covered / n_variants < min_peak_variant_coverage:
                n_links_low_coverage += 1
                continue
            n_links_kept += 1
            kept_peaks.append((peak_id, beta_link, se_link))
        if len(kept_peaks) == 0:
            n_genes_no_peaks += 1

        # Build caqtl matrices and peak-gene vectors (nan where variant not tested in peak's caqtl scan)
        K = len(kept_peaks)
        p = n_variants
        caqtl_effects = np.full((K, p), np.nan)
        caqtl_se = np.full((K, p), np.nan)
        peak_gene_effects = np.zeros(K)
        peak_gene_se = np.zeros(K)
        peak_ids = []
        for peak_iter, (peak_id, beta_link, se_link) in enumerate(kept_peaks):
            peak_ids.append(peak_id)
            peak_gene_effects[peak_iter] = beta_link
            peak_gene_se[peak_iter] = se_link
            for variant_iter, variant_id in enumerate(variant_ids):
                if variant_id in peak_id_to_caqtl_sumstats[peak_id]:
                    caqtl_effects[peak_iter, variant_iter], caqtl_se[peak_iter, variant_iter] = peak_id_to_caqtl_sumstats[peak_id][variant_id]
        peak_ids = np.array(peak_ids)

        # Save to files
        gene_output_stem = sum_stats_fm_input_dir + cell_type + '_caqtl_mediated_gene_' + gene_id
        peak_ids_file = gene_output_stem + '_peak_ids.txt'
        caqtl_effects_file = gene_output_stem + '_caqtl_effects.npy'
        caqtl_se_file = gene_output_stem + '_caqtl_se.npy'
        peak_gene_effects_file = gene_output_stem + '_peak_gene_effects.npy'
        peak_gene_se_file = gene_output_stem + '_peak_gene_se.npy'
        np.savetxt(peak_ids_file, peak_ids, fmt='%s')
        np.save(caqtl_effects_file, caqtl_effects)
        np.save(caqtl_se_file, caqtl_se)
        np.save(peak_gene_effects_file, peak_gene_effects)
        np.save(peak_gene_se_file, peak_gene_se)

        # Write the file names to the summary file (variant id and eqtl files are those from the eqtl summary file)
        t.write('\t'.join([gene_id, gene_chrom_num, str(gene_tss), str(p), str(K), variant_ids_file, eqtl_effects_file, eqtl_se_file, peak_ids_file, caqtl_effects_file, caqtl_se_file, peak_gene_effects_file, peak_gene_se_file]) + '\n')
        n_genes_written += 1
    t.close()
    print(str(n_genes_written) + ' genes written to ' + caqtl_mediated_summary_file + ' (' + str(n_genes_no_peaks) + ' with no peaks)')
    print('Peak-gene links: ' + str(n_links_kept) + ' kept, ' + str(n_links_low_coverage) + ' removed by ' + str(min_peak_variant_coverage) + ' variant coverage filter, ' + str(n_links_no_caqtl) + ' removed because peak has no caqtl sumstats')
    return


def load_ld_for_gene(fingen_ld_dir, gene_chrom_num, variant_ids, ldstore_binary, tmp_stem):
    # Load LD for the gene's variants from the finngen LDstore v1.1 bcor files (one per chromosome: FG_LD_chr<N>.bcor)
    # Runs ldstore twice on the genomic range spanned by the gene's variants: --meta (variant info) and --table (stored pairs)
    # Returns (ld_variant_ids, ld_mat): variant ids as chr_pos_A_allele_B_allele from the panel, and the correlation matrix
    # Pairs not in the table (|r| below the panel's storage threshold, or beyond its window) are set to 0
    # The panel may cover extra variants, be in a different order, or have alleles swapped relative to the QTL files;
    # align_ld_to_qtl_variants handles all of that
    bcor_file = fingen_ld_dir + 'FG_LD_chr' + gene_chrom_num + '.bcor'
    positions = np.array([int(variant_id.split('_')[1]) for variant_id in variant_ids])
    incl_range = str(np.min(positions)) + '-' + str(np.max(positions))
    meta_file = tmp_stem + '_ldstore.meta'
    table_file = tmp_stem + '_ldstore.table'
    for extract_flag, extract_file in [('--meta', meta_file), ('--table', table_file)]:
        result = subprocess.run([ldstore_binary, '--bcor', bcor_file, '--incl-range', incl_range, extract_flag, extract_file], capture_output=True, text=True)
        if result.returncode != 0:
            print('assumption error: ldstore failed for ' + bcor_file + ' range ' + incl_range)
            print(result.stdout)
            print(result.stderr)
            pdb.set_trace()

    # ldstore exits 0 but writes no meta file when the range contains no panel variants (e.g. regions the panel does not cover)
    if not os.path.exists(meta_file):
        print('warning: ldstore wrote no meta file for ' + bcor_file + ' range ' + incl_range + ' (no panel variants in range); gene will be skipped')
        if os.path.exists(table_file):
            os.remove(table_file)
        return np.array([], dtype=str), np.zeros((0, 0))

    # Parse meta file: one row per variant in range
    ld_variant_ids = []
    rsid_to_index = {}
    f = open(meta_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split()
        if len(data) == 0:
            continue
        if head_count == 0:
            head_count += 1
            header = data
            if 'RSID' not in header or 'position' not in header or 'chromosome' not in header or 'A_allele' not in header or 'B_allele' not in header:
                print('assumption error: unexpected ldstore meta header: ' + ' '.join(header))
                pdb.set_trace()
            rsid_col = header.index('RSID')
            position_col = header.index('position')
            chromosome_col = header.index('chromosome')
            a_allele_col = header.index('A_allele')
            b_allele_col = header.index('B_allele')
            continue
        rsid = data[rsid_col]
        if rsid in rsid_to_index:
            print('assumption error: duplicate RSID in ldstore meta file ' + rsid)
            pdb.set_trace()
        rsid_to_index[rsid] = len(ld_variant_ids)
        chrom = data[chromosome_col]
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        chrom = str(int(chrom))  # e.g. '01' -> '1'
        ld_variant_ids.append('chr' + chrom + '_' + str(int(data[position_col])) + '_' + data[a_allele_col] + '_' + data[b_allele_col])
    f.close()
    ld_variant_ids = np.array(ld_variant_ids)

    # Parse table file: one row per stored variant pair (no table file = no stored pairs in range)
    n_ld_variants = len(ld_variant_ids)
    ld_mat = np.eye(n_ld_variants)
    if not os.path.exists(table_file):
        os.remove(meta_file)
        return ld_variant_ids, ld_mat
    f = open(table_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split()
        if len(data) == 0:
            continue
        if head_count == 0:
            head_count += 1
            header = data
            if 'RSID1' not in header or 'RSID2' not in header or 'correlation' not in header:
                print('assumption error: unexpected ldstore table header: ' + ' '.join(header))
                pdb.set_trace()
            rsid1_col = header.index('RSID1')
            rsid2_col = header.index('RSID2')
            correlation_col = header.index('correlation')
            continue
        rsid1 = data[rsid1_col]
        rsid2 = data[rsid2_col]
        if rsid1 not in rsid_to_index or rsid2 not in rsid_to_index:
            print('assumption error: RSID in ldstore table not in meta file: ' + rsid1 + ' ' + rsid2)
            pdb.set_trace()
        index1 = rsid_to_index[rsid1]
        index2 = rsid_to_index[rsid2]
        correlation = float(data[correlation_col])
        ld_mat[index1, index2] = correlation
        ld_mat[index2, index1] = correlation
    f.close()
    os.remove(meta_file)
    os.remove(table_file)
    return ld_variant_ids, ld_mat


def align_ld_to_qtl_variants(variant_ids, ld_variant_ids, ld_mat):
    # Align LD panel to the QTL variant ids (chr_pos_ref_alt) and orient it to the QTL effect allele (alt)
    # Returns (ld, found): ld is (n_found x n_found) correlation matrix in the order of variant_ids[found],
    # found is boolean mask over variant_ids of variants present in the LD panel (either orientation)
    # If the panel stores a variant as chr_pos_alt_ref, its LD row and column signs are flipped
    ld_id_to_index = {}
    for ld_index, ld_variant_id in enumerate(ld_variant_ids):
        if ld_variant_id in ld_id_to_index:
            print('assumption error: duplicate variant id in LD panel ' + ld_variant_id)
            pdb.set_trace()
        ld_id_to_index[ld_variant_id] = ld_index
    indices = []
    signs = []
    found = np.zeros(len(variant_ids), dtype=bool)
    for variant_iter, variant_id in enumerate(variant_ids):
        chrom, pos, ref, alt = variant_id.split('_')
        flipped_variant_id = chrom + '_' + pos + '_' + alt + '_' + ref
        if variant_id in ld_id_to_index:
            indices.append(ld_id_to_index[variant_id])
            signs.append(1.0)
            found[variant_iter] = True
        elif flipped_variant_id in ld_id_to_index:
            indices.append(ld_id_to_index[flipped_variant_id])
            signs.append(-1.0)
            found[variant_iter] = True
    indices = np.array(indices, dtype=int)
    signs = np.array(signs)
    ld = ld_mat[np.ix_(indices, indices)] * np.outer(signs, signs)
    return ld, found


def make_ld_positive_semidefinite(ld):
    # Project a (possibly indefinite, e.g. thresholded/quantized reference panel) correlation matrix onto the nearest
    # positive semidefinite matrix by clipping negative eigenvalues at 0, then rescale to unit diagonal
    # Returns (ld_psd, eigenvalues, eigenvectors, min_eigenvalue_before); eigen-decomposition is of the returned matrix
    eigenvalues, eigenvectors = np.linalg.eigh(ld)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue < 0.0:
        ld = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        d = np.sqrt(np.diag(ld))
        ld = ld / np.outer(d, d)
        ld = 0.5 * (ld + ld.T)
        np.fill_diagonal(ld, 1.0)
        eigenvalues, eigenvectors = np.linalg.eigh(ld)
    return ld, eigenvalues, eigenvectors, min_eigenvalue


def estimate_ld_mismatch_s(z, eigenvalues, eigenvectors, s_min=1e-4, s_max=1.0):
    # LD-mismatch parameter s of Zou et al. 2022: maximize log N(z; 0, (1-s) R + s I) over s (golden-section on log s)
    # Note: assumes null z; strong signals push the estimate toward 0, so a floor is applied by the caller
    uz2 = (eigenvectors.T @ z)**2
    def negloglik(log_s):
        s = np.exp(log_s)
        v = (1.0 - s) * eigenvalues + s
        return 0.5 * np.sum(np.log(v) + uz2 / v)
    lo = np.log(s_min)
    hi = np.log(s_max)
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - golden * (hi - lo)
    x2 = lo + golden * (hi - lo)
    f1 = negloglik(x1)
    f2 = negloglik(x2)
    for _ in range(60):
        if f1 > f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + golden * (hi - lo)
            f2 = negloglik(x2)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - golden * (hi - lo)
            f1 = negloglik(x1)
    return float(np.exp(0.5 * (lo + hi)))


def impute_missing_caqtl_sumstats(caqtl_effects, caqtl_se, ld, ridge):
    # Impute missing (nan) caqtl effects/ses for each peak from typed variants using LD (ImpG-style, Pasaniuc et al. 2014)
    #   z_miss = R_mt (R_tt + ridge*I)^-1 z_typed ; r2pred = diag(R_mt (R_tt + ridge*I)^-1 R_tm)
    # Imputed se is the median se of the typed variants in that peak (se ~ constant across variants on standardized-genotype scale)
    # Imputed beta = z_miss * se. No r2pred cutoff: poorly tagged cells are shrunk toward zero by the ridge (fully non-missing output)
    # Returns (caqtl_effects, caqtl_se, imputed_mask, r2pred); r2pred is 1.0 for typed cells
    K, p = caqtl_effects.shape
    caqtl_effects = np.copy(caqtl_effects)
    caqtl_se = np.copy(caqtl_se)
    imputed_mask = np.isnan(caqtl_effects) | np.isnan(caqtl_se)
    r2pred = np.ones((K, p))
    for k in range(K):
        missing = imputed_mask[k, :]
        if np.sum(missing) == 0:
            continue
        typed = ~missing
        if np.sum(typed) == 0:
            print('assumption error: peak ' + str(k) + ' has no typed variants to impute from')
            pdb.set_trace()
        z_typed = caqtl_effects[k, typed] / caqtl_se[k, typed]
        R_tt = ld[np.ix_(typed, typed)] + ridge * np.eye(int(np.sum(typed)))
        R_mt = ld[np.ix_(missing, typed)]
        W = np.linalg.solve(R_tt, R_mt.T).T  # R_mt (R_tt + ridge I)^-1
        z_miss = W @ z_typed
        se_miss = np.median(caqtl_se[k, typed])
        caqtl_effects[k, missing] = z_miss * se_miss
        caqtl_se[k, missing] = se_miss
        r2pred[k, missing] = np.sum(W * R_mt, axis=1)
    return caqtl_effects, caqtl_se, imputed_mask, r2pred


def generate_fine_mapping_input(caqtl_mediated_summary_file, fingen_ld_dir, sum_stats_fm_input_dir, LD_fm_input_dir, cell_type, ldstore_binary, ld_impute_ridge, ld_regularization_floor, fm_input_summary_file):
    # For each gene in caqtl-mediated summary file, get LD for the gene's variants (oriented to the QTL effect allele),
    # drop variants not in the LD panel, and write files + summary in the format used by the fine-mapping simulation
    t = open(fm_input_summary_file, 'w')
    t.write('Gene_Name\tVariant_IDs_File\tLD_File\teQTL_Effects_File\teQTL_SE_File\tcaQTL_Effects_File\tcaQTL_SE_File\tEstimated_Peak_Gene_Effects_File\tEstimated_Peak_Gene_SE_File\tcaQTL_Imputed_Mask_File\tcaQTL_R2pred_File\tLD_Min_Eigenvalue_Before_Repair\tLD_Regularization_s\n')

    n_genes_written = 0
    n_variants_total = 0
    n_variants_missing_ld = 0
    n_caqtl_cells_total = 0
    n_caqtl_cells_imputed = 0
    n_ld_indefinite = 0
    ld_s_all = []
    f = open(caqtl_mediated_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip().split('\t')
        if head_count == 0:
            head_count += 1
            continue
        gene_id = data[0]
        gene_chrom_num = data[1]
        variant_ids = np.loadtxt(data[5], dtype=str, ndmin=1)
        eqtl_effects = np.load(data[6])
        eqtl_se = np.load(data[7])
        peak_ids = np.loadtxt(data[8], dtype=str, ndmin=1) if os.path.getsize(data[8]) > 0 else np.array([], dtype=str)  # empty file for genes with no peaks
        caqtl_effects = np.load(data[9])
        caqtl_se = np.load(data[10])
        peak_gene_effects = np.load(data[11])
        peak_gene_se = np.load(data[12])

        # Load LD and align to QTL variants
        ld_variant_ids, ld_mat = load_ld_for_gene(fingen_ld_dir, gene_chrom_num, variant_ids, ldstore_binary, LD_fm_input_dir + cell_type + '_fm_input_gene_' + gene_id)
        ld, found = align_ld_to_qtl_variants(variant_ids, ld_variant_ids, ld_mat)
        n_variants_total += len(variant_ids)
        n_variants_missing_ld += int(np.sum(~found))
        if np.sum(found) == 0:
            print('Skipping gene ' + gene_id + ' because none of its variants are in the LD panel')
            continue

        # Drop variants not in LD panel
        variant_ids = variant_ids[found]
        eqtl_effects = eqtl_effects[found]
        eqtl_se = eqtl_se[found]
        caqtl_effects = caqtl_effects[:, found]
        caqtl_se = caqtl_se[:, found]

        # Make LD positive semidefinite (thresholded, quantized reference panel LD is generally indefinite)
        ld, eigenvalues, eigenvectors, min_eigenvalue = make_ld_positive_semidefinite(ld)
        if min_eigenvalue < 0.0:
            n_ld_indefinite += 1

        # Regularize LD toward the identity, R <- (1-s) R + s I, to remove zero-curvature directions along which
        # SuSiE-RSS drifts with mismatched reference LD; s = max(estimate from eQTL z-scores, floor)
        ld_s = max(estimate_ld_mismatch_s(eqtl_effects / eqtl_se, eigenvalues, eigenvectors), ld_regularization_floor)
        ld = (1.0 - ld_s) * ld + ld_s * np.eye(len(variant_ids))
        ld_s_all.append(ld_s)

        # Impute caqtl sumstats for variants not tested in a peak's caqtl scan (nan) using LD
        caqtl_effects, caqtl_se, caqtl_imputed_mask, caqtl_r2pred = impute_missing_caqtl_sumstats(caqtl_effects, caqtl_se, ld, ld_impute_ridge)
        n_caqtl_cells_total += caqtl_effects.size
        n_caqtl_cells_imputed += int(np.sum(caqtl_imputed_mask))

        # Save to files
        gene_output_stem = sum_stats_fm_input_dir + cell_type + '_fm_input_gene_' + gene_id
        variant_ids_file = gene_output_stem + '_variant_ids.txt'
        eqtl_effects_file = gene_output_stem + '_eqtl_effects.npy'
        eqtl_se_file = gene_output_stem + '_eqtl_se.npy'
        peak_ids_file = gene_output_stem + '_peak_ids.txt'
        caqtl_effects_file = gene_output_stem + '_caqtl_effects.npy'
        caqtl_se_file = gene_output_stem + '_caqtl_se.npy'
        peak_gene_effects_file = gene_output_stem + '_peak_gene_effects.npy'
        peak_gene_se_file = gene_output_stem + '_peak_gene_se.npy'
        caqtl_imputed_mask_file = gene_output_stem + '_caqtl_imputed_mask.npy'
        caqtl_r2pred_file = gene_output_stem + '_caqtl_r2pred.npy'
        ld_file = LD_fm_input_dir + cell_type + '_fm_input_gene_' + gene_id + '_ld.npy'
        np.savetxt(variant_ids_file, variant_ids, fmt='%s')
        np.save(eqtl_effects_file, eqtl_effects)
        np.save(eqtl_se_file, eqtl_se)
        np.savetxt(peak_ids_file, peak_ids, fmt='%s')
        np.save(caqtl_effects_file, caqtl_effects)
        np.save(caqtl_se_file, caqtl_se)
        np.save(peak_gene_effects_file, peak_gene_effects)
        np.save(peak_gene_se_file, peak_gene_se)
        np.save(caqtl_imputed_mask_file, caqtl_imputed_mask)
        np.save(caqtl_r2pred_file, caqtl_r2pred)
        np.save(ld_file, ld)

        # Write the file names to the summary file (same columns as fine-mapping simulation sumstats summary)
        t.write('\t'.join([gene_id, variant_ids_file, ld_file, eqtl_effects_file, eqtl_se_file, caqtl_effects_file, caqtl_se_file, peak_gene_effects_file, peak_gene_se_file, caqtl_imputed_mask_file, caqtl_r2pred_file, str(min_eigenvalue), str(ld_s)]) + '\n')
        t.flush()
        n_genes_written += 1
    f.close()
    t.close()
    print(str(n_genes_written) + ' genes written to ' + fm_input_summary_file)
    print('Variants: ' + str(n_variants_total - n_variants_missing_ld) + ' kept, ' + str(n_variants_missing_ld) + ' dropped because not in LD panel')
    print('caQTL cells: ' + str(n_caqtl_cells_imputed) + ' of ' + str(n_caqtl_cells_total) + ' imputed from LD')
    print('LD matrices: ' + str(n_ld_indefinite) + ' of ' + str(n_genes_written) + ' were indefinite and projected to positive semidefinite')
    print('LD regularization s: median ' + str(np.median(ld_s_all)) + ', 90th pct ' + str(np.percentile(ld_s_all, 90)) + ', max ' + str(np.max(ld_s_all)) + ' (floor ' + str(ld_regularization_floor) + ')')
    return


parser = argparse.ArgumentParser()
parser.add_argument('--eqtl_file', type=str)
parser.add_argument('--caqtl_file', type=str)
parser.add_argument('--peak_gene_links_file', type=str)
parser.add_argument('--fingen_ld_dir', type=str)
parser.add_argument('--cell_type', type=str)
parser.add_argument('--gene_annotation_file', type=str)
parser.add_argument('--sum_stats_fm_input_dir', type=str)
parser.add_argument('--LD_fm_input_dir', type=str)
parser.add_argument('--distance_window', type=int, default=50000)  # Cis window (bp) around gene TSS
parser.add_argument('--min_peak_variant_coverage', type=float, default=0.8)  # Min fraction of a gene's cis variants a linked peak must have caqtl sumstats for
parser.add_argument('--ldstore_binary', type=str, default='/lab-share/CHIP-Strober-e2/Public/finngen/ld/ldstore_v1.1_x86_64/ldstore')  # Path to LDstore v1.1 executable (reads finngen bcor files)
parser.add_argument('--ld_impute_ridge', type=float, default=0.1)  # Ridge added to LD of typed variants when imputing missing caqtl sumstats
parser.add_argument('--ld_regularization_floor', type=float, default=0.05)  # Minimum LD-mismatch s: LD <- (1-s) LD + s I
args = parser.parse_args()

eqtl_file = args.eqtl_file
caqtl_file = args.caqtl_file
peak_gene_links_file = args.peak_gene_links_file
fingen_ld_dir = args.fingen_ld_dir
cell_type = args.cell_type
gene_annotation_file = args.gene_annotation_file
sum_stats_fm_input_dir = args.sum_stats_fm_input_dir  # Output directory for sum stats fine mapping input files
LD_fm_input_dir = args.LD_fm_input_dir  # Output directory for LD fine mapping input files
distance_window = args.distance_window  # Cis window (bp) around gene TSS
min_peak_variant_coverage = args.min_peak_variant_coverage
ldstore_binary = args.ldstore_binary
ld_impute_ridge = args.ld_impute_ridge
ld_regularization_floor = args.ld_regularization_floor




ens_id_to_chrom_num_and_tss = create_ensamble_id_to_chrom_and_tss_mapping(gene_annotation_file)


##################################
# First generate eqtl summary stats
##################################
eqtl_summary_file = sum_stats_fm_input_dir + cell_type + '_eqtl_sumstats_summary.txt'
generate_eqtl_sumstats(eqtl_file, ens_id_to_chrom_num_and_tss, sum_stats_fm_input_dir, cell_type, distance_window, eqtl_summary_file)


##################################
# Second generate caqtl and peak-gene link summary stats (for caqtl-mediated fine-mapping)
##################################
caqtl_mediated_summary_file = sum_stats_fm_input_dir + cell_type + '_caqtl_mediated_sumstats_summary.txt'
generate_caqtl_and_peak_gene_link_sumstats(eqtl_summary_file, caqtl_file, peak_gene_links_file, sum_stats_fm_input_dir, cell_type, min_peak_variant_coverage, caqtl_mediated_summary_file)


##################################
# Third get LD for each gene and generate fine-mapping input files (matching fine-mapping simulation format)
##################################
fm_input_summary_file = sum_stats_fm_input_dir + cell_type + '_fine_mapping_input_summary.txt'
generate_fine_mapping_input(caqtl_mediated_summary_file, fingen_ld_dir, sum_stats_fm_input_dir, LD_fm_input_dir, cell_type, ldstore_binary, ld_impute_ridge, ld_regularization_floor, fm_input_summary_file)


