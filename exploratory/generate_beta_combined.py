import argparse
import numpy as np
import os
import sys
import pdb
import gzip

def load_in_caqtl_data(fingen_caqtl_file, peak_to_scaling_factor):
    f = gzip.open(fingen_caqtl_file, 'rt')
    head_count = 0
    obj = {}
    line_num = 0
    for line in f:
        line_num += 1
        if line_num % 1000000 == 0:
            print(line_num, flush=True)
        if head_count == 0:
            head_count += 1
            header = line.strip().split('\t')
            continue
        data = line.strip().split('\t')

        peak_id = data[3]
        variant_id = data[4]
        if data[11] == "NA" or data[12] == "NA":
            continue

        if peak_id not in peak_to_scaling_factor:
            print("Peak id %s not found in peak re-scaling file" % peak_id)
            sys.exit(1)

        scaling_factor = peak_to_scaling_factor[peak_id]


        beta = float(data[11])*scaling_factor
        beta_se = float(data[12])*scaling_factor
        af = float(data[7])

        if peak_id not in obj:
            obj[peak_id] = {}
        if variant_id not in obj[peak_id]:
            obj[peak_id][variant_id] = (beta, beta_se, af)
        else:
            print("Duplicate peak_id, variant_id pair found in caQTL file: %s, %s" % (peak_id, variant_id))
            sys.exit(1)
    f.close()
    return obj

def generate_beta_combined_all_links(fingen_peak_gene_links_file, caqtl_data, link_version):
    # For each (gene, variant) pair:
    #   beta_pred = sum_k beta_caqtl_k * beta_link_k            (k = peaks linked to the gene)
    #   var_pred  = sum_k beta_link_k^2 * se_caqtl_k^2 + beta_caqtl_k^2 * se_link_k^2 + se_caqtl_k^2 * se_link_k^2
    #   var_pred_unbiased = same with the last term subtracted rather than added
    # var_pred plugs the estimated betas into the exact variance of a product of independent normals. Since
    # E[beta_hat^2] = beta^2 + se^2, that overstates the variance by 2 * se_caqtl^2 * se_link^2 per peak (threefold
    # for a null pair). var_pred_unbiased has the right expectation but can be negative for near-null pairs, so it is
    # for moment-based estimators (noise-corrected correlation), not for z-scores; se_combined stays conservative.
    # link_version == 'count_only':      beta_link / se_link are from the count component of the hurdle model
    # link_version == 'hurdle_combined': beta_link = beta_count + (1 - p) * beta_zero, p = expr_cell_num / total_cell_num
    #                                    se_link^2 = se_count^2 + (1 - p)^2 * se_zero^2 (components treated as independent)
    #                                    from d log(p * mu)/dx; assumes beta_zero is on the logit P(expressing) scale
    f = gzip.open(fingen_peak_gene_links_file, 'rt')
    head_count = 0
    beta_combined_data = {}
    seen_links = set()
    line_num = 0
    n_nonfinite_links = 0
    max_se_link = 1.0
    se_links_kept = []
    for line in f:
        line_num += 1
        if line_num % 100000 == 0:
            print(line_num, flush=True)
        data = line.strip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            beta_link_col = header.index('hurdle_count_beta')
            se_link_col = header.index('hurdle_count_se')
            beta_zero_col = header.index('hurdle_zero_beta')
            se_zero_col = header.index('hurdle_zero_se')
            total_cell_col = header.index('total_cell_num')
            expr_cell_col = header.index('expr_cell_num')
            continue

        peak_id = data[0]
        gene_id = data[1]
        if data[beta_link_col] == "NA" or data[se_link_col] == "NA":
            continue
        beta_link = float(data[beta_link_col])
        se_link = float(data[se_link_col])
        if link_version == 'hurdle_combined':
            if data[beta_zero_col] == "NA" or data[se_zero_col] == "NA":
                continue
            p_expr = float(data[expr_cell_col]) / float(data[total_cell_col])
            beta_link = beta_link + (1.0 - p_expr) * float(data[beta_zero_col])
            se_link = np.sqrt(se_link ** 2 + ((1.0 - p_expr) ** 2) * (float(data[se_zero_col]) ** 2))
        elif link_version != 'count_only':
            print("Invalid link version: %s" % link_version)
            sys.exit(1)
        if not (np.isfinite(beta_link) and np.isfinite(se_link)) or se_link > max_se_link:
            # Degenerate fits (e.g. a near-separated logistic zero component) give infinite or absurd SEs. Counts are ~0/1,
            # so an SE above max_se_link per read cannot come from a real fit. A few such links dominate the mean error
            # variance across all pairs and break the noise-corrected correlation.
            n_nonfinite_links += 1
            continue
        se_links_kept.append(se_link)

        if (peak_id, gene_id) in seen_links:
            print("Duplicate peak_id, gene_id pair found in peak-gene links file: %s, %s" % (peak_id, gene_id))
            sys.exit(1)
        seen_links.add((peak_id, gene_id))

        if peak_id not in caqtl_data:
            continue

        if gene_id not in beta_combined_data:
            beta_combined_data[gene_id] = {}
        gene_dict = beta_combined_data[gene_id]

        beta_link_sq = beta_link ** 2
        se_link_sq = se_link ** 2
        for variant_id, (beta_caqtl, se_caqtl, af) in caqtl_data[peak_id].items():
            se_caqtl_sq = se_caqtl ** 2
            beta_pred_k = beta_caqtl * beta_link
            cross_term = se_caqtl_sq * se_link_sq
            var_pred_k = beta_link_sq * se_caqtl_sq + (beta_caqtl ** 2) * se_link_sq
            if variant_id not in gene_dict:
                # [beta_pred, var_pred, af, n_peaks, var_pred_unbiased, |top contribution|, beta_caqtl_top, beta_link_top]
                # The last three track the single peak contributing the largest |beta_caqtl * beta_link| to the sum
                gene_dict[variant_id] = [beta_pred_k, var_pred_k + cross_term, af, 1, var_pred_k - cross_term,
                                         abs(beta_pred_k), beta_caqtl, beta_link]
            else:
                entry = gene_dict[variant_id]
                entry[0] += beta_pred_k
                entry[1] += var_pred_k + cross_term
                entry[3] += 1
                entry[4] += var_pred_k - cross_term
                if abs(beta_pred_k) > entry[5]:
                    entry[5] = abs(beta_pred_k)
                    entry[6] = beta_caqtl
                    entry[7] = beta_link
    f.close()
    print("Peak-gene links skipped for non-finite or > %.1f se_link: %d" % (max_se_link, n_nonfinite_links), flush=True)
    se_links_kept = np.array(se_links_kept)
    print("se_link among kept links: median = %.3g, 99th = %.3g, 99.9th = %.3g, 99.99th = %.3g, max = %.3g" % (
        np.median(se_links_kept), np.percentile(se_links_kept, 99), np.percentile(se_links_kept, 99.9),
        np.percentile(se_links_kept, 99.99), se_links_kept.max()), flush=True)

    # Convert accumulated variance to standard error
    for gene_id in beta_combined_data:
        gene_dict = beta_combined_data[gene_id]
        for variant_id, entry in gene_dict.items():
            gene_dict[variant_id] = (entry[0], np.sqrt(entry[1]), entry[2], entry[3], entry[4], entry[6], entry[7])
    return beta_combined_data


def generate_beta_combined(fingen_peak_gene_links_file, caqtl_data, combination_version):
    if combination_version == "all_links":
        beta_combined_data = generate_beta_combined_all_links(fingen_peak_gene_links_file, caqtl_data, 'count_only')
    elif combination_version == "all_links_hurdle":
        beta_combined_data = generate_beta_combined_all_links(fingen_peak_gene_links_file, caqtl_data, 'hurdle_combined')
    else:
        print("Invalid combination version: %s" % combination_version)
        sys.exit(1)
    return beta_combined_data


def write_beta_combined_output(fingen_eqtl_file, beta_combined_data, output_file):
    # eQTL cis_nominal columns: #CHR, POS, cell_type, phenotype_id, MarkerID, AF_Allele2, BETA, SE, p.value
    # Every variant-gene pair in the eQTL file is written; pairs with no prediction get NA for beta_combined / se_combined /
    # var_combined_unbiased and 0 for n_peaks. beta_caqtl_top / beta_link_top are the components of the peak contributing
    # the largest |beta_caqtl * beta_link| to the pair's sum (NA when there is no prediction)
    # af is AF_Allele2 from the eQTL file (alternate allele frequency, same orientation as the caQTL af)
    f = gzip.open(fingen_eqtl_file, 'rt')
    t = gzip.open(output_file, 'wt')
    t.write('\t'.join(['variant_id', 'gene_id', 'beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined', 'af', 'n_peaks',
                       'var_combined_unbiased', 'beta_caqtl_top', 'beta_link_top']) + '\n')
    head_count = 0
    line_num = 0
    n_written = 0
    n_with_prediction = 0
    for line in f:
        line_num += 1
        if line_num % 1000000 == 0:
            print(line_num, flush=True)
        data = line.strip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            gene_col = header.index('phenotype_id')
            variant_col = header.index('MarkerID')
            af_col = header.index('AF_Allele2')
            beta_col = header.index('BETA')
            se_col = header.index('SE')
            continue

        gene_id = data[gene_col]
        variant_id = data[variant_col]
        af = data[af_col]
        beta_eqtl_hat = data[beta_col]
        beta_eqtl_se = data[se_col]
        if beta_eqtl_hat == "NA" or beta_eqtl_se == "NA":
            continue

        if gene_id in beta_combined_data and variant_id in beta_combined_data[gene_id]:
            beta_combined, se_combined, _, n_peaks, var_unbiased, beta_caqtl_top, beta_link_top = beta_combined_data[gene_id][variant_id]
            beta_combined = str(beta_combined)
            se_combined = str(se_combined)
            n_peaks = str(n_peaks)
            var_unbiased = str(var_unbiased)
            beta_caqtl_top = str(beta_caqtl_top)
            beta_link_top = str(beta_link_top)
            n_with_prediction += 1
        else:
            beta_combined = "NA"
            se_combined = "NA"
            n_peaks = "0"
            var_unbiased = "NA"
            beta_caqtl_top = "NA"
            beta_link_top = "NA"

        t.write('\t'.join([variant_id, gene_id, beta_eqtl_hat, beta_eqtl_se, beta_combined, se_combined, af, n_peaks, var_unbiased,
                           beta_caqtl_top, beta_link_top]) + '\n')
        n_written += 1
    f.close()
    t.close()
    print("eQTL variant-gene pairs written: %d; with beta_combined prediction: %d" % (n_written, n_with_prediction), flush=True)

def create_mapping_from_peak_to_scaling_factor(peak_re_scaling_file, scaling_version):
    mapping = {}
    peak_to_a_p = {}
    f = gzip.open(peak_re_scaling_file, 'rt')
    scaling_factors = []
    head_count = 0
    for line in f:
        if head_count == 0:
            head_count += 1
            header = np.copy(line.strip().split('\t'))
            continue
        data = line.strip().split('\t')
        if len(data) != len(header):
            print("Line in peak re-scaling file has different number of columns than header: %s" % line.strip())
            sys.exit(1)
        peak_id = data[2]
        a_p = float(data[15]) # Mean ATAC counts per nucleus in the peak (across all nuclei in the cell type))
        robust_sd_p = float(data[11]) # Robust across-donor standard deviation of the pre-INT accessibility phenotype for that peak
        sd_p = float(data[8]) # Across-donor standard deviation of the pre-INT accessibility phenotype for that peak
        # scaling_version == 'a_p_robust_sd': delta-count per allele (first order) is ln2 * a_p * robust_sd_p * beta_caqtl.
        #     beta_link is a per-count slope on natural-log expression, so the product is in ln units; dividing by ln2 to put it
        #     on the log2 scale of the pre-INT eQTL phenotype cancels the ln2 here. Multiply by ln2 to recover natural-log units.
        # scaling_version == 'robust_sd': beta_caqtl in log2 accessibility units (no count conversion). Under a sparse 0/1 count
        #     model the a_p in the count conversion cancels the attenuation of the per-count link slope, so this is expected to
        #     be the most stable across peaks.
        # scaling_version == 'none': raw INT-scale beta_caqtl.
        if scaling_version == 'a_p_robust_sd':
            scaling_factor = a_p*robust_sd_p
        elif scaling_version == 'robust_sd':
            scaling_factor = robust_sd_p
        elif scaling_version == 'none':
            scaling_factor = 1.0
        else:
            print("Invalid scaling version: %s" % scaling_version)
            sys.exit(1)
        if peak_id in mapping:
            print("Duplicate peak_id found in peak re-scaling file: %s" % peak_id)
            sys.exit(1)
        mapping[peak_id] = scaling_factor
        peak_to_a_p[peak_id] = a_p
        scaling_factors.append(scaling_factor)
    scaling_factors = np.array(scaling_factors)
    return mapping, peak_to_a_p


def run_peak_gene_link_sanity_checks(fingen_peak_gene_links_file, peak_to_a_p):
    # Check 1 (units of the link betas): regress log(hurdle_count_se) on log(sd_peak_acc_expr) across links.
    #   slope ~ -1 => betas are per unit count; slope ~ 0 => betas are per SD of accessibility
    # Check 2 (predictor is the raw per-nucleus count): if so, sd_peak_acc ~ sqrt(a_p * (1 - a_p)) with a_p from the re-scaling file
    # Check 3 (sign of the zero component): among links with sig_zero and sig_count both true, the two betas should mostly
    #   share a sign if hurdle_zero_beta is on the P(expressing) scale; mostly opposite => it is on the P(zero) scale
    f = gzip.open(fingen_peak_gene_links_file, 'rt')
    head_count = 0
    log_se = []
    log_sd = []
    sd_ratio = []
    n_agree = 0
    n_disagree = 0
    strat_bins = [0, 2, 4, 6, 10, 20, np.inf]  # bins of min(zero_nlog10p, count_nlog10p) among doubly-sig links
    strat_agree = np.zeros(len(strat_bins) - 1)
    strat_total = np.zeros(len(strat_bins) - 1)
    mode_agree = {}
    mode_total = {}
    true_strings = set(['TRUE', 'True', 'true', '1'])
    for line in f:
        data = line.strip().split('\t')
        if head_count == 0:
            head_count += 1
            header = data
            count_beta_col = header.index('hurdle_count_beta')
            count_se_col = header.index('hurdle_count_se')
            zero_beta_col = header.index('hurdle_zero_beta')
            sd_acc_col = header.index('sd_peak_acc')
            sd_acc_expr_col = header.index('sd_peak_acc_expr')
            sig_zero_col = header.index('sig_zero')
            sig_count_col = header.index('sig_count')
            zero_nlog10p_col = header.index('hurdle_zero_nlog10p')
            count_nlog10p_col = header.index('hurdle_count_nlog10p')
            mode_col = header.index('mode')
            continue
        peak_id = data[0]
        if data[count_se_col] != "NA" and data[sd_acc_expr_col] != "NA":
            se = float(data[count_se_col])
            sd = float(data[sd_acc_expr_col])
            if se > 0 and sd > 0:
                log_se.append(np.log(se))
                log_sd.append(np.log(sd))
        if data[sd_acc_col] != "NA" and peak_id in peak_to_a_p:
            a_p = peak_to_a_p[peak_id]
            if 0 < a_p < 1:
                sd_ratio.append(float(data[sd_acc_col]) / np.sqrt(a_p * (1.0 - a_p)))
        if data[sig_zero_col] in true_strings and data[sig_count_col] in true_strings:
            if data[count_beta_col] != "NA" and data[zero_beta_col] != "NA":
                agree = np.sign(float(data[count_beta_col])) == np.sign(float(data[zero_beta_col]))
                if agree:
                    n_agree += 1
                else:
                    n_disagree += 1
                if data[zero_nlog10p_col] != "NA" and data[count_nlog10p_col] != "NA":
                    min_nlog10p = min(float(data[zero_nlog10p_col]), float(data[count_nlog10p_col]))
                    b = min(np.searchsorted(strat_bins, min_nlog10p, side='right') - 1, len(strat_bins) - 2)  # inf (p = 0) goes in top bin
                    strat_total[b] += 1
                    strat_agree[b] += agree
                mode = data[mode_col]
                mode_total[mode] = mode_total.get(mode, 0) + 1
                mode_agree[mode] = mode_agree.get(mode, 0) + int(agree)
    f.close()
    log_se = np.array(log_se)
    log_sd = np.array(log_sd)
    slope = np.cov(log_sd, log_se)[0, 1] / np.var(log_sd, ddof=1)
    print("Check 1: slope of log(hurdle_count_se) on log(sd_peak_acc_expr) = %.3f (n = %d); ~-1 => per-count, ~0 => per-SD" % (slope, len(log_se)), flush=True)
    sd_ratio = np.array(sd_ratio)
    print("Check 2: sd_peak_acc / sqrt(a_p * (1 - a_p)): median = %.3f, 5th-95th pct = %.3f-%.3f (n = %d); ~1 => raw per-nucleus count" % (np.median(sd_ratio), np.percentile(sd_ratio, 5), np.percentile(sd_ratio, 95), len(sd_ratio)), flush=True)
    print("Check 3: among links with sig_zero and sig_count: sign agree = %d, disagree = %d; mostly agree => beta_zero on P(expressing) scale" % (n_agree, n_disagree), flush=True)
    for b in range(len(strat_bins) - 1):
        if strat_total[b] > 0:
            print("  Check 3 by min(zero, count) -log10p in [%s, %s): fraction agree = %.3f (n = %d)" % (strat_bins[b], strat_bins[b + 1], strat_agree[b] / strat_total[b], strat_total[b]), flush=True)
    for mode in sorted(mode_total.keys()):
        print("  Check 3 by mode = %s: fraction agree = %.3f (n = %d)" % (mode, mode_agree[mode] / mode_total[mode], mode_total[mode]), flush=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingen_eqtl_file', type=str, required=True)
    parser.add_argument('--fingen_caqtl_file', type=str, required=True)
    parser.add_argument('--fingen_peak_gene_links_file', type=str, required=True)
    parser.add_argument('--combination_version', type=str, required=True)
    parser.add_argument('--peak_re_scaling_file', type=str, required=True)
    parser.add_argument('--scaling_version', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    fingen_eqtl_file = args.fingen_eqtl_file
    fingen_caqtl_file = args.fingen_caqtl_file
    fingen_peak_gene_links_file = args.fingen_peak_gene_links_file
    combination_version = args.combination_version
    peak_re_scaling_file = args.peak_re_scaling_file
    scaling_version = args.scaling_version
    output_file = args.output_file

    # Zeroth, create mapping from peak id to scaling factor
    peak_to_scaling_factor, peak_to_a_p = create_mapping_from_peak_to_scaling_factor(peak_re_scaling_file, scaling_version)

    # Sanity checks on the peak-gene links file (units of link betas, predictor scale, sign of zero component)
    #run_peak_gene_link_sanity_checks(fingen_peak_gene_links_file, peak_to_a_p)

    # First load in caqtl effect sizes and standard errors
    caqtl_data = load_in_caqtl_data(fingen_caqtl_file, peak_to_scaling_factor)

    # Second load in peak-gene links, and generate beta combined
    beta_combined_data = generate_beta_combined(fingen_peak_gene_links_file, caqtl_data, combination_version)


    # Loop through eqtl file, and write out file with variant_id, gene_id, beta_eqtl_hat, beta_eqtl_se, beta_combined, se_combined, af, n_peaks, var_combined_unbiased
    write_beta_combined_output(fingen_eqtl_file, beta_combined_data, output_file)





