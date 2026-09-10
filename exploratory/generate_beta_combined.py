import argparse
import numpy as np
import os
import sys
import pdb
import gzip

def load_in_caqtl_data(fingen_caqtl_file):
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
        beta = float(data[11])
        beta_se = float(data[12])
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

def generate_beta_combined_all_links(fingen_peak_gene_links_file, caqtl_data):
    # For each (gene, variant) pair:
    #   beta_pred = sum_k beta_caqtl_k * beta_link_k            (k = peaks linked to the gene)
    #   var_pred  = sum_k beta_link_k^2 * se_caqtl_k^2 + beta_caqtl_k^2 * se_link_k^2 + se_caqtl_k^2 * se_link_k^2
    # beta_link / se_link are from the negative binomial count component of the hurdle model
    f = gzip.open(fingen_peak_gene_links_file, 'rt')
    head_count = 0
    beta_combined_data = {}
    seen_links = set()
    line_num = 0
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
            continue

        peak_id = data[0]
        gene_id = data[1]
        if data[beta_link_col] == "NA" or data[se_link_col] == "NA":
            continue
        beta_link = float(data[beta_link_col])
        se_link = float(data[se_link_col])

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
            var_pred_k = beta_link_sq * se_caqtl_sq + (beta_caqtl ** 2) * se_link_sq + se_caqtl_sq * se_link_sq
            if variant_id not in gene_dict:
                # [beta_pred, var_pred, af, n_peaks]
                gene_dict[variant_id] = [beta_pred_k, var_pred_k, af, 1]
            else:
                entry = gene_dict[variant_id]
                entry[0] += beta_pred_k
                entry[1] += var_pred_k
                entry[3] += 1
    f.close()

    # Convert accumulated variance to standard error
    for gene_id in beta_combined_data:
        gene_dict = beta_combined_data[gene_id]
        for variant_id, entry in gene_dict.items():
            gene_dict[variant_id] = (entry[0], np.sqrt(entry[1]), entry[2], entry[3])
    return beta_combined_data


def generate_beta_combined(fingen_peak_gene_links_file, caqtl_data, combination_version):
    if combination_version == "all_links":
        beta_combined_data = generate_beta_combined_all_links(fingen_peak_gene_links_file, caqtl_data)
    else:
        print("Invalid combination version: %s" % combination_version)
        sys.exit(1)
    return beta_combined_data


def write_beta_combined_output(fingen_eqtl_file, beta_combined_data, output_file):
    # eQTL cis_nominal columns: #CHR, POS, cell_type, phenotype_id, MarkerID, AF_Allele2, BETA, SE, p.value
    # Every variant-gene pair in the eQTL file is written; pairs with no prediction get NA for beta_combined / se_combined and 0 for n_peaks
    # af is AF_Allele2 from the eQTL file (alternate allele frequency, same orientation as the caQTL af)
    f = gzip.open(fingen_eqtl_file, 'rt')
    t = gzip.open(output_file, 'wt')
    t.write('\t'.join(['variant_id', 'gene_id', 'beta_eqtl_hat', 'beta_eqtl_se', 'beta_combined', 'se_combined', 'af', 'n_peaks']) + '\n')
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
            beta_combined, se_combined, _, n_peaks = beta_combined_data[gene_id][variant_id]
            beta_combined = str(beta_combined)
            se_combined = str(se_combined)
            n_peaks = str(n_peaks)
            n_with_prediction += 1
        else:
            beta_combined = "NA"
            se_combined = "NA"
            n_peaks = "0"

        t.write('\t'.join([variant_id, gene_id, beta_eqtl_hat, beta_eqtl_se, beta_combined, se_combined, af, n_peaks]) + '\n')
        n_written += 1
    f.close()
    t.close()
    print("eQTL variant-gene pairs written: %d; with beta_combined prediction: %d" % (n_written, n_with_prediction), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingen_eqtl_file', type=str, required=True)
    parser.add_argument('--fingen_caqtl_file', type=str, required=True)
    parser.add_argument('--fingen_peak_gene_links_file', type=str, required=True)
    parser.add_argument('--combination_version', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    fingen_eqtl_file = args.fingen_eqtl_file
    fingen_caqtl_file = args.fingen_caqtl_file
    fingen_peak_gene_links_file = args.fingen_peak_gene_links_file
    combination_version = args.combination_version
    output_file = args.output_file

    # First load in caqtl effect sizes and standard errors
    caqtl_data = load_in_caqtl_data(fingen_caqtl_file)

    # Second load in peak-gene links, and generate beta combined
    beta_combined_data = generate_beta_combined(fingen_peak_gene_links_file, caqtl_data, combination_version)


    # Loop through eqtl file, and write out file with variant_id, gene_id, beta_eqtl_hat, beta_eqtl_se, beta_combined, se_combined, af, n_peaks
    write_beta_combined_output(fingen_eqtl_file, beta_combined_data, output_file)





