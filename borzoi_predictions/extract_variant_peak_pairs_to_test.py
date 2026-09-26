import numpy as np
import os
import sys
import pdb
import pyarrow.parquet as pq
import pandas as pd
import pyarrow.compute as pc
import gzip



def extract_ordered_list_of_all_variant_ids_on_this_chromosome(gtex_sumstats_dir, chrom_string, pc_genes):
	variants = {}
	pc_genes_set = set(pc_genes)

	for file_name in os.listdir(gtex_sumstats_dir):
		if not file_name.endswith('chr' + str(chrom_string) + '.parquet'):
			continue

		print(file_name)
		pf = pq.ParquetFile(gtex_sumstats_dir + file_name)

		for rg in range(pf.num_row_groups):
			table = pf.read_row_group(
				rg,
				columns=['gene_id', 'variant_id', 'tss_distance', 'af']
			)

			if table.num_columns == 0:
				continue

			gene_col = table['gene_id']
			variant_col = table['variant_id']
			dist_col = table['tss_distance']
			af_col = table['af']

			mask = pc.less_equal(pc.abs(dist_col), 100000)

			filt = table.filter(mask)

			if filt.num_rows == 0:
				continue

			gene_ids = filt['gene_id'].to_pylist()
			variant_ids = filt['variant_id'].to_pylist()


			for gene_id, variant_id in zip(gene_ids, variant_ids):
				if gene_id.split('.', 1)[0] not in pc_genes_set:
					continue
				var_info = variant_id.split('_')
				# Skip variants that are not SNPs (i.e., indels or structural variants)
				if len(var_info[2]) > 1 or len(var_info[3]) > 1:
					continue
				variants[variant_id] = 1

	return variants

def extract_dictionary_list_of_protein_coding_genes(pc_genes_gtf):
	f = open(pc_genes_gtf)
	dicti = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		ens_id = data[8].split(';')[0].split('"')[1]
		if ens_id.startswith('ENSG') == False:
			print('assumption oernroro')
			pdb.set_trace()
		dicti[ens_id.split('.')[0]] = 1

	f.close()

	return dicti



def make_variant_vcf_file(variant_gene_pair_file, vcf_file):
	f = open(variant_gene_pair_file)
	head_count = 0
	var_dicti = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		var_dicti[data[0]] = 1
	f.close()

	tupler = []
	for var_id in [*var_dicti]:
		if len(var_id.split('_')) == 1:
			print(var_id)
			continue
		chromer = int(var_id.split('_')[0].split('hr')[1])
		position = int(var_id.split('_')[1])
		tupler.append((chromer, position, var_id))

	tupler.sort(key=lambda x: (x[0], x[1]))

	t = open(vcf_file,'w')

	for tup in tupler:
		var_id = tup[2]
		chromer, pos, a1, a2, garbage = var_id.split('_')

		if len(a1) != 1 or len(a2) != 1:
			continue

		t.write(chromer + '\t' + pos + '\t' + var_id + '\t' + a1 + '\t' + a2 + '\n')

	t.close()

	return

def extract_peaks_on_this_chromosome(peak_gene_reference_file, chrom_string):
	f = gzip.open(peak_gene_reference_file, 'rt')
	peak_set = {}
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		peak_id = data[0]
		peak_chrom = peak_id.split('-')[0].split('hr')[1]
		if peak_chrom != chrom_string:
			continue
		peak_set[peak_id] = 1
	f.close()
	return peak_set

def extract_variant_peak_pairs(variant_set, peak_set, window=10000):
	# Sort variants by position so each peak can be queried with a binary search
	variant_ids = np.asarray([*variant_set])
	variant_pos = np.asarray([int(var_id.split('_')[1]) for var_id in variant_ids])
	ordering = np.argsort(variant_pos)
	variant_ids = variant_ids[ordering]
	variant_pos = variant_pos[ordering]

	variant_peak_pairs = {}
	for peak_id in [*peak_set]:
		peak_start = int(peak_id.split('-')[1])
		peak_end = int(peak_id.split('-')[2])
		# Variants with position in [peak_start - window, peak_end + window]
		lo = np.searchsorted(variant_pos, peak_start - window, side='left')
		hi = np.searchsorted(variant_pos, peak_end + window, side='right')
		for variant_id in variant_ids[lo:hi]:
			variant_peak_pairs[variant_id + ':' + peak_id] = 1

	return variant_peak_pairs






# Command line args
eqtl_sumstats_dir = sys.argv[1]
variant_peak_pair_file = sys.argv[2]
variant_output_stem = sys.argv[3]
pc_genes_gtf = sys.argv[4]
peak_gene_reference_file = sys.argv[5]



# Extract dictionary list of protein coding genes
pc_genes = extract_dictionary_list_of_protein_coding_genes(pc_genes_gtf)

t = open(variant_peak_pair_file,'w')
t.write('variant_id\tpeak_id\n')

for chrom_num in range(1,23):
	print(chrom_num)

	# Extract list of peaks on this chromosome
	peak_set = extract_peaks_on_this_chromosome(peak_gene_reference_file, str(chrom_num))

	# Extract  list of variant ids in this chromosome
	variant_set = extract_ordered_list_of_all_variant_ids_on_this_chromosome(eqtl_sumstats_dir, str(chrom_num), pc_genes)

	# Extract variant-peak pairs where the variant is within 25KB of a peak boundary
	variant_peak_pairs = extract_variant_peak_pairs(variant_set, peak_set, window=25000)

	
	for vp_pair in [*variant_peak_pairs]:
		variant_id, peak_id = vp_pair.split(':')

		t.write(variant_id + '\t' + peak_id + '\n')

t.close()

# Make variant vcf file
vcf_file = variant_output_stem + 'all_variant.vcf'
make_variant_vcf_file(variant_peak_pair_file, vcf_file)
