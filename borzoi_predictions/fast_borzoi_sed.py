#!/usr/bin/env python
# Copyright 2022 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
from collections import OrderedDict
import json
import pickle
import os
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pysam
from scipy.special import rel_entr
import time
import pdb
from numba import njit, prange

from baskerville.gene import Transcriptome
from baskerville import dataset
from baskerville import seqnn
from baskerville import vcf as bvcf
from baskerville import dna
import tensorflow as tf
import gzip

"""
borzoi_sed.py

Compute SNP Expression Difference (SED) scores for SNPs in a VCF file,
relative to gene exons in a GTF file.
"""

# =====================================================
# 1. Precompute a 256x4 lookup table ONCE.
# =====================================================

DNA_LUT = np.zeros((256, 4), dtype=np.float16)

# A,C,G,T maps
DNA_LUT[ord("A")] = [1, 0, 0, 0]
DNA_LUT[ord("C")] = [0, 1, 0, 0]
DNA_LUT[ord("G")] = [0, 0, 1, 0]
DNA_LUT[ord("T")] = [0, 0, 0, 1]

# lowercase (optional if input sometimes lowercase)
DNA_LUT[ord("a")] = [1, 0, 0, 0]
DNA_LUT[ord("c")] = [0, 1, 0, 0]
DNA_LUT[ord("g")] = [0, 0, 1, 0]
DNA_LUT[ord("t")] = [0, 0, 0, 1]

# N → uniform
DNA_LUT[ord("N")] = [0.25, 0.25, 0.25, 0.25]
DNA_LUT[ord("n")] = [0.25, 0.25, 0.25, 0.25]



mapping2 = {}
mapping2['A'] = [1, 0, 0, 0]
mapping2['C'] = [0, 1, 0, 0]
mapping2['G'] = [0, 0, 1, 0]
mapping2['T'] = [0, 0, 0, 1]
mapping2['a'] = [1, 0, 0, 0]
mapping2['c'] = [0, 1, 0, 0]
mapping2['g'] = [0, 0, 1, 0]
mapping2['t'] = [0, 0, 0, 1]
mapping2['N'] = [0.25, 0.25, 0.25, 0.25]
mapping2['n'] = [0.25, 0.25, 0.25, 0.25]


def dna_1hot_ultrafast(seq, seq_len=None):
	"""Fastest version assuming n_uniform=True always."""
	
	L = len(seq)

	# --- Center trim/pad ---
	if seq_len is None:
		seq_len = L
		seq_start = 0
	else:
		if seq_len <= L:
			trim = (L - seq_len) // 2
			seq = seq[trim : trim + seq_len]
			L = seq_len
			seq_start = 0
		else:
			seq_start = (seq_len - L) // 2

	# --- allocate output ---
	out = np.zeros((seq_len, 4), dtype=np.float16)

	if L == 0:
		return out

	region = slice(seq_start, seq_start + L)

	# Convert to uint8 view directly — NO .upper(), NO masks
	seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

	# Lookup all rows at once
	out[region] = DNA_LUT[seq_bytes]

	return out


def load_in_vg_pairs_to_test(vgpairfile):
	dicti = {}
	f = open(vgpairfile)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 128
			continue
		if len(data) != 2:
			continue
		variant_id = data[0]
		# Strip gene version if present; no-op for peak ids (no dots)
		gene_id = data[1].split('.')[0]
		vg_pair = variant_id + ':' + gene_id
		dicti[vg_pair] = 1
	f.close()
	return dicti

def dna_1hot_fast(
	seq: str, seq_len: int = None, n_uniform: bool = False, n_sample: bool = False
):
	"""Vectorized version of dna_1hot."""
	# -------------------------
	# 1. Handle length / centering
	# -------------------------
	if seq_len is None:
		seq_len = len(seq)
		seq_start = 0
	else:
		if seq_len <= len(seq):
			# trim the sequence (centered)
			seq_trim = (len(seq) - seq_len) // 2
			seq = seq[seq_trim : seq_trim + seq_len]
			seq_start = 0
		else:
			# pad (centered) -> string stays same, we shift where we write
			seq_start = (seq_len - len(seq)) // 2

	seq = seq.upper()
	L = len(seq)  # length after optional trimming

	# -------------------------
	# 2. Allocate output
	# -------------------------
	if n_uniform:
		seq_code = np.zeros((seq_len, 4), dtype="float16")
	else:
		seq_code = np.zeros((seq_len, 4), dtype="bool")

	if L == 0:
		return seq_code

	# slice in seq_code that corresponds to the original sequence
	region = slice(seq_start, seq_start + L)
	region_view = seq_code[region]

	# -------------------------
	# 3. Vectorized base mapping
	# -------------------------
	# Turn the sequence into a NumPy array of bytes
	bases = np.frombuffer(seq.encode("ascii"), dtype="S1")

	mask_A = bases == b"A"
	mask_C = bases == b"C"
	mask_G = bases == b"G"
	mask_T = bases == b"T"

	# Set A/C/G/T positions
	region_view[mask_A, 0] = 1
	region_view[mask_C, 1] = 1
	region_view[mask_G, 2] = 1
	region_view[mask_T, 3] = 1

	# -------------------------
	# 4. Handle N / other bases
	# -------------------------
	valid_mask = mask_A | mask_C | mask_G | mask_T
	n_mask = ~valid_mask

	if n_uniform:
		# represent N's as 0.25 across A/C/G/T
		region_view[n_mask, :] = 0.25
	elif n_sample:
		# randomly sample a base for each N
		n_idx = np.where(n_mask)[0]
		if n_idx.size > 0:
			# choose a random column 0..3 for each N
			rand_cols = np.random.randint(0, 4, size=n_idx.size)
			region_view[n_idx, rand_cols] = 1

	return seq_code


def dna_length_1hot(seq, length):
	"""Adjust the sequence length and compute
	a 1hot coding."""

	if length < len(seq):
		# trim the sequence
		seq_trim = (len(seq) - length) // 2
		seq = seq[seq_trim : seq_trim + length]

	elif length > len(seq):
		# extend with N's
		nfront = (length - len(seq)) // 2
		nback = length - len(seq) - nfront
		seq = "N" * nfront + seq + "N" * nback

	# n_uniform required to avoid different
	#   random nucleotides for each allele
	#seq_1hot_orig = dna.dna_1hot(seq, n_uniform=True)
	seq_1hot = dna_1hot_fast(seq, n_uniform=True)


	return seq_1hot, seq

def snp_seq1(snp, seq_len, genome_open):
	"""Produce one hot coded sequences for a SNP.

	Attrs:
		snp [SNP] :
		seq_len (int) : sequence length to code
		genome_open (File) : open genome FASTA file

	Return:
		seq_vecs_list [array] : list of one hot coded sequences surrounding the
		SNP
	"""
	left_len = seq_len // 2 - 1
	right_len = seq_len // 2

	# initialize one hot coded vector list
	seq_vecs_list = []

	# specify positions in GFF-style 1-based
	seq_start = snp.pos - left_len
	seq_end = snp.pos + right_len + max(0, len(snp.ref_allele) - snp.longest_alt())

	# extract sequence as BED style
	if seq_start < 0:
		seq = "N" * (1 - seq_start) + genome_open.fetch(snp.chr, 0, seq_end).upper()
	else:
		seq = genome_open.fetch(snp.chr, seq_start - 1, seq_end).upper()

	# extend to full length
	if len(seq) < seq_len:
		seq += "N" * (seq_len - len(seq))

	# verify that ref allele matches ref sequence
	seq_ref = seq[left_len : left_len + len(snp.ref_allele)]
	ref_found = True
	if seq_ref != snp.ref_allele:
		# search for reference allele in alternatives
		ref_found = False

		# for each alternative allele
		for alt_al in snp.alt_alleles:
			# grab reference sequence matching alt length
			seq_ref_alt = seq[left_len : left_len + len(alt_al)]
			if seq_ref_alt == alt_al:
				# found it!
				ref_found = True

				# warn user
				print(
					"WARNING: %s - alt (as opposed to ref) allele matches reference genome; changing reference genome to match."
					% (snp.rsid),
					file=sys.stderr,
				)

				# remove alt allele and include ref allele
				seq = seq[:left_len] + snp.ref_allele + seq[left_len + len(alt_al) :]
				break

	if not ref_found:
		print(
			"WARNING: %s - reference genome does not match any allele" % snp.rsid,
			file=sys.stderr,
		)

	else:
		# one hot code ref allele
		seq_vecs_ref = dna_1hot_ultrafast(seq)

		seq_vecs_list.append(seq_vecs_ref)

		for alt_al in snp.alt_alleles:
			# remove ref allele and include alt allele
			seq_alt = seq[:left_len] + alt_al + seq[left_len + len(snp.ref_allele) :]

			# one hot code
			seq_vecs_alt = np.copy(seq_vecs_ref)
			seq_vecs_alt[left_len,:] = mapping2[alt_al]
			#seq_vecs_alt = dna_1hot_ultrafast(seq_alt)
			'''
			if np.array_equal(seq_vecs_alt, seq_vecs_alt_orig) == False:
				print('assumption eroeoror')
				pdb.set_trace()
			'''
			seq_vecs_list.append(seq_vecs_alt)


	return seq_vecs_list

@njit(parallel=True, fastmath=True)
def untransform_preds_numba(preds, cs, sqrt_idx):
	n_rows, n_cols = preds.shape

	for i in prange(n_rows):
		for j in range(n_cols):
			if preds[i, j] > cs[j]:
				x = preds[i, j] - cs[j] + 1.0
				preds[i, j] = cs[j] - 1.0 + x * x

		for kk in range(len(sqrt_idx)):
			j = sqrt_idx[kk]
			x = preds[i, j] + 1.0
			preds[i, j] = x * x - 1.0


	return preds



def untransform_preds(preds, targets_df, cs, sqrt_idx, unscale=False, unclip=True):
	"""Undo the squashing transformations performed for the tasks.

	Args:
	  preds (np.array): Predictions LxT.
	  targets_df (pd.DataFrame): Targets information table.

	Returns:
	  preds (np.array): Untransformed predictions LxT.
	"""
	# clip soft
	if unclip:
		'''
		if np.any(preds > single_c):
			cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
			preds_unclip = cs - 1 + (preds - cs + 1) ** 2
			preds = np.where(preds > cs, preds_unclip, preds)
		'''
		#cs = np.asarray(targets_df.clip_soft, dtype=preds.dtype)
		rows, cols = np.nonzero(preds > cs[None, :])
		if len(rows) > 0:
			tmp = preds[rows, cols].copy()
			tmp -= cs[cols] - 1
			np.square(tmp, out=tmp)
			tmp += cs[cols] - 1
			preds[rows, cols] = tmp

	# sqrt
	#sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])
	#preds[:, sqrt_mask] = -1 + (preds[:, sqrt_mask] + 1) ** 2  # (4 / 3)
	#np.add(preds, 1, out=preds)      # preds = preds + 1
	#np.square(preds, out=preds)      # preds = preds ** 2
	#preds -= 1 

	sub = preds[:, sqrt_idx].copy()
	np.add(sub, 1, out=sub)
	np.square(sub, out=sub)
	sub -= 1
	preds[:, sqrt_idx] = sub


	# scale
	if unscale:
		scale = np.expand_dims(np.array(targets_df.scale), axis=0)
		preds = preds / scale

	return preds


################################################################################
# main
################################################################################
def main():
	usage = "usage: %prog [options] <params_file> <model_file> <vcf_file>"
	parser = OptionParser(usage)
	parser.add_option(
		"-b",
		dest="bedgraph",
		default=False,
		action="store_true",
		help="Write ref/alt predictions as bedgraph [Default: %default]",
	)
	parser.add_option(
		"-f",
		dest="genome_fasta",
		default="%s/assembly/ucsc/hg38.fa" % os.environ.get('BORZOI_HG38', 'hg38'),
		help="Genome FASTA for sequences [Default: %default]",
	)
	parser.add_option(
		"-g",
		dest="genes_gtf",
		default="%s/genes/gencode41/gencode41_basic_nort.gtf" % os.environ.get('BORZOI_HG38', 'hg38'),
		help="GTF for gene definition [Default %default]",
	)
	parser.add_option(
		"-o",
		dest="output_file",
		default="sed",
		help="Output file [Default: %default]",
	)
	parser.add_option(
		"-v",
		dest="vgpairfile",
		default="sed",
		help="Output file [Default: %default]",
	)
	parser.add_option(
		"-p",
		dest="processes",
		default=None,
		type="int",
		help="Number of processes, passed by multi script",
	)
	parser.add_option(
		"--rc",
		dest="rc",
		default=False,
		action="store_true",
		help="Average forward and reverse complement predictions [Default: %default]",
	)
	parser.add_option(
		"--windowspan",
		dest="windowspan",
		default=False,
		action="store_true",
		help="Average forward and reverse complement predictions [Default: %default]",
	)

	parser.add_option(
		"--shifts",
		dest="shifts",
		default="0",
		type="str",
		help="Ensemble prediction shifts [Default: %default]",
	)
	parser.add_option(
		"--span",
		dest="span",
		default=False,
		action="store_true",
		help="Aggregate entire gene span [Default: %default]",
	)
	parser.add_option(
		"--stats",
		dest="sed_stats",
		default="SED",
		help="Comma-separated list of stats to save. [Default: %default]",
	)
	parser.add_option(
		"-t",
		dest="targets_file",
		default=None,
		type="str",
		help="File specifying target indexes and labels in table format",
	)
	parser.add_option(
		"-u",
		dest="untransform_old",
		default=False,
		action="store_true",
	)
	parser.add_option(
		"--no_untransform",
		dest="no_untransform",
		default=False,
		action="store_true",
	)
	parser.add_option(
		"--batch_size",
		dest="batch_size",
		default=10,
	)

	parser.add_option(
		"--no_unclip",
		dest="no_unclip",
		default=False,
		action="store_true",
	)
	(options, args) = parser.parse_args()

	if len(args) == 3:
		# single worker
		params_file = args[0]
		model_file = args[1]
		vcf_file = args[2]

	elif len(args) == 4:
		# multi separate
		options_pkl_file = args[0]
		params_file = args[1]
		model_file = args[2]
		vcf_file = args[3]

		# save out dir
		out_dir = options.out_dir

		# load options
		options_pkl = open(options_pkl_file, "rb")
		options = pickle.load(options_pkl)
		options_pkl.close()

		# update output directory
		options.out_dir = out_dir

	elif len(args) == 5:
		# multi worker
		options_pkl_file = args[0]
		params_file = args[1]
		model_file = args[2]
		vcf_file = args[3]
		worker_index = int(args[4])

		# load options
		options_pkl = open(options_pkl_file, "rb")
		options = pickle.load(options_pkl)
		options_pkl.close()

		# update output directory
		options.out_dir = "%s/job%d" % (options.out_dir, worker_index)

	else:
		parser.error("Must provide parameters/model, VCF, and genes GTF")



	options.shifts = [int(shift) for shift in options.shifts.split(",")]
	options.sed_stats = options.sed_stats.split(",")


	# Load in vg-pairs to test
	vg_pairs = load_in_vg_pairs_to_test(options.vgpairfile)

	#################################################################
	# read parameters and targets

	# read model parameters
	with open(params_file) as params_open:
		params = json.load(params_open)
	params_model = params["model"]
	params_train = params["train"]
	seq_len = params_model["seq_length"]

	if options.targets_file is None:
		parser.error("Must provide targets table to properly handle strands.")
	else:
		targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

	# prep st jrand
	targets_strand_df = dataset.targets_prep_strand(targets_df)

	# set strand pairs (using new indexing)
	orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
	targets_strand_pair = np.array(
		[orig_new_index[ti] for ti in targets_df.strand_pair]
	)
	params_model["strand_pair"] = [targets_strand_pair]

	#################################################################
	# setup model

	seqnn_model = seqnn.SeqNN(params_model)
	seqnn_model.restore(model_file)
	seqnn_model.build_slice(targets_df.index)
	seqnn_model.build_ensemble(options.rc, options.shifts)

	model_stride = seqnn_model.model_strides[0]
	out_seq_len = seqnn_model.target_lengths[0] * model_stride

	###***
	# Grab underlying Keras model
	if seqnn_model.ensemble is not None:
		keras_model = seqnn_model.ensemble
	else:
		keras_model = seqnn_model.model

	# Warmup
	dummy = tf.zeros((1, seq_len, 4), dtype=tf.float32)
	_ = keras_model(dummy)

	@tf.function(jit_compile=True)
	def predict_step_tf(x):
		# x: (batch, seq_len, 4)
		x = tf.cast(x, tf.float32)
		preds = keras_model(x, training=False)
		return preds
	###***

	
	#################################################################
	# read SNPs / genes

	# filter for worker SNPs
	if options.processes is not None:
		# determine boundaries
		num_snps = bvcf.vcf_count(vcf_file)
		worker_bounds = np.linspace(0, num_snps, options.processes + 1, dtype="int")

		# read SNPs form VCF
		snps_all = bvcf.vcf_snps(
			vcf_file,
			start_i=worker_bounds[worker_index],
			end_i=worker_bounds[worker_index + 1],
		)

	else:
		# read SNPs form VCF
		#snps_all = bvcf.vcf_snps(vcf_file)
		total_n_snps_in_file = bvcf.vcf_count(vcf_file)


	# read genes
	transcriptome = Transcriptome(options.genes_gtf)
	gene_strand = {}
	for gene_id, gene in transcriptome.genes.items():
		gene_strand[gene_id] = gene.strand

	#################################################################
	# predict SNP scores, write output

	# create SNP seq generator
	genome_open = pysam.Fastafile(options.genome_fasta)

	# Precomputed quantities (no need to recompute for each gene)
	pos_gene_strand_mask = targets_df.strand != "-"
	neg_gene_strand_mask = targets_df.strand != "+"
	sqrt_idx = np.flatnonzero(["_sqrt" in ss for ss in targets_df.sum_stat])
	cs = np.asarray(targets_df.clip_soft)
	#cs = cs[None, :]
	#cs = np.expand_dims(np.array(targets_df.clip_soft), axis=0)
	'''
	if len(np.unique(cs)) != 1:
		print('clip soft assumption error: assumed all were the same. need to fix')
		pdb.set_trace()
	'''
	#single_c = np.max(cs)
	#sqrt_mask = np.array([ss.find("_sqrt") != -1 for ss in targets_df.sum_stat])


	B = int(options.batch_size)

	# Open output file handle
	# Open output file handle
	str_dtype = h5py.string_dtype(encoding='utf-8')
	t = h5py.File(options.output_file, 'w')

	snp_chrom_ds = t.create_dataset(
		'snp_chrom',
		shape=(0,),
		maxshape=(None,),
		dtype=str_dtype,
		compression='gzip',
		compression_opts=4
	)
	snp_pos_ds = t.create_dataset(
		'snp_pos',
		shape=(0,),
		maxshape=(None,),
		dtype=np.int64,
		compression='gzip',
		compression_opts=4
	)
	snp_ds = t.create_dataset(
		'snp',
		shape=(0,),
		maxshape=(None,),
		dtype=str_dtype,
		compression='gzip',
		compression_opts=4
	)
	gene_ds = t.create_dataset(
		'gene',
		shape=(0,),
		maxshape=(None,),
		dtype=str_dtype,
		compression='gzip',
		compression_opts=4
	)

	logRef_ds = None
	logAlt_ds = None

	chunk_size = 10_000
	orig_time = time.time()
	for jj in range(0, total_n_snps_in_file, chunk_size):

		print('Iteration ' + str(jj))
		tmp_time = time.time()
		run_time = tmp_time - orig_time
		if jj > 10000:
			print('runtime (seconds): ' + str(run_time/chunk_size))
		orig_time = time.time()

		snps = bvcf.vcf_snps(vcf_file, start_i=jj, end_i=np.min([jj+chunk_size, total_n_snps_in_file]))
		#snps = snps_all[jj : jj + chunk_size]

		# map SNP sequences to gene positions
		print('mapping')
		snpseq_gene_slice = map_snpseq_genes(
			snps, out_seq_len, transcriptome, model_stride, options.span
		)

		# remove SNPs w/o genes
		print('remove snps w/o genes')
		num_snps_pre = len(snps)
		snp_gene_mask = np.array([len(sgs) > 0 for sgs in snpseq_gene_slice])
		snps = [snps[si] for si in range(num_snps_pre) if snp_gene_mask[si]]
		snpseq_gene_slice = [
			snpseq_gene_slice[si] for si in range(num_snps_pre) if snp_gene_mask[si]
		]

		# SNP/gene index
		#for si, snp in tqdm(enumerate(snps), total=len(snps)):
		prev_time = time.time()
		for ii in range(0, len(snps), int(options.batch_size)):
			print(ii)
			cur_time = time.time()
			#print('########', flush=True)
			#print(ii, flush=True)
			#print(cur_time-prev_time, flush=True)
			delta = cur_time - prev_time
			prev_time = cur_time
			if np.mod(ii, 100) == 0 and ii > 0:
				print(str(ii) + ' : ' + str(delta), flush=True)
				#t.flush()
			# Get batch of snps
			snp_batch = snps[ii : ii + int(options.batch_size)]


			###**
			# Preallocate 1-hot batch
			n_in_batch = len(snp_batch)
			snps_1hot = np.zeros((2 * B, seq_len, 4), dtype=np.float32)

			for idx, snp in enumerate(snp_batch):
				ref_1hot, alt_1hot = snp_seq1(snp, seq_len, genome_open)
				snps_1hot[2 * idx] = ref_1hot
				snps_1hot[2 * idx + 1] = alt_1hot

			# To TF once
			snps_1hot_tf = tf.convert_to_tensor(snps_1hot, dtype=tf.float32)

			# Fast compiled forward pass
			snp_preds = predict_step_tf(snps_1hot_tf).numpy() 

			# Keep only the valid predictions (drop padded entries)
			if n_in_batch != B:
				snp_preds = snp_preds[: 2 * n_in_batch]

			###**
			# Loop throug each batched snp
			for snp_iter, snp in enumerate(snp_batch):
				#t1 = time.time()
				# Get predictions for batched snps
				ref_preds, alt_preds = snp_preds[2*snp_iter], snp_preds[((2*snp_iter) + 1)]

				global_snp_index = ii + snp_iter

				# untransform predictions
				if options.targets_file is not None:
					if not options.no_untransform:
						if options.untransform_old:
							ref_preds = dataset.untransform_preds1(ref_preds, targets_df, unclip=not options.no_unclip)
							alt_preds = dataset.untransform_preds1(alt_preds, targets_df, unclip=not options.no_unclip)
						else:
							#ref_preds = untransform_preds(ref_preds, targets_df, cs, sqrt_idx, unclip=not options.no_unclip)
							#alt_preds = untransform_preds(alt_preds, targets_df, cs, sqrt_idx, unclip=not options.no_unclip)

							ref_preds = untransform_preds_numba(ref_preds, cs, sqrt_idx)
							alt_preds = untransform_preds_numba(alt_preds, cs, sqrt_idx)

				#t2 = time.time()

				# buffers for this SNP's overlapping genes
				snp_chrom_buf = []
				snp_pos_buf = []
				snp_buf = []
				gene_buf = []
				logRef_buf = []
				logAlt_buf = []

				# for each overlapping gene
				for gene_id, gene_slice in snpseq_gene_slice[global_snp_index].items():

					# Strip gene version if present; no-op for peak ids (no dots)
					vg_pair = snp.rsid + ':' + gene_id.split('.')[0]
					if vg_pair not in vg_pairs:
						continue

					if len(gene_slice) > len(set(gene_slice)):
						print("WARNING: %d %s has overlapping bins" % (global_snp_index, gene_id))
					# slice gene positions
					ref_preds_gene = ref_preds[gene_slice]
					alt_preds_gene = alt_preds[gene_slice]

					if options.windowspan:
						ref_preds_gene = np.copy(ref_preds)
						alt_preds_gene = np.copy(alt_preds)

					# slice relevant strand targets
					if gene_strand[gene_id] == "+":
						ref_preds_gene = ref_preds_gene[..., pos_gene_strand_mask]
						alt_preds_gene = alt_preds_gene[..., pos_gene_strand_mask]
					else:
						ref_preds_gene = ref_preds_gene[..., neg_gene_strand_mask]
						alt_preds_gene = alt_preds_gene[..., neg_gene_strand_mask]



					# sum across length
					ref_preds_sum = ref_preds_gene.sum(axis=0)
					alt_preds_sum = alt_preds_gene.sum(axis=0)
					altLog = np.log2(alt_preds_sum + 1)
					refLog = np.log2(ref_preds_sum + 1)
					#log_sed = altLog - refLog

					snp_chrom_buf.append(str(snp.chr))
					snp_pos_buf.append(int(snp.pos))
					snp_buf.append(snp.rsid)
					gene_buf.append(gene_id)
					logRef_buf.append(refLog.astype(np.float32))
					logAlt_buf.append(altLog.astype(np.float32))

				if len(logRef_buf) > 0:
					logRef_arr = np.asarray(logRef_buf, dtype=np.float32)
					logAlt_arr = np.asarray(logAlt_buf, dtype=np.float32)

					if logRef_ds is None:
						n_targets = logRef_arr.shape[1]
						logRef_ds = t.create_dataset(
							'logRef',
							shape=(0, n_targets),
							maxshape=(None, n_targets),
							dtype=np.float32,
							chunks=(128, n_targets),
							compression='gzip',
							compression_opts=4
						)
						logAlt_ds = t.create_dataset(
							'logAlt',
							shape=(0, n_targets),
							maxshape=(None, n_targets),
							dtype=np.float32,
							chunks=(128, n_targets),
							compression='gzip',
							compression_opts=4
						)

					n_old = snp_chrom_ds.shape[0]
					n_add = len(snp_chrom_buf)
					n_new = n_old + n_add

					snp_chrom_ds.resize((n_new,))
					snp_pos_ds.resize((n_new,))
					snp_ds.resize((n_new,))
					gene_ds.resize((n_new,))
					logRef_ds.resize((n_new, logRef_ds.shape[1]))
					logAlt_ds.resize((n_new, logAlt_ds.shape[1]))

					snp_chrom_ds[n_old:n_new] = snp_chrom_buf
					snp_pos_ds[n_old:n_new] = snp_pos_buf
					snp_ds[n_old:n_new] = snp_buf
					gene_ds[n_old:n_new] = gene_buf
					logRef_ds[n_old:n_new, :] = logRef_arr
					logAlt_ds[n_old:n_new, :] = logAlt_arr

				#t3 = time.time()

				#print(t3-t2)
				#print(t2-t1)
	t.close()

	# close genome
	genome_open.close()

	return


def clip_float(x, dtype=np.float16):
	return np.clip(x, np.finfo(dtype).min, np.finfo(dtype).max)


def initialize_output_h5(out_dir: str, sed_stats, snps, snpseq_gene_slice, targets_df):
	"""Initialize an output HDF5 file for SAD stats.

	Args:
		out_dir (str): Output directory.
		sed_stats (list): List of SAD stats to compute.
		snps ([bvcf.SNP]): SNP list.
		snpseq_gene_slice ([dict]): List of dicts mapping gene_ids
		  to their exon-overlapping positions for each sequence.
		targets_df (pandas.DataFrame): Targets table.
	"""
	sed_out = h5py.File("%s/sed.h5" % out_dir, "w")

	# collect identifier tuples
	snp_indexes = []
	gene_ids = []
	snp_ids = []
	for si, gene_slice in enumerate(snpseq_gene_slice):
		snp_genes = list(gene_slice.keys())
		gene_ids += snp_genes
		snp_indexes += [si] * len(snp_genes)
	num_scores = len(snp_indexes)

	# write SNP indexes
	snp_indexes = np.array(snp_indexes)
	sed_out.create_dataset("si", data=snp_indexes)

	# write genes
	gene_ids = np.array(gene_ids, "S")
	sed_out.create_dataset("gene", data=gene_ids)

	# write SNPs
	snp_ids = np.array([snp.rsid for snp in snps], "S")
	sed_out.create_dataset("snp", data=snp_ids)

	# write SNP chr
	snp_chr = np.array([snp.chr for snp in snps], "S")
	sed_out.create_dataset("chr", data=snp_chr)

	# write SNP pos
	snp_pos = np.array([snp.pos for snp in snps], dtype="uint32")
	sed_out.create_dataset("pos", data=snp_pos)

	# write SNP reference allele
	snp_refs = []
	snp_alts = []
	for snp in snps:
		if snp.flipped:
			print("SNP %s is flipped. How did that happen?" % snp.rsid)
			snp_refs.append(snp.alt_alleles[0])
			snp_alts.append(snp.ref_allele)
		else:
			snp_refs.append(snp.ref_allele)
			snp_alts.append(snp.alt_alleles[0])
	snp_refs = np.array(snp_refs, "S")
	snp_alts = np.array(snp_alts, "S")
	sed_out.create_dataset("ref_allele", data=snp_refs)
	sed_out.create_dataset("alt_allele", data=snp_alts)

	# write targets
	sed_out.create_dataset("target_ids", data=np.array(targets_df.identifier, "S"))
	sed_out.create_dataset("target_labels", data=np.array(targets_df.description, "S"))

	# initialize SED stats
	num_targets = targets_df.shape[0]
	for sed_stat in sed_stats:
		sed_out.create_dataset(
			sed_stat, shape=(num_scores, num_targets), dtype="float16"
		)

	return sed_out


def make_snpseq_bedt(snps, seq_len: int):
	"""Make a BedTool object for all SNP sequences, where seq_len considers cropping."""
	num_snps = len(snps)
	left_len = seq_len // 2
	right_len = seq_len // 2

	snpseq_bed_lines = []
	for si in range(num_snps):
		# bound sequence start at 0 (true sequence will be N padded)
		snpseq_start = max(0, snps[si].pos - left_len)
		snpseq_end = snps[si].pos + right_len
		# correct end for alternative indels
		snpseq_end += max(0, len(snps[si].ref_allele) - snps[si].longest_alt())
		snpseq_bed_lines.append(
			"%s %d %d %d" % (snps[si].chr, snpseq_start, snpseq_end, si)
		)

	snpseq_bedt = pybedtools.BedTool("\n".join(snpseq_bed_lines), from_string=True)
	return snpseq_bedt


def map_snpseq_genes(
	snps,
	seq_len: int,
	transcriptome,
	model_stride: int,
	span: bool,
	majority_overlap: bool = True,
	intron1: bool = False,
):
	"""Intersect SNP sequences with gene exons, constructing a list
	mapping sequence indexes to dictionaries of gene_ids to their
	exon-overlapping positions in the sequence.

	Args:
	   snps ([bvcf.SNP]): SNP list.
	   seq_len (int): Sequence length, after model cropping.
	   transcriptome (Transcriptome): Transcriptome.
	   model_stride (int): Model stride.
	   span (bool): If True, use gene span instead of exons.
	   majority_overlap (bool): If True, only consider bins for which
		 the majority of the space overlaps an exon.
	   intron1 (bool): If True, include intron bins adjacent to junctions.
	"""

	# make gene BEDtool
	if span:
		genes_bedt = transcriptome.bedtool_span()
	else:
		genes_bedt = transcriptome.bedtool_exon()

	# make SNP sequence BEDtool
	snpseq_bedt = make_snpseq_bedt(snps, seq_len)

	# map SNPs to genes
	snpseq_gene_slice = []
	for snp in snps:
		snpseq_gene_slice.append(OrderedDict())

	for overlap in genes_bedt.intersect(snpseq_bedt, wo=True):
		
		gene_id = overlap[3]
		gene_start = int(overlap[1])
		gene_end = int(overlap[2])
		seq_start = int(overlap[7])
		seq_end = int(overlap[8])
		si = int(overlap[9])

		# adjust for left overhang padded
		seq_len_chop = seq_end - seq_start
		seq_start -= seq_len - seq_len_chop

		# clip left boundaries
		gene_seq_start = max(0, gene_start - seq_start)
		gene_seq_end = max(0, gene_end - seq_start)

		if majority_overlap:
			# requires >50% overlap
			bin_start = int(np.round(gene_seq_start / model_stride))
			bin_end = int(np.round(gene_seq_end / model_stride))
		else:
			# any overlap
			bin_start = int(np.floor(gene_seq_start / model_stride))
			bin_end = int(np.ceil(gene_seq_end / model_stride))

		if intron1:
			bin_start -= 1
			bin_end += 1

		# clip boundaries
		bin_max = int(seq_len / model_stride)
		bin_start = min(bin_start, bin_max)
		bin_end = min(bin_end, bin_max)
		bin_start = max(0, bin_start)
		bin_end = max(0, bin_end)

		if bin_end - bin_start > 0:
			# save gene bin positions
			snpseq_gene_slice[si].setdefault(gene_id, []).extend(
				range(bin_start, bin_end)
			)

	# handle possible overlaps
	for si in range(len(snps)):
		for gene_id, gene_slice in snpseq_gene_slice[si].items():
			snpseq_gene_slice[si][gene_id] = np.unique(gene_slice)

	return snpseq_gene_slice


def write_pct(sed_out, sed_stats):
	"""Compute percentile values for each target and write to HDF5."""
	# define percentiles
	d_fine = 0.001
	d_coarse = 0.01
	percentiles_neg = np.arange(d_fine, 0.1, d_fine)
	percentiles_base = np.arange(0.1, 0.9, d_coarse)
	percentiles_pos = np.arange(0.9, 1, d_fine)

	percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
	sed_out.create_dataset("percentiles", data=percentiles)
	pct_len = len(percentiles)

	for sad_stat in sed_stats:
		if sad_stat not in ["REF", "ALT"]:
			sad_stat_pct = "%s_pct" % sad_stat

			# compute
			sad_pct = np.percentile(sed_out[sad_stat], 100 * percentiles, axis=0).T
			sad_pct = sad_pct.astype("float16")

			# save
			sed_out.create_dataset(sad_stat_pct, data=sad_pct, dtype="float16")


def write_bedgraph_snp(snp, ref_preds, alt_preds, out_dir: str, model_stride: int):
	"""Write full predictions around SNP as BedGraph.

	Args:
	  snp (bvcf.SNP): SNP.
	  ref_preds (np.ndarray): Reference predictions.
	  alt_preds (np.ndarray): Alternate predictions.
	  out_dir (str): Output directory.
	  model_stride (int): Model stride.
	"""
	target_length, num_targets = ref_preds.shape

	# mean across targets
	ref_preds = ref_preds.mean(axis=-1, dtype="float32")
	alt_preds = alt_preds.mean(axis=-1, dtype="float32")
	diff_preds = alt_preds - ref_preds

	# initialize raw predictions/targets
	ref_out = open("%s/%s_ref.bedgraph" % (out_dir, snp.rsid), "w")
	alt_out = open("%s/%s_alt.bedgraph" % (out_dir, snp.rsid), "w")
	diff_out = open("%s/%s_diff.bedgraph" % (out_dir, snp.rsid), "w")

	# specify positions
	seq_len = target_length * model_stride
	left_len = seq_len // 2 - 1
	right_len = seq_len // 2
	seq_start = snp.pos - left_len - 1
	seq_end = snp.pos + right_len + max(0, len(snp.ref_allele) - snp.longest_alt())

	# write values
	bin_start = seq_start
	for bi in range(target_length):
		bin_end = bin_start + model_stride
		cols = [snp.chr, str(bin_start), str(bin_end), str(ref_preds[bi])]
		print("\t".join(cols), file=ref_out)
		cols = [snp.chr, str(bin_start), str(bin_end), str(alt_preds[bi])]
		print("\t".join(cols), file=alt_out)
		cols = [snp.chr, str(bin_start), str(bin_end), str(diff_preds[bi])]
		print("\t".join(cols), file=diff_out)
		bin_start = bin_end

	ref_out.close()
	alt_out.close()
	diff_out.close()


def write_snp(ref_preds, alt_preds, sed_out, xi: int, sed_stats, pseudocounts):
	"""Write SNP predictions to HDF, assuming the length dimension has
	  been maintained.

	Args:
	  ref_preds (np.ndarray): Reference predictions, (gene length x tasks)
	  alt_preds (np.ndarray): Alternate predictions, (gene length x tasks)
	  sed_out (h5py.File): HDF5 output file.
	  xi (int): SNP index.
	  sed_stats (list): SED statistics to compute.
	  pseudocounts (np.ndarray): Target pseudocounts for safe logs.
	"""

	# ref/alt_preds is L x T
	seq_len, num_targets = ref_preds.shape

	# log/sqrt
	ref_preds_log = np.log2(ref_preds+1)
	alt_preds_log = np.log2(alt_preds+1)

	# sum across length
	ref_preds_sum = ref_preds.sum(axis=0)
	alt_preds_sum = alt_preds.sum(axis=0)

	# difference of sums
	'''
	if 'SED' in sed_stats:
		sed = alt_preds_sum - ref_preds_sum
		sed_out['SED'][xi] = clip_float(sed).astype('float16')
	if 'logSED' in sed_stats:
		log_sed = np.log2(alt_preds_sum + 1) - np.log2(ref_preds_sum + 1)
		sed_out['logSED'][xi] = log_sed.astype('float16')
	if 'refLogSed' in sed_stats:
		ref_log_sed = np.log2(ref_preds_sum + 1)
		sed_out['refLogSed'][xi] = ref_log_sed.astype('float16')
	if 'altLogSed' in sed_stats:
		alt_log_sed = np.log2(alt_preds_sum + 1)
		sed_out['altLogSed'][xi] = alt_log_sed.astype('float16')

	# difference L1 norm
	if "D1" in sed_stats:
		diff_abs = np.abs(ref_preds - alt_preds)
		diff_norm1 = diff_abs.sum(axis=0)
		sed_out["D1"][xi] = clip_float(diff_norm1).astype("float16")
	if 'logD1' in sed_stats:
		diff1_log = np.abs(ref_preds_log - alt_preds_log, 2)
		diff_log_norm1 = diff1_log.sum(axis=0)
		sed_out['logD1'][xi] = clip_float(diff_log_norm1).astype('float16')

	# difference L2 norm
	if 'D2' in sed_stats:
		diff2 = np.power(ref_preds - alt_preds, 2)
		diff_norm2 = np.sqrt(diff2.sum(axis=0))
		sed_out['D2'][xi] = clip_float(diff_norm2).astype('float16')
	if 'logD2' in sed_stats:
		diff2_log = np.power(ref_preds_log - alt_preds_log, 2)
		diff_log_norm2 = np.sqrt(diff2_log.sum(axis=0))
		sed_out['logD2'][xi] = clip_float(diff_log_norm2).astype('float16')

	# normalized scores
	ref_preds_norm = ref_preds + pseudocounts
	ref_preds_norm /= ref_preds_norm.sum(axis=0)
	alt_preds_norm = alt_preds + pseudocounts
	alt_preds_norm /= alt_preds_norm.sum(axis=0)

	# compare normalized squared difference
	if "nD2" in sed_stats:
		ndiff2 = np.power(ref_preds_norm - alt_preds_norm, 2)
		ndiff_norm2 = np.sqrt(ndiff2.sum(axis=0))
		sed_out["nD2"][xi] = ndiff_norm2.astype("float16")

	# compare normalized abs max
	if "nDi" in sed_stats:
		ndiff_abs = np.abs(ref_preds_norm - alt_preds_norm)
		ndiff_normi = ndiff_abs.max(axis=0)
		sed_out["nDi"][xi] = ndiff_normi.astype("float16")

	# compare normalized JS
	if "JS" in sed_stats:
		ref_alt_entr = rel_entr(ref_preds_norm, alt_preds_norm).sum(axis=0)
		alt_ref_entr = rel_entr(alt_preds_norm, ref_preds_norm).sum(axis=0)
		js_dist = (ref_alt_entr + alt_ref_entr) / 2
		sed_out["JS"][xi] = js_dist.astype("float16")

	# predictions
	if "REF" in sed_stats:
		sed_out["REF"][xi] = ref_preds_sum.astype("float16")
	if "ALT" in sed_stats:
		sed_out["ALT"][xi] = alt_preds_sum.astype("float16")
	'''


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
	main()