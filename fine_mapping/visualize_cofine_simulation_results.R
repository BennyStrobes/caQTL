args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
options(warn=1)

# Plots of CoFine simulation results organized across seeds by organize_cofine_simulation_results.py
#   args[1]: organized results stem (the --output_stem of the organizer); args[2]: output directory for the pdfs
# Calibration pools variants across seeds within each probability bin (binomial 95% CIs). Power and false discovery
# proportion are computed within each seed and averaged across seeds (95% CIs from the across-seed standard error).
# Hyper-parameter recovery is shown as the ratio of each seed's posterior mean to the true value.
organized_stem <- args[1]
viz_dir <- args[2]
dir.create(viz_dir, showWarnings=FALSE, recursive=TRUE)

method_colors <- c("CoFine"="#D06A4B", "CoFine without chromatin"="#4C78A8")

figure_theme <- function() {
	return(
		theme_minimal(base_size=11) +
		theme(
			plot.title=element_text(face="bold", size=12, hjust=.5),
			axis.title=element_text(size=11, color="#25313B"),
			axis.text=element_text(size=10, color="#33424F"),
			strip.text=element_text(size=10, face="bold"),
			panel.grid.minor=element_blank(),
			panel.grid.major.x=element_blank(),
			axis.line.x=element_line(colour="#7E8A97", linewidth=.35),
			axis.line.y=element_line(colour="#7E8A97", linewidth=.35),
			legend.position="bottom",
			legend.title=element_blank()
		)
	)
}

calibration_df <- read.table(paste0(organized_stem, "_calibration.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
power_df <- read.table(paste0(organized_stem, "_power.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
hyperparameter_df <- read.table(paste0(organized_stem, "_hyperparameters.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
n_seeds <- length(unique(power_df$seed))


#####################
# Calibration
#####################
make_calibration_plot <- function(df, title, x_label, y_label) {
	pooled <- aggregate(cbind(n_variants, sum_predicted, n_true) ~ method + bin_lower + bin_upper, data=df, FUN=sum)
	pooled <- pooled[pooled$n_variants > 0, ]
	pooled$mean_predicted <- pooled$sum_predicted / pooled$n_variants
	pooled$fraction_true <- pooled$n_true / pooled$n_variants
	se <- sqrt(pooled$fraction_true * (1 - pooled$fraction_true) / pooled$n_variants)
	pooled$ci_lower <- pmax(pooled$fraction_true - 1.96 * se, 0)
	pooled$ci_upper <- pmin(pooled$fraction_true + 1.96 * se, 1)
	p <- ggplot(pooled, aes(x=mean_predicted, y=fraction_true, color=method)) +
		geom_abline(slope=1, intercept=0, linetype="dashed", color="grey50") +
		geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), width=.02, position=position_dodge(width=.02)) +
		geom_point(size=2, position=position_dodge(width=.02)) +
		scale_color_manual(values=method_colors) +
		coord_cartesian(xlim=c(0, 1), ylim=c(0, 1)) +
		labs(x=x_label, y=y_label, title=title) +
		figure_theme()
	return(p)
}

eqtl_calibration_plot <- make_calibration_plot(calibration_df[calibration_df$quantity == "eqtl_pip", ], "eQTL PIP calibration", "Mean PIP in bin", "Fraction causal")
path_calibration_plot <- make_calibration_plot(calibration_df[calibration_df$quantity == "path_prob", ], "Chromatin path calibration", "Mean P(H = 1) in bin", "Fraction on a true path")
ggsave(paste0(viz_dir, "cofine_simulation_calibration.pdf"), plot_grid(eqtl_calibration_plot, path_calibration_plot, ncol=2), width=9, height=4.5, units="in")


#####################
# Power and false discovery proportion
#####################
power_df$power <- power_df$n_true_called / power_df$n_causal
power_df$fdp <- ifelse(power_df$n_called > 0, 1 - power_df$n_true_called / power_df$n_called, NA)

summarize_across_seeds <- function(df, value_column) {
	df <- df[!is.na(df[[value_column]]), ]
	summary_df <- aggregate(df[[value_column]], by=df[c("method", "stratum", "pip_threshold")], FUN=function(x) c(mean=mean(x), se=ifelse(length(x) > 1, sd(x)/sqrt(length(x)), 0)))
	summary_df <- do.call(data.frame, summary_df)
	colnames(summary_df)[(ncol(summary_df)-1):ncol(summary_df)] <- c("mean", "se")
	summary_df$ci_lower <- pmax(summary_df$mean - 1.96 * summary_df$se, 0)
	summary_df$ci_upper <- pmin(summary_df$mean + 1.96 * summary_df$se, 1)
	summary_df$threshold_label <- paste0("PIP > ", summary_df$pip_threshold)
	summary_df$stratum <- factor(summary_df$stratum, levels=c("All variants", "On a true chromatin path", "Off a true chromatin path"))
	return(summary_df)
}

make_bar_plot <- function(summary_df, y_label, title, hline=NULL) {
	p <- ggplot(summary_df, aes(x=stratum, y=mean, fill=method)) +
		geom_col(position=position_dodge(width=.8), width=.75) +
		geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), position=position_dodge(width=.8), width=.25) +
		facet_wrap(~threshold_label) +
		scale_fill_manual(values=method_colors) +
		scale_x_discrete(labels=function(x) gsub(" a true ", "\na true ", x)) +
		labs(x="", y=y_label, title=title) +
		figure_theme()
	if (!is.null(hline)) {
		p <- p + geom_hline(yintercept=hline, linetype="dashed", color="grey50")
	}
	return(p)
}

power_plot <- make_bar_plot(summarize_across_seeds(power_df, "power"), "Power", paste0("Power (", n_seeds, " simulations)"))
fdp_plot <- make_bar_plot(summarize_across_seeds(power_df, "fdp"), "False discovery proportion", paste0("False discovery proportion (", n_seeds, " simulations)"))
ggsave(paste0(viz_dir, "cofine_simulation_power.pdf"), power_plot, width=8, height=4.5, units="in")
ggsave(paste0(viz_dir, "cofine_simulation_fdp.pdf"), fdp_plot, width=8, height=4.5, units="in")


#####################
# Hyper-parameter recovery (CoFine)
#####################
hyperparameter_order <- c("lambda", "tau2", "pi0", "sigma2_0", "pi1", "pi_delta", "sigma2_delta", "pi_gamma", "sigma2_gamma")
hyperparameter_df$ratio <- hyperparameter_df$posterior_mean / hyperparameter_df$true_value
hyperparameter_df$hyperparameter <- factor(hyperparameter_df$hyperparameter, levels=hyperparameter_order)
hyperparameter_df$covered <- hyperparameter_df$posterior_95_lb <= hyperparameter_df$true_value & hyperparameter_df$posterior_95_ub >= hyperparameter_df$true_value
coverage_df <- aggregate(covered ~ hyperparameter, data=hyperparameter_df, FUN=mean)
coverage_df$label <- paste0(round(100 * coverage_df$covered), "%")
hyperparameter_plot <- ggplot(hyperparameter_df, aes(x=hyperparameter, y=ratio)) +
	geom_hline(yintercept=1, linetype="dashed", color="grey50") +
	geom_boxplot(outlier.shape=NA, fill="#F3D9CF", color="#D06A4B", width=.6) +
	geom_jitter(width=.12, height=0, size=.8, alpha=.6, color="#25313B") +
	geom_text(data=coverage_df, aes(x=hyperparameter, y=Inf, label=label), vjust=1.5, size=3, color="#33424F") +
	labs(x="", y="Posterior mean / true value", title=paste0("CoFine hyper-parameter recovery (", n_seeds, " simulations; % = 95% interval coverage)")) +
	figure_theme()
ggsave(paste0(viz_dir, "cofine_simulation_hyperparameters.pdf"), hyperparameter_plot, width=9, height=4.5, units="in")

# Everything in one file
combined <- plot_grid(plot_grid(eqtl_calibration_plot, path_calibration_plot, ncol=2), power_plot, fdp_plot, hyperparameter_plot, ncol=1, rel_heights=c(1, 1, 1, 1))
ggsave(paste0(viz_dir, "cofine_simulation_summary.pdf"), combined, width=9, height=18, units="in")
print(paste0("Plots of ", n_seeds, " simulations written to ", viz_dir))
