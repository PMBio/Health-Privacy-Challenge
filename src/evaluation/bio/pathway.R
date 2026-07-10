library(GSVA)
library(msigdbr)
library(dplyr)
library(clusterProfiler)
library(org.Hs.eg.db)


# ----------------------------------------------------------------
# Prepare HALLMARK gene sets
# Returns a named list of gene vectors (symbols)
# ----------------------------------------------------------------
get_hallmark_genesets <- function() {
  msigdbr(species = "Homo sapiens", category = "H") %>%
    split(x = .$gene_symbol, f = .$gs_name)
}

# ----------------------------------------------------------------
# Convert ENSEMBL rownames to gene symbols if needed
# Pass convert_ids = TRUE if your matrix uses ENSEMBL IDs
# ----------------------------------------------------------------
convert_ensembl_to_symbol <- function(mat) {
  mapped <- bitr(rownames(mat),
                 fromType = "ENSEMBL",
                 toType   = "SYMBOL",
                 OrgDb    = org.Hs.eg.db,
                 drop     = TRUE)
  mat <- mat[mapped$ENSEMBL, , drop = FALSE]
  rownames(mat) <- mapped$SYMBOL
  # Remove duplicate symbols (keep first)
  mat[!duplicated(rownames(mat)), , drop = FALSE]
}

# ----------------------------------------------------------------
# Run ssGSEA
# Input:  mat — genes x samples matrix (VST counts)
#         genesets — named list of gene vectors
# Output: pathways x samples score matrix
# ----------------------------------------------------------------
run_ssgsea <- function(mat, genesets) {
  gsva(as.matrix(mat),
       genesets,
       method       = "ssgsea",
       ssgsea.norm  = TRUE,
       verbose      = FALSE)
}

# ----------------------------------------------------------------
# Compare pathway score distributions: real vs synthetic
# Input:  scores_real, scores_synth — pathways x samples matrices
# Output: per-pathway metrics dataframe
# ----------------------------------------------------------------
compare_pathway_scores <- function(scores_real, scores_synth) {
  pathways <- rownames(scores_real)

  results <- lapply(pathways, function(pw) {
    r <- scores_real[pw, ]
    s <- scores_synth[pw, ]

    ks      <- ks.test(r, s)
    meanD   <- mean(s) - mean(r)
    varR    <- round(var(r), 4)
    varS    <- round(var(s), 4)
    # Simple effect size: mean difference / pooled SD
    pooled_sd <- sqrt((var(r) + var(s)) / 2)
    effect_size <- if (pooled_sd > 0) meanD / pooled_sd else NA

    data.frame(
      pathway      = pw,
      mean_real    = round(mean(r), 4),
      mean_synth   = round(mean(s), 4),
      mean_diff    = round(meanD, 4),
      effect_size  = round(effect_size, 4),
      var_real     = varR,
      var_synth    = varS,
      var_ratio    = round(varS / varR, 4),   # >1 = overdispersed, <1 = underdispersed
      ks_stat      = round(ks$statistic, 4),
      ks_pval      = round(ks$p.value, 4),
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, results)
}

# ----------------------------------------------------------------
# Summary metrics across all pathways
# ----------------------------------------------------------------
summarise_pathway_comparison <- function(pathway_metrics) {
  list(
    mean_abs_effect    = round(mean(abs(pathway_metrics$effect_size), na.rm = TRUE), 4),
    mean_ks_stat       = round(mean(pathway_metrics$ks_stat, na.rm = TRUE), 4),
    mean_var_ratio     = round(mean(pathway_metrics$var_ratio, na.rm = TRUE), 4),
    n_sig_pathways     = sum(pathway_metrics$ks_pval < 0.05, na.rm = TRUE),  # pathways where distributions differ
    n_total_pathways   = nrow(pathway_metrics)
  )
}

# ----------------------------------------------------------------
# call this per generative model
# real_mat, synth_mat: genes x samples log-normalised matrices
# convert_ids: set TRUE if rownames are ENSEMBL IDs
# ----------------------------------------------------------------
run_pathway_fidelity <- function(real_mat, 
                                 synth_mat,
                                 #model_name  = "model",
                                 split_no    = 1,
                                 convert_ids = FALSE,
                                 output_dir  = "pathway_results") {

  cat(sprintf("\nRunning ssGSEA pathway fidelity...\n"))

  if (convert_ids) {
    real_mat  <- convert_ensembl_to_symbol(real_mat)
    synth_mat <- convert_ensembl_to_symbol(synth_mat)
  }

  genesets <- get_hallmark_genesets()

  scores_real  <- run_ssgsea(real_mat,  genesets)
  scores_synth <- run_ssgsea(synth_mat, genesets)

  pathway_metrics <- compare_pathway_scores(scores_real, scores_synth)
  summary_metrics <- summarise_pathway_comparison(pathway_metrics)

  cat(sprintf("  Mean |effect size|:  %.4f\n", summary_metrics$mean_abs_effect))
  cat(sprintf("  Mean KS statistic:   %.4f\n", summary_metrics$mean_ks_stat))
  cat(sprintf("  Mean variance ratio: %.4f\n", summary_metrics$mean_var_ratio))
  cat(sprintf("  Pathways with sig. distribution shift: %d / %d\n",
              summary_metrics$n_sig_pathways,
              summary_metrics$n_total_pathways))

  # Save per-pathway results
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  out_file <- file.path(output_dir,
                        sprintf("pathway_metrics_split_%d.csv", split_no))
  #pathway_metrics$model    <- model_name
  pathway_metrics$split_no <- split_no
  write.csv(pathway_metrics, out_file, row.names = FALSE)
  cat(sprintf("  Saved to: %s\n", out_file))

  list(
    per_pathway = pathway_metrics,
    summary     = summary_metrics
  )
}

#### RUN IT


args <- commandArgs(trailingOnly = TRUE)
home_dir <- args[1]
split_no <- as.integer(args[2])
dataset_name <- args[3]
generator_name <- args[4]
param_dir <- args[5]

######################

real_data_dir <- file.path(home_dir, "data_splits", dataset_name, "real")
synthetic_data_dir <- file.path(
  home_dir, "data_splits", dataset_name,
  "synthetic", generator_name, param_dir
)
meta_dir <- file.path(home_dir, "data/meta")
bio_res_dir <- file.path(home_dir, "results/bio", dataset_name, generator_name)
anno_colname_class <- "Subtype"


# Load synthetic and real data
real_data <- read.csv(file.path(
  real_data_dir,
  paste("X_train_real_split_", split_no, ".csv", sep = "")
))
rownames(real_data) <- paste0("P", seq_len(nrow(real_data)))

real_annots <- read.csv(file.path(
  real_data_dir,
  paste("y_train_real_split_", split_no, ".csv", sep = "")
))
rownames(real_annots) <- paste0("P", seq_len(nrow(real_annots)))
colnames(real_annots) <- gsub(
  "Subtype_Selected",
  "Subtype", colnames(real_annots)
)
colnames(real_annots) <- gsub(
  "cancer_type",
  "Subtype", colnames(real_annots)
)


synthetic_data <- read.csv(file.path(
  synthetic_data_dir,
  paste("synthetic_data_split_", split_no, ".csv", sep = "")
))
rownames(synthetic_data) <- paste0("P", seq_len(nrow(synthetic_data)))

synthetic_annots <- read.csv(file.path(
  synthetic_data_dir,
  paste("synthetic_labels_split_", split_no, ".csv", sep = "")
))
### this was missing...
colnames(real_annots) <- gsub(
  "Subtype_Selected",
  "Subtype", colnames(real_annots)
)
colnames(synthetic_annots) <- gsub(
  "cancer_type",
  "Subtype", colnames(real_annots)
)
rownames(synthetic_annots) <- paste0("P", seq_len(nrow(synthetic_annots)))

# Load means and standard deviations for reverse standardization
# scaling_params <- read.csv(file.path(real_data_dir,
#    paste("X_train_scale_params_split_", split_no, ".csv", sep = "")), row.names =1)


# transpose to switch gene names to rows
synthetic_data <- t(synthetic_data)
synthetic_data <- as.data.frame(synthetic_data)

real_data <- t(real_data)
real_data <- as.data.frame(real_data)


### output dir 
output_dir <- file.path(bio_res_dir, param_dir)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


run_pathway_fidelity(real_data,
                    synthetic_data,
                    split_no,
                    convert_ids = TRUE,
                    output_dir = output_dir)

