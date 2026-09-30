#!/usr/bin/env Rscript
# Install R dependencies for the bio-evaluation scripts (DE, coexpression, pathway).
# Run once inside R/4.3.2 before running the analysis scripts.
#
# NOTE: coexpression also requires hcocena, which is NOT installed here — see the
# message at the end and https://github.com/MarieOestreich/hCoCena

# --- CRAN ---
cran <- c("dplyr", "ggplot2", "msigdbr", "remotes", "BiocManager")
install.packages(setdiff(cran, rownames(installed.packages())),
                 repos = "https://cloud.r-project.org")

# --- Bioconductor ---
bioc <- c("org.Hs.eg.db", "AnnotationDbi", "scran",
          "GSVA", "clusterProfiler")
BiocManager::install(setdiff(bioc, rownames(installed.packages())),
                     update = FALSE, ask = FALSE)

# --- sanity check ---
for (p in c("org.Hs.eg.db", "AnnotationDbi", "dplyr", "ggplot2",
            "scran", "GSVA", "msigdbr", "clusterProfiler")) {
  if (!requireNamespace(p, quietly = TRUE)) stop("MISSING: ", p)
}
cat("All CRAN/Bioconductor dependencies present.\n")

cat("\nNOTE: coexpression additionally requires hcocena, installed separately:\n")
cat("  https://github.com/MarieOestreich/hCoCena\n")