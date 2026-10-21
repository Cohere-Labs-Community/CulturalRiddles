# Fit the riddle-difficulty decomposition models.
# Usage: Rscript fit_lme4.R <frame.csv> <output_dir>
# Called by prepare.py --fit. The two fits run in parallel processes.
suppressPackageStartupMessages({
  library(lme4); library(data.table); library(parallel)
})
args <- commandArgs(trailingOnly = TRUE)
frame <- fread(args[1])
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TIERS <- c("high", "medium", "low", "extremely-low")
frame[, tier := factor(language_resourcedness, levels = TIERS)]

main <- frame[complete_block == TRUE & is_scored == TRUE & grader_flag == FALSE]

ctl <- glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
fit_glmm <- function(formula, data) glmer(formula, data = data, family = binomial, nAGQ = 0, control = ctl)

jobs <- list(
  # Every facet as a random effect, including model-by-facet interactions.
  variance = function() fit_glmm(correct ~ 1 + (1|model_name) + (1|riddle_id) + (1|language_code) +
    (1|community) + (1|topic_core) + (1|model_name:language_code) + (1|model_name:community) +
    (1|model_name:topic_core), main),
  # Model is a fixed effect: 28 clustered abilities need no normality assumption.
  tier = function() fit_glmm(correct ~ 0 + model_name + tier + (1|riddle_id) + (1|language_code) +
    (1|community) + (1|topic_core) + (1|model_name:community), main)
)
fits <- mclapply(jobs, function(job) job(), mc.cores = min(length(jobs), max(1, detectCores() - 2)),
                 mc.preschedule = FALSE)
failed <- names(fits)[vapply(fits, inherits, logical(1), "try-error")]
if (length(failed)) stop("Fits failed: ", paste(failed, collapse = ", "))
for (name in names(fits)) saveRDS(fits[[name]], file.path(out_dir, paste0(name, ".rds")))

# 1. Variance decomposition on the latent logit scale (residual = pi^2 / 3).
v <- as.data.frame(VarCorr(fits$variance))
vc <- rbind(data.table(component = v$grp, variance = v$vcov),
            data.table(component = "residual", variance = pi^2 / 3))
vc[, `:=`(share = variance / sum(variance), sd = sqrt(variance))]
item_side <- c("riddle_id", "language_code", "community", "topic_core")
vc[component %in% item_side, share_of_benchmark_difficulty := variance / sum(variance)]
fwrite(vc[order(-variance)], file.path(out_dir, "variance_components.csv"))

# 2. Resource-tier effects on the odds of a correct answer.
mt <- fits$tier; b <- fixef(mt); se <- sqrt(diag(vcov(mt)))
tier_terms <- paste0("tier", TIERS[-1])
fwrite(data.table(tier = TIERS[-1], log_odds_vs_high = b[tier_terms], se = se[tier_terms],
                  odds_ratio = exp(b[tier_terms]), ci_low = exp(b[tier_terms] - 1.96 * se[tier_terms]),
                  ci_high = exp(b[tier_terms] + 1.96 * se[tier_terms])), file.path(out_dir, "tier_effects.csv"))
cat("Wrote outputs to", out_dir, "\n")
