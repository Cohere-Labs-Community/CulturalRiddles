# Fit the riddle-feature models of a correct answer: odds ratios of riddle and
# answer features, and how much riddle-level difficulty those features explain.
# Usage: Rscript fit_riddle_features.R <riddle_feature_frame.csv> <output_dir>
# Called by riddle_features.py --fit. The two fits run in parallel processes.
suppressPackageStartupMessages({
  library(lme4); library(data.table); library(parallel)
})
args <- commandArgs(trailingOnly = TRUE)
d <- fread(args[1])
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TIERS <- c("high", "medium", "low", "extremely-low")
d[, `:=`(tier = factor(language_resourcedness, levels = TIERS),
         answer_kind = relevel(factor(answer_kind), "common_word"),
         riddle_form = relevel(factor(riddle_form), "third_person_description"),
         log2_words_c = log2_words - median(log2_words))]

ctl <- glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
fit_glmm <- function(formula, data) glmer(formula, data = data, family = binomial, nAGQ = 0, control = ctl)
RE <- "(1|riddle_id) + (1|language_code) + (1|community) + (1|topic_core)"
FEATURES <- "answer_kind + riddle_form + multi_clue + multiline + log2_words_c"
fixed <- function(terms) as.formula(paste("correct ~ 0 + model_name +", terms, "+", RE))

jobs <- list(
  accuracy_tier = function() fit_glmm(fixed("tier"), d),
  accuracy_features = function() fit_glmm(fixed(paste("tier +", FEATURES)), d)
)
fits <- mclapply(jobs, function(job) job(), mc.cores = min(length(jobs), max(1, detectCores() - 2)),
                 mc.preschedule = FALSE)
failed <- names(fits)[vapply(fits, inherits, logical(1), "try-error")]
if (length(failed)) stop("Fits failed: ", paste(failed, collapse = ", "))
for (name in names(fits)) saveRDS(fits[[name]], file.path(out_dir, paste0(name, ".rds")))

# 1. Odds ratios of the features (model fixed effects omitted).
b <- fixef(fits$accuracy_features); se <- sqrt(diag(vcov(fits$accuracy_features)))
keep <- !startsWith(names(b), "model_name")
fwrite(data.table(term = names(b)[keep], log_odds = b[keep], se = se[keep],
                  odds_ratio = exp(b[keep]), ci_low = exp(b[keep] - 1.96 * se[keep]),
                  ci_high = exp(b[keep] + 1.96 * se[keep]), p = 2 * pnorm(-abs(b[keep] / se[keep]))),
       file.path(out_dir, "riddle_feature_effects.csv"))

# 2. Riddle variance explained by the features, relative to the tier-only fit.
rv <- sapply(names(jobs), function(nm) {
  v <- as.data.frame(VarCorr(fits[[nm]])); v$vcov[v$grp == "riddle_id"]
})
fwrite(data.table(fit = names(rv), riddle_variance = rv,
                  pct_riddle_variance_explained = 100 * (1 - rv / rv[1]),
                  AIC = sapply(names(rv), function(nm) AIC(fits[[nm]]))),
       file.path(out_dir, "riddle_feature_variance_explained.csv"))
cat("Wrote riddle-feature outputs to", out_dir, "\n")
