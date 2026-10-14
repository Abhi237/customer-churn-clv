#!/usr/bin/env Rscript
# Survival analysis of time-to-inactivity (90-day lapse) for UCI Online Retail II customers.
# Input : data/processed/survival_customers.csv (built by sql/08_survival.sql)
# Output: docs/figures/km_*.png, artifacts/r_survival/*.csv, artifacts/r_survival/summary.txt
suppressPackageStartupMessages({
  library(survival)
  library(survminer)
  library(dplyr)
  library(readr)
  library(ggplot2)
})

root <- normalizePath(Sys.getenv("CHURN_CLV_ROOT", "."))
fig_dir <- file.path(root, "docs", "figures")
out_dir <- file.path(root, "artifacts", "r_survival")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

surv <- read_csv(file.path(root, "data/processed/survival_customers.csv"), show_col_types = FALSE) |>
  mutate(
    country_group = relevel(factor(country_group), ref = "United Kingdom"),
    first_value_tier = factor(first_value_quartile, levels = 1:4,
                              labels = c("Q1 (lowest)", "Q2", "Q3", "Q4 (highest)")),
    acq_quarter = factor(paste0("Q", ceiling(as.integer(format(acquisition_cohort, "%m")) / 3))),
    log_first_value = log1p(first_order_value),
    log_first_products = log1p(first_order_products),
    first_cohort_2009 = format(acquisition_cohort, "%Y-%m") == "2009-12"
  )

# Customers first seen in Dec-2009 are left-truncated (data starts then); main analysis excludes them.
d <- filter(surv, !first_cohort_2009)
lines <- c(sprintf("customers (excl. 2009-12 cohort): %d; lapse events: %d (%.1f%%)",
                   nrow(d), sum(d$event), 100 * mean(d$event)))

theme_set(theme_minimal(base_size = 11))
save_km <- function(fit, data, name, title, legend_title) {
  if (!is.null(fit$strata)) names(fit$strata) <- sub(".*=", "", names(fit$strata))
  p <- ggsurvplot(fit, data = data, conf.int = TRUE, risk.table = TRUE, xlim = c(0, 600),
                  break.time.by = 90, xlab = "Days since first purchase",
                  ylab = "P(still active: no 90-day lapse yet)", title = title,
                  legend.title = legend_title, risk.table.height = 0.28, censor = FALSE,
                  palette = "Dark2", ggtheme = theme_minimal(base_size = 11))
  png(file.path(fig_dir, paste0(name, ".png")), width = 1500, height = 1100, res = 150)
  print(p)
  invisible(dev.off())
}

# ---- Overall KM
km_all <- survfit(Surv(duration_days, event) ~ 1, data = d)
km_tab <- summary(km_all, times = c(1, 2, 30, 90, 180, 365))
write_csv(tibble(time = km_tab$time, survival = km_tab$surv, lower = km_tab$lower, upper = km_tab$upper,
                 n_risk = km_tab$n.risk), file.path(out_dir, "km_overall.csv"))
lines <- c(lines, sprintf("KM S(t) at day %d: %.3f", km_tab$time, km_tab$surv))
lines <- c(lines, sprintf("KM median active days: %s", as.character(summary(km_all)$table["median"])))
save_km(km_all, d, "km_overall", "Time to first 90-day lapse - all customers", "All")

# ---- KM + log-rank by segments
logrank <- list()
for (v in c("first_value_tier", "country_group", "acq_quarter")) {
  f <- as.formula(paste("Surv(duration_days, event) ~", v))
  fit <- survfit(f, data = d)
  fit$call$formula <- f
  lr <- survdiff(f, data = d)
  p <- pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)
  logrank[[v]] <- tibble(variable = v, chisq = lr$chisq, df = length(lr$n) - 1, p_value = p)
  lines <- c(lines, sprintf("log-rank %s: chisq=%.1f df=%d p=%.3g", v, lr$chisq, length(lr$n) - 1, p))
  st <- summary(fit, times = c(90, 365))
  write_csv(tibble(strata = st$strata, time = st$time, survival = st$surv, n_risk = st$n.risk),
            file.path(out_dir, paste0("km_by_", v, ".csv")))
  save_km(fit, d, paste0("km_by_", v), paste("Time to first 90-day lapse by", gsub("_", " ", v)), v)
}
write_csv(bind_rows(logrank), file.path(out_dir, "logrank_tests.csv"))

# ---- Cox PH
cox <- coxph(Surv(duration_days, event) ~ log_first_value + log_first_products + country_group + acq_quarter,
             data = d, ties = "efron")
cs <- summary(cox)
cox_tab <- tibble(term = rownames(cs$coefficients), hr = cs$conf.int[, 1], lower95 = cs$conf.int[, 3],
                  upper95 = cs$conf.int[, 4], p_value = cs$coefficients[, 5])
write_csv(cox_tab, file.path(out_dir, "cox_hazard_ratios.csv"))
lines <- c(lines, sprintf("Cox concordance: %.3f (se %.3f); n=%d events=%d", cs$concordance[1],
                          cs$concordance[2], cs$n, cs$nevent))
lines <- c(lines, sprintf("HR %s = %.3f [%.3f, %.3f] p=%.3g", cox_tab$term, cox_tab$hr, cox_tab$lower95,
                          cox_tab$upper95, cox_tab$p_value))

zph <- cox.zph(cox)
zt <- tibble(term = rownames(zph$table), chisq = zph$table[, 1], df = zph$table[, 2], p_value = zph$table[, 3])
write_csv(zt, file.path(out_dir, "cox_zph.csv"))
lines <- c(lines, sprintf("cox.zph %s: chisq=%.1f p=%.3g", zt$term, zt$chisq, zt$p_value))
png(file.path(fig_dir, "cox_zph_schoenfeld.png"), width = 1600, height = 900, res = 150)
par(mfrow = c(1, 2))
plot(zph[1], resid = FALSE, main = "Schoenfeld: log first order value")
abline(h = coef(cox)[1], col = "red", lty = 2)
plot(zph[2], resid = FALSE, main = "Schoenfeld: log first order products")
abline(h = coef(cox)[2], col = "red", lty = 2)
invisible(dev.off())

# ---- Remedy for the PH violation (acquisition quarter): stratified Cox, separate baseline hazard per quarter
cox_st <- coxph(Surv(duration_days, event) ~ log_first_value + log_first_products + country_group +
                  strata(acq_quarter), data = d, ties = "efron")
cst <- summary(cox_st)
st_tab <- tibble(term = rownames(cst$coefficients), hr = cst$conf.int[, 1], lower95 = cst$conf.int[, 3],
                 upper95 = cst$conf.int[, 4], p_value = cst$coefficients[, 5])
write_csv(st_tab, file.path(out_dir, "cox_stratified_hazard_ratios.csv"))
zst <- cox.zph(cox_st)
write_csv(tibble(term = rownames(zst$table), chisq = zst$table[, 1], p_value = zst$table[, 3]),
          file.path(out_dir, "cox_stratified_zph.csv"))
lines <- c(lines, "Stratified Cox (strata = acquisition quarter):",
           sprintf("  HR %s = %.3f [%.3f, %.3f] p=%.3g", st_tab$term, st_tab$hr, st_tab$lower95,
                   st_tab$upper95, st_tab$p_value),
           sprintf("  cox.zph %s: p=%.3g", rownames(zst$table), zst$table[, 3]))

# ---- Sensitivity: include the left-truncated 2009-12 cohort
cox_all <- coxph(Surv(duration_days, event) ~ log_first_value + log_first_products + country_group + acq_quarter,
                 data = surv)
lines <- c(lines, sprintf("Sensitivity incl. 2009-12 cohort: HR log_first_value = %.3f (main: %.3f)",
                          exp(coef(cox_all)[1]), exp(coef(cox)[1])))

writeLines(lines, file.path(out_dir, "summary.txt"))
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
cat(paste(lines, collapse = "\n"), "\n")
