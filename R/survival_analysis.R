#!/usr/bin/env Rscript
# Survival analysis of time-to-lapse (first inter-purchase gap > 90 days) for the Complete Journey households.
# Input : data/processed/survival_households.csv (built by sql/09_survival.sql; landmark design, see its header)
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

tier <- function(x, k, labels) {
  # rank-based tiers so that ties cannot collapse a tier
  factor(labels[ceiling(rank(x, ties.method = "first") / length(x) * k)], levels = labels)
}

d <- read_csv(file.path(root, "data/processed/survival_households.csv"), show_col_types = FALSE,
              col_types = cols(household_id = col_character(), age = col_character(), income = col_character())) |>
  mutate(
    trip_tier = tier(baseline_trips, 3, c("Low (T1)", "Mid (T2)", "High (T3)")),
    basket_tier = tier(baseline_avg_basket_spend, 4, c("Q1 (smallest)", "Q2", "Q3", "Q4 (largest)")),
    coupon_user = factor(ifelse(baseline_coupon_share > 0, "Used coupons", "No coupon use"),
                         levels = c("No coupon use", "Used coupons")),
    store_pattern = factor(ifelse(baseline_n_stores > 1, "2+ stores", "1 store"), levels = c("1 store", "2+ stores")),
    entry_group = factor(case_when(format(first_purchase_date, "%Y-%m") == "2017-01" ~ "Jan 2017",
                                   format(first_purchase_date, "%Y-%m") %in% c("2017-02", "2017-03") ~ "Feb-Mar 2017",
                                   TRUE ~ "Apr-Sep 2017"),
                         levels = c("Jan 2017", "Feb-Mar 2017", "Apr-Sep 2017")),
    log_trips = log1p(baseline_trips),
    log_basket = log(baseline_avg_basket_spend),
    fresh_10pts = baseline_fresh_share * 10
  )

lines <- c(sprintf("households analysed: %d; lapse events: %d (%.1f%%); max follow-up %d days",
                   nrow(d), sum(d$event), 100 * mean(d$event), max(d$duration_days)))

theme_set(theme_minimal(base_size = 11))
km_curves <- list()
curve_df <- function(fit, seg) {
  st <- if (is.null(fit$strata)) rep("All", length(fit$time)) else rep(sub(".*=", "", names(fit$strata)), fit$strata)
  tibble(segment_var = seg, stratum = st, time = fit$time, surv = fit$surv, lower = fit$lower,
         upper = fit$upper, n_risk = fit$n.risk)
}
save_km <- function(fit, data, name, title, legend_title) {
  km_curves[[name]] <<- curve_df(fit, legend_title)
  if (!is.null(fit$strata)) names(fit$strata) <- sub(".*=", "", names(fit$strata))
  p <- ggsurvplot(fit, data = data, conf.int = TRUE, risk.table = TRUE, xlim = c(0, 250),
                  break.time.by = 30, xlab = "Days since landmark (first observed purchase + 28 days)",
                  ylab = "P(no 90-day lapse yet)", title = title,
                  legend.title = legend_title, risk.table.height = 0.28, censor = FALSE,
                  palette = "Dark2", ggtheme = theme_minimal(base_size = 11))
  png(file.path(fig_dir, paste0(name, ".png")), width = 1500, height = 1100, res = 150)
  print(p)
  invisible(dev.off())
}

# ---- Overall KM
km_all <- survfit(Surv(duration_days, event) ~ 1, data = d)
tt <- c(30, 60, 90, 120, 180, 240)
km_tab <- summary(km_all, times = tt)
write_csv(tibble(time = km_tab$time, survival = km_tab$surv, lower = km_tab$lower, upper = km_tab$upper,
                 n_risk = km_tab$n.risk), file.path(out_dir, "km_overall.csv"))
lines <- c(lines, sprintf("KM S(t) at day %d: %.3f", km_tab$time, km_tab$surv))
med <- summary(km_all)$table["median"]
lines <- c(lines, sprintf("KM median time to lapse: %s", ifelse(is.na(med), "not reached", as.character(med))))
save_km(km_all, d, "km_overall", "Time to first 90-day lapse - all households", "All")

# ---- KM + log-rank by segments
logrank <- list()
for (v in c("trip_tier", "basket_tier", "coupon_user", "entry_group")) {
  f <- as.formula(paste("Surv(duration_days, event) ~", v))
  fit <- survfit(f, data = d)
  fit$call$formula <- f
  lr <- survdiff(f, data = d)
  p <- pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)
  logrank[[v]] <- tibble(variable = v, chisq = lr$chisq, df = length(lr$n) - 1, p_value = p)
  lines <- c(lines, sprintf("log-rank %s: chisq=%.1f df=%d p=%.3g", v, lr$chisq, length(lr$n) - 1, p))
  st <- summary(fit, times = c(90, 180))
  write_csv(tibble(strata = sub(".*=", "", as.character(st$strata)), time = st$time, survival = st$surv,
                   n_risk = st$n.risk), file.path(out_dir, paste0("km_by_", v, ".csv")))
  save_km(fit, d, paste0("km_by_", v), paste("Time to first 90-day lapse by", gsub("_", " ", v)), v)
}
write_csv(bind_rows(km_curves), file.path(out_dir, "km_curves.csv"))
write_csv(bind_rows(logrank), file.path(out_dir, "logrank_tests.csv"))

# ---- Cox PH: baseline covariates measured in the first 28 days (landmark design)
cox_formula <- Surv(duration_days, event) ~ log_trips + log_basket + fresh_10pts + coupon_user + store_pattern +
  entry_group
cox <- coxph(cox_formula, data = d, ties = "efron")
cs <- summary(cox)
tab_of <- function(s) tibble(term = rownames(s$coefficients), hr = s$conf.int[, 1], lower95 = s$conf.int[, 3],
                             upper95 = s$conf.int[, 4], p_value = s$coefficients[, 5])
cox_tab <- tab_of(cs)
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
par(mfrow = c(1, 3))
for (i in 1:3) {
  plot(zph[i], resid = FALSE, main = paste("Schoenfeld:", rownames(zph$table)[i]))
  abline(h = coef(cox)[i], col = "red", lty = 2)
}
invisible(dev.off())

# ---- Remedy if entry group violates PH: stratified Cox (separate baseline hazard per entry group)
cox_st <- coxph(Surv(duration_days, event) ~ log_trips + log_basket + fresh_10pts + coupon_user + store_pattern +
                  strata(entry_group), data = d, ties = "efron")
cst <- summary(cox_st)
st_tab <- tab_of(cst)
write_csv(st_tab, file.path(out_dir, "cox_stratified_hazard_ratios.csv"))
zst <- cox.zph(cox_st)
write_csv(tibble(term = rownames(zst$table), chisq = zst$table[, 1], p_value = zst$table[, 3]),
          file.path(out_dir, "cox_stratified_zph.csv"))
lines <- c(lines, "Stratified Cox (strata = entry group):",
           sprintf("  HR %s = %.3f [%.3f, %.3f] p=%.3g", st_tab$term, st_tab$hr, st_tab$lower95,
                   st_tab$upper95, st_tab$p_value),
           sprintf("  cox.zph %s: p=%.3g", rownames(zst$table), zst$table[, 3]),
           sprintf("  concordance %.3f", cst$concordance[1]))

# ---- Sensitivity 1: lapse threshold is fixed at 90 days in SQL; check robustness to excluding Feb-Sep entrants
d_jan <- filter(d, entry_group == "Jan 2017")
cox_jan <- coxph(Surv(duration_days, event) ~ log_trips + log_basket + fresh_10pts + coupon_user + store_pattern,
                 data = d_jan, ties = "efron")
lines <- c(lines, sprintf("Sensitivity (Jan-2017 entrants only, n=%d, events=%d): HR log_trips = %.3f (main stratified: %.3f)",
                          nrow(d_jan), sum(d_jan$event), exp(coef(cox_jan))["log_trips"], exp(coef(cox_st))["log_trips"]))

# ---- Sensitivity 2: demographics subset (PARTIAL coverage; descriptive, small number of events)
dd <- filter(d, has_demographics, !is.na(income), !is.na(age)) |>
  mutate(income_band = factor(case_when(income %in% c("Under 15K", "15-24K", "25-34K") ~ "Under $35K",
                                        income %in% c("35-49K", "50-74K") ~ "$35K-$74K",
                                        TRUE ~ "$75K+"), levels = c("$35K-$74K", "Under $35K", "$75K+")),
         age_group = factor(case_when(age %in% c("19-24", "25-34") ~ "19-34", age %in% c("35-44", "45-54") ~ "35-54",
                                      TRUE ~ "55+"), levels = c("35-54", "19-34", "55+")))
lines <- c(lines, sprintf("Demographics subset: n=%d households with income and age, events=%d (of %d analysed)",
                          nrow(dd), sum(dd$event), nrow(d)))
if (sum(dd$event) >= 30) {
  cox_dm <- coxph(Surv(duration_days, event) ~ log_trips + income_band + age_group + strata(entry_group),
                  data = dd, ties = "efron")
  cm <- summary(cox_dm)
  dm_tab <- tab_of(cm)
  write_csv(dm_tab, file.path(out_dir, "cox_demographics_subset.csv"))
  lines <- c(lines, sprintf("  HR %s = %.3f [%.3f, %.3f] p=%.3g", dm_tab$term, dm_tab$hr, dm_tab$lower95,
                            dm_tab$upper95, dm_tab$p_value))
  lr_inc <- survdiff(Surv(duration_days, event) ~ income_band, data = dd)
  lines <- c(lines, sprintf("  log-rank income band (demographics subset): chisq=%.1f p=%.3g", lr_inc$chisq,
                            pchisq(lr_inc$chisq, length(lr_inc$n) - 1, lower.tail = FALSE)))
}

writeLines(lines, file.path(out_dir, "summary.txt"))
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))
cat(paste(lines, collapse = "\n"), "\n")
