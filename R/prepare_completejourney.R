#!/usr/bin/env Rscript
# Convert the Complete Journey .rda/.rds objects (downloaded by scripts/download_data.py) to CSV.
# Usage: Rscript R/prepare_completejourney.R <src_dir> <out_dir>
# Timestamps are written as naive America/New_York local time ("YYYY-MM-DD HH:MM:SS").
args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args) >= 1) args[1] else "data/raw/completejourney/src"
out <- if (length(args) >= 2) args[2] else "data/raw/completejourney"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

write_csv_fast <- function(x, name) {
  x <- as.data.frame(x)
  for (j in seq_along(x)) if (inherits(x[[j]], "POSIXct")) {
    x[[j]] <- format(x[[j]], "%Y-%m-%d %H:%M:%S", tz = "America/New_York")
  }
  path <- file.path(out, paste0(name, ".csv"))
  if (requireNamespace("data.table", quietly = TRUE)) {
    data.table::fwrite(x, path, quote = FALSE)
  } else {
    utils::write.csv(x, path, row.names = FALSE, quote = FALSE)
  }
  cat(sprintf("wrote %-24s %10d rows\n", name, nrow(x)))
}

load_rda <- function(f) {
  e <- new.env()
  load(file.path(src, f), envir = e)
  get(ls(e)[1], envir = e)
}

# promotions.csv (20.9M rows) is deliberately not converted: the project does not use it.
for (n in c("campaign_descriptions", "campaigns", "coupon_redemptions", "coupons", "demographics", "products")) {
  write_csv_fast(load_rda(paste0(n, ".rda")), n)
}
write_csv_fast(readRDS(file.path(src, "transactions.rds")), "transactions")
