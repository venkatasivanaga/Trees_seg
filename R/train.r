#' Train the vegseg model (uses existing Python env)
#' @param cfg list from vegseg_config()
#' @param setup_env logical; keep FALSE since you already have a Python env
#' @return list(best_oa, best_epoch, ckpt_path)
#' @export
vegseg_train <- function(cfg, setup_env = FALSE) {
  stopifnot(is.list(cfg))
  # import the Python trainer
  py_train <- reticulate::import("vegseg.train", delay_load = FALSE)
  
  # Make sure numeric fields are not strings
  cfg$batch_size    <- as.integer(cfg$batch_size)
  cfg$epochs        <- as.integer(cfg$epochs)
  cfg$learning_rate <- as.numeric(cfg$learning_rate)
  cfg$weight_decay  <- as.numeric(cfg$weight_decay)
  cfg$block_size    <- as.numeric(cfg$block_size)
  cfg$stride        <- as.numeric(cfg$stride)
  cfg$sample_n      <- as.integer(cfg$sample_n)
  cfg$repeat_per_tile <- as.integer(cfg$repeat_per_tile)
  cfg$min_pts_tile  <- as.integer(cfg$min_pts_tile)
  cfg$cell_size     <- as.numeric(cfg$cell_size)
  cfg$quantile      <- as.numeric(cfg$quantile)
  
  message(">> Calling Python vegseg.train.train_model(config)")
  res <- py_train$train_model(cfg)  # <-- single config list passed
  # Expect a dict: {"best_oa": float, "best_epoch": int, "ckpt_path": str}
  if (is.null(res)) stop("Python returned NULL. Check console for Python errors.")
  return(res)
}
