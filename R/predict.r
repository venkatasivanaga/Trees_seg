#' Run prediction on cfg$las_path and write output LAS
#' @param cfg list from vegseg_config()
#' @param mode "overwrite" or "extra"
#' @param setup_env logical; FALSE if you already have Python env
#' @return output LAS path (character)
#' @export
vegseg_predict <- function(cfg, mode = c("overwrite", "extra"), setup_env = FALSE) {
  mode <- match.arg(mode)
  stopifnot(is.list(cfg))
  
  py_model  <- reticulate::import("vegseg.model",  delay_load = FALSE)
  py_infer  <- reticulate::import("vegseg.infer",  delay_load = FALSE)
  torch     <- reticulate::import("torch",         delay_load = FALSE)
  
  device <- if (is.null(cfg$device)) {
    if (torch$cuda$is_available()) "cuda" else "cpu"
  } else cfg$device
  
  message(">> Loading model weights: ", cfg$model_path)
  # Build the model skeleton to load weights
  num_classes <- if (!is.null(cfg$num_classes)) as.integer(cfg$num_classes) else 3L
  mdl <- py_model$HeightAwarePointNetTiny(in_ch = 4L, num_classes = num_classes, k = 16L,
                                          z_idx = 2L, hag_idx = 3L, use_height_prior = TRUE)
  mdl <- mdl$to(device)
  state_dict <- torch$load(cfg$model_path, map_location = device)
  mdl$load_state_dict(state_dict)
  mdl$eval()
  
  message(">> Running inference on ", cfg$las_path, " ...")
  y_pred <- py_infer$infer_on_las_path(
    model          = mdl,
    las_path       = cfg$las_path,
    BLOCK_SIZE     = as.numeric(cfg$block_size),
    STRIDE         = as.numeric(cfg$stride),
    SAMPLE_N       = as.integer(cfg$sample_n),
    REPEAT_PER_TILE= as.integer(cfg$repeat_per_tile),
    MIN_PTS_TILE   = as.integer(cfg$min_pts_tile),
    CELL_SIZE      = as.numeric(cfg$cell_size),
    QUANTILE       = as.numeric(cfg$quantile),
    DEVICE         = device
  )
  
  # decide output path
  out_dir <- cfg$out_pred_dir
  if (is.null(out_dir) || !nzchar(out_dir)) out_dir <- "data/output_predictions"
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  out_las <- file.path(out_dir, "trees_predicted.las")
  
  message(">> Writing predictions to: ", out_las, " (mode=", mode, ")")
  py_infer$write_predictions_to_las(
    in_las  = cfg$las_path,
    out_las = out_las,
    y_pred  = y_pred,
    mode    = mode
  )
  
  invisible(out_las)
}
