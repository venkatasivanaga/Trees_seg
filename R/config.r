#' Default config for vegseg (override values as needed)
#' @export
vegseg_config <- function(
    las_path      = "data/trees.las",
    out_dir       = "data/ds_hag4",
    out_pred_dir  = "data/output_predictions",
    model_path    = "data/model/best_model.pth",
    device        = NULL,     # NULL => Python picks cuda/cpu
    block_size    = 6.0,
    stride        = 1.0,
    sample_n      = 4096,
    repeat_per_tile = 4,
    min_pts_tile  = 512,
    val_split     = 0.15,
    test_split    = 0.10,
    seed          = 42,
    batch_size    = 16,
    epochs        = 20,
    learning_rate = 1e-5,
    weight_decay  = 1e-4,
    cell_size     = 0.25,   # HAG grid size (m)
    quantile      = 0.05
) { as.list(environment()) }
