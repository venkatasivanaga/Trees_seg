
# vegseg: An R Package for ---------------------------- from LiDAR Data

**Authors:** Venkata Siva Reddy Naga, Carlos Alberto Silva, et al.,

```r
library(vegseg)
library(reticulate)
use_condaenv("pointnext", required = TRUE)

cfg <- vegseg_config(
  las_path     = "data/trees.las",
  out_dir      = "data/ds_hag4",
  out_pred_dir = "data/output_predictions",
  model_path   = "data/model/best_model.pth",
  epochs       = 2, batch_size = 16,
  learning_rate = 1e-5, weight_decay = 1e-4,
  block_size = 6, stride = 1, sample_n = 4096,
  repeat_per_tile = 4, min_pts_tile = 512,
  cell_size = 0.25, quantile = 0.05
)

res <- vegseg_train(cfg, setup_env = FALSE)        # trains & saves best .pth
vegseg_predict(cfg, mode = "overwrite", setup_env = FALSE)  # writes trees_predicted.las
```



```r
library(vegseg)
library(reticulate)
use_condaenv("pointnext", required = TRUE)

cfg <- vegseg_config(
  las_path     = "data/trees2.las",  # any LAS you want to segment
  out_pred_dir = "data/output_predictions",
  model_path   = "data/model/best_model.pth"       # your pre-trained checkpoint
)

vegseg_predict(cfg, mode = "overwrite", setup_env = FALSE)
# or keep original classification and add 'pred_label':
# vegseg_predict(cfg, mode = "extra", setup_env = FALSE)
```
