
# vegseg: Tree Segmentation with LiDAR + PyTorch (R + Python)

**Authors:** Venkata Siva Reddy Naga, Carlos Alberto Silva, et al.,


`vegseg` is an R wrapper around a PyTorch point-cloud model for **tree / vegetation segmentation** from LiDAR `.las` files.  

It provides a clean R interface:

- Build a training dataset from a labeled `.las`
- Train a height-aware Pointnext model in Python (via **reticulate**)
- Run inference on new `.las` files and write predictions back to LAS
- Keep everything orchestrated from R, so it can be used inside R scripts, notebooks, or RStudio.


The R side handles:

- configuration and paths  
- calling Python training / inference  
- writing predicted classes back to LAS

The Python side (in `src/`) handles:

- dataset tiling and feature building (XYZ + HAG)  
- training a height-aware point model  
- full-scene inference and LAS writing

---

## 1. Getting Started

### 1.1 Installation of the vegseg package


```r
install.packages(
  "vegseg",
  repos = c("https://venkatasivanaga.r-universe.dev",
            "https://cloud.r-project.org")
)

library(vegseg)

```

---

### 1.2 Manual install (official Anaconda)

**i) Download**

- Anaconda (full distribution, includes many packages):  
  <https://www.anaconda.com/download>
- Miniconda (lightweight, only Conda + Python):  
  <https://docs.conda.io/en/latest/miniconda.html>

**ii) Install**

1. Download the Windows installer (**64-bit**) for Python 3.x.
2. Run the installer:
   - Accept the license.
   - Choose **“Just Me”** (recommended) unless you know you need “All Users”.
   - Keep the default install location (e.g. `C:\Users\<you>\anaconda3`).
   - *Optional but convenient:* check **“Add Anaconda to my PATH”** if you want
     to use `conda` from a normal Command Prompt.
3. Click **Next → Install** and wait for the installation to finish.
4. Open **Anaconda Prompt** from the Start menu and run:

   ```bash
   conda --version

### 1.3 Create the `pointnext` Conda environment (from R)

You can create the Python environment directly from R using **reticulate** and
install all Python dependencies from `requirements.txt`.

```r
# 1) Install and load reticulate (once)
install.packages("reticulate")
library(reticulate)

# 2) Tell reticulate where Conda lives (optional on most Anaconda installs)
#    Adjust the path if your Anaconda is somewhere else.
use_condaenv("base", required = FALSE)

# 3) Create a new Conda environment called "pointnext" with Python 3.10
conda_create(
  envname  = "pointnext",
  packages = "python=3.10"
)

# 4) Install Python dependencies from requirements.txt using pip
#    Path to your requirements file inside this repo/package.
#    Example if it lives at: src/requirements.txt
req_file <- file.path(getwd(), "src", "requirements.txt")
# If you're running this from inside the installed package, you can also use:
# req_file <- system.file("python", "requirements.txt", package = "vegseg")

conda_install(
  envname  = "pointnext",
  packages = c("-r", normalizePath(req_file)),
  pip      = TRUE
)

# 5) Activate this environment for the current R session
use_condaenv("pointnext", required = TRUE)

# Sanity check: should show Python from the "pointnext" env
py_config()

```


## 3. Predict on a new LAS using a pre-trained model

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

## 4. Train a new model on your own labelled LAS data


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

## 5 Predicted results

The figure below shows an example of the vegetation segmentation applied to a labeled LAS file.
Each point is colored by its predicted class (e.g., ground/understory, stem, canopy foliage).

![Example segmentation output](readme/trees.png)

![Example segmentation output](readme/trees1.png)

In this example, the model was trained on `trees.las` and then used to predict labels for the
same scene. The output LAS (`trees_predicted.las`) stores predictions in the `classification`
field, which can be visualized in tools like CloudCompare or QGIS using a class-based color ramp.
