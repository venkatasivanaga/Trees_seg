#' Install Python dependencies into a conda environment
#'
#' This helper uses `reticulate::conda_install()` with pip to install
#' all Python packages needed by vegseg into a given conda env.
#'
#' @param envname Name of the conda environment (default: "pointnext").
#'   The environment must already exist (e.g., created with `conda_create()`).
#' @return Invisibly returns TRUE on success.
#' @export
vegseg_install_py_deps <- function(envname = "pointnext") {
  # Make sure reticulate is available
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Please install it first.")
  }
  
  # Requirements from your requirements.txt, expressed as pip arguments
  pip_args <- c(
    # Deep learning (CUDA 12.1 builds) – only works on a CUDA 12.1-capable GPU machine
    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
    "torch==2.5.1+cu121",
    "torchvision==0.20.1+cu121",
    "torchaudio==2.5.1+cu121",
    
    # Core numerics
    "numpy~=2.2",
    "scipy~=1.15",
    "scikit-learn~=1.7",
    "tqdm>=4.66",
    
    # Point cloud IO
    "laspy~=2.6",
    "lazrs~=0.7",
    
    # Metrics / plots
    "matplotlib~=3.10",
    "seaborn~=0.13"
  )
  
  message(">> Installing Python deps into conda env '", envname, "' using pip...")
  
  reticulate::conda_install(
    envname  = envname,
    packages = pip_args,
    pip      = TRUE
  )
  
  message(">> Finished installing Python deps into '", envname, "'.")
  invisible(TRUE)
}
