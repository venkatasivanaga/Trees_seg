## 1. Getting Started
## 1.1 Installation of the FuelDeep3D package

install.packages(
  "FuelDeep3D",
  repos = c(
    "https://venkatasivanaga.r-universe.dev",
    "https://cloud.r-project.org"
  )
)

library(FuelDeep3D)

## 1.2 Manual install of Anaconda / Miniconda
## (Download and install Anaconda or Miniconda from their official websites.)

## 1.3 Create the `pointnext` Conda environment (from R)

install.packages("reticulate")
library(reticulate)
library(FuelDeep3D)

# Optional: let reticulate find conda
reticulate::use_condaenv("base", required = FALSE)

# Create a new Conda environment called "pointnext" with Python 3.10
reticulate::conda_create(
  envname  = "pointnext",
  packages = "python=3.10"
)

# Install all Python dependencies into that env (helper you defined in FuelDeep3D)
install_py_deps("pointnext")

# Activate this environment for the current R session
reticulate::use_condaenv("pointnext", required = TRUE)

# Sanity check: should show Python from the "pointnext" env
py_config()



## 3. Predict on a new LAS using a pre-trained model

library(FuelDeep3D)
library(reticulate)
reticulate::use_condaenv("pointnext", required = TRUE)

cfg <- config(
  las_path      = system.file("extdata","las", "trees.las", package = "FuelDeep3D"),
  out_pred_dir  = getwd(),
  model_path    = system.file("extdata", "model","best_model.pth", package = "FuelDeep3D"),
)


FuelDeep3D::predict(cfg, mode = "overwrite", setup_env = FALSE)


require(lidR)

p0<-readLAS(system.file("extdata","las", "trees.las", package = "FuelDeep3D"))
writeLAS(p0, "C:\\Users\\vs.naga\\Documents\\Github\\FuelDeep3D\\inst\\extdata\\las\\trees.laz" )

p<-readLAS("C:/Users/vs.naga/Documents/Github/FuelDeep3D/trees_predicted.las")

head(p@data)

summary(p@data$Classification)
summary(p0@data$label)

p@data$Classification[p@data$Classification==0]<-3

plot(p0, color ="label" )

library(rgl)

# Option A: use class codes directly as palette indices (1,2,3,...)
rgl::points3d(p@data[,c(1:3)], col = p@data$Classification, size = 2)


# or keep original classification and add 'pred_label':
# predict(cfg, mode = "extra", setup_env = FALSE)

cols <- c(
  "1" = "blue",  # class 1
  "2" = "sienna",       # class 2
  "3" = "gray40"        # class 3
)

rgl::points3d(
  p@data[,c(1:3)],
  col  = cols[as.character(p@data$Classification)],
  size = 2
)


p3<-function(las,cols,...){
  
  rgl::points3d(
    las@data[,c(1:3)],
    col  = cols[as.character(las@data$Classification)],
    ...
  )

}

p3(p,cols, size=2)
## 4. Train a new model on your own labelled LAS data

library(FuelDeep3D)
library(reticulate)
use_condaenv("pointnext", required = TRUE)

cfg <- config(
  las_path      = system.file("extdata", "trees.las", package = "FuelDeep3D"),
  out_dir       = getwd(),
  out_pred_dir  = getwd(),
  model_path    = system.file("extdata", "best_model.pth", package = "FuelDeep3D"),
  epochs         = 2,
  batch_size     = 16,
  learning_rate  = 1e-5,
  weight_decay   = 1e-4,
  block_size     = 6,
  stride         = 1,
  sample_n       = 4096,
  repeat_per_tile = 4,
  min_pts_tile   = 512,
  cell_size      = 0.25,
  quantile       = 0.05
)

res <- FuelDeep3D_train(cfg, setup_env = FALSE)                  # trains & saves best .pth
predict(cfg, mode = "overwrite", setup_env = FALSE)   # writes trees_predicted.las



## 5. Predicted results
## (Visualize `trees_predicted.las` in CloudCompare or QGIS using classification colors.)
