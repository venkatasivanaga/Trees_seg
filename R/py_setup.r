#' Create/use a venv and install Python deps listed in inst/python/requirements.txt
#' @export
vegseg_py_setup <- function(envdir = file.path(system.file("python", package = "vegseg"), ".venv")) {
  reticulate::virtualenv_create(envdir, python = NULL)
  req <- system.file("python", "requirements.txt", package = "vegseg")
  if (file.exists(req)) {
    reticulate::virtualenv_install(envdir, packages = sprintf("-r %s", req), ignore_installed = TRUE)
  }
  reticulate::use_virtualenv(envdir, required = TRUE)
  invisible(envdir)
}
