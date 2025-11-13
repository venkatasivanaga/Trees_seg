.onLoad <- function(libname, pkgname) {
  pkg_py <- system.file("python", package = pkgname)
  if (nzchar(pkg_py)) {
    old <- Sys.getenv("PYTHONPATH", "")
    new <- if (nzchar(old)) paste(pkg_py, old, sep = .Platform$path.sep) else pkg_py
    Sys.setenv(PYTHONPATH = new)
  }
}
