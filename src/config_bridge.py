def apply_cfg(py_cfg, ns):
  for k, v in py_cfg.items():
    setattr(ns, k, v)
    setattr(ns, k.upper(), v)
  from pathlib import Path
  ns.OUT_DIR = Path(ns.out_dir)
  return ns
