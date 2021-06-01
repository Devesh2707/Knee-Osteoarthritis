from trainer import run
from config import CONFIG

cfg = CONFIG()

for fold in range(cfg.nfolds):
    run(fold)