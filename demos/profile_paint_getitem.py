import sys
import time
[sys.path.append(i) for i in ['.', '..']]

from easydict import EasyDict as edict
from lib.util import load_config
from datasets.paint_talus import paintTalusDataset

LOG_PATH = "demos/profile_paint_getitem_out.txt"
f = open(LOG_PATH, "w", buffering=1)

def log(msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()

config = edict(load_config("configs/talus_paint_10ep.yaml"))
ds = paintTalusDataset(config, "train")

log(f"num samples: {len(ds)}")

N = 5
t_total0 = time.time()
for i in range(N):
    t0 = time.time()
    data = ds.get_input_train(i)
    dt = time.time() - t0
    log(f"[{i}] took {dt:.3f}s  src={data[0].shape} tgt={data[1].shape} n_corr={data[6].shape}")
log(f"avg per item: {(time.time() - t_total0) / N} s")
f.close()
