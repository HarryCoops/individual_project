import torch
import sys
from pyinstrument import Profiler
import timeit 
import pandas as pd

BATCH_SIZE=32
IMAGE_SIZE = 256
IMAGE_DIM = 3

def convert_separate(n_agents, tensor):
    for tens in tensor:
        tens.float()

def convert_batched(n_agents, tensor):
    tensor.float()

torch.cuda.init()
torch.rand(1, device="cuda:2")
pr = Profiler()
pr.start()
times = {}
for f in ("separate", "batched"):
    _times = []
    for n in range(5, 250, 5):
        if f == "batched":
            tensor = torch.randint(
                0,
                255,
                (n, BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM), 
                device="cuda:2"
            )
        else:
            tensor = [
                torch.randint(
                    0,
                    255,
                    (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM)
                    , device="cuda:2"
                ) for i in range(n)
            ]
        t = timeit.timeit(
            f"convert_{f}({n}, tensor)", 
            setup=f"from __main__ import convert_{f}, tensor",
            number=20
        )
        del(tensor)
        _times.append(t)
        torch.cuda.empty_cache()
    times[f] = _times

times = pd.DataFrame(times, index=[n for n in range(5, 250, 5)])
print(times)
pr.stop()
print(pr.output_text(unicode=True, color=True))
ax = times.plot(xlabel="Number of 'agents'", ylabel="Mean convert time (s)")
fig = ax.get_figure()
fig.savefig("type_conversion_plot_gpu.png")
