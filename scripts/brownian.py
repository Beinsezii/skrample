#! /usr/bin/env python

from time import perf_counter_ns

import torch

from skrample.common import Step
from skrample.pytorch.noise import Brownian

print("device\tdtype\tshape\tsteps\tmedian_ms")
with torch.inference_mode():
    for device in torch.device("cuda:0"), torch.device("cpu"):
        for dtype in torch.bfloat16, torch.float32:
            if dtype == torch.bfloat16 and device.type == "cpu":
                continue
            for shape in (1, 4, 512 // 8, 512 // 8), (2, 16, 1280 // 8, 720 // 8):
                for steps in 10, 50, 200:
                    rng = Brownian.from_inputs(shape, torch.Generator(device).manual_seed(42), dtype=dtype)

                    clocks: list[int] = []

                    for n in range(steps):
                        step = Step.from_int(n, steps)
                        t = perf_counter_ns()
                        _ = rng.generate(step)
                        clocks.append(perf_counter_ns() - t)

                    print(f"{device}\t{dtype}\t{shape}\t{steps}\t{sorted(clocks)[len(clocks) // 2] / 1e6:.2f}")
