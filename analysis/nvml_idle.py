"""Sample the node's idle GPU power floor, so benchmark energy can be net of it.

    python analysis/nvml_idle.py                        # 30 s, all GPUs, JSON to stdout
    python analysis/nvml_idle.py --seconds 60 -o idle.json

Run this with NOTHING else on the node. On a shared or display-attached GPU the
floor includes whatever else is resident, which is exactly why it has to be
measured per machine rather than assumed.

Two numbers come out per device and they are not redundant: `power_w_avg` is the
mean of instantaneous samples, `power_w_from_counter` divides the hardware energy
counter's delta by wall time. They should agree closely on an idle device; a gap
means the sampling interval is too coarse to see what the device is doing, which
matters far more once there is a real load on it.

Needs pynvml (`pip install nvidia-ml-py`); everything else here is stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def sample(seconds: float, interval: float) -> dict:
    import pynvml

    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]

    def energy_j(handle):
        # Volta and newer only. Where it is missing the counter cross-check is
        # simply absent rather than wrong, so the sampled mean still stands.
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1e3
        except Exception:
            return None

    start_j = [energy_j(h) for h in handles]
    series: list[list[float]] = [[] for _ in handles]

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for i, h in enumerate(handles):
            try:
                series[i].append(pynvml.nvmlDeviceGetPowerUsage(h) / 1e3)
            except Exception:
                pass
        time.sleep(interval)
    elapsed = time.perf_counter() - t0

    devices = []
    for i, h in enumerate(handles):
        s = sorted(series[i])
        if not s:
            continue
        end = energy_j(h)
        counter_j = (
            end - start_j[i] if (end is not None and start_j[i] is not None) else None
        )
        devices.append(
            {
                "index": i,
                "name": _name(pynvml, h),
                "n_samples": len(s),
                "power_w_avg": round(sum(s) / len(s), 2),
                "power_w_min": round(s[0], 2),
                "power_w_max": round(s[-1], 2),
                "power_w_p50": round(s[len(s) // 2], 2),
                "energy_j_counter": round(counter_j, 2) if counter_j is not None else None,
                "power_w_from_counter": (
                    round(counter_j / elapsed, 2) if counter_j is not None else None
                ),
            }
        )

    total = sum(d["power_w_avg"] for d in devices)
    return {
        "duration_s": round(elapsed, 2),
        "device_count": len(devices),
        # The node floor is the sum over devices: a job that leaves 3 of 4 A100s
        # untouched still pays for them, and that is the number to subtract when
        # asking what a NODE-hour of this work cost.
        "node_idle_w": round(total, 2),
        "per_device_idle_w": round(total / len(devices), 2) if devices else None,
        "devices": devices,
    }


def _name(pynvml, handle) -> str:
    name = pynvml.nvmlDeviceGetName(handle)
    return name.decode() if isinstance(name, bytes) else name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("-o", "--output", help="write JSON here as well as stdout")
    args = ap.parse_args()

    try:
        result = sample(args.seconds, args.interval)
    except ImportError:
        print("pynvml missing -- pip install nvidia-ml-py", file=sys.stderr)
        raise SystemExit(2)

    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
