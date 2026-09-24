import os
import json
import time
import torch
import psutil
import platform


class ResourceProfiler:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Profiler not started")

        elapsed = time.time() - self.start_time
        return {
            "elapsed_sec": elapsed,
            **self.get_peak_memory(),
            **self.get_gpu_info()
        }

    def get_gpu_info(self):
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 3),
                "gpu_index": idx
            }
        return {"gpu_name": "CPU", "gpu_total_memory_gb": None}
    

    def get_cpu_info(self):
        return {
            "cpu_name": platform.processor(),
            "cpu_cores_available": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2)
        }

    def get_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return {
                "peak_memory_gb": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                ),
                "memory_type": "GPU"
            }

        proc = psutil.Process(os.getpid())
        return {
            "peak_memory_gb": round(proc.memory_info().rss / (1024**3), 3),
            "memory_type": "RAM"
        }

    def count_parameters(self, model):
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return round(total / 1e6, 3)

    def save_metrics(self, metrics, split_no, generator_name):
        out_dir = "resource_logs_" + generator_name
        os.makedirs(out_dir, exist_ok=True)

        fname = f"{out_dir}/split{split_no}.json"

        with open(fname, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[resource] saved -> {fname}")