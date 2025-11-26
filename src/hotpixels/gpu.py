import torch
import pynvml
import psutil
from typing import Tuple


class GPUInfo:

    def __init__(self):
        try:
            self.cudaAvailable = torch.cuda.is_available()
        except Exception as e:
            self.cudaAvailable = False

        try:
            self.mpsAvailable = torch.backends.mps.is_available()
        except Exception as e:
            self.mpsAvailable = False

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                self.useNVML = True
                self.nvmlHandle = pynvml.nvmlDeviceGetHandleByIndex(0)
            else:
                self.useNVML = False
                self.nvmlHandle = None
        except Exception as e:
            self.useNVML = False
            self.nvmlHandle = None

    def get_gpu_names(self):
        try:
            if self.mpsAvailable:
                return [["mps", "Apple Silicon GPU (MPS)"]]
            if self.cudaAvailable:
                return [
                    [f"cuda:{i}", torch.cuda.get_device_name(i)]
                    for i in range(torch.cuda.device_count())
                ]
        except Exception as e:
            pass
        return []

    def get_gpu_present(self):
        return self.cudaAvailable or self.mpsAvailable

    def get_gpu_memory(self) -> Tuple[float, float]:  # Return (total GB, available GB)
        try:
            if self.mpsAvailable:
                total_bytes = psutil.virtual_memory().total
                free_bytes = psutil.virtual_memory().available
                return float(to_gb(total_bytes)), float(to_gb(free_bytes))
            if self.cudaAvailable:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                return float(to_gb(total_bytes)), float(to_gb(free_bytes))
        except Exception as e:
            pass
        return None
    
    def get_gpu_memory_total(self) -> float:
        try:
            if self.mpsAvailable:
                total_bytes = psutil.virtual_memory().total
                return float(to_gb(total_bytes))
            if self.cudaAvailable:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                return float(to_gb(total_bytes))
        except Exception as e:
            pass
        return None
    
    def get_gpu_memory_available(self) -> float:
        try:
            if self.mpsAvailable:
                free_bytes = psutil.virtual_memory().available
                return float(to_gb(free_bytes))
            if self.cudaAvailable:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                return float(to_gb(free_bytes))
        except Exception as e:
            pass
        return None

    def get_gpu_utilization(self) -> float:
        try:
            if self.mpsAvailable:
                return None
            if self.cudaAvailable and self.nvmlHandle:
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.nvmlHandle).gpu / 100.0
                return gpu_utilization
        except Exception as e:
            pass
        return None


def to_gb(bytes):
    return bytes / (1024 ** 3)
