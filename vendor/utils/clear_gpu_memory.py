"""
Utility script to clear GPU memory
"""

import torch
import gc


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ GPU memory cleared")
        
        # Print memory info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9
            free = total - reserved
            
            print(f"\nGPU {i} ({props.name}):")
            print(f"  Total: {total:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {free:.2f} GB")
    else:
        print("No CUDA device available")


if __name__ == "__main__":
    clear_gpu_memory()

