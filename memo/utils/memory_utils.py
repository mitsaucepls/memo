import torch
import gc
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages GPU/CPU memory by offloading models when needed"""
    
    def __init__(self, 
                 enable_offload: bool = False,
                 threshold_gb: float = 14.0,
                 offload_models: Optional[List[str]] = None):
        self.enable_offload = enable_offload
        self.threshold_gb = threshold_gb
        self.offload_models = offload_models or []
        self.model_locations: Dict[str, str] = {}  # Track where each model is
        self.models: Dict[str, Any] = {}  # Store model references
        
    def register_model(self, name: str, model: Any):
        """Register a model for memory management"""
        self.models[name] = model
        self.model_locations[name] = "gpu" if next(model.parameters()).is_cuda else "cpu"
        
    def get_gpu_memory_usage_gb(self) -> float:
        """Get current GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024**3)
    
    def should_offload(self) -> bool:
        """Check if we should start offloading models"""
        if not self.enable_offload:
            return False
        return self.get_gpu_memory_usage_gb() > self.threshold_gb
    
    def offload_model_to_cpu(self, name: str):
        """Move a specific model to CPU"""
        if name not in self.models:
            logger.warning(f"Model {name} not registered for offloading")
            return
            
        if self.model_locations[name] == "cpu":
            return  # Already on CPU
            
        logger.info(f"Offloading {name} to CPU to free GPU memory")
        model = self.models[name]
        model.to("cpu")
        self.model_locations[name] = "cpu"
        
        # Clear GPU cache after offloading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_model_to_gpu(self, name: str, device: str = "cuda", dtype: Optional[torch.dtype] = None):
        """Move a specific model to GPU"""
        if name not in self.models:
            logger.warning(f"Model {name} not registered")
            return
            
        if self.model_locations[name] == "gpu":
            return  # Already on GPU
            
        logger.info(f"Loading {name} to GPU")
        model = self.models[name]
        if dtype:
            model.to(device=device, dtype=dtype)
        else:
            model.to(device)
        self.model_locations[name] = "gpu"
    
    def auto_offload_if_needed(self):
        """Automatically offload models if memory usage is too high"""
        if not self.should_offload():
            return
            
        logger.info(f"GPU memory usage ({self.get_gpu_memory_usage_gb():.2f}GB) exceeds threshold ({self.threshold_gb}GB)")
        
        # Offload models in priority order
        for model_name in self.offload_models:
            if model_name in self.models and self.model_locations[model_name] == "gpu":
                self.offload_model_to_cpu(model_name)
                
                # Check if we freed enough memory
                if not self.should_offload():
                    logger.info("Sufficient memory freed")
                    break
    
    @contextmanager
    def model_context(self, name: str, device: str = "cuda", dtype: Optional[torch.dtype] = None):
        """Context manager to temporarily load a model to GPU and optionally offload after use"""
        was_on_cpu = self.model_locations.get(name) == "cpu"
        
        try:
            # Load to GPU if needed
            if was_on_cpu:
                self.load_model_to_gpu(name, device, dtype)
            yield self.models[name]
        finally:
            # Offload back to CPU if it was originally there and offloading is enabled
            if was_on_cpu and self.enable_offload:
                self.offload_model_to_cpu(name)
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        stats = {
            "gpu_memory_gb": self.get_gpu_memory_usage_gb(),
            "model_locations": self.model_locations.copy(),
            "offload_enabled": self.enable_offload,
            "threshold_gb": self.threshold_gb
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        
        return stats
    
    def print_memory_stats(self):
        """Print current memory statistics"""
        stats = self.get_memory_stats()
        logger.info("Memory Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")


# Global memory manager instance
memory_manager = MemoryManager()


def setup_memory_manager(config):
    """Setup memory manager from config"""
    global memory_manager
    memory_manager = MemoryManager(
        enable_offload=getattr(config, 'enable_cpu_offload', False),
        threshold_gb=getattr(config, 'cpu_offload_threshold_gb', 14.0),
        offload_models=getattr(config, 'offload_models', [])
    )
    return memory_manager