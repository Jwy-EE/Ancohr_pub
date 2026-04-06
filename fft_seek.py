import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt

class PeriodEstimator:
    """
    Explicit Frequency-Domain Physical Prior Extractor for ANCHOR Architecture.
    Adapted input format: (Batch, Channel, Length)
    """
    def __init__(self, top_k=3):
        self.top_k = top_k

    def __call__(self, x):
        """
        Args:
            x: Input data [B, C, L]
        Returns:
            periods: List of top-k physical periods [p1, p2, p3]
        """
        # 1. RFFT transform to extract explicit dominant periods
        xf = torch.fft.rfft(x, dim=-1) 
        
        # 2. Globally unified spectral energy distribution
        frequency_list = torch.abs(xf).mean(0).mean(0)
        
        # 3. Remove DC component (detrending)
        frequency_list[0] = 0
        
        # 4. Select Top-K frequencies
        _, top_list = torch.topk(frequency_list, self.top_k)
        
        # 5. Convert to Numpy
        top_list = top_list.detach().cpu().numpy()
        
        # 6. Mapped physical period set
        L = x.shape[-1]
        periods = L // top_list
        
        # 7. Deduplicate and sort descending for FGDM dynamic tracking
        periods = [int(p) for p in periods]
        periods = sorted(list(set(periods)), reverse=True) 
        
        return periods