import scipy.ndimage as ndi
import numpy as np
import torch

# 在文件开头，import之后，parse_args之前添加：

def filter_largest_component(mask: torch.Tensor) -> torch.Tensor:
    """
    只保留最大的连通域
    
    Args:
        mask: torch.Tensor, shape (D, H, W), bool或int
    Returns:
        过滤后的mask，同类型
    """
    if mask.sum().item() == 0:
        return mask
    
    mask_np = mask.cpu().numpy().astype(np.uint8)
    
    # 连通域分析
    labeled, num_features = ndi.label(mask_np)  #type:ignore
    
    if num_features <= 1:
        # 0个或1个连通域，不需要过滤
        return mask
    
    # 找最大的连通域
    component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    largest_idx = np.argmax(component_sizes) + 1
    
    # 只保留最大的
    filtered_np = (labeled == largest_idx).astype(mask_np.dtype)
    
    return torch.from_numpy(filtered_np).to(mask.dtype).to(mask.device)