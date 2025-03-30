import torch

def move_dict_to_device(data_dict, device):
    """
    Moves all tensor values in a dictionary to the specified device.

    Args:
        data_dict (dict): Dictionary containing tensors (and possibly other values).
        device (torch.device or str): Target device ('cuda', 'cpu', etc.)

    Returns:
        dict: A new dictionary with all tensors moved to the target device.
    """
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_dict.items()}