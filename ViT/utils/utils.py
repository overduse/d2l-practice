import torch

def check_shape(a, shape):
    """Check the shape of a tensor. """
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

def cpu():
    """Get the CPU device. """
    return torch.device('cpu')

def gpu(i=0):
    """Get a GPU device. """
    return torch.device(f'cuda:{i}')

def num_gpus():
    """Get the number of available GPUs. """
    return torch.cuda.device_count()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu(). """
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists. """
    return [gpu(i) for i in range(num_gpus())]

reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
