def check_shape(a, shape):
    """Check the shape of a tensor"""
    assert a.shape == shape, \
        f'tensor\'s shape {a.shape} != expected shape {shape}'
