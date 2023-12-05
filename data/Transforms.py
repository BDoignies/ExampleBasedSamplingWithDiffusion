import numpy as np

def compose(transforms):
    def wrapper(x):
        for t in transforms:
            x = t(x)
        return x
    return wrapper

def batch(func):
    def wrapper(points):
        if points.ndim == 4:
            rslts = []
            for p in points:
                rslts.append(func(p)[None, ...])
            return np.vstack(rslts)
        else:
            return func(points)
    return wrapper

@batch
def to_image(points):
    N = points.shape[0]
    D = points.shape[1]
    n = int(round((N ** (1 / D))))
    
    if (n ** D) != N:
        raise ValueError(f"Can't perform stratified imaging:  {N} != {n} ** {D}")
        
    strata_factor = np.cumprod(n ** np.arange(0, D)).astype(np.int64)

    # Integer coordinate of the strata
    strata_coords = np.floor(points * n).astype(np.int64)

    # Compute index of strata from its coords (scan)
    indices = np.sum(strata_coords * strata_factor, axis=1)

    if np.unique(indices).shape != indices.shape:
        print("Warning: pointset is not stratified... Reordering is likely to fail")
        raise ValueError("Warning: pointset is not stratified... Reordering is likely to fail")

    # Check how to order original array from those indices
    indices = np.argsort(indices)

    points = points[indices]
    points = points.reshape((*[n] * D, D))

    return points

@batch
def to_image_stratified(points):
    """
        points: shape (N, D)
    """
    N, D = points.shape
    points = to_image(points)

    n = points.shape[0]
    # Do not use mgrid for compatibility with to_image ordering
    grid = np.linspace(0, 1, n, endpoint=False)
    grid = np.meshgrid(*[grid] * D)
    grid = [g.ravel() for g in grid]
    grid = np.vstack(grid).T.reshape((*[n] * D, D))

    points = points - grid
    # Put to [0, 1] then to [-1, 1]
    points = points * n
    points = 2. * points - 1.

    points = np.swapaxes(np.swapaxes(points, 2, 1), 1, 0)
    return points

@batch
def to_pointset_stratified(points):
    """
        points: shape (D, n, ..., n)
    """
    init_shape = points.shape
    D = points.shape[0]
    n = points.shape[1]
    N = int(n ** D)

    # Do not use mgrid for compatibility with to_image ordering
    grid = np.linspace(0, 1, n, endpoint=False)
    grid = np.meshgrid(*[grid] * D)
    grid = [g.ravel() for g in grid]
    grid = np.vstack(grid).T.reshape((*[n] * D, D))

    # Make last channel become first
    for i in range(D):
        points = np.swapaxes(points, i, i + 1)

    points = (points + 1.) * 0.5
    points = points / n
    points = points + grid
    points = np.reshape(points, (N, D))
    return points

@batch
def to_image_optimal_transport(points):
    """
        Take pointset of shape [N, D] and 
        return an image of shape (n, n, ..., n, D)
        (where N = n ** D) in the range [-1, 1].

        The image is ordered accodring to optimal
        transport to a grid (where points are in
        the middle of cells) 
    """
    import ot

    N = points.shape[0]
    D = points.shape[1]
    n = int(round((N ** (1 / D))))

    if (n ** D) != N:
        raise ValueError(f"Can't perform OT:  {N} != {n} ** {D}")
    
    # Do not use mgrid for compatibility with other methods
    grid = np.linspace(0, 1, n, endpoint=False)
    grid = np.meshgrid(*[grid] * D)
    grid = [g.ravel() for g in grid]
    grid = np.vstack(grid).T.reshape((N, D))
    grid = grid + (1 / (2 * n))
    
    M = ot.dist(grid, points)
    a, b = np.ones((2, N)) / N
    G0 = ot.emd(a, b, M)

    # Reorder according to grid
    points = points[np.nonzero(G0)[1]]
    # points = to_image(points)

    # To [-1, 1]
    points = (points - grid) * n
    # points = np.clip(points, -1, 1)
    points = points.T.reshape(D, *[n] * D)
    return points

@batch
def to_pointset_optimal_transport(points):
    n = points.shape[-1]
    D = points.shape[0]

    N = int(n ** D)

    grid = np.linspace(0, 1, n, endpoint=False)
    grid = np.meshgrid(*[grid] * D)
    grid = [g.ravel() for g in grid]
    grid = np.vstack(grid).T.reshape((N, D))
    grid = grid + (1 / (2 * n))

    points = points.reshape((D, N)).T
    points = (points / n) + grid
    points = np.clip(points, 0, 1.0)
    
    points = points.T # Shape (D, N)
    points = points.reshape((D, *[n] * D))
    return points


@batch
def to_image_ramping_optimal_transport(points):
    """
        Take pointset of shape [N, D] and 
        return an image of shape (n, n, ..., n, D)
        (where N = n ** D) in the range [-1, 1].

        The image is ordered accodring to optimal
        transport to a grid (where points are in
        the middle of cells) 
    """
    import ot

    points[:, 0] = np.sqrt(points[:, 0])

    N = points.shape[0]
    D = points.shape[1]
    n = int((N ** (1 / D)))

    if (n ** D) != N:
        raise ValueError(f"Can't perform OT:  {N} != {n} ** {D}")
    
    # Do not use mgrid for compatibility with other methods
    grid = np.linspace(0, 1, n, endpoint=False)
    grid = np.meshgrid(*[grid] * D)
    grid = [g.ravel() for g in grid]
    grid = np.vstack(grid).T.reshape((N, D))
    grid = grid + (1 / (2 * n))
    
    M = ot.dist(grid, points)
    a, b = np.ones((2, N)) / N
    G0 = ot.emd(a, b, M)

    # Reorder according to grid
    points = points[np.nonzero(G0)[1]]
    # points = to_image(points)

    # To [-1, 1]
    points = (points - grid) * n
    # points = np.clip(points, -1, 1)
    points = points.T.reshape(D, *[n] * D)
    return points

@batch
def to_pointset_ramping_optimal_transport(points):
    n = points.shape[-1]
    D = points.shape[0]

    N = int(n ** D)

    grid = np.linspace(0, 1, n, endpoint=False)
    grid = np.meshgrid(*[grid] * D)
    grid = [g.ravel() for g in grid]
    grid = np.vstack(grid).T.reshape((N, D))
    grid = grid + (1 / (2 * n))

    points = points.reshape((D, N)).T
    points = (points / n) + grid
    points = np.clip(points, 0, 1.0)
    
    points = points.T # Shape (D, N)
    points = points.reshape((D, *[n] * D))
    return points

