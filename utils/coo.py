import torch


def create_matrix_from_coo(indices, shape, length=None):
    """
    Build a sparse binary matrix from a list of tuples and given shape (tuple)
    passing length will remove that amount of "padding"
    """
    rows, cols = zip(*indices)
    values = torch.ones(len(indices))
    sparse_tensor = torch.sparse_coo_tensor(torch.tensor([rows, cols]), values, size=shape)
    dense_tensor = sparse_tensor.to_dense()
    if length != None:
        r, c = shape
        extended_tensor = torch.zeros((length, length))
        extended_tensor[:r, :c] = dense_tensor
        dense_tensor = extended_tensor
    return dense_tensor


def matrix_to_coo(matrix, mask=None):
    """
    Given a dense matrix with optional attention mask trims and converts to COO
    """
    if mask is not None:
        cutoff = (mask == 0).float().argmax()
        matrix = matrix[:cutoff, :cutoff]
    indices = torch.nonzero(matrix, as_tuple=False)
    coo_format = {
        'indices': indices,
        'shape': matrix.shape
    }
    return coo_format


def create_matrix_from_coo_batch(indices, shapes, length=None, device='cpu'):
    """
    create_matrix_from_coo but compatible with a batch dimension
    """
    dense_tensors = []
    for index, shape in zip(indices, shapes):
        rows, cols = zip(*index)
        values = torch.ones(len(rows))
        sparse_tensor = torch.sparse_coo_tensor(torch.tensor([rows, cols]), values, size=shape).to(device)
        dense_tensor = sparse_tensor.to_dense()
        if length != None:
            r, c = shape
            extended_tensor = torch.full((length, length), -100.0, dtype=torch.float32, device=device)
            extended_tensor[:r, :c] = dense_tensor
            dense_tensor = extended_tensor
        dense_tensors.append(dense_tensor)
    return torch.stack(dense_tensors)


def matrix_to_coo_batch(batch, masks=None):
    """
    matrix_to_coo but compatible with a batch dimension
    """
    coo_formats = []
    for matrix, mask in zip(batch, masks):
        if mask is not None:
            cutoff = (mask == 0).float().argmax()
            matrix_batch = matrix[:cutoff, :cutoff]
        else:
            matrix_batch = matrix
        indices = torch.nonzero(matrix_batch, as_tuple=False)
        coo_format = {
            'indices': indices,
            'shape': matrix_batch.shape
        }
        coo_formats.append(coo_format)
    return coo_formats


if __name__ == '__main__':
    pass ### TODO
