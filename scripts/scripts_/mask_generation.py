import torch
import numpy as np

import time


def time_counter(func):
    def fun(*args, **argv):
        t = time.perf_counter()
        result = func(*args, **argv)
        print(f"function '{func.__name__}' cost {time.perf_counter()-t}. ")
        return result

    return fun


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def buffered_future_mask(tensor):
        dim = tensor.size(0)
        inf_mask = fill_with_neg_inf(torch.zeros([dim, dim]))
        print(inf_mask)
        future_mask = torch.triu(
                inf_mask, 1
            )
        future_mask = future_mask.to(tensor)
        return future_mask[:dim, :dim]
@time_counter
def gen_sequence(length):
    list = []
    for i in range(1, 2 * length + 1):
        for _ in range(i if i <= length else 2 * length - i):
            list.append(i)
        if i < length:
            list.append(0) 
    list.insert(0, 0)
    # return torch.tensor(list, dtype=torch.float)
    return list[:-1]

@time_counter
def gen_seq(n):
    res = []    
    res = [i for i in range(1, 2 * n + 1) for _ in range(i if i <= n else 2 * n - i)]
    # return torch.tensor(res, dtype=torch.float)

def gen_mask(length, mask, seq):
    fill_seq = [i for i in range(2, 2 *length- 1)]
    print(fill_seq)
    for i in fill_seq:
        index = seq.index(i)
        print(index)
        mask[index:index+i, index:index+i] = torch.tensor(0)

    return mask

length = 3
seq = gen_sequence(length)
mask = torch.ones([len(seq),len(seq)])
mask = torch.tril(mask, 0)

print(seq)
print(mask)
mask = gen_mask(length, mask, seq) 
diag = torch.diag_embed(torch.diag(mask))
print(diag)
mask = mask - diag + torch.eye(len(seq))
print(mask)

mask = torch.where(mask == 0, torch.tensor(float("-inf")), mask) - 1
print(mask)
