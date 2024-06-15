
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange
import torch
import torch.nn.functional as F

def select_multi_from_tensor(logits: Float[Tensor, 'b h'], choice_ids: Int[Tensor, 'b ...']) -> Float[Tensor, 'b ...']:
    """select from the 2nd dim of a tensor"""
    inds = torch.arange(logits.shape[0]).to(logits.device)
    for _ in range(choice_ids.ndim - 1):
        inds = inds.unsqueeze(-1)
    r = logits[inds, choice_ids.long()]
    return r


# def select_multi_from_tensor2(logits, choice_ids):
#     """using for loops"""
#     r = []
#     for i in range(logits.shape[0]):
#         r.append(logits[i, choice_ids[i].long()])
#     return torch.stack(r)


# b= 2
# h = 10
# logits = torch.rand((b, h)).float()
# choices = torch.randint(h, (b,))

# y = select_multi_from_tensor(logits, choices)
# y2 = select_multi_from_tensor2(logits, choices)
# np.testing.assert_array_almost_equal(y, y2)
# y, y2
