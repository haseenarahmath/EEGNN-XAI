import torch
from captum.attr import Saliency

def explain_saliency(model, x, adj, y, device):
    """
    Returns attributions wrt input features via raw gradients: shape [N, B].
    """
    class Wrapper(torch.nn.Module):
        def __init__(self, m, A): super().__init__(); self.m, self.A = m, A
        def forward(self, feats): return self.m(feats, self.A)

    model.eval()
    wrapper = Wrapper(model, adj).to(device)
    x = x.detach().requires_grad_(True)
    sal = Saliency(wrapper)
    attrs = sal.attribute(x, target=y.to(device))
    return attrs.abs()
