"""
Grad-CAM style feature attribution for GNN backbone.

Note:
- Classical Grad-CAM is defined for CNN spatial feature maps.
- For GNN-based HSI with dense-adj GraphConv, we adapt the idea:
  * take gradients wrt the last hidden representation (after norm+relu),
  * compute channel weights via global averaging over nodes,
  * propagate importance back to input via the first linear layer.
The output is an approximate band-importance map [N, B], comparable to IG/Saliency.
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def _hook_hidden_activations(model, x, adj):
    """Run a forward pass and cache the last hidden representation after norm+relu."""
    h = model.fc_in(x)
    last = None
    for conv in model.convs:
        h = model.drop(h)
        h = conv(h, adj)
        h = model.norm(h)
        h = model.relu(h)
        last = h
    return last  # [N, H]

def explain_gradcam_hidden(model, x, adj, y, device):
    model.eval()
    x = x.detach()
    x.requires_grad_(True)  # to map back to input bands

    # Forward pass with gradient tracking
    h = model.fc_in(x)
    hidden_list = []
    for conv in model.convs:
        h = model.drop(h)
        h = conv(h, adj)
        h = model.norm(h)
        h = model.relu(h)
        hidden_list.append(h)  # store after activation
    hidden = hidden_list[-1]           # [N, H]
    logits = model.fc_out(hidden)      # [N, C]

    # Select target per-node class
    target = y.to(device)
    selected = logits.gather(1, target.view(-1, 1)).sum()

    # Backprop to get dLogit/dHidden
    model.zero_grad(set_to_none=True)
    grads = torch.autograd.grad(selected, hidden, retain_graph=False, create_graph=False)[0]  # [N, H]

    # Global average pooling over nodes -> channel weights
    weights = grads.mean(dim=0, keepdim=True)  # [1, H]

    # Grad-CAM heatmap on hidden: ReLU(weighted hidden)
    cam_hidden = F.relu(hidden * weights)  # [N, H]

    # Map hidden importance back to input bands via first layer weights
    W_in = model.fc_in.weight  # [H, B]
    # distribute hidden importance to input: approx band score per node
    band_scores = torch.matmul(cam_hidden, W_in)  # [N, B]
    return band_scores.abs()
