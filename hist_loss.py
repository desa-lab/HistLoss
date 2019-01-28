import torch
import torch.nn as nn

class Hist(torch.autograd.Function):
  
  @staticmethod
  def forward(ctx, sim, n_bins, w):

    # linearly transform similarity values to the range between 0 and 1
    sim = sim.data
    max_, min_ = torch.max(sim), torch.min(sim)     
    sim = (sim - min_) / (max_ - min_)

    # compute the step size in the histogram
    step = 1. / n_bins
    idx = sim / step

    lower = idx.floor()
    upper = idx.ceil()

    delta_u = idx - lower
    delta_l = upper - idx

    lower = lower.long()
    upper = upper.long()

    hist = torch.bincount(upper, delta_u * w, n_bins + 1) + torch.bincount( lower, delta_l * w, n_bins + 1)
    w_sum = w.sum()
    hist = hist / w_sum

    ctx.save_for_backward(upper, lower, w, w_sum)

    return hist
    

  @staticmethod
  def backward(ctx, grad_hist):
    upper, lower, w, w_sum = ctx.saved_tensors
    grad_sim = None
  
    grad_hist = grad_hist / w_sum
    grad_sim = (grad_hist[upper] - grad_hist[lower]) * w

    return grad_sim, None, None


class HistogramLoss(nn.Module):
  def __init__(self):
    super(HistogramLoss, self).__init__()
    
    self.hist = Hist.apply

  def forward(self, sim_pos, sim_neg, n_bins, w_pos=None, w_neg=None):  
 
    sim_pos = sim_pos.flatten()
    sim_neg = sim_neg.flatten()

    if w_pos is not None:
      w_pos = w_pos.data.flatten()
      assert sim.size() == w.size(), "Please make sure the size of the similarity tensor matches that of the weight tensor."
    else:
      w_pos = torch.ones_like(sim_pos)

    if w_neg is not None:
      w_neg = w_neg.data.flatten()
      assert sim.size() == w.size(), "Please make sure the size of the similarity tensor matches that of the weight tensor."
    else:
      w_neg = torch.ones_like(sim_neg)
 
    pdf_pos = self.hist(sim_pos, n_bins, w_pos)
    pdf_neg = self.hist(sim_neg, n_bins, w_neg)

    cdf_pos = torch.cumsum(pdf_pos, dim=0)
    loss = (cdf_pos * pdf_neg).sum()

    return loss




