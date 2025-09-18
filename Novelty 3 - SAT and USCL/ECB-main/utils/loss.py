import torch
import torch.nn.functional as F


def ce_loss(logits, targets, reduction="none"):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == "none":
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


###### Loss Fixed Threshold ######
# def consistency_loss(logits_u_str, logits_u_w, threshold=0.6):
#     """
#     Consistency regularization for fixed threshold loss in semi-supervised learning.
#     Args:
#         logits_u_str: logits of strong augmented unlabeled samples
#         logits_u_w: logits of weak augmented unlabeled samples
#         threshold: fixed threshold
#     Returns:
#         loss: consistency regularization loss
#     """
#     pseudo_label = torch.softmax(logits_u_w, dim=1)
#     max_probs, targets_u = torch.max(pseudo_label, dim=-1)
#     mask = max_probs.ge(threshold).float()
#     loss = (ce_loss(logits_u_str, targets_u, reduction="none") * mask).mean()
#     return loss
def consistency_loss(logits_u_str, logits_u_w, threshold=0.6, use_threshold=True):
    """
    Consistency loss with optional confidence-aware weighting.
    
    Args:
        logits_u_str: logits of strong augmented unlabeled samples (student)
        logits_u_w: logits of weak augmented unlabeled samples (teacher)
        threshold: confidence threshold (used only if use_threshold=True)
        use_threshold: if True, filter low-confidence pseudo-labels
    
    Returns:
        Weighted consistency loss
    """
    with torch.no_grad():
        pseudo_probs = torch.softmax(logits_u_w, dim=1)
        max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)

        if use_threshold:
            mask = max_probs.ge(threshold).float()
            weights = max_probs * mask  # zero out samples below threshold
        else:
            weights = max_probs  # soft confidence weighting for all samples
            
    weights = torch.clamp(max_probs, min=0.6)

    loss_raw = ce_loss(logits_u_str, pseudo_labels, reduction='none')
    loss = (loss_raw * weights).sum() / (weights.sum() + 1e-6)  # normalize by total confidence

    return loss

##################################


def contrastive_loss(features, labels, temperature=0.5):
    # Normalize features
    features = F.normalize(features, dim=1)
    batch_size = features.size(0)
    labels = labels.contiguous().view(-1, 1)

    # Create mask where positives match in label
    mask = torch.eq(labels, labels.T).float().to(features.device)

    logits = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(features.device)
    mask = mask * logits_mask

    # log prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    # Mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    # Loss
    loss = -mean_log_prob_pos.mean()
    return loss

