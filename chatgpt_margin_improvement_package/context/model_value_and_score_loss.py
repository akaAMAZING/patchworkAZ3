# EXCERPT from src/network/model.py
# Value head output and loss computation (value MSE, score Huber).
# Full path in repo: src/network/model.py

# --- ValueHead forward (lines ~355-399) ---
# Value head outputting scalar in [-1, 1]. Optional score head outputs tanh(margin).
# value = tanh(fc2(hidden))
# score = tanh(score_head_linear(hidden))   # tanh-normalised margin
# Returns (value, score, z_lin) when with_score_head=True; z_lin used for Huber in logit space.

# --- get_loss() relevant parts (lines ~818-924) ---
# Inputs: target_policy, target_value, target_score (optional), policy_weight, value_weight, score_loss_weight
# Policy loss: cross-entropy
# Value loss: F.mse_loss(value, target_value)   <-- target_value is currently BINARY ±1/0
# Score loss (when target_score is not None and score_loss_weight > 0):
#   - target_score and predicted score are tanh-normalised margin in (-1, 1)
#   - Loss computed in logit space: z_pred = atanh(predicted), z_t = atanh(target)
#   - score_loss = F.huber_loss(z_pred, z_t, delta=1.0)  # ~30 pts in raw-margin units
# total_loss = policy_weight * policy_loss + value_weight * value_loss + score_loss_weight * score_loss + ...

# Code snippet (actual implementation):
"""
        # Value loss (MSE)
        value_loss = F.mse_loss(value, target_value)

        # Score loss in logit space (Huber on z) to avoid gradient saturation for large margins.
        score_loss = torch.tensor(0.0, device=state.device)
        if target_score is not None and score_loss_weight > 0 and self.value_head.score_head is not None:
            predicted_flat = score.squeeze(-1)
            target_flat = target_score if target_score.dim() <= 1 else target_score.squeeze(-1)
            if z_lin is not None:
                z_pred = z_lin.squeeze(-1)
            else:
                eps = torch.finfo(predicted_flat.dtype).eps
                clamp_val = 1.0 - 2.0 * eps
                z_pred = torch.atanh(predicted_flat.float().clamp(-clamp_val, clamp_val).to(predicted_flat.dtype))
            eps_t = torch.finfo(target_flat.dtype).eps
            clamp_val_t = 1.0 - 2.0 * eps_t
            z_t = torch.atanh(target_flat.float().clamp(-clamp_val_t, clamp_val_t).to(target_flat.dtype))
            score_loss = F.huber_loss(z_pred, z_t, delta=1.0)

        total_loss = (
            policy_weight * policy_loss
            + value_weight * value_loss
            + score_loss_weight * score_loss
            + ownership_weight * ownership_loss
        )
"""
