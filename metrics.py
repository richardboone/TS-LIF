import torch
import math

def calculate_rate_plasticity(spikes_before, spikes_after):
    rate_before = torch.sum(spikes_before).float()
    rate_after = torch.sum(spikes_after).float()
    rate_change = torch.abs(rate_after - rate_before) / (rate_before + 1e-7)
    return rate_change.item()

def calculate_absolute_rate_plasticity(spikes_before, spikes_after, T):
    # Calculate total spikes per neuron across the time dimension
    rate_before = spikes_before.float().sum(dim=1)
    rate_after = spikes_after.float().sum(dim=1)
    
    # Raw absolute difference in spike counts
    rate_change = torch.abs(rate_after - rate_before)
    
    # Normalize by T to get the change in firing frequency per timestep.
    # (e.g., if a neuron fired 2 times before and 3 times after over 10 steps, 
    # the change is 1 / 10 = 0.1)
    rate_change_per_step = rate_change / T
    
    # Optional but recommended: Mask out completely dead neurons 
    # so the massive amount of zeros doesn't drag your average to zero.
    active_mask = torch.max(rate_before, rate_after) > 0
    
    if not active_mask.any():
        return 0.0
        
    return rate_change_per_step[active_mask].mean().item()

def calculate_temporal_plasticity_wasserstein(spikes_before, spikes_after, T):
    spikes_before_flat = spikes_before.flatten(2)
    spikes_after_flat = spikes_after.flatten(2)
    dist_before = spikes_before_flat / (spikes_before_flat.sum(dim=1, keepdim=True) + 1e-9)
    dist_after = spikes_after_flat / (spikes_after_flat.sum(dim=1, keepdim=True) + 1e-9)
    cdf_before = torch.cumsum(dist_before, dim=1)
    cdf_after = torch.cumsum(dist_after, dim=1)
    wasserstein_distance = torch.sum(torch.abs(cdf_before - cdf_after), dim=1)
    normalized_distance = torch.mean(wasserstein_distance) / (T)
    return normalized_distance.item()

def calculate_population_entropy(spikes: torch.Tensor) -> float:
    if spikes.ndim > 3:
        spikes = spikes.flatten(2)
    spike_counts = spikes.sum(dim=1).float()
    total_spikes_per_item = spike_counts.sum(dim=1, keepdim=True)
    batch_size = spikes.shape[0]
    entropy_bits = torch.zeros(batch_size, device=spikes.device)
    non_silent_mask = (total_spikes_per_item > 0).squeeze()
    if not non_silent_mask.any():
        return entropy_bits.mean().item()
    p_neurons = spike_counts[non_silent_mask] / total_spikes_per_item[non_silent_mask]
    entropy_nats_terms = torch.special.entr(p_neurons)
    entropy_nats = entropy_nats_terms.sum(dim=1)
    entropy_bits_for_non_silent = entropy_nats / torch.log(torch.tensor(2.0, device=spikes.device))
    entropy_bits[non_silent_mask] = entropy_bits_for_non_silent
    return entropy_bits.mean().item()



def calculate_temporal_entropy(spikes: torch.Tensor, time_dim: int = 1) -> float:
    """
    Calculates the average temporal entropy of active neurons in a spiking layer.
    
    Args:
        spikes: Tensor of spikes. Default shape assumption is (Batch, Time, Neurons...).
        time_dim: The index of the time dimension. 
                  Use 1 for (Batch, Time, ...) or 0 for (Time, Batch, ...).
    """
    # 1. Standardize shape to (Batch, Time, Neurons...)
    if time_dim == 0:
        spikes = spikes.transpose(0, 1)
    
    # Flatten spatial/neuron dims so we just have (Batch, Time, Neurons)
    if spikes.ndim > 3:
        spikes = spikes.flatten(2)
        
    # 2. Get total spikes per neuron across the time dimension
    # Shape: (Batch, 1, Neurons)
    spike_counts = spikes.sum(dim=1, keepdim=True).float()
    
    # Create a mask to track which neurons actually fired
    # Shape: (Batch, Neurons)
    active_neurons_mask = (spike_counts > 0).squeeze(1)
    
    # If the whole layer is dead, temporal entropy is 0
    if not active_neurons_mask.any():
        return 0.0
        
    # 3. Calculate probabilities. 
    # We use a safe denominator trick to prevent 0/0 NaNs on silent neurons.
    # Because torch.special.entr(0) == 0, we can safely pretend silent neurons 
    # had a denominator of 1.
    safe_counts = spike_counts.clone()
    safe_counts[safe_counts == 0] = 1.0
    
    # p_time represents the probability distribution of spikes over time for each neuron
    # Shape: (Batch, Time, Neurons)
    p_time = spikes / safe_counts
    
    # 4. Calculate Shannon entropy over the time dimension (dim=1)
    # Shape after sum: (Batch, Neurons)
    entropy_nats = torch.special.entr(p_time).sum(dim=1)
    
    # Convert nats to bits
    entropy_bits = entropy_nats / math.log(2.0)
    
    # 5. Average the entropy, but ONLY across the neurons that actually fired.
    # Including dead neurons (which have 0 entropy) would artificially drag down 
    # the metric, punishing layers that are just sparsely active.
    mean_active_temporal_entropy = entropy_bits[active_neurons_mask].mean().item()
    
    return mean_active_temporal_entropy


def calculate_l1_plasticity(spikes_before, spikes_after, T):
    """
    Measures the L1 distance (total spike changes) between two spike tensors,
    normalized by the number of neurons and timesteps.
    A value of 0.1 means that on average, 10% of the spikes in a neuron's timeline changed.
    """
    # Total number of spike additions and removals
    l1_distance = torch.sum(torch.abs(spikes_before.float() - spikes_after.float()))

    # Normalize by the number of neurons and timesteps to get an average
    num_neurons = spikes_before.numel() / T
    normalized_l1 = l1_distance / (num_neurons * T)
    
    return normalized_l1.item()

@torch.jit.script
def calculate_vp_distance_optimized(spikes_before: torch.Tensor, spikes_after: torch.Tensor, q: float, T: int, sample_ratio: float = 0.01) -> float:
    """
    An optimized VP distance calculation that combines JIT compilation, neuron sub-sampling,
    and a time-band optimization to prune the inner loop search space.
    """
    device = spikes_before.device
    spikes_before = spikes_before.float()
    spikes_after = spikes_after.float()

    if spikes_before.ndim > 3:
        spikes_before = spikes_before.flatten(2)
    if spikes_after.ndim > 3:
        spikes_after = spikes_after.flatten(2)

    B, _, N = spikes_before.shape
    total_distance = 0.0

    num_to_sample = int(N * sample_ratio)
    if num_to_sample == 0 and N > 0:
        num_to_sample = 1
    
    # --- Time-Band Optimization ---
    # Calculate the max time difference where a shift is cheaper than delete+insert
    max_dist = 2.0 / q if q > 0 else float('inf')

    if num_to_sample > 0:
        for b in range(B):
            sampled_neuron_indices = torch.randperm(N, device=device)[:num_to_sample]
            
            for n in sampled_neuron_indices:
                st1 = torch.nonzero(spikes_before[b, :, n]).squeeze(-1)
                st2 = torch.nonzero(spikes_after[b, :, n]).squeeze(-1)

                n1 = st1.size(0)
                n2 = st2.size(0)

                if n1 == 0 and n2 == 0: continue
                if n1 == 0: total_distance += float(n2); continue
                if n2 == 0: total_distance += float(n1); continue

                cost_matrix = torch.zeros((n1 + 1, n2 + 1), device=device)
                cost_matrix[0, :] = torch.arange(n2 + 1, device=device).float()
                cost_matrix[:, 0] = torch.arange(n1 + 1, device=device).float()

                # --- Optimized Inner Loop ---
                j_start = 1
                for i in range(1, n1 + 1):
                    # For each spike in st1, find the relevant start for spikes in st2
                    while j_start <= n2 and st2[j_start - 1] < st1[i - 1] - max_dist:
                        j_start += 1
                        
                    for j in range(j_start, n2 + 1):
                        # If we've moved past the relevant time-band for this i, break
                        if st2[j - 1] > st1[i - 1] + max_dist:
                            break

                        cost_del = cost_matrix[i - 1, j] + 1.0
                        cost_ins = cost_matrix[i, j - 1] + 1.0
                        cost_shift = cost_matrix[i - 1, j - 1] + q * torch.abs(st1[i - 1] - st2[j - 1])
                        
                        min_cost = torch.min(cost_del, cost_ins)
                        min_cost = torch.min(min_cost, cost_shift)
                        cost_matrix[i, j] = min_cost
                
                total_distance += cost_matrix[n1, n2].item()
        
        return total_distance / (B * num_to_sample)
    else:
        return 0.0
