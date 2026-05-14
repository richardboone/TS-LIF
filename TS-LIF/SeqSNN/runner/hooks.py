import torch
from collections import defaultdict
import sys
import os

# Add root to path to ensure we can import the top-level metrics module if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from metrics import (
        calculate_rate_plasticity,
        calculate_absolute_rate_plasticity,
        calculate_temporal_plasticity_wasserstein,
        calculate_l1_plasticity,
        calculate_temporal_entropy,
        calculate_population_entropy
    )
except ImportError:
    # Fallback for different sys paths
    sys.path.append(os.path.abspath('.'))
    from metrics import (
        calculate_rate_plasticity,
        calculate_absolute_rate_plasticity,
        calculate_temporal_plasticity_wasserstein,
        calculate_l1_plasticity,
        calculate_temporal_entropy,
        calculate_population_entropy
    )

class SpikeMetricsTracker:
    """
    Manages PyTorch forward hooks to capture step-by-step spikes from layers,
    assembles them into full sequence representations, and calculates temporal
    plasticity and entropy metrics.
    """
    def __init__(self, model, enabled=False):
        self.model = model
        self.enabled = enabled
        self.hooks = []
        self.spikes = defaultdict(list)
        self._register_hooks()

    def _register_hooks(self):
        try:
            from SeqSNN.network.snn.TSLIF import BaseNode, BaseNode1
        except ImportError:
            BaseNode, BaseNode1 = None, None

        try:
            from snntorch import Leaky
        except ImportError:
            Leaky = None

        target_classes = []
        if BaseNode is not None:
            target_classes.append(BaseNode)
        if BaseNode1 is not None:
            target_classes.append(BaseNode1)
        if Leaky is not None:
            target_classes.append(Leaky)

        if not target_classes:
            print("[SpikeMetricsTracker] WARNING: No spike node classes could be imported!")
            return

        target_classes = tuple(target_classes)

        registered_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, target_classes):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
                registered_count += 1
        
        print(f"[SpikeMetricsTracker] Successfully registered hooks on {registered_count} spiking layers.")

    def _make_hook(self, name):
        def hook_fn(module, inputs, outputs):
            if not self.enabled:
                return
            
            # Unpack outputs if they are (spikes, mem)
            spikes = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            if isinstance(spikes, torch.Tensor):
                # Detach to ensure we don't track gradients and save VRAM
                self.spikes[name].append(spikes.detach().cpu())
        return hook_fn

    def reset(self):
        self.spikes.clear()

    def get_assembled_spikes(self):
        """
        Converts a list of step-by-step spike tensors into a single aligned 
        tensor with a defined time dimension: (Batch, Time, Neurons...).
        Assumes standard single-step inputs stack easily.
        """
        assembled = {}
        for name, spikes_list in self.spikes.items():
            if not spikes_list:
                continue
            
            # Stack over the time dimension (dim=1). 
            # Tensors generally are (Batch, Neurons...) so stacking makes (Batch, Time, Neurons...)
            try:
                # Handle variable shapes gracefully
                full_spikes = torch.stack(spikes_list, dim=1)
                assembled[name] = full_spikes
            except Exception as e:
                print(f"[SpikeMetricsTracker] ERROR Assembling spikes for {name}: {e}")
                
        return assembled

    def compute_metrics(self, spikes_before, spikes_after, T):
        """
        Computes absolute rate, L1, and Wasserstein plasticity along with temporal/population entropy.
        """
        results = {}
        for layer_name in spikes_before.keys():
            if layer_name not in spikes_after:
                continue
            
            sb = spikes_before[layer_name]
            sa = spikes_after[layer_name]
            
            # Basic checks to match shapes
            if sb.shape != sa.shape:
                print(f"[SpikeMetricsTracker] WARNING: Shape mismatch for layer {layer_name}: {sb.shape} vs {sa.shape}")
                continue

            try:
                # 1. Core Plasticity Metrics
                results[f"Plasticity/{layer_name}/L1"] = calculate_l1_plasticity(sb, sa, T)
                results[f"Plasticity/{layer_name}/AbsoluteRate"] = calculate_absolute_rate_plasticity(sb, sa, T)
                results[f"Plasticity/{layer_name}/Wasserstein"] = calculate_temporal_plasticity_wasserstein(sb, sa, T)
                results[f"Plasticity/{layer_name}/RelativeRateChange"] = calculate_rate_plasticity(sb, sa)

                # 2. Temporal States (computed on 'before' spikes to save computation)
                results[f"Entropy/{layer_name}/Temporal"] = calculate_temporal_entropy(sb, time_dim=1)
                results[f"Entropy/{layer_name}/Population"] = calculate_population_entropy(sb)
            except Exception as e:
                print(f"[SpikeMetricsTracker] ERROR evaluating layer {layer_name}: {e}")

        return results

    def remove(self):
        """Removes all registered hooks to prevent memory leaks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
