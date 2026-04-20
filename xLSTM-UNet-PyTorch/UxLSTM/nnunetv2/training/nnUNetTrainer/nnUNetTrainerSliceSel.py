"""
Custom nnUNet trainer that computes loss only on selected slices.

Usage:
    nnUNetv2_train 702 2d all -tr nnUNetTrainerSliceSel -lr 0.005 -bs 30

Environment variable required:
    SLICE_SELECTION_FILE=path/to/slices.json

This trainer inherits from nnUNetTrainerUxLSTMBot and only modifies
the loss computation to use selected slices. Architecture, optimizer,
scheduler, and learning rate are unchanged.
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUxLSTMBot import nnUNetTrainerUxLSTMBot
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerSliceSel(nnUNetTrainerUxLSTMBot):
    """Trainer with slice selection for limited supervision."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 learning_rate, batchsize, unpack_dataset=True,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                        learning_rate, batchsize, unpack_dataset, device)

        self.slice_selection_file = os.environ.get('SLICE_SELECTION_FILE', None)
        self.selected_slices = {}

        if self.slice_selection_file and os.path.exists(self.slice_selection_file):
            with open(self.slice_selection_file, 'r') as f:
                self.selected_slices = json.load(f)
            print(f"Loaded slice selection from: {self.slice_selection_file}")
            print(f"  Cases: {len(self.selected_slices)}")
        elif self.slice_selection_file:
            print(f"WARNING: Slice selection file not found: {self.slice_selection_file}")
            print("  Using all slices for supervision")

    def _get_selected_for_case(self, case_key: str, num_slices: int) -> Optional[List[int]]:
        """Get selected slice indices for a case. Returns None if all slices should be used."""
        if not self.selected_slices:
            return None
        if case_key in self.selected_slices:
            return self.selected_slices[case_key]
        return None

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)

            # Debug resolutions
            if isinstance(output, (list, tuple)):
                 out_res = [o.shape[2] for o in output]
                 tar_res = [t.shape[2] for t in target] if isinstance(target, (list, tuple)) else [target.shape[2]]
                 print(f"DEBUG PRE-MASK: Output levels depth: {out_res}, Target levels depth: {tar_res}")

            # Apply slice selection for loss computation
            output, target = self._apply_slice_selection(output, target, batch)

            # Debug shapes post-masking
            if isinstance(output, (list, tuple)):
                 out_res = [o.shape[2] for o in output]
                 tar_res = [t.shape[2] for t in target] if isinstance(target, (list, tuple)) else [target.shape[2]]
                 print(f"DEBUG POST-MASK: Output levels depth: {out_res}, Target levels depth: {tar_res}")
            else:
                 print(f"DEBUG POST-MASK: output shape: {output.shape}, target shape: {target.shape}")

            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': loss.detach().cpu().numpy()}

    def _apply_slice_selection(self, output, target, batch):
        """
        Apply slice masking for limited supervision.
        For 2D: already slice-by-slice, so selection is handled by dataloader
        For 3D: mask loss to selected slices along depth dimension
        """
        # Check if we have slice selection active
        if not self.selected_slices:
            return output, target

        # Get case keys
        case_keys = batch.get('keys', [])
        if len(case_keys) == 0:
            return output, target

        # Check dimension: if output is 4D (B,C,H,W), it's 2D - slices already handled
        if isinstance(output, (list, tuple)):
            out_shape = output[0].shape
        else:
            out_shape = output.shape

        # 2D case: 4D tensor - dataloader already selects slices
        # The slice selection is enforced by the training strategy itself
        # No additional masking needed
        if len(out_shape) == 4:
            return output, target

        # 3D case: 5D tensor (B,C,D,H,W) - apply masking
        if len(out_shape) == 5:
            # Get selected slices for first case in batch (simplification)
            case_key = case_keys[0]
            selected = self._get_selected_for_case(case_key, out_shape[2])

            if selected is not None and len(selected) > 0:
                sel = sorted(selected)
                if isinstance(output, (list, tuple)):
                    # Deep supervision: mask each output level
                    masked_output = []
                    for i, o in enumerate(output):
                        # Downscale slice indices for lower resolution
                        scale_factor = o.shape[2] / out_shape[2]
                        sel_scaled = [int(s * scale_factor) for s in sel]
                        sel_scaled = [s for s in sel_scaled if s < o.shape[2]]
                        sel_scaled = sorted(list(set(sel_scaled)))
                        if sel_scaled:
                            masked_output.append(o[:, :, sel_scaled, :, :])
                        else:
                            masked_output.append(o)

                    masked_target = target
                    if isinstance(target, (list, tuple)):
                        masked_target = []
                        for i, t in enumerate(target):
                            # In nnU-Net, target usually has the same number of levels as output
                            # Each level i in target should correspond to level i in output
                            # and should be masked with the same scale_factor relative to the highest resolution
                            scale_factor = t.shape[2] / out_shape[2]
                            sel_scaled = [int(s * scale_factor) for s in sel]
                            sel_scaled = [s for s in sel_scaled if s < t.shape[2]]
                            sel_scaled = sorted(list(set(sel_scaled)))
                            if sel_scaled:
                                masked_target.append(t[:, :, sel_scaled, :, :])
                            else:
                                # Fallback if no slices left after scaling
                                masked_target.append(t)
                    elif target.ndim == 5:
                        scale_factor = target.shape[2] / out_shape[2]
                        sel_scaled = [int(s * scale_factor) for s in sel]
                        sel_scaled = [s for s in sel_scaled if s < target.shape[2]]
                        sel_scaled = sorted(list(set(sel_scaled)))
                        if sel_scaled:
                            masked_target = target[:, :, sel_scaled, :, :]

                    return masked_output, masked_target
                else:
                    return output[:, :, sel, :, :], target[:, :, sel, :, :]

        return output, target
