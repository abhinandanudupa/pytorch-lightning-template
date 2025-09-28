from typing import Any, Dict, List, Tuple, Union

import torch

loss_clses: Dict[str, Union[torch.nn.Module, Any]] = {}


class LossModule(torch.nn.Module):
    def __init__(
        self,
        losses: Union[List[Dict[str, Dict[str, Any]]], List[Dict[str, Dict[str, Any]]]],
    ):
        self.losses = []
        for name, config in losses:
            wt = config["weight"]
            params = config["params"]
            fn = loss_clses[name](**params)
            self.losses.append(
                {
                    "name": name,
                    "weight": wt,
                    "fn": fn,
                }
            )

    def forward(self, x, y):
        loss: Dict[str, torch.Tensor] = {"total": torch.Tensor(0.0, device=x.device)}
        for loss_fn in self.losses:
            loss[name] = loss_fn["fn"](x, y)
            loss["total"] += loss[name] * loss_fn["weight"]

        return loss
