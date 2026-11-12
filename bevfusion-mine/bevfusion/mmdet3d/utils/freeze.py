"""Frozen-backbone fine-tuning helpers for memory-friendly fine-tuning.

Two operations are needed to freeze part of a model:
  1. ``freeze_model(model, names)``      - switch frozen modules to eval() and
     set ``requires_grad=False`` on all their parameters.  This both removes the
     frozen weights from the optimizer and stops autograd from retaining the
     intermediate activations of those modules (saving a lot of VRAM).
  2. ``FrozenBackboneEvalHook``          - mmcv runner keeps calling
     ``model.train()`` at the start of every epoch, which would flip the frozen
     modules back into train mode (and let their BN running stats drift). This
     hook re-asserts eval() on the frozen modules at ``before_train_epoch``.

Example (also see configs/.../resnet50/convfuser_frozenfinetune.yaml)::

    freeze:
      - encoders.camera.backbone
      - encoders.lidar.backbone
"""

from mmcv.runner import HOOKS, Hook as BaseHook


def _match(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def freeze_model(model, names):
    """Freeze submodules (eval + requires_grad=False) by dot-path prefixes.

    Args:
        model (nn.Module): full model, e.g. the BEVFusion instance.
        names (list[str] | str | None): module-path prefixes to freeze, e.g.
            ``["encoders.camera.backbone", "encoders.lidar.backbone"]``.

    Returns:
        list[nn.Module]: the frozen module objects (convenient for inspection).
    """
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    names = list(names)

    stats = {t: [0, 0] for t in names}  # [num_submodules, num_params]
    frozen_modules = []
    for name, module in model.named_modules():
        if _match(name, names):
            module.eval()
            for p in module.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    for t in names:
                        if name == t or name.startswith(t + "."):
                            stats[t][1] += p.numel()
                            break
            for t in names:
                if name == t or name.startswith(t + "."):
                    stats[t][0] += 1
                    break
            frozen_modules.append(module)

    total = 0
    for t, (n_mod, n_param) in stats.items():
        total += n_param
        print(
            f"[freeze] '{t}': {n_mod} submodule(s), "
            f"{n_param / 1e6:.2f} M params -> eval() + requires_grad=False"
        )
    print(f"[freeze] total frozen params: {total / 1e6:.2f} M")
    return frozen_modules


@HOOKS.register_module()
class FrozenBackboneEvalHook(BaseHook):
    """Re-assert eval() on frozen modules after the runner calls model.train().

    Register one instance after the runner is built (see mmdet3d/apis/train.py).
    """

    def __init__(self, names):
        if isinstance(names, str):
            names = [names]
        self.names = list(names)

    def before_train_epoch(self, runner):
        model = getattr(runner.model, "module", runner.model)
        for name, module in model.named_modules():
            if module.training and _match(name, self.names):
                module.eval()