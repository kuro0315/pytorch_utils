import torch.nn as nn


def dropout_train_only(model: nn.Module) -> None:
    """Recursively set only dropout layers to train mode.

    Args:
        model (nn.Module): The model to modify.

    Every module in ``model`` is first put into evaluation mode. Then, any
    modules that are instances of dropout layers are switched back to train
    mode. This function modifies ``model`` in place.
    """
    # Put everything in eval mode first
    model.eval()

    dropout_types = (nn.modules.dropout._DropoutNd, nn.AlphaDropout)
    if hasattr(nn, "FeatureAlphaDropout"):
        dropout_types += (nn.FeatureAlphaDropout,)

    def _apply(module: nn.Module) -> None:
        for child in module.children():
            if isinstance(child, dropout_types):
                child.train()
            _apply(child)

    _apply(model)
