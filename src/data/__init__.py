"""Dataset loaders, deterministic concept operators, Macenko stain
normalization, and augmentation pipelines for the EX-IDA-CBM project.

Submodules are not auto-imported — some depend on torch/cellpose and we
don't want to force those deps on callers that only need the stateless
concept operators. Import what you need directly, e.g.::

    from src.data.concepts import compute_concept_vector
    from src.data.datasets import AneRBCIDataset
    from src.data.stain_norm import MacenkoNormalizer
"""
