"""Canonical experiment names for HMACD-ECG runs.

The name must cover every dimension the experiment sweeps. If it does not, two
configurations collide on the same checkpoint path and the earlier one is
silently overwritten -- not an error, just a missing run. K and seed were
covered; the two loss weights were not, so any lambda sweep at fixed K and seed
kept only its last configuration.

When a new dimension enters the search space, add it here. This is the single
place that defines the naming, so downstream scripts can rely on it.
"""


def _fmt_weight(v):
    """Short, stable text for a loss weight: 0.1 -> '0.1', 0.05 -> '0.05', 0 -> '0'."""
    return f"{float(v):.3g}"


def experiment_name(config):
    """Build the run name from a full config dict.

    >>> experiment_name({"model": {"num_concepts": 32},
    ...                  "training": {"lambda_compact": 0.1,
    ...                               "lambda_diverse": 0.05, "seed": 42}})
    'hmacd_k32_lc0.1_ld0.05_seed42'
    >>> experiment_name({"model": {"num_concepts": 32},
    ...                  "training": {"lambda_compact": 0.1, "lambda_diverse": 0.05,
    ...                               "lambda_decor": 0.02, "seed": 42}})
    'hmacd_k32_lc0.1_ld0.05_ldec0.02_seed42'
    """
    model = config.get("model", {})
    train = config.get("training", {})
    name = (
        f"hmacd_k{int(model.get('num_concepts', 32))}"
        f"_lc{_fmt_weight(train.get('lambda_compact', 0.1))}"
        f"_ld{_fmt_weight(train.get('lambda_diverse', 0.05))}"
    )
    # `_ldec` is appended ONLY when the decorrelation weight is nonzero, so
    # existing zero-decor configs keep their old names (checkpoint rename in
    # the project status notes depends on this).
    if float(train.get("lambda_decor", 0.0)) > 0:
        name += f"_ldec{_fmt_weight(train.get('lambda_decor'))}"
    # evidence-span arm: `_sc{n}` = scattered n positions (arm 1 control),
    # `_sg{m}` = m contiguous segments (arm 2). Arm 0 (global) adds nothing,
    # so historical checkpoints keep their names.
    arm = model.get("arm", "global")
    if arm == "scatter":
        name += f"_sc{int(model.get('arm_positions', 8))}"
    elif arm == "segment":
        name += f"_sg{int(model.get('segment_m', 4))}"
    # concept-bank attachment layer: `_l2`/`_l3`; layer4 (default) adds
    # nothing so historical checkpoints keep their names.
    attach = model.get("attach_layer", "layer4")
    if attach != "layer4":
        name += f"_{attach.replace('layer', 'l')}"
    name += f"_seed{int(train.get('seed', 42))}"
    return name
