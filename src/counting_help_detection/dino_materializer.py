"""Memory-bounded construction and strict source mapping for official DINOv3 detection.

This adapter invokes the unmodified DINOv3 architecture by Meta Platforms, Inc.
and affiliates. Upstream source and its DINOv3 license remain separate:
https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751
https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE
The tensor mapping and allocation code in this module is an independent adapter.
"""
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bindings(model, kind):
    values = (model.named_parameters(remove_duplicate=False) if kind == "parameter"
              else model.named_buffers(remove_duplicate=False))
    groups = defaultdict(list)
    objects = {}
    for name, value in values:
        groups[id(value)].append(name)
        objects[id(value)] = value
    return [(sorted(names), objects[key]) for key, names in groups.items()]


def _owner(model, name):
    parent, _, leaf = name.rpartition(".")
    return model.get_submodule(parent), leaf


def _bind(model, names, value, kind):
    attribute = "_parameters" if kind == "parameter" else "_buffers"
    for name in names:
        owner, leaf = _owner(model, name)
        getattr(owner, attribute)[leaf] = value


def prepare_meta(source):
    """Preserve native architecture and aliases; predeclare mixed storage only."""
    import torch
    from torch import nn
    sys.path.insert(0, str(Path(source).resolve()))
    from dinov3.hub.detectors import dinov3_vit7b16_de
    with torch.device("meta"):
        model = dinov3_vit7b16_de(pretrained=False)
    groups = _bindings(model, "parameter")
    aliases_before = sorted(names for names, _ in groups if len(names) > 1)
    legacy = [dict(names=names, shape=list(value.shape),
                   bytes=value.numel()*value.element_size())
              for names, value in groups if not value.is_meta]
    assert sum(row["bytes"] for row in legacy) <= 3072
    assert all(row["names"] == ["detector.transformer.level_embed"] for row in legacy)
    eligible = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    policy = {}
    for names, old in groups:
        owners = [_owner(model, name)[0] for name in names]
        cast = old.is_floating_point() and all(isinstance(owner, eligible) for owner in owners)
        dtype = torch.bfloat16 if cast else old.dtype
        new = nn.Parameter(torch.empty(old.shape, device="meta", dtype=dtype), requires_grad=False)
        _bind(model, names, new, "parameter")
        for name in names:
            policy[name] = dict(target_dtype=str(dtype), cast_linear_or_conv_storage=cast,
                                owner_class=type(_owner(model, name)[0]).__name__)
    assert sorted(names for names, _ in _bindings(model, "parameter") if len(names) > 1) == aliases_before
    assert all(value.is_meta for value in model.state_dict().values())
    assert model.detector.num_queries == model.postprocessor.topk == 1500
    model.eval()
    return model, dict(parameter_policy=policy, normalized_native_cpu_factories=legacy,
                       alias_groups=aliases_before,
                       precision="Linear/Conv weights and biases use BF16 storage and CUDA BF16 autocast. "
                                 "All other parameters and buffers retain native dtype; preserve FP32 "
                                 "LayerNorm, LayerScale gamma, ROPE frequencies, tokens and embeddings.")


def strict_plan(model, backbone_metadata, head_state):
    """Map every unique parameter/buffer and every checkpoint key before allocation."""
    backbone = {row["key"]: row for row in backbone_metadata["tensors"]}
    head = {key: dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix("torch."),
                      storage_key=str(value.untyped_storage()._cdata),
                      storage_offset=value.storage_offset(), stride=list(value.stride()))
            for key, value in head_state.items()}
    patch_names = [name for name in model.state_dict()
                   if name.endswith("patch_embed.proj.weight")]
    assert len(patch_names) == 1
    backbone_prefix = patch_names[0][:-len("patch_embed.proj.weight")]
    covered = {"backbone": set(), "head": set()}
    plan = []
    target_names = set()
    for kind in ("parameter", "buffer"):
        for names, target in _bindings(model, kind):
            sources = {}
            for name in names:
                if name.startswith(backbone_prefix):
                    candidate = name[len(backbone_prefix):]
                    if candidate in backbone:
                        sources["backbone", candidate] = backbone[candidate]
                if name.startswith("detector."):
                    candidate = name[len("detector."):]
                    if candidate in head:
                        sources["head", candidate] = head[candidate]
            if not sources:
                raise ValueError("Unmapped native target: " + str(names))
            for (namespace, key), descriptor in sources.items():
                if descriptor["shape"] != list(target.shape):
                    raise ValueError("Checkpoint shape differs: " + namespace + ":" + key)
                covered[namespace].add(key)
            if len(sources) > 1:
                views = {(namespace, row["storage_key"], row["storage_offset"],
                          tuple(row["shape"]), tuple(row["stride"]), row["dtype"])
                         for (namespace, _), row in sources.items()}
                assert len(views) == 1, "Aliased targets have inconsistent source tensor views"
            namespace, key = sorted(sources)[0]
            plan.append(dict(kind=kind, target_names=names, source_namespace=namespace,
                             source_key=key, shape=list(target.shape),
                             target_dtype=str(target.dtype).removeprefix("torch."),
                             source_dtype=sources[namespace, key]["dtype"],
                             numel=target.numel(), target_bytes=target.numel()*target.element_size()))
            target_names.update(names)
    assert not (set(backbone)-covered["backbone"]), "Unused backbone checkpoint keys"
    assert not (set(head)-covered["head"]), "Unused head checkpoint keys"
    assert set(model.state_dict()) <= target_names, "Unmapped persistent native state"
    return plan, dict(backbone_keys_used=len(covered["backbone"]), head_keys_used=len(covered["head"]),
                      target_unique_objects=len(plan), persistent_target_state_keys=len(model.state_dict()),
                      target_storage_bytes=sum(row["target_bytes"] for row in plan),
                      missing_target_keys=0, unused_checkpoint_keys=0,
                      backbone_parameter_prefix=backbone_prefix)


def inspect(source, backbone_metadata, head_path):
    """Metadata-only CPU admission. No learned tensor data or GPU allocations."""
    import torch
    assert not torch.cuda.is_initialized()
    model, policy = prepare_meta(source)
    saved = torch.load(head_path, map_location="meta", mmap=True, weights_only=True)
    plan, coverage = strict_plan(model, backbone_metadata, saved["model"])
    record = dict(status="OFFICIAL_META_ARCHITECTURE_AND_ALL_CHECKPOINT_SHAPES_PASS",
                  model_parameter_numel=sum(value.numel() for value in model.parameters()),
                  max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  GPU_initialized=torch.cuda.is_initialized(), learned_tensor_values_read=False,
                  model_forward_executed=False, **coverage, **policy)
    assert record["max_RSS_bytes"] < 1024**3
    return record, plan


def materialize(source, backbone_path, head_path, backbone_metadata, device="cuda:0",
                gpu_limit_GiB=22):
    """Load admitted mmap states one unique object at a time; caller profiles forward."""
    import torch
    expected = [
        (backbone_path, 26864407683, "a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7"),
        (head_path, 1549231874, "b0235ff7ea0a037b521f58eb7915d3a022c75319e93307dd48dfc5fd85c6f031"),
    ]
    signatures = {}
    for path, size, digest in expected:
        path = Path(path)
        stat = path.stat()
        assert stat.st_size == size and sha256_file(path) == digest
        signatures[str(path)] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    target_device = torch.device(device)
    total = torch.cuda.get_device_properties(target_device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., gpu_limit_GiB*1024**3/total), target_device)
    model, policy = prepare_meta(source)
    states = {"backbone": torch.load(backbone_path, map_location="cpu", mmap=True, weights_only=True)}
    head_checkpoint = torch.load(head_path, map_location="cpu", mmap=True, weights_only=True)
    states["head"] = head_checkpoint["model"]
    del head_checkpoint
    plan, coverage = strict_plan(model, backbone_metadata, states["head"])
    with torch.inference_mode():
        for row in plan:
            original = states[row["source_namespace"]][row["source_key"]]
            assert list(original.shape) == row["shape"]
            dtype = getattr(torch, row["target_dtype"])
            real = original.to(device=target_device, dtype=dtype, non_blocking=False)
            if row["kind"] == "parameter":
                real = torch.nn.Parameter(real, requires_grad=False)
            _bind(model, row["target_names"], real, row["kind"])
        torch.cuda.synchronize(target_device)
    del original, real, states
    gc.collect()
    assert all(not value.is_meta for value in model.state_dict().values())
    assert all(value.device == target_device for value in model.state_dict().values())
    for path, signature in signatures.items():
        stat = Path(path).stat()
        assert signature == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    model.eval()
    return model, dict(status="MMAP_PARAMETER_BY_PARAMETER_LOAD_COMPLETE",
                       **coverage, precision=policy["precision"],
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(target_device),
                       peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(target_device),
                       model_forward_executed=False)


def backbone_metadata_from_checkpoint(path):
    """Read only tensor headers using mmap/meta and canonicalize shared-storage IDs."""
    import torch
    state = torch.load(path, map_location="meta", mmap=True, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("The approved backbone must be a flat tensor state dictionary.")
    storage_ids = {}
    tensors = []
    for key, value in sorted(state.items()):
        identity = value.untyped_storage()._cdata
        if identity not in storage_ids:
            storage_ids[identity] = "storage_" + str(len(storage_ids))
        tensors.append(dict(key=key, shape=list(value.shape),
                            dtype=str(value.dtype).removeprefix("torch."),
                            storage_key=storage_ids[identity], storage_offset=value.storage_offset(),
                            stride=list(value.stride())))
    del state
    return dict(format="mmap_meta_tensor_headers", learned_tensor_values_read=False, tensors=tensors)
