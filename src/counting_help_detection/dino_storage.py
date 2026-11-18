# Copyright (c) Meta Platforms, Inc. and affiliates.
# Native WindowsWrapper and backbone operations below derive from the pinned DINOv3 source.
# This software may be used and distributed in accordance with the DINOv3 License Agreement.
# Backbone lineage additionally credits Plain-DETR (MIT), Deformable DETR (Apache2.0), and DETR.
"""Dormant, reversible BF16 destination-storage adapter. Activate only after parity admission."""
from collections import defaultdict
from hashlib import file_digest, sha256
import inspect
import json
from pathlib import Path
from types import MethodType

SOURCE_COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'
SOURCE_ROOT_URL = 'https://github.com/facebookresearch/dinov3/blob/' + SOURCE_COMMIT + '/dinov3/eval/detection/models/'
SOURCE_FILES = {
    'windows.py': 'fc1a62a3bafdab4345b6c3232a978d03aef0e72ea1101c7da52ef6eca231e30d',
    'backbone.py': 'e739d35f02af46fed6afd8f824e47060bd4d09b73228460bf0063ad8bb974edc',
    'position_encoding.py': 'f885b7bfd40df50cb012c8038979042ff2f5c427a5f6bbe36028e84735132d4a',
}
SOURCE_FUNCTIONS = {
    'WindowsWrapper.forward': '00becf6c6e6fb70e01aa78078156b169630899071e5bf2fd95b97c600709a045',
    'BackboneWithPositionEncoding.forward': '8190f8bb24316414b1a0061b31e1e7d2338936de8a29100c435a1f67eb32fafa',
    'PositionEmbeddingSine.forward': '979dcbdbfb29cdbebe6a7a7c60068d66b1029763520f8cb5b1f2caf36f9bab19',
}


def _file_sha(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def _require_native_autocast():
    import torch
    if torch.is_grad_enabled() or not torch.is_autocast_enabled('cuda'):
        raise RuntimeError('This storage adapter requires inference without gradients and CUDA BF16 autocast.')
    if torch.get_autocast_dtype('cuda') != torch.bfloat16:
        raise RuntimeError('The admitted destination storage is BF16, never a new inference precision.')


def _windows_forward(self, tensor_list):
    """Copy native crop/backbone/resize operations; quantize only cat destinations."""
    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import v2
    from dinov3.eval.detection.util.misc import NestedTensor
    _require_native_autocast()
    tensors = tensor_list.tensors
    original_h, original_w = tensors.shape[2], tensors.shape[3]
    window_h = math.ceil((original_h // self._n_windows_h) / self._patch_size) * self._patch_size
    window_w = math.ceil((original_w // self._n_windows_w) / self._patch_size) * self._patch_size
    all_h = [window_h] * (self._n_windows_h - 1) + [original_h - window_h * (self._n_windows_h - 1)]
    all_w = [window_w] * (self._n_windows_w - 1) + [original_w - window_w * (self._n_windows_w - 1)]
    all_h_cumsum = [0] + list(np.cumsum(all_h))
    all_w_cumsum = [0] + list(np.cumsum(all_w))
    tiles = [[None for _ in range(self._n_windows_w)] for _ in range(self._n_windows_h)]
    for ih in range(self._n_windows_h):
        for iw in range(self._n_windows_w):
            window_tensor = v2.functional.crop(tensors, top=all_h_cumsum[ih], left=all_w_cumsum[iw],
                                               height=all_h[ih], width=all_w[iw])
            window_mask = v2.functional.crop(tensor_list.mask, top=all_h_cumsum[ih], left=all_w_cumsum[iw],
                                             height=all_h[ih], width=all_w[iw])
            feature = self._backbone(NestedTensor(tensors=window_tensor, mask=window_mask))[0]
            if feature.tensors.dtype != torch.float32:
                raise RuntimeError('The native local feature must remain FP32 through the backbone.')
            tiles[ih][iw] = feature.tensors.to(dtype=torch.bfloat16)
            del feature, window_tensor, window_mask
    rows = [torch.cat(tiles[ih], dim=-1) for ih in range(len(tiles))]
    window_tensors = torch.cat(rows, dim=-2)
    for row in tiles:
        row.clear()
    tiles.clear()
    rows.clear()
    del tiles, rows

    resized_global_tensor = v2.functional.resize(tensors, size=(window_h, window_w))
    global_features = self._backbone(NestedTensor(tensors=resized_global_tensor, mask=tensor_list.mask))
    if global_features[0].tensors.dtype != torch.float32:
        raise RuntimeError('The native global backbone feature must retain FP32.')
    resized_global_fp32 = v2.functional.resize(global_features[0].tensors, size=window_tensors.shape[-2:])
    if resized_global_fp32.dtype != torch.float32:
        raise RuntimeError('Global interpolation must finish in native FP32 before destination casting.')
    resized_global_bf16 = resized_global_fp32.to(dtype=torch.bfloat16)
    del resized_global_fp32, global_features, resized_global_tensor
    concat_tensors = torch.cat([resized_global_bf16, window_tensors], dim=1)
    del resized_global_bf16, window_tensors
    global_mask = F.interpolate(tensor_list.mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
    return [NestedTensor(tensors=concat_tensors, mask=global_mask)]


def _joiner_forward(self, tensor_list):
    """Keep the original sine path FP32 after feature destination storage becomes BF16."""
    import torch
    _require_native_autocast()
    out = list(self[0](tensor_list))
    positions = []
    for index, feature in enumerate(out):
        position = self[1][index](feature)
        if position.dtype != torch.float32:
            raise RuntimeError('Native sine positional encodings must remain FP32.')
        positions.append(position)
    return out, positions


def _state_identity(model):
    rows = []
    for name, value in model.state_dict(keep_vars=True).items():
        if value.is_meta:
            raise RuntimeError('Install only after the strict learned-state materializer has finished.')
        try:
            version = value._version
        except RuntimeError as error:
            if 'Inference tensors do not track version counter' not in str(error):
                raise
            version = 'native_inference_tensor_without_version_counter'
        rows.append((name, tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device),
                     value.storage_offset(), value.data_ptr(), value.untyped_storage()._cdata, version))
    aliases = defaultdict(list)
    for name, value in model.named_parameters(remove_duplicate=False):
        aliases[id(value)].append(name)
    alias_groups = sorted(sorted(names) for names in aliases.values() if len(names) > 1)
    return tuple(rows), alias_groups


class Installation:
    def __init__(self, model, windows, joiner, before, aliases, receipt):
        self.model, self.windows, self.joiner = model, windows, joiner
        self.before, self.aliases = before, aliases
        self.saved = [(module, 'forward' in module.__dict__, module.__dict__.get('forward'))
                      for module in (windows, joiner)]
        self.receipt = receipt
        windows.forward = MethodType(_windows_forward, windows)
        joiner.forward = MethodType(_joiner_forward, joiner)
        after, after_aliases = _state_identity(model)
        if before != after or aliases != after_aliases:
            self.uninstall()
            raise RuntimeError('Installing a destination-storage method must not modify any state or aliases.')
        self.receipt['state_identity_unchanged_after_install'] = True

    def uninstall(self):
        if not self.receipt['active']:
            return
        if (getattr(self.windows.forward, '__func__', None) is not _windows_forward
                or getattr(self.joiner.forward, '__func__', None) is not _joiner_forward):
            raise RuntimeError('Another caller changed the active methods; do not overwrite its changes.')
        for module, had_instance_forward, original in self.saved:
            if had_instance_forward:
                module.forward = original
            else:
                del module.__dict__['forward']
        after, aliases = _state_identity(self.model)
        if after != self.before or aliases != self.aliases:
            raise RuntimeError('The learned-state storage identity changed while this adapter was installed.')
        self.receipt['state_identity_unchanged_after_uninstall'] = True
        self.receipt['active'] = False


def install(model):
    """Return a reversible handle. Caller must first admit actual BF16/input/full-prediction parity."""
    import torch
    from dinov3.eval.detection.models.windows import WindowsWrapper
    from dinov3.eval.detection.models.backbone import BackboneWithPositionEncoding
    from dinov3.eval.detection.models.position_encoding import PositionEmbeddingSine
    classes = [WindowsWrapper, BackboneWithPositionEncoding, PositionEmbeddingSine]
    functions = [WindowsWrapper.forward, BackboneWithPositionEncoding.forward, PositionEmbeddingSine.forward]
    verified = {}
    for cls, function in zip(classes, functions):
        path = Path(inspect.getsourcefile(cls)).resolve()
        key = cls.__name__ + '.forward'
        function_sha = sha256(inspect.getsource(function).encode()).hexdigest()
        if _file_sha(path) != SOURCE_FILES[path.name] or function_sha != SOURCE_FUNCTIONS[key]:
            raise ValueError('The native source or unadapted forward differs from the pinned revision.')
        verified[path.name] = dict(file_sha256=SOURCE_FILES[path.name], function_sha256=function_sha,
                                   source_url=SOURCE_ROOT_URL + path.name)
    windows = [(name, module) for name, module in model.named_modules() if isinstance(module, WindowsWrapper)]
    joiners = [(name, module) for name, module in model.named_modules() if isinstance(module, BackboneWithPositionEncoding)]
    if len(windows) != 1 or len(joiners) != 1 or joiners[0][1][0] is not windows[0][1]:
        raise ValueError('Expected exactly the native WindowsWrapper and its position-encoding joiner.')
    if not all(isinstance(position, PositionEmbeddingSine) for position in joiners[0][1][1]):
        raise ValueError('This admitted position rule is only for the unchanged native FP32 sine modules.')
    if any(parameter.dtype != torch.float32 for position in joiners[0][1][1] for parameter in position.parameters()):
        raise ValueError('The positional module parameter dtype changed.')
    if not hasattr(model, 'detector') or not all(layer[0].weight.dtype == torch.bfloat16 for layer in model.detector.input_proj):
        raise ValueError('This storage rule requires the already-admitted BF16 input projections.')
    before, aliases = _state_identity(model)
    receipt = dict(status='INSTALLED_FOR_CONDITIONAL_PARITY_TEST_ONLY', active=True,
        source_commit=SOURCE_COMMIT, source_functions=verified, helper_sha256=_file_sha(__file__),
        windows_module=windows[0][0], joiner_module=joiners[0][0],
        policy='FP32 native backbone and global interpolation unchanged. Store completed local tiles/global-resize results BF16 only for concatenation destinations; downstream BF16 input-projection autocast is unchanged.',
        position_policy='Use the same native sine module/mask/device/shape and preserve its FP32 output; do not cast positions to the BF16 destination-feature dtype.',
        state_entries=len(before), alias_groups=aliases,
        state_signature_sha256=sha256(repr(before).encode()).hexdigest(),
        signature_scope='Keys/shapes/strides/dtypes/devices/storage pointers/version counters; no learned values copied or modified.',
        public_claim=False, AP_selected=False, parity_admission_required_before_production=True)
    return Installation(model, windows[0][1], joiners[0][1], before, aliases, receipt)
