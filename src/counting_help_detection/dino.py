"""Native DINOv3 detection exporter with strict low-memory loading and independent prediction seals."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from hashlib import file_digest, sha256
import fcntl
import gc
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback
from uuid import uuid4

SOURCE_COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'
WEIGHTS = {
    'backbone': dict(bytes=26864407683, sha256='a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7'),
    'head': dict(bytes=1549231874, sha256='b0235ff7ea0a037b521f58eb7915d3a022c75319e93307dd48dfc5fd85c6f031'),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def rows_from_manifest(path):
    value = read_json(path)
    if isinstance(value, dict):
        if 'annotations' in value:
            raise ValueError('Inference requires an image-only manifest.')
        value = value['images']
    ids = []
    for row in value:
        if any(key in row for key in ('annotations', 'boxes', 'labels', 'ground_truth')):
            raise ValueError('The inference manifest contains annotation fields.')
        identifier = row.get('image_id', row.get('id'))
        if type(identifier) is not int or identifier < 0:
            raise ValueError('Image IDs must be nonnegative integers.')
        row['image_id'] = identifier
        ids.append(identifier)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Image IDs are empty or duplicated.')
    return value


def image_path(row, root):
    if root is None and row.get('image_path'):
        return Path(row['image_path']).resolve()
    relative = PurePosixPath(row['file_name'])
    if root is None or relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Supply a safe image root or the frozen legacy image_path.')
    return (root / relative).resolve()


def resized_shape(height, width, edge):
    denominator = min(height, width) * 16
    return tuple(((value * edge + denominator - 1) // denominator) * 16
                 for value in (height, width))


def source_snapshot(root):
    mapping = {}
    for path in sorted(root.rglob('*')):
        if (path.is_file() and path.suffix in {'.py', '.yaml', '.yml'}
                and not any(part in {'.git', '__pycache__'} for part in path.parts)):
            mapping[str(path.relative_to(root))] = digest(path)
    git_revision = None
    if (root / '.git').exists():
        process = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                 check=True, capture_output=True, text=True)
        git_revision = process.stdout.strip()
        if git_revision != SOURCE_COMMIT:
            raise ValueError('The DINO source checkout differs from the declared revision.')
    return dict(git_revision=git_revision, files=mapping,
                fingerprint_sha256=sha256(json.dumps(mapping, sort_keys=True,
                                                     separators=(',', ':')).encode()).hexdigest())


def signature(path):
    state = Path(path).stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


def gpu_admission(required_gib=22., device='cuda:0'):
    if not math.isfinite(required_gib) or required_gib <= 0:
        raise ValueError('The requested GPU memory budget must be positive and finite.')
    import torch
    selected = torch.device(device)
    if selected.type != 'cuda':
        raise ValueError('Native DINO inference requires a CUDA device.')
    if selected.index is None:
        selected = torch.device('cuda', torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(selected)
    if free < math.ceil(required_gib * 1024**3):
        raise RuntimeError('The requested GPU memory budget exceeds the currently available selected-device memory.')
    torch.cuda.set_device(selected)
    apps, diagnostic_error = None, None
    try:
        apps = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True,
            timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        diagnostic_error = type(error).__name__ + ': ' + str(error)
    return dict(utc=datetime.now(timezone.utc).isoformat(), device=str(selected),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                free_bytes=free, total_bytes=total, free_mib=free // 1024**2,
                used_mib=(total - free) // 1024**2, existing_compute_processes=apps,
                optional_all_gpu_process_diagnostics_error=diagnostic_error)


def canonical_rows(image_id, labels, boxes, scores, categories):
    if not len(labels) == len(boxes) == len(scores) == 1500:
        raise ValueError('The native postprocessor must return1500 candidates before eligibility filtering.')
    predictions = []
    native_predictions = []
    ignored_predictions = []
    excluded = Counter()
    nonpositive = 0
    for index, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        if type(label) is not int or not 0 <= label < 91:
            raise ValueError('The native91-slot head returned an unexpected class ID.')
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError('The native model returned a nonfinite coordinate.')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('The native model returned an invalid score.')
        native_row = dict(image_id=image_id, category_id=label, candidate_index=index,
                          bbox_xyxy=box, score=score)
        native_predictions.append(native_row)
        if label not in categories:
            excluded[label] += 1
            ignored_predictions.append(native_row)
            continue
        # Native PostProcess already clamps/rescales. Keep its original-pixel result unchanged.
        nonpositive += int(box[2] <= box[0] or box[3] <= box[1])
        predictions.append(native_row)
    return predictions, dict(excluded), nonpositive, native_predictions, ignored_predictions


def seal_stream(root, rows, chunks, protocol_sha, manifest_sha, scope, extra):
    root.mkdir(parents=True, exist_ok=True)
    exports = {}
    for filename, field in [('predictions.jsonl', 'predictions'),
                            ('native_predictions.jsonl', 'native_predictions'),
                            ('ignored_native_slots.jsonl', 'ignored_native_slots')]:
        target = root / filename
        pending = target.with_name('.' + target.name + '.' + uuid4().hex)
        exports[field] = dict(target=target, pending=pending,
                              stream=None if target.exists() else pending.open('xb'),
                              digest=sha256(), rows=0)
    count = 0
    invalid = 0
    exclusions = Counter()
    receipts = []
    try:
        for image in rows:
            path = chunks / f"{image['image_id']:012d}.json"
            data = read_json(path)
            if data['image_id'] != image['image_id'] or data['protocol_sha256'] != protocol_sha:
                raise ValueError('A chunk belongs to a different image or frozen protocol.')
            receipts.append(dict(image_id=image['image_id'], sha256=digest(path)))
            invalid += data['nonpositive_native_boxes_retained']
            exclusions.update({int(key): value for key, value in data['excluded_unused_slots'].items()})
            if (len(data['native_predictions']) != 1500
                    or len(data['predictions']) + len(data['ignored_native_slots']) != 1500):
                raise ValueError('Native1500 archive and eligible/ignored partition differ.')
            for field, export in exports.items():
                for prediction in data[field]:
                    encoded = (json.dumps(prediction, separators=(',', ':'), allow_nan=False) + '\n').encode()
                    export['digest'].update(encoded)
                    if export['stream'] is not None:
                        export['stream'].write(encoded)
                    export['rows'] += 1
            count += len(data['predictions'])
        for export in exports.values():
            if export['stream'] is not None:
                export['stream'].flush()
                os.fsync(export['stream'].fileno())
    finally:
        for export in exports.values():
            if export['stream'] is not None:
                export['stream'].close()
    for export in exports.values():
        if export['target'].exists():
            if digest(export['target']) != export['digest'].hexdigest():
                raise ValueError('Existing flat predictions differ from the frozen chunks.')
        else:
            export['pending'].rename(export['target'])
    if (exports['native_predictions']['rows'] != len(rows) * 1500
            or count + exports['ignored_native_slots']['rows'] != len(rows) * 1500):
        raise ValueError('The complete native stream lost a candidate during eligibility partitioning.')
    result = dict(status='PREDICTIONS_SEALED_BEFORE_GT', ground_truth_read=False,
                  scope=scope, images=len(rows), native_candidates_per_image=1500,
                  native_candidate_rows=len(rows) * 1500, candidate_rows=count,
                  excluded_unused_slots={str(key): value for key, value in sorted(exclusions.items())},
                  nonpositive_native_boxes_retained=invalid,
                  protocol_sha256=protocol_sha, image_manifest_sha256=manifest_sha,
                  prediction_sha256=exports['predictions']['digest'].hexdigest(),
                  native_prediction_sha256=exports['native_predictions']['digest'].hexdigest(),
                  ignored_native_slots_sha256=exports['ignored_native_slots']['digest'].hexdigest(),
                  ignored_native_rows=exports['ignored_native_slots']['rows'],
                  chunks=receipts, **extra)
    seal = root / 'GT_FREE_COMPLETE.json'
    if seal.exists():
        if read_json(seal) != result:
            raise ValueError('The saved prediction seal differs from the frozen chunks.')
    else:
        atomic_json(seal, result)
    return {key: value for key, value in result.items() if key != 'chunks'}


def verify_storage_admission(args):
    admission = read_json(args.storage_admission)
    if admission.get('status') not in {'NATIVE_SINGLE_SCALE_STORAGE_ADMISSION_PASS', 'NATIVE2048_STORAGE_ADMISSION_PASS'}:
        raise ValueError('Feature storage requires an actual all-geometry admission receipt.')
    if (admission['short_edge'] != args.short_edge or args.short_edge not in {1536, 2048}
            or admission['source_revision'] != SOURCE_COMMIT
            or admission['gpu_limit_gib'] != args.gpu_limit_gib
            or admission['backbone_sha256'] != WEIGHTS['backbone']['sha256']
            or admission['head_sha256'] != WEIGHTS['head']['sha256']
            or admission.get('labels_read') is not False or admission.get('AP_evaluated') is not False):
        raise ValueError('The admitted geometry, source, weights, memory cap or GT-free scope differs.')
    if digest(args.storage_helper) != admission['helper_sha256']:
        raise ValueError('The admitted feature-storage helper changed.')
    complete_path = Path(admission['profile_complete_path'])
    parity_path = Path(admission['parity_pass_path'])
    if digest(complete_path) != admission['profile_complete_sha256'] or digest(parity_path) != admission['parity_pass_sha256']:
        raise ValueError('An actual profile/parity proof changed.')
    complete = read_json(complete_path)
    parity = read_json(parity_path)
    expected_ids = [139, 42102, 44590, 298251]
    if (complete.get('status') != f'PARITY_VERIFIED_NATIVE{args.short_edge}_STORAGE_PROFILE_COMPLETE'
            or [row['image_id'] for row in complete['images']] != expected_ids
            or admission['image_ids'] != expected_ids
            or not all(row['native_candidates'] == 1500 and row['status'] == f'GT_FREE_NATIVE{args.short_edge}_FORWARD_PASS' for row in complete['images'])
            or complete.get('labels_read') is not False or complete.get('AP_evaluated') is not False):
        raise ValueError('All four original geometry probes must actually complete with native1500.')
    if (parity.get('status') != 'EXACT_ALL1500_NATIVE_PREDICTION_AND_BF16_CONSUMER_INPUT_PARITY_PASS'
            or parity['native1500_sha256'] != admission['native_reference_sha256']
            or parity['native_input']['sha256'] != parity['storage_input']['sha256']
            or parity['native_input']['sha256'] != admission['projection_bf16_sha256']
            or parity['native_input']['logical_shape'] != parity['storage_input']['logical_shape']
            or parity.get('labels_read') is not False or parity.get('AP_evaluated') is not False):
        raise ValueError('The exact native1500 and BF16 consumer-input parity proof is absent.')
    if 'PYTORCH_CUDA_ALLOC_CONF' not in admission['allocator_env']:
        raise ValueError('The admitted allocator setting must be explicit, including an unset value if native.')
    for key, value in admission['allocator_env'].items():
        if os.environ.get(key) != value:
            raise ValueError('The allocator environment differs from its admitted resource profile.')
    profile_protocol_path = complete_path.with_name('PROFILE_PROTOCOL.json')
    install_receipt_path = complete_path.with_name('STORAGE_REMEDY_INSTALL_RECEIPT.json')
    profile_protocol = read_json(profile_protocol_path)
    installed = read_json(install_receipt_path)
    if (profile_protocol['resolution'] != args.short_edge
            or profile_protocol['gpu_limit_GiB'] != args.gpu_limit_gib
            or installed['helper_sha256'] != admission['helper_sha256']
            or installed.get('state_identity_unchanged_after_install') is not True):
        raise ValueError('The admitted method, state-preservation or resource profile differs.')
    locks = {str(path): digest(path) for path in (args.storage_helper, args.storage_admission,
             complete_path, parity_path, profile_protocol_path, install_receipt_path)}
    return admission, locks


def verify_published_storage_policy(args, configuration):
    """Record the verified implementation and requested resource scope without private proofs."""
    policy = configuration['feature_storage']
    if policy['validation_status'] != 'ADMITTED_NATIVE_PARITY_GEOMETRY_AND_PUBLIC_ENTRY':
        raise ValueError('The published feature-storage implementation has not passed its recorded validation.')
    if digest(args.storage_helper) != policy['helper_sha256']:
        raise ValueError('The feature-storage implementation differs from the published verified version.')
    if args.short_edge not in {1536, 2048} or not math.isfinite(args.gpu_limit_gib) or args.gpu_limit_gib <= 0:
        raise ValueError('Choose an official single-scale setting and a positive memory budget.')
    receipt = dict(
        status='PUBLISHED_FEATURE_STORAGE_IMPLEMENTATION_VERIFIED',
        helper_sha256=policy['helper_sha256'],
        verified_short_edge=policy['verified_short_edge'], requested_short_edge=args.short_edge,
        requested_setting_verified_here=args.short_edge == policy['verified_short_edge'],
        setting_scope=('VERIFIED_NATIVE1536_PUBLIC_ENTRY' if args.short_edge == policy['verified_short_edge']
                       else 'SEPARATE_OFFICIAL2048_SETTING_NOT_BENCHMARKED_HERE'),
        reference_native1500_sha256=policy['native1536_reference_sha256'],
        source_admission_sha256=policy['source_admission_sha256'],
        verified_gpu_limit_gib=policy['verified_gpu_limit_gib'], requested_gpu_limit_gib=args.gpu_limit_gib,
        verified_allocator_environment=policy['verified_allocator_environment'],
        allocator_env={'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF')},
        model_math_changed=False, preprocessing_matches_verified_setting=args.short_edge == policy['verified_short_edge'],
        labels_read=False, AP_evaluated=False)
    atomic_json(args.out / ('PUBLISHED_STORAGE_POLICY_' + uuid4().hex + '.json'), receipt)
    return receipt, {str(args.storage_helper): digest(args.storage_helper)}


def prepare_scope(args, configuration):
    images = rows_from_manifest(args.images)
    manifest_sha = digest(args.images)
    categories = {row['category_id'] for row in read_json(args.vocabulary)}
    if sorted(categories) != configuration['valid_category_ids']:
        raise ValueError('Use the original80 sparse COCO IDs, without a contiguous-label remap.')
    preprocessing = dict(short_edge=args.short_edge, round_height_and_width_up_to=16,
        preserve_aspect_ratio_until_patch_rounding=True, long_edge_cap=None,
        normalization_mean=[.485,.456,.406], normalization_std=[.229,.224,.225],
        batch_size=1, single_scale=True, test_time_augmentation=False)
    if args.protocol is not None:
        declaration = read_json(args.protocol)
        if (declaration['source_commit'] != SOURCE_COMMIT
                or declaration['input_preprocessing'] != preprocessing
                or declaration['image_manifest_sha256'] != manifest_sha
                or declaration['image_count'] != len(images)
                or declaration['category_ids'] != sorted(categories)):
            raise ValueError('The source, scope or preprocessing differs from the supplied frozen declaration.')
    else:
        declaration = dict(source_commit=SOURCE_COMMIT, input_preprocessing=preprocessing)
    pilot_declaration_sha = None
    if args.pilot_protocol is not None:
        pilot_declaration = read_json(args.pilot_protocol)
        path = args.pilot_images or Path(pilot_declaration['image_manifest'])
        pilot_images = rows_from_manifest(path)
        pilot_manifest_sha = digest(path)
        if pilot_manifest_sha != pilot_declaration['image_manifest_sha256']:
            raise ValueError('The supplied frozen prefix declaration does not match its image manifest.')
        pilot_declaration_sha = digest(args.pilot_protocol)
    elif args.pilot_images is not None:
        path = args.pilot_images
        pilot_images = rows_from_manifest(path)
        pilot_manifest_sha = digest(path)
    elif args.pilot_count:
        if not 0 < args.pilot_count <= len(images):
            raise ValueError('The explicit prefix count must fit the input image manifest.')
        pilot_images = images[:args.pilot_count]
        path = args.out / 'pilot_images_gt_free.json'
        if not path.exists():
            atomic_json(path, pilot_images)
        elif read_json(path) != pilot_images:
            raise ValueError('The frozen prefix manifest changed.')
        pilot_manifest_sha = digest(path)
    else:
        pilot_images, pilot_manifest_sha = [], None
    if len(pilot_images) > len(images) or [row['image_id'] for row in images[:len(pilot_images)]] != [row['image_id'] for row in pilot_images]:
        raise ValueError('A prefix must retain the exact supplied input order; no new image selection.')
    for first, pilot in zip(images, pilot_images):
        if any(first[key] != pilot[key] for key in ('image_id','width','height')):
            raise ValueError('The frozen prefix image metadata differs from its parent manifest.')
        if first.get('image_sha256') and pilot.get('image_sha256') and first['image_sha256'] != pilot['image_sha256']:
            raise ValueError('The prefix image bytes differ from the parent input.')
    args._pilot_manifest_path = path if pilot_images else None
    return declaration, images, manifest_sha, categories, pilot_images, pilot_manifest_sha, pilot_declaration_sha


def run(args):
    if args.out.exists() and not args.resume:
        raise FileExistsError('Use a new output or --resume with the same frozen protocol.')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / 'CONTROLLER_PID.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()) + '\n')
    lock.flush()
    os.nice(10)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    configuration = read_json(args.model_config)
    args.storage_helper = Path(__file__).with_name('dino_storage.py')
    if args.feature_storage == 'bf16-concat' and args.storage_admission is not None:
        storage_admission, storage_locks = verify_storage_admission(args)
    elif args.feature_storage == 'bf16-concat':
        storage_admission, storage_locks = verify_published_storage_policy(args, configuration)
    else:
        storage_admission = None
        storage_locks = {}
    import torch
    from PIL import Image
    from torchvision.transforms import functional as tvf, InterpolationMode
    from .dino_materializer import materialize, backbone_metadata_from_checkpoint

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261118)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    declaration, images, manifest_sha, categories, pilot_images, pilot_manifest_sha, pilot_declaration_sha = prepare_scope(args, configuration)
    if args.metadata is None:
        generated = args.out / 'backbone_tensor_metadata.json'
        if not generated.exists():
            atomic_json(generated, backbone_metadata_from_checkpoint(args.backbone))
        args.metadata = generated
    for key, path in [('backbone', args.backbone), ('head', args.head)]:
        if path.stat().st_size != WEIGHTS[key]['bytes']:
            raise ValueError('A full approved checkpoint is incomplete; preserve its transfer and wait.')
    materializer = Path(__file__).with_name('dino_materializer.py')
    source = source_snapshot(args.source.resolve())
    if source['fingerprint_sha256'] != configuration['source_fingerprint_sha256']:
        raise ValueError('The upstream executable files differ from the verified source snapshot.')
    small_locks = {str(path): digest(path) for path in (
        args.model_config, args.protocol, args.pilot_protocol, args.images, args.vocabulary,
        args.metadata, materializer, Path(__file__), getattr(args, '_pilot_manifest_path', None)) if path is not None}
    small_locks.update(storage_locks)
    protocol = dict(
        model='DINOv3 ViT-7B/16 + Plain-DETR', source_commit=SOURCE_COMMIT,
        upstream_source=source, source_path=str(args.source.resolve()),
        declaration_sha256=digest(args.protocol) if args.protocol else None, primary_pilot_declaration_sha256=pilot_declaration_sha,
        image_manifest_sha256=manifest_sha, image_count=len(images),
        pilot_image_manifest_sha256=pilot_manifest_sha, vocabulary_sha256=digest(args.vocabulary),
        backbone_metadata_sha256=digest(args.metadata), weights=WEIGHTS,
        input_preprocessing=declaration['input_preprocessing'],
        precision='Native CUDA BF16 autocast; only Linear/Conv parameter storage BF16; all other parameters/buffers retain native dtype',
        device=args.device, gpu_limit_GiB=args.gpu_limit_gib,
        strict_mapping=dict(unique_objects=932, persistent_state_keys=988, unused_checkpoint_keys=0),
        native_topk=1500, raw_class_slots=91, category_ids=sorted(categories),
        feature_storage=args.feature_storage,
        feature_storage_admission_sha256=digest(args.storage_admission) if args.storage_admission else None,
        feature_storage_helper_sha256=digest(args.storage_helper) if storage_admission else None,
        allocator_environment={'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF')},
        unused_class_slots=sorted(set(range(91)) - categories),
        candidate_policy='Archive all native1500 rows; derive eligible80-category stream by dropping only unused COCO slots after top1500 without refill; preserve native indices and all finite boxes',
        coordinate_policy='Call detector core, native PostProcess with resized_HW and original_HW; no additional coordinate transform',
        materializer_sha256=digest(materializer), exporter_sha256=digest(__file__),
        locked_small_files=small_locks,
        torch_version=torch.__version__, cpu_threads=1, seed=20261118,
        matmul_allow_tf32=False, cudnn_benchmark=False,
        cuda_cache_policy='gc.collect and empty_cache before first forward and after every image',
        ground_truth_read=False, AP_computed=False)
    protocol_path = args.out / 'PROTOCOL_PRE_GT.json'
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError('The source, model, preprocessing or manifest changed since this run began.')
    else:
        atomic_json(protocol_path, protocol)
    protocol_sha = digest(protocol_path)
    chunks = args.out / 'chunks'
    chunks.mkdir(exist_ok=True)
    pending = [row for row in images if not (chunks / f"{row['image_id']:012d}.json").exists()]
    model = None
    weight_signatures = {str(path): signature(path) for path in (args.backbone, args.head)}
    def verify_small_locks_and_weights():
        for path, expected in small_locks.items():
            if digest(path) != expected:
                raise ValueError('A frozen declaration, preprocessing input or wrapper file changed.')
        for path, old in weight_signatures.items():
            if signature(path) != old:
                raise ValueError('A learned checkpoint file changed during inference.')
    if pending:
        admission = gpu_admission(args.gpu_limit_gib, args.device)
        atomic_json(args.out / ('GPU_ADMISSION_' + uuid4().hex + '.json'), admission)
        model, loading = materialize(args.source, args.backbone, args.head,
                                     read_json(args.metadata), device=args.device,
                                     gpu_limit_GiB=args.gpu_limit_gib)
        if (loading['target_unique_objects'] != 932 or loading['persistent_target_state_keys'] != 988
                or loading['missing_target_keys'] or loading['unused_checkpoint_keys']):
            raise ValueError('The materialized native architecture differs from the admitted strict map.')
        atomic_json(args.out / ('STRICT_WEIGHT_LOAD_' + uuid4().hex + '.json'), loading)
        if storage_admission is not None:
            from . import dino_storage
            storage_handle = dino_storage.install(model)
            if storage_handle.receipt['helper_sha256'] != storage_admission['helper_sha256']:
                storage_handle.uninstall()
                raise ValueError('The activated helper differs from the admitted runtime method.')
            atomic_json(args.out / ('ADMITTED_FEATURE_STORAGE_INSTALL_' + uuid4().hex + '.json'), storage_handle.receipt)
        gc.collect()
        torch.cuda.empty_cache()
    started = time.monotonic()
    pilot_done = False
    current_image = None
    try:
        for index, row in enumerate(images, 1):
            current_image = row['image_id']
            path = image_path(row, args.image_root)
            raw_bytes = path.read_bytes()
            image_sha = sha256(raw_bytes).hexdigest()
            if row.get('image_sha256') and image_sha != row['image_sha256']:
                raise ValueError('An input image changed from its frozen content hash.')
            chunk = chunks / f"{row['image_id']:012d}.json"
            if chunk.exists():
                previous = read_json(chunk)
                if previous['protocol_sha256'] != protocol_sha or previous['image_sha256'] != image_sha:
                    raise ValueError('A cached image chunk differs from the frozen protocol or image.')
            else:
                original_height, original_width = row['height'], row['width']
                height, width = resized_shape(original_height, original_width, args.short_edge)
                torch.cuda.reset_peak_memory_stats()
                frame_started = time.monotonic()
                with Image.open(BytesIO(raw_bytes)) as picture:
                    if picture.size != (original_width, original_height):
                        raise ValueError('Decoded dimensions differ from the image-only manifest.')
                    picture = tvf.resize(picture.convert('RGB'), [height, width],
                                         interpolation=InterpolationMode.BILINEAR, antialias=True)
                    tensor = tvf.normalize(tvf.to_tensor(picture), [.485, .456, .406], [.229, .224, .225])
                tensor = tensor.to(args.device)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model.detector([tensor])
                    if list(output['pred_logits'].shape) != [1, 1500, 91]:
                        raise ValueError('The native decoder changed its query count or class slots.')
                    result, = model.postprocessor(
                        output, target_sizes=torch.tensor([[height, width]], device=args.device),
                        original_target_sizes=torch.tensor([[original_height, original_width]], device=args.device))
                predictions, excluded, nonpositive, native_predictions, ignored_predictions = canonical_rows(
                    row['image_id'], result['labels'].cpu().tolist(),
                    result['boxes'].float().cpu().tolist(), result['scores'].float().cpu().tolist(), categories)
                torch.cuda.synchronize(args.device)
                if digest(path) != image_sha:
                    raise ValueError('An image changed while its native forward was running.')
                atomic_json(chunk, dict(
                    image_id=row['image_id'], image_sha256=image_sha, protocol_sha256=protocol_sha,
                    original_HW=[original_height, original_width], resized_HW=[height, width],
                    native_candidates=1500, eligible_candidates=len(predictions),
                    excluded_unused_slots=excluded, nonpositive_native_boxes_retained=nonpositive,
                    seconds=time.monotonic() - frame_started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device), predictions=predictions,
                    native_predictions=native_predictions, ignored_native_slots=ignored_predictions))
                del output, result, tensor, picture, predictions, native_predictions, ignored_predictions
                gc.collect()
                torch.cuda.empty_cache()
            if pilot_images and index == len(pilot_images):
                verify_small_locks_and_weights()
                pilot = seal_stream(args.out / ('pilot' + str(len(pilot_images))), images[:len(pilot_images)], chunks, protocol_sha,
                                    pilot_manifest_sha, 'EXPLICITLY_FROZEN_PREFIX_IMAGE_PANEL',
                                    dict(short_edge=args.short_edge, full_worker_continues=len(images) > len(pilot_images),
                                         primary_pilot_protocol_sha256=pilot_declaration_sha))
                pilot_done = True
                print('PILOT_GT_FREE_COMPLETE', json.dumps(pilot), flush=True)
            if index % 10 == 0 or index == len(images):
                print(json.dumps(dict(images=index, total=len(images), short_edge=args.short_edge,
                                      elapsed_seconds=time.monotonic() - started, pilot500_sealed=pilot_done)), flush=True)
        if {int(path.stem) for path in chunks.glob('*.json')} != {row['image_id'] for row in images}:
            raise ValueError('Chunk coverage does not exactly match the declared manifest.')
        if digest(args.images) != manifest_sha or source_snapshot(args.source.resolve()) != source:
            raise ValueError('The input manifest or upstream executable source changed during inference.')
        if digest(materializer) != protocol['materializer_sha256']:
            raise ValueError('The strict materializer source changed during inference.')
        verify_small_locks_and_weights()
        summary = seal_stream(args.out, images, chunks, protocol_sha, manifest_sha,
                              'COMPLETE_DECLARED_MANIFEST_NATIVE_DETECTION',
                              dict(short_edge=args.short_edge, pilot500_sealed=pilot_done))
        print('GT_FREE_COMPLETE', json.dumps(summary), flush=True)
    except BaseException as error:
        failure = dict(status='FAILED_OR_INTERRUPTED_COMPLETED_CHUNKS_AND_PREFIX_PRESERVED',
                       error_type=type(error).__name__, error=str(error), image_id=current_image,
                       traceback=traceback.format_exc(), pilot500_sealed=pilot_done,
                       ground_truth_read=False, AP_computed=False, protocol_sha256=protocol_sha,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if torch.cuda.is_initialized() else 0,
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if torch.cuda.is_initialized() else 0)
        atomic_json(args.out / ('FAILURE_' + uuid4().hex + '.json'), failure)
        raise


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ('source', 'backbone', 'head', 'images', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('metadata', 'protocol', 'pilot-protocol', 'pilot-images', 'storage-admission'):
        parser.add_argument('--' + name, type=Path)
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument('--model-config', type=Path, default=repository / 'configs/dino_vit7b16.json')
    parser.add_argument('--vocabulary', type=Path, default=repository / 'configs/benchmarks/coco_val2017/vocabulary.json')
    parser.add_argument('--pilot-count', type=int, default=0)
    parser.add_argument('--feature-storage', choices=('native', 'bf16-concat'), default='bf16-concat')
    parser.add_argument('--allocator-config')
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--short-edge', type=int, choices=(1536, 2048), required=True)
    parser.add_argument('--gpu-limit-gib', type=float, default=22.)
    parser.add_argument('--device', default='cuda:0', help='Logical CUDA device, for example cuda:0 or cuda:1; respects CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.gpu_limit_gib) or args.gpu_limit_gib <= 0:
        parser.error('Use a positive allocation within the available device capacity and any supplied admission receipt.')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        parser.error('Use an indexed logical CUDA device, for example cuda:0 or cuda:1.')
    if args.out.exists() and not args.resume:
        parser.error('Use a new output directory or explicitly resume the existing frozen run.')
    if args.allocator_config is not None:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = args.allocator_config
    try:
        run(args)
    except BaseException as error:
        # Failures before the frame loop (including strict loading) preserve their own receipt.
        if args.out.exists() and not any(args.out.glob('FAILURE_*.json')):
            atomic_json(args.out / ('INITIALIZATION_FAILURE_' + uuid4().hex + '.json'), dict(
                status='INITIALIZATION_FAILED_WITHOUT_PARAMETER_OR_PROTOCOL_FALLBACK',
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                ground_truth_read=False, AP_computed=False))
        raise


if __name__ == '__main__':
    main()
