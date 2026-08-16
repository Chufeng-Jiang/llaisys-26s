import argparse
import math
import os
import sys


# ============================================================
# Repository paths
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
PYTHON_DIR = os.path.join(REPO_ROOT, 'python')

sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, TEST_DIR)


# ============================================================
# Imports
# ============================================================

import torch

import llaisys

from llaisys.triton import execution_context
from llaisys.triton.backends.registry import get_triton_backend
from llaisys.triton.ops import argmax as triton_argmax

from test_utils import (
    BenchmarkRecorder,
    benchmark,
    build_experiment_output_path,
    check_equal,
    collect_backend_metadata,
    llaisys_device,
    llaisys_dtype,
    random_tensor,
    reference_torch_device,
    torch_dtype,
    torch_to_llaisys_memcpy_kind,
    zero_tensor,
)


# ============================================================
# Constants
# ============================================================

DTYPE_BYTES = {
    'f32': 4,
    'f16': 2,
    'bf16': 2,
}

TEST_DTYPES = ['f32', 'f16', 'bf16']


# ============================================================
# PyTorch reference
# ============================================================


def torch_argmax(max_idx, max_val, vals):
    # PyTorch semantics used by LLAISYS Argmax:
    #
    # - reduce the complete 1D tensor;
    # - return both value and index;
    # - ties select the first occurrence;
    # - NaN selection follows torch.max semantics.
    torch.max(
        vals,
        dim=-1,
        keepdim=True,
        out=(max_val, max_idx),
    )


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_argmax(max_idx, max_val, vals, backend):
    if backend == 'native':
        llaisys.Ops.argmax(max_idx, max_val, vals)
        return

    if backend == 'triton':
        triton_argmax(max_idx, max_val, vals)
        return

    raise ValueError(f'Unsupported Argmax backend: {backend}')


# ============================================================
# Effective configuration
# ============================================================


def _parse_env_config_value(name, default='default'):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return value


def get_argmax_config(vals, backend):
    numel = math.prod(vals.shape())

    if backend == 'native':
        return 'requested_or_backend_policy', {
            'BLOCK_SIZE': _parse_env_config_value('LLAISYS_BLOCK_SIZE'),
        }

    if backend == 'triton':
        triton_backend = get_triton_backend(vals.device_type())
        config = triton_backend.argmax_config(numel)

        return 'effective', {
            'STAGE1_BLOCK_SIZE': config['STAGE1_BLOCK_SIZE'],
            'STAGE1_NUM_WARPS': config['STAGE1_NUM_WARPS'],
            'STAGEN_BLOCK_SIZE': config['STAGEN_BLOCK_SIZE'],
            'STAGEN_NUM_WARPS': config['STAGEN_NUM_WARPS'],
        }

    raise ValueError(f'Unsupported Argmax backend: {backend}')


def get_argmax_config_label(vals, backend):
    _, config = get_argmax_config(vals, backend)
    values = ', '.join(f'{key}={value}' for key, value in config.items())
    return f'config[{values}]'


def get_argmax_output_filename_config(backend):
    if backend == 'native':
        return {
            'BLOCK_SIZE': _parse_env_config_value('LLAISYS_BLOCK_SIZE'),
        }

    if backend == 'triton':
        return {
            'STAGE1_BLOCK_SIZE': _parse_env_config_value(
                'LLAISYS_TRITON_STAGE1_BLOCK_SIZE'
            ),
            'STAGE1_NUM_WARPS': _parse_env_config_value(
                'LLAISYS_TRITON_STAGE1_NUM_WARPS'
            ),
            'STAGEN_BLOCK_SIZE': _parse_env_config_value(
                'LLAISYS_TRITON_STAGEN_BLOCK_SIZE'
            ),
            'STAGEN_NUM_WARPS': _parse_env_config_value(
                'LLAISYS_TRITON_STAGEN_NUM_WARPS'
            ),
        }

    raise ValueError(f'Unsupported Argmax backend: {backend}')


# ============================================================
# Logical traffic / derived metrics
# ============================================================
#
# Minimum operator-level logical traffic:
#
#     read  numel input values
#     write one output value
#     write one int64 output index
#
# Multi-stage implementations may perform additional workspace traffic.
# Therefore this is explicitly a minimum logical/equivalent traffic model,
# not measured DRAM traffic.
# ============================================================


def get_argmax_minimum_logical_io_traffic_bytes(numel, dtype_name):
    value_bytes = DTYPE_BYTES[dtype_name]
    return numel * value_bytes + value_bytes + 8


def get_effective_bandwidth_gbs(traffic_bytes, median_ms):
    return traffic_bytes / median_ms / 1_000_000.0


def get_input_throughput_gelem_s(numel, median_ms):
    return numel / median_ms / 1_000_000.0


def get_argmax_derived_metrics(stats, numel, dtype_name):
    traffic_bytes = get_argmax_minimum_logical_io_traffic_bytes(
        numel,
        dtype_name,
    )

    llaisys_stats = stats['llaisys']
    torch_stats = stats.get('torch')

    derived = {
        'minimum_logical_io_traffic_bytes': traffic_bytes,
        'llaisys_effective_io_bandwidth_gbs': get_effective_bandwidth_gbs(
            traffic_bytes,
            llaisys_stats['median_ms'],
        ),
        'llaisys_input_throughput_gelem_s': get_input_throughput_gelem_s(
            numel,
            llaisys_stats['median_ms'],
        ),
    }

    if torch_stats is not None:
        derived.update(
            {
                'torch_equivalent_io_bandwidth_gbs': get_effective_bandwidth_gbs(
                    traffic_bytes,
                    torch_stats['median_ms'],
                ),
                'torch_input_throughput_gelem_s': get_input_throughput_gelem_s(
                    numel,
                    torch_stats['median_ms'],
                ),
            }
        )

    return derived


def print_argmax_derived_metrics(
    derived,
    device_name,
    show_bandwidth,
    show_throughput,
):
    if show_bandwidth:
        print(
            f'        LLAISYS {device_name} effective minimum-I/O bandwidth: '
            f"{derived['llaisys_effective_io_bandwidth_gbs']:.2f} GB/s"
        )

        torch_bandwidth = derived.get('torch_equivalent_io_bandwidth_gbs')

        if torch_bandwidth is not None:
            print(
                f'        Torch {device_name} equivalent minimum-I/O bandwidth: '
                f'{torch_bandwidth:.2f} GB/s'
            )

    if show_throughput:
        print(
            f'        LLAISYS {device_name} input throughput: '
            f"{derived['llaisys_input_throughput_gelem_s']:.3f} GElem/s"
        )

        torch_throughput = derived.get('torch_input_throughput_gelem_s')

        if torch_throughput is not None:
            print(
                f'        Torch {device_name} input throughput: '
                f'{torch_throughput:.3f} GElem/s'
            )


# ============================================================
# Deterministic tensor construction
# ============================================================


def copy_reference_to_llaisys(torch_tensor, dtype_name, device_name, device_id=0):
    tensor = llaisys.Tensor(
        tuple(torch_tensor.shape),
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()

    api.memcpy_sync(
        tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_llaisys_memcpy_kind(device_name),
    )

    return tensor


def tensor_pair_from_values(values, dtype_name, device_name, device_id=0):
    reference_device = reference_torch_device(device_name, device_id)

    torch_tensor = torch.tensor(
        values,
        dtype=torch_dtype(dtype_name),
        device=reference_device,
    ).contiguous()

    llaisys_tensor = copy_reference_to_llaisys(
        torch_tensor,
        dtype_name,
        device_name,
        device_id,
    )

    return torch_tensor, llaisys_tensor


def tensor_pair_from_sparse_case(
    numel,
    dtype_name,
    device_name,
    *,
    fill_value=-1000.0,
    assignments=None,
    device_id=0,
):
    reference_device = reference_torch_device(device_name, device_id)

    torch_tensor = torch.full(
        (numel,),
        fill_value,
        dtype=torch_dtype(dtype_name),
        device=reference_device,
    )

    for index, value in assignments or []:
        torch_tensor[index] = value

    torch_tensor = torch_tensor.contiguous()

    llaisys_tensor = copy_reference_to_llaisys(
        torch_tensor,
        dtype_name,
        device_name,
        device_id,
    )

    return torch_tensor, llaisys_tensor


# ============================================================
# Result verification
# ============================================================


def check_argmax_result(
    *,
    name,
    torch_max_idx,
    torch_max_val,
    llaisys_max_idx,
    llaisys_max_val,
    dtype_name,
    device_name,
    backend,
):
    assert check_equal(llaisys_max_idx, torch_max_idx, strict=True), (
        f'Argmax index mismatch: case={name}, dtype={dtype_name}, '
        f'device={device_name}, backend={backend}'
    )

    reference_is_nan = bool(torch.isnan(torch_max_val).item())

    if reference_is_nan:
        # check_equal normally treats NaN != NaN. The selected index is the
        # semantic requirement that determines which NaN won the reduction.
        print(f'      NaN reference selected at index {int(torch_max_idx.item())}')
        return

    assert check_equal(llaisys_max_val, torch_max_val, strict=True), (
        f'Argmax value mismatch: case={name}, dtype={dtype_name}, '
        f'device={device_name}, backend={backend}'
    )


# ============================================================
# Random differential correctness / performance case
# ============================================================


def test_op_argmax(
    shape,
    dtype_name='f32',
    device_name='cpu',
    backend='native',
    profile=False,
    backend_variant='unspecified',
    backend_implementation=None,
    suite='correctness',
    seed=0,
    warmup=10,
    repeat=100,
    rounds=10,
    benchmark_order='alternating',
    show_config=False,
    show_bandwidth=False,
    show_throughput=False,
    recorder=None,
    device_metadata=None,
):
    numel = math.prod(shape)

    print(
        f'   random shape {shape} '
        f'numel {numel} '
        f'dtype <{dtype_name}> '
        f'device <{device_name}> '
        f'backend <{backend}>'
    )

    torch_vals, llaisys_vals = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    torch_max_idx, llaisys_max_idx = zero_tensor(
        (1,),
        'i64',
        device_name,
    )

    torch_max_val, llaisys_max_val = zero_tensor(
        (1,),
        dtype_name,
        device_name,
    )

    torch_argmax(
        torch_max_idx,
        torch_max_val,
        torch_vals,
    )

    run_llaisys_argmax(
        llaisys_max_idx,
        llaisys_max_val,
        llaisys_vals,
        backend,
    )

    check_argmax_result(
        name=f'random_{numel}',
        torch_max_idx=torch_max_idx,
        torch_max_val=torch_max_val,
        llaisys_max_idx=llaisys_max_idx,
        llaisys_max_val=llaisys_max_val,
        dtype_name=dtype_name,
        device_name=device_name,
        backend=backend,
    )

    if not profile:
        return

    if recorder is None:
        recorder = BenchmarkRecorder()

    benchmark_argmax(
        torch_max_idx,
        torch_max_val,
        torch_vals,
        llaisys_max_idx,
        llaisys_max_val,
        llaisys_vals,
        backend,
        backend_variant,
        backend_implementation,
        device_name,
        dtype_name,
        suite,
        seed,
        warmup,
        repeat,
        rounds,
        benchmark_order,
        show_config,
        show_bandwidth,
        show_throughput,
        recorder,
        device_metadata or {},
    )


# ============================================================
# Deterministic semantic correctness
# ============================================================


def test_semantic_tensor_case(
    name,
    torch_vals,
    llaisys_vals,
    expected_index,
    dtype_name,
    device_name,
    backend,
):
    print(
        f'   semantic {name} '
        f'numel {torch_vals.numel()} '
        f'dtype <{dtype_name}> '
        f'device <{device_name}> '
        f'backend <{backend}>'
    )

    torch_max_idx, llaisys_max_idx = zero_tensor(
        (1,),
        'i64',
        device_name,
    )

    torch_max_val, llaisys_max_val = zero_tensor(
        (1,),
        dtype_name,
        device_name,
    )

    torch_argmax(
        torch_max_idx,
        torch_max_val,
        torch_vals,
    )

    actual_reference_index = int(torch_max_idx.item())

    assert actual_reference_index == expected_index, (
        f'Invalid semantic test definition: name={name}, '
        f'dtype={dtype_name}, expected={expected_index}, '
        f'PyTorch={actual_reference_index}'
    )

    run_llaisys_argmax(
        llaisys_max_idx,
        llaisys_max_val,
        llaisys_vals,
        backend,
    )

    check_argmax_result(
        name=name,
        torch_max_idx=torch_max_idx,
        torch_max_val=torch_max_val,
        llaisys_max_idx=llaisys_max_idx,
        llaisys_max_val=llaisys_max_val,
        dtype_name=dtype_name,
        device_name=device_name,
        backend=backend,
    )


def test_semantic_case(
    name,
    values,
    expected_index,
    dtype_name,
    device_name,
    backend,
):
    torch_vals, llaisys_vals = tensor_pair_from_values(
        values,
        dtype_name,
        device_name,
    )

    test_semantic_tensor_case(
        name,
        torch_vals,
        llaisys_vals,
        expected_index,
        dtype_name,
        device_name,
        backend,
    )


def run_semantic_tests(device_name, dtype_name, backend):
    cases = [
        ('single_element', [5.0], 0),
        ('normal', [1.0, 4.0, 2.0, 3.0], 1),
        ('maximum_first', [9.0, 1.0, 2.0, 3.0], 0),
        ('maximum_last', [1.0, 2.0, 3.0, 9.0], 3),
        ('all_equal', [7.0, 7.0, 7.0, 7.0], 0),
        ('duplicate_maximum', [1.0, 9.0, 3.0, 9.0], 1),
        ('all_negative', [-9.0, -3.0, -7.0], 1),
        ('all_negative_infinity', [float('-inf')] * 5, 0),
        ('positive_infinity', [1.0, float('inf'), 100.0], 1),
        ('duplicate_positive_infinity', [1.0, float('inf'), 2.0, float('inf')], 1),
        ('negative_infinity_mixed', [float('-inf'), -2.0, -3.0], 1),
        ('signed_zero_tie', [-0.0, 0.0, -1.0], 0),
        (
            'wide_duplicate_maximum',
            [1.0, 2.0, 99.0, 4.0, 5.0, 6.0, 7.0, 99.0],
            2,
        ),
        ('single_nan', [1.0, float('nan'), 100.0], 1),
        ('multiple_nan', [float('nan'), 1.0, float('nan')], 0),
        ('nan_after_numeric_max', [1000.0, 999.0, float('nan')], 2),
        ('nan_before_numeric_max', [float('nan'), 999.0, 1000.0], 0),
        (
            'wide_multiple_nan',
            [1.0, 2.0, float('nan'), 4.0, 5.0, 6.0, float('nan'), 1000.0],
            2,
        ),
        (
            'cross_block_duplicate_maximum_fixed',
            [99.0 if i in (100, 1100) else float(i % 17) for i in range(1200)],
            100,
        ),
        (
            'cross_block_multiple_nan_fixed',
            [float('nan') if i in (123, 1123) else float(i % 23) for i in range(1200)],
            123,
        ),
    ]

    for name, values, expected_index in cases:
        test_semantic_case(
            name,
            values,
            expected_index,
            dtype_name,
            device_name,
            backend,
        )


# ============================================================
# Triton reduction-boundary semantic tests
# ============================================================
#
# These cases are generated from the EFFECTIVE Triton configuration so
# environment overrides continue to test the actual launch boundaries.
# ============================================================


def get_triton_boundary_config(device_name):
    backend = get_triton_backend(llaisys_device(device_name))
    config = backend.argmax_config(1 << 20)

    return {
        'STAGE1_BLOCK_SIZE': int(config['STAGE1_BLOCK_SIZE']),
        'STAGEN_BLOCK_SIZE': int(config['STAGEN_BLOCK_SIZE']),
    }


def run_triton_boundary_tests(device_name, dtype_name, backend):
    if backend != 'triton':
        return

    config = get_triton_boundary_config(device_name)
    stage1 = config['STAGE1_BLOCK_SIZE']
    stagen = config['STAGEN_BLOCK_SIZE']

    print(
        f'   Triton reduction boundaries: '
        f'stage1={stage1}, stagen={stagen}, dtype=<{dtype_name}>'
    )

    # --------------------------------------------------------
    # Stage-1 boundary: duplicate maximum on opposite sides of
    # the boundary. The first ORIGINAL input index must win.
    # --------------------------------------------------------

    numel = stage1 + 3
    first = stage1 - 1
    second = stage1

    torch_vals, llaisys_vals = tensor_pair_from_sparse_case(
        numel,
        dtype_name,
        device_name,
        fill_value=-100.0,
        assignments=[
            (first, 1000.0),
            (second, 1000.0),
        ],
    )

    test_semantic_tensor_case(
        'stage1_boundary_duplicate_maximum',
        torch_vals,
        llaisys_vals,
        first,
        dtype_name,
        device_name,
        backend,
    )

    # --------------------------------------------------------
    # Stage-1 boundary: NaN on both sides. The first NaN must
    # win exactly like torch.max.
    # --------------------------------------------------------

    torch_vals, llaisys_vals = tensor_pair_from_sparse_case(
        numel,
        dtype_name,
        device_name,
        fill_value=-100.0,
        assignments=[
            (first, float('nan')),
            (second, float('nan')),
        ],
    )

    test_semantic_tensor_case(
        'stage1_boundary_multiple_nan',
        torch_vals,
        llaisys_vals,
        first,
        dtype_name,
        device_name,
        backend,
    )

    # --------------------------------------------------------
    # Partial final stage-1 block: the last valid lane must not
    # be confused with masked -inf lanes.
    # --------------------------------------------------------

    last = numel - 1

    torch_vals, llaisys_vals = tensor_pair_from_sparse_case(
        numel,
        dtype_name,
        device_name,
        fill_value=-100.0,
        assignments=[(last, 1000.0)],
    )

    test_semantic_tensor_case(
        'stage1_partial_block_last_maximum',
        torch_vals,
        llaisys_vals,
        last,
        dtype_name,
        device_name,
        backend,
    )

    # --------------------------------------------------------
    # All -inf with a partial block. Masked lanes also use -inf,
    # so this explicitly verifies that an invalid lane never wins.
    # --------------------------------------------------------

    torch_vals, llaisys_vals = tensor_pair_from_sparse_case(
        numel,
        dtype_name,
        device_name,
        fill_value=float('-inf'),
    )

    test_semantic_tensor_case(
        'stage1_partial_block_all_negative_infinity',
        torch_vals,
        llaisys_vals,
        0,
        dtype_name,
        device_name,
        backend,
    )

    # --------------------------------------------------------
    # Multi-stage boundary.
    #
    # A stage-1 output count greater than STAGEN_BLOCK_SIZE forces
    # at least two stage-N reduction launches:
    #
    #     stage1 -> stageN -> final stageN
    #
    # Keep this bounded so extreme user overrides do not accidentally
    # allocate an enormous semantic-test tensor.
    # --------------------------------------------------------

    multi_stage_numel = stage1 * stagen + 1
    max_multi_stage_test_numel = 4 * 1024 * 1024 + 1

    if multi_stage_numel <= max_multi_stage_test_numel:
        first = stage1 - 1
        second = multi_stage_numel - 1

        torch_vals, llaisys_vals = tensor_pair_from_sparse_case(
            multi_stage_numel,
            dtype_name,
            device_name,
            fill_value=-100.0,
            assignments=[
                (first, 1000.0),
                (second, 1000.0),
            ],
        )

        test_semantic_tensor_case(
            'multi_stage_duplicate_maximum',
            torch_vals,
            llaisys_vals,
            first,
            dtype_name,
            device_name,
            backend,
        )
    else:
        print(
            '      skip multi_stage_duplicate_maximum: '
            f'generated numel={multi_stage_numel} exceeds '
            f'test cap={max_multi_stage_test_numel}'
        )


# ============================================================
# Benchmark
# ============================================================


def benchmark_argmax(
    torch_max_idx,
    torch_max_val,
    torch_vals,
    llaisys_max_idx,
    llaisys_max_val,
    llaisys_vals,
    backend,
    backend_variant,
    backend_implementation,
    device_name,
    dtype_name,
    suite,
    seed,
    warmup,
    repeat,
    rounds,
    benchmark_order,
    show_config,
    show_bandwidth,
    show_throughput,
    recorder,
    device_metadata,
):
    shape = llaisys_vals.shape()
    numel = math.prod(shape)
    config_status, config = get_argmax_config(llaisys_vals, backend)

    label = (
        f'Argmax shape={shape} '
        f'numel={numel} '
        f'dtype={dtype_name} '
        f'backend={backend}'
    )

    if show_config:
        label += f' {get_argmax_config_label(llaisys_vals, backend)}'

    print(f'        {label}:')

    torch_fn = lambda: torch_argmax(
        torch_max_idx,
        torch_max_val,
        torch_vals,
    )

    llaisys_fn = lambda: run_llaisys_argmax(
        llaisys_max_idx,
        llaisys_max_val,
        llaisys_vals,
        backend,
    )

    if backend == 'native':
        stats = benchmark(
            torch_fn,
            llaisys_fn,
            device_name,
            warmup=warmup,
            repeat=repeat,
            rounds=rounds,
            benchmark_order=benchmark_order,
        )
    elif backend == 'triton':
        with execution_context(
            llaisys_vals.device_type(),
            llaisys_vals.device_id(),
        ):
            stats = benchmark(
                torch_fn,
                llaisys_fn,
                device_name,
                warmup=warmup,
                repeat=repeat,
                rounds=rounds,
                benchmark_order=benchmark_order,
            )
    else:
        raise ValueError(f'Unsupported Argmax backend: {backend}')

    derived = get_argmax_derived_metrics(
        stats,
        numel,
        dtype_name,
    )

    if show_bandwidth or show_throughput:
        print_argmax_derived_metrics(
            derived,
            device_name,
            show_bandwidth,
            show_throughput,
        )

    recorder.record_microbenchmark(
        op='argmax',
        backend_name=backend,
        backend_variant=backend_variant,
        backend_implementation=backend_implementation,
        suite=suite,
        device_name=device_name,
        device_id=llaisys_vals.device_id(),
        shape=shape,
        numel=numel,
        dtype_name=dtype_name,
        seed=seed,
        config=config,
        config_status=config_status,
        warmup=warmup,
        repeat=repeat,
        rounds=rounds,
        benchmark_order=benchmark_order,
        stats=stats,
        derived=derived,
        workload_metadata={
            'reduction_length': numel,
            'torch_reference': 'torch.max_out_1d',
            'tie_semantics': 'first_occurrence',
            'nan_semantics': 'torch_max_first_nan',
            'input_distribution': 'uniform[-1,1)',
        },
        device_metadata=device_metadata,
    )


# ============================================================
# Profiler helpers
# ============================================================


def _torch_profiler_synchronize(device_name):
    if device_name in ('nvidia', 'amd'):
        torch.cuda.synchronize()


def _begin_profiler_range(label, device_name):
    if device_name not in ('nvidia', 'amd'):
        return False

    if not torch.cuda.is_available():
        return False

    try:
        torch.cuda.nvtx.range_push(label)
        return True
    except Exception:
        return False


def _end_profiler_range(range_pushed):
    if not range_pushed:
        return

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def run_argmax_profiler_case(
    *,
    numel,
    dtype_name,
    device_name,
    backend,
    backend_variant,
    profiler_target,
    profiler_warmup,
    profiler_launches,
    profiler_check,
    show_config,
):
    print()
    print('=== Profiler single case ===')
    print(
        f'   target <{profiler_target}> '
        f'numel {numel} '
        f'dtype <{dtype_name}> '
        f'device <{device_name}> '
        f'backend <{backend}>'
    )

    torch_vals, llaisys_vals = random_tensor(
        (numel,),
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    torch_max_idx, llaisys_max_idx = zero_tensor(
        (1,),
        'i64',
        device_name,
    )

    torch_max_val, llaisys_max_val = zero_tensor(
        (1,),
        dtype_name,
        device_name,
    )

    if profiler_target == 'torch':
        if device_name == 'metax':
            raise ValueError(
                'Torch profiler target is unavailable for MetaX because '
                'the current MetaX reference tensor is hosted on CPU.'
            )

        target_fn = lambda: torch_argmax(
            torch_max_idx,
            torch_max_val,
            torch_vals,
        )
        synchronize = lambda: _torch_profiler_synchronize(device_name)
        config_status = 'reference'
        config = {}

        target_label = (
            f'LLAISYS_PROFILE:argmax:torch:{device_name}:'
            f'numel={numel}:dtype={dtype_name}'
        )

        for _ in range(profiler_warmup):
            target_fn()
        synchronize()

        range_pushed = _begin_profiler_range(target_label, device_name)

        try:
            for _ in range(profiler_launches):
                target_fn()
            synchronize()
        finally:
            _end_profiler_range(range_pushed)

        if profiler_check:
            run_llaisys_argmax(
                llaisys_max_idx,
                llaisys_max_val,
                llaisys_vals,
                backend,
            )

            check_argmax_result(
                name='profiler_case',
                torch_max_idx=torch_max_idx,
                torch_max_val=torch_max_val,
                llaisys_max_idx=llaisys_max_idx,
                llaisys_max_val=llaisys_max_val,
                dtype_name=dtype_name,
                device_name=device_name,
                backend=backend,
            )
    else:
        config_status, config = get_argmax_config(
            llaisys_vals,
            backend,
        )

        if show_config:
            print(f'        {get_argmax_config_label(llaisys_vals, backend)}')

        target_fn = lambda: run_llaisys_argmax(
            llaisys_max_idx,
            llaisys_max_val,
            llaisys_vals,
            backend,
        )

        api = llaisys.RuntimeAPI(llaisys_vals.device_type())
        synchronize = api.device_synchronize

        config_tag = ','.join(
            f'{key}={value}'
            for key, value in config.items()
        )

        target_label = (
            f'LLAISYS_PROFILE:argmax:{backend}:{backend_variant}:{device_name}:'
            f'numel={numel}:dtype={dtype_name}:{config_tag}'
        )

        def execute_target():
            for _ in range(profiler_warmup):
                target_fn()
            synchronize()

            range_pushed = _begin_profiler_range(
                target_label,
                device_name,
            )

            try:
                for _ in range(profiler_launches):
                    target_fn()
                synchronize()
            finally:
                _end_profiler_range(range_pushed)

        if backend == 'triton':
            with execution_context(
                llaisys_vals.device_type(),
                llaisys_vals.device_id(),
            ):
                execute_target()
        else:
            execute_target()

        if profiler_check:
            torch_argmax(
                torch_max_idx,
                torch_max_val,
                torch_vals,
            )
            _torch_profiler_synchronize(device_name)

            check_argmax_result(
                name='profiler_case',
                torch_max_idx=torch_max_idx,
                torch_max_val=torch_max_val,
                llaisys_max_idx=llaisys_max_idx,
                llaisys_max_val=llaisys_max_val,
                dtype_name=dtype_name,
                device_name=device_name,
                backend=backend,
            )

    print(f'Profiler target range: {target_label}')
    print(
        f'Profiler launches: warmup={profiler_warmup}, '
        f'target={profiler_launches}'
    )

    if profiler_target == 'llaisys' and backend == 'triton':
        print(
            'Profiler note: multi-block Triton Argmax is a multi-kernel '
            'pipeline (stage1 plus one or more stage-N kernels). Profile the '
            'NVTX operator range with a timeline tool first; for NCU, discover '
            'the exact kernel sequence before applying launch-count filters.'
        )

    if profiler_check:
        print('Profiler post-check: passed')

    return {
        'target': profiler_target,
        'numel': numel,
        'dtype': dtype_name,
        'config_status': config_status,
        'config': config,
        'warmup': profiler_warmup,
        'launches': profiler_launches,
        'range': target_label,
    }


# ============================================================
# CLI helpers
# ============================================================


def _unique_lengths(values):
    result = []
    seen = set()

    for value in values:
        value = int(value)

        if value <= 0 or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def build_correctness_lengths(device_name, backend):
    lengths = [
        1,
        2,
        3,
        4,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        32000,
        128256,
        151936,
        512 * 4096,
    ]

    if backend == 'triton':
        config = get_triton_boundary_config(device_name)
        stage1 = config['STAGE1_BLOCK_SIZE']
        stagen = config['STAGEN_BLOCK_SIZE']

        lengths.extend(
            [
                stage1 - 1,
                stage1,
                stage1 + 1,
                2 * stage1 - 1,
                2 * stage1,
                2 * stage1 + 1,
                stage1 * stagen - 1,
                stage1 * stagen,
                stage1 * stagen + 1,
            ]
        )

    return _unique_lengths(lengths)


def get_profile_lengths(profile_suite):
    sweep = [
        4,
        32,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        2097152,
    ]

    llm = [
        32000,
        128256,
        151936,
    ]

    if profile_suite == 'sweep':
        return [('sweep', value) for value in sweep]

    if profile_suite == 'llm':
        return [('llm', value) for value in llm]

    return (
        [('sweep', value) for value in sweep]
        + [('llm', value) for value in llm]
    )


# ============================================================
# Main
# ============================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'nvidia', 'metax', 'amd'],
        type=str,
    )

    parser.add_argument(
        '--backend',
        default='native',
        choices=['native', 'triton'],
        type=str,
    )

    parser.add_argument(
        '--backend-variant',
        default='unspecified',
        type=str,
        help=(
            'Experiment variant label, for example baseline, tuned, '
            'autotuned, or vendor-specific.'
        ),
    )

    parser.add_argument(
        '--backend-implementation',
        default=None,
        type=str,
        help=(
            'Optional implementation override. Native normally maps to '
            'cpu/cuda/maca/hip and Triton maps to triton.'
        ),
    )

    execution_mode = parser.add_mutually_exclusive_group()

    execution_mode.add_argument(
        '--profile',
        action='store_true',
        help='Run the paper-oriented microbenchmark suite.',
    )

    execution_mode.add_argument(
        '--profiler-mode',
        action='store_true',
        help=(
            'Run one controlled Argmax workload for ncu/nsys/mcProfiler/rocprof. '
            'This does not run the normal benchmark suite.'
        ),
    )

    parser.add_argument(
        '--case-numel',
        default=None,
        type=int,
        help='Reduction length for --profiler-mode. Required in profiler mode.',
    )

    parser.add_argument(
        '--case-dtype',
        default='f16',
        choices=TEST_DTYPES,
        type=str,
        help='Profiler-case dtype. Default: f16.',
    )

    parser.add_argument(
        '--profiler-target',
        default='llaisys',
        choices=['llaisys', 'torch'],
        type=str,
    )

    parser.add_argument(
        '--profiler-warmup',
        default=1,
        type=int,
    )

    parser.add_argument(
        '--profiler-launches',
        default=1,
        type=int,
    )

    parser.add_argument(
        '--profiler-check',
        action='store_true',
    )

    parser.add_argument(
        '--show-config',
        action='store_true',
    )

    parser.add_argument(
        '--show-bandwidth',
        action='store_true',
    )

    parser.add_argument(
        '--show-throughput',
        action='store_true',
    )

    parser.add_argument(
        '--skip-correctness',
        action='store_true',
    )

    parser.add_argument(
        '--skip-semantic',
        action='store_true',
    )

    parser.add_argument(
        '--skip-boundary-semantic',
        action='store_true',
        help='Skip Triton configuration-derived reduction-boundary semantic cases.',
    )

    parser.add_argument(
        '--profile-suite',
        default='all',
        choices=['sweep', 'llm', 'all'],
        type=str,
    )

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--repeat', default=100, type=int)
    parser.add_argument('--rounds', default=10, type=int)

    parser.add_argument(
        '--benchmark-order',
        default='alternating',
        choices=[
            'llaisys_then_torch',
            'torch_then_llaisys',
            'alternating',
        ],
        type=str,
    )

    parser.add_argument(
        '--output-dir',
        default='results',
        type=str,
    )

    parser.add_argument(
        '--no-record',
        action='store_true',
    )

    parser.add_argument(
        '--output',
        default=None,
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument('--run-id', default=None, type=str)
    parser.add_argument('--run-note', default=None, type=str)

    args = parser.parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if args.backend == 'triton' and args.device == 'cpu':
        raise ValueError('Triton Argmax requires a GPU device')

    if args.warmup < 0:
        raise ValueError('--warmup must be non-negative')

    if args.repeat <= 0:
        raise ValueError('--repeat must be greater than zero')

    if args.rounds <= 0:
        raise ValueError('--rounds must be greater than zero')

    if args.profiler_warmup < 0:
        raise ValueError('--profiler-warmup must be non-negative')

    if args.profiler_launches <= 0:
        raise ValueError('--profiler-launches must be greater than zero')

    if args.profiler_mode and (args.case_numel is None or args.case_numel <= 0):
        raise ValueError('--case-numel > 0 is required with --profiler-mode')

    torch.manual_seed(args.seed)

    if args.device in ('nvidia', 'amd') and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    backend_metadata = collect_backend_metadata(
        args.backend,
        args.device,
        variant=args.backend_variant,
        implementation=args.backend_implementation,
    )

    filename_config = get_argmax_output_filename_config(args.backend)

    if args.output is not None:
        output_path = args.output
    elif args.profile and not args.no_record:
        output_path = build_experiment_output_path(
            args.output_dir,
            op='argmax',
            device_name=args.device,
            backend=backend_metadata,
            config=filename_config,
        )
    else:
        output_path = None

    run_metadata = {
        'profile_suite': args.profile_suite,
        'benchmark_order': args.benchmark_order,
        'note': args.run_note,
        'reference': {
            'torch': 'torch.max_out_1d',
        },
        'semantics': {
            'tie': 'first_occurrence',
            'nan': 'torch.max-compatible first NaN selection',
        },
        'input_distribution': 'uniform[-1,1)',
        'profiler_mode': args.profiler_mode,
        'profiler_case': {
            'numel': args.case_numel,
            'dtype': args.case_dtype,
            'target': args.profiler_target,
            'warmup': args.profiler_warmup,
            'launches': args.profiler_launches,
        },
        'output': {
            'automatic': args.output is None,
            'directory': args.output_dir,
            'filename_config': filename_config,
        },
    }

    recorder = BenchmarkRecorder(
        output_path=output_path,
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        run_metadata=run_metadata,
    )

    print(
        f'Testing Ops.argmax on {args.device} '
        f'with {args.backend} backend'
    )
    print(
        f"Backend identity: name={backend_metadata['name']}, "
        f"implementation={backend_metadata['implementation']}, "
        f"variant={backend_metadata['variant']}"
    )
    print(f'Random seed: {args.seed}')
    print(
        f'Benchmark protocol: warmup={args.warmup}, '
        f'repeat={args.repeat}, rounds={args.rounds}, '
        f'order={args.benchmark_order}'
    )
    print(f'Using llaisys from: {llaisys.__file__}')

    if output_path is not None:
        print(f'Recording JSONL: {output_path}')
        print(f'Run ID: {recorder.run_id}')

    if args.profiler_mode:
        run_argmax_profiler_case(
            numel=args.case_numel,
            dtype_name=args.case_dtype,
            device_name=args.device,
            backend=args.backend,
            backend_variant=args.backend_variant,
            profiler_target=args.profiler_target,
            profiler_warmup=args.profiler_warmup,
            profiler_launches=args.profiler_launches,
            profiler_check=args.profiler_check,
            show_config=args.show_config,
        )

        print()
        print('\033[92mProfiler run completed!\033[0m')
        raise SystemExit(0)

    # ========================================================
    # Random differential correctness
    # ========================================================

    if not args.skip_correctness:
        print()
        print('=== Correctness: random differential / boundary lengths ===')

        correctness_lengths = build_correctness_lengths(
            args.device,
            args.backend,
        )

        for numel in correctness_lengths:
            for dtype_name in TEST_DTYPES:
                test_op_argmax(
                    (numel,),
                    dtype_name=dtype_name,
                    device_name=args.device,
                    backend=args.backend,
                    profile=False,
                )

    # ========================================================
    # Deterministic semantics
    # ========================================================

    if not args.skip_semantic:
        print()
        print('=== Correctness: deterministic semantics ===')

        for dtype_name in TEST_DTYPES:
            run_semantic_tests(
                args.device,
                dtype_name,
                args.backend,
            )

    # ========================================================
    # Dynamic Triton reduction-boundary semantics
    # ========================================================

    if (
        args.backend == 'triton'
        and not args.skip_boundary_semantic
    ):
        print()
        print('=== Correctness: Triton reduction-stage boundaries ===')

        for dtype_name in TEST_DTYPES:
            run_triton_boundary_tests(
                args.device,
                dtype_name,
                args.backend,
            )

    # ========================================================
    # Performance
    # ========================================================

    if args.profile:
        print()
        print('=== Performance ===')

        for suite, numel in get_profile_lengths(args.profile_suite):
            for dtype_name in TEST_DTYPES:
                test_op_argmax(
                    (numel,),
                    dtype_name=dtype_name,
                    device_name=args.device,
                    backend=args.backend,
                    profile=True,
                    backend_variant=args.backend_variant,
                    backend_implementation=args.backend_implementation,
                    suite=suite,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    rounds=args.rounds,
                    benchmark_order=args.benchmark_order,
                    show_config=args.show_config,
                    show_bandwidth=args.show_bandwidth,
                    show_throughput=args.show_throughput,
                    recorder=recorder,
                    device_metadata={},
                )

    print()
    print('\033[92mTest passed!\033[0m')