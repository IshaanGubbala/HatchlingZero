"""Diagnostic: probe MLX 0.32 `mx.fast.metal_kernel` dispatch semantics.

For each (grid_shape, threadgroup_shape) combination, dispatch a tiny
body-only kernel that writes each thread's `thread_position_in_grid`,
`threadgroup_position_in_grid`, and `thread_position_in_threadgroup`
into a unique output slot. Count how many threads actually wrote to
confirm how MLX interprets grid vs threadgroup.
"""

import mlx.core as mx

_BODY_TEMPLATE = """
constexpr uint Dv     = {Dv};
constexpr uint TOTAL_G = {TOTAL_G};
constexpr uint MAX_DV  = 64;
constexpr uint OUT_SIZE = {OUT_SIZE};

uint gid  = thread_position_in_grid.x;
uint gpid = threadgroup_position_in_grid.x;
uint lid  = thread_position_in_threadgroup.x;

if (gid >= TOTAL_G || lid >= Dv) return;

uint slot = gpid * MAX_DV + lid;
if (slot < OUT_SIZE) {{
    info[slot * 4 + 0] = (float)gid;
    info[slot * 4 + 1] = (float)gpid;
    info[slot * 4 + 2] = (float)lid;
    info[slot * 4 + 3] = (float)(lid & 31u);  // simd_lane_id (Apple SIMD group = 32)
}}
"""


def probe(grid_shape, threadgroup_shape, OUT_SIZE=1024):
    g_x = grid_shape[0]
    tg_x = threadgroup_shape[0]
    src = _BODY_TEMPLATE.format(Dv=tg_x, TOTAL_G=g_x, OUT_SIZE=OUT_SIZE)
    k = mx.fast.metal_kernel(
        name="dispatch_probe",
        input_names=["info_in"],
        output_names=["info"],
        source=src,
        header="",
        atomic_outputs=False,
        ensure_row_contiguous=True,
    )
    info_in = mx.zeros((OUT_SIZE * 4,), dtype=mx.float32)
    (info,) = k(
        inputs=[info_in],
        output_shapes=[(OUT_SIZE * 4,)],
        output_dtypes=[mx.float32],
        grid=grid_shape,
        threadgroup=threadgroup_shape,
    )
    mx.eval(info)
    return info


def decode(info, OUT_SIZE=1024):
    import numpy as np
    arr = np.array(info).reshape(OUT_SIZE, 4)
    rows = []
    for s in range(OUT_SIZE):
        if arr[s, 0] != 0 or arr[s, 1] != 0 or arr[s, 2] != 0:
            rows.append(tuple(int(v) for v in arr[s]))
    return rows


def main():
    cases = [
        # (label, grid, threadgroup, expected_total_threads)
        ("grid=(4,1,1), tg=(8,1,1)",  (4, 1, 1), (8, 1, 1),  32),
        ("grid=(2,1,1), tg=(8,1,1)",  (2, 1, 1), (8, 1, 1),  16),
        ("grid=(8,1,1), tg=(1,1,1)",  (8, 1, 1), (1, 1, 1),   8),
        ("grid=(16,1,1), tg=(2,1,1)", (16, 1, 1), (2, 1, 1), 32),
        ("grid=(1,1,1), tg=(8,1,1)",  (1, 1, 1), (8, 1, 1),   8),
        ("grid=(1,1,1), tg=(32,1,1)", (1, 1, 1), (32, 1, 1), 32),
        ("grid=(1,1,1), tg=(64,1,1)", (1, 1, 1), (64, 1, 1), 64),
    ]
    for label, grid, tg, expected in cases:
        info = probe(grid, tg)
        rows = decode(info)
        running = len(rows)
        gids = sorted(set(r[0] for r in rows))
        gpids = sorted(set(r[1] for r in rows))
        lids = sorted(set(r[2] for r in rows))
        print(f"{label}: running={running} expected={expected}  "
              f"gids={gids[:8]}{'..' if len(gids) > 8 else ''}  "
              f"gpids={gpids[:8]}{'..' if len(gpids) > 8 else ''}  "
              f"lids={lids[:16]}{'..' if len(lids) > 16 else ''}",
              flush=True)
        # Group rows by gpid
        from collections import defaultdict
        per_gpid = defaultdict(list)
        for r in rows:
            per_gpid[r[1]].append(r[2])
        for gpid in sorted(per_gpid.keys())[:4]:
            print(f"   gpid={gpid}: lids={sorted(per_gpid[gpid])}", flush=True)


if __name__ == "__main__":
    main()
