# Frozen Optimized BBS Primitive Capacity Dataset (`2026-11-27-opt`)

Canonical frozen benchmark package for the `opt` branch of `cuBBS-Cred` (commit `9cad7ca`), replacing the historical `2026-10-28-matched` snapshot on the `opt` branch.

## Provenance & Hardware

- **Git branch / commit**: `opt` (`9cad7ca`)
- **CUDA toolchain**: `nvcc 13.2` (`-O3 --std=c++17 -arch=native` for Blackwell `sm_120`)
- **Execution guard**: `CUBBS_BBS_FORCE_GPU=1 CUBBS_BBS_REQUIRE_HARDWARE_GPU=1` (`0` failed items across all 32 workloads)
- **GPU conformance**: 34/34 unit & hardware GPU conformance tests passed on RTX 5060 Ti; 31/31 standalone hardware GPU tests passed on RTX 5090
- **RTX 5060 Ti host**: Local NVIDIA GeForce RTX 5060 Ti (16 GB GDDR7, 36 SMs, 180W TDP) + Intel Core Ultra 7 265K (20C)
- **RTX 5090 host**: Vast.ai NVIDIA GeForce RTX 5090 (32 GB GDDR7, 170 SMs, 400W host power limit, PCIe Gen5 x16) + AMD Ryzen 9 9950X3D2 (32T, 62 GB RAM)
- **CPU baselines**: Retained verbatim from the canonical EPYC 7763 (64C/128T) and EPYC 9554 (64C/128T) columns of `table_bbs_capacity.csv`

## Headline Summary (32-Row Geometric Mean, 3-Rep Median)

| Platform | GMean vs. Prior Frozen Orig | GMean vs. EPYC 9554 | Peak (`256 Unique Verify`) | Mean GPU SM% | Mean Power |
|---|---:|---:|---:|---:|---:|
| **RTX 5060 Ti Opt** | **`4.78×`** | **`173×`** | `415,299 ops/s` (`36.39×` vs. 5060 Ti Orig, `8.64×` vs. 5090 Orig) | `96.7%` (`30/32 ≥ 90%`) | `144.7W / 180W` |
| **RTX 5090 Opt** | **`3.52×`** | **`388×`** | `644,652 ops/s` (`13.41×` vs. 5090 Orig) | `94.3%` (Median `100%`) | `297.9W / 400W` |
| **RTX 5090 Opt / RTX 5060 Ti Opt** | **`2.24×`** | — | — | — | — |

## Directory Layout

- `tables/table_bbs_capacity.csv` — Exact 39-column canonical schema matching the paper Table 2 CSV format (`epyc7763_*`, `epyc9554_*`, `rtx5060ti_*`, `rtx5090_*`, `rtx5090_over_rtx5060ti`). Note: `rtx5060ti` ran 3 repetitions on 28 rows and 2 repetitions on the 4 `m65536_*` rows; `rtx5090` ran 3 repetitions on all 32 rows (on-host median serialized; `min`/`max` mirror the median in the canonical CSV).
- `tables/table_bbs_capacity_telemetry.csv` — Complete 32-row hardware telemetry (100ms time-aligned `nvidia-smi` SM%, Mem%, Power W, SM Clock MHz, CPU cores, CPU s/1M items, and speedup vs. prior frozen baseline) for both GPUs.
- `tables/table_bbs_capacity.md` & `tables/table_bbs_capacity.tex` — Rendered paper Markdown and LaTeX tables.
- `tables/render_table.py` — Standalone validator & renderer for this frozen package.
- `figures/table_bbs_capacity.{png,pdf,svg}` — Canonical paper-style rendered Table 2.
- `figures/table2_rtx5060ti_opt_vs_orig.png` — Single-GPU RTX 5060 Ti Opt vs. prior frozen baseline comparison chart.
- `figures/table2_rtx5060ti_and_rtx5090_opt_vs_orig.png` — Dual-GPU RTX 5060 Ti & RTX 5090 Opt vs. prior frozen baseline comparison chart.
- `raw/table2_rtx5060ti_opt_32rows.json` & `raw/table2_rtx5090_opt_32rows.json` — Raw 32-row benchmark + telemetry JSON records.
- `SHA256SUMS` — SHA-256 verification manifest for all files in this directory.
