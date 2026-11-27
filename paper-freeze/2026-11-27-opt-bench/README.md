# cuBBS-Cred Complete Refreshed Paper Dataset (`opt-bench`, 2026-11-27)

Complete end-to-end re-measurement of all paper figures and tables (**Table 1, Table 2, Fig. 1, Fig. 2, Fig. 3, Fig. 4, Fig. 8**) on the optimized `opt` codebase (with lightweight MSM route ablation hooks isolated on branch `opt-bench`, keeping `opt` spotless).

## Summary of Measurements (0 failed across all runs)

- **Table 1 & Fig. 1** (`table1/`, `fig1/`): 32-row BBS primitive capacity on RTX 5060 Ti (local, `4.78×` GMean vs Orig, `173×` vs EPYC 9554) and RTX 5090 (Vast.ai Ryzen 9 9950X3D2, `3.52×` GMean vs Orig, `388×` vs EPYC 9554). M=64 speedups vs EPYC 9554: Sign `612× / 210×`, Verify `880× / 724×`, ProofGen `352× / 123×`, ProofVerify `355× / 113×`.
- **Fig. 2** (`fig2/`): RTX 5060 Ti MSM route ablation (`msm_route_bench`, BLST-verified):
  - Panel (a) `K=3` variable-base (`B=1..65,536`): `R1 (1t/row GLV)` beats `R4 (universal GLV)` from `B≥256` (`1.21×`) up to `97.3k rows/s` (`5.35×`).
  - Panel (b) `B=1024` shared-point (`M=16..8192`): `R2 (Lim-Lee, K≤2100)` achieves `8.1–24.3×` (`440.8k` at `M=16`, `11.3k` at `M=2048`), and `R3 (10-bit GLV Pippenger, K>2100)` achieves `6.2–7.3×` (`2.65k` at `M=4096`, `1.56k` at `M=8192`).
- **Fig. 3** (`fig3/`): RTX 5060 Ti message-count scalability (`M=16..65,536`, fixed `B=24, S=8`, 3 repeats): `M=65,536` endpoints reach Sign `294 ops/s`, Verify `2.52k ops/s`, ProofGen `160 ops/s`, ProofVerify `225 ops/s`.
- **Fig. 4** (`fig4/`): RTX 5060 Ti mixed-shape (`M={16,64,128}`, `B=512, S=1`) issuer compatibility & Verify pre-B aggregation ablation (6 GPU conditions × 3 repeats vs archived EPYC 9554 NUMA1 reference):
  - Unique-PK Verify jumps to `34.8k ops/s` (`12.7×` CPU, `12.0×` vs Orig), matching Shared-PK (`12.6×`, `U_issuer = 0.99×`).
  - Verify pre-B aggregation control: `12.41×` (shared), `19.03×` (unique).
- **Table 2 & Fig. 8** (`table2/`, `fig8/`): W3C VC Data Integrity synchronous HTTP service 2×2 ablation (`B0/C0`, `B0/C1`, `B1/C0`, `B1/C1`, `0` failed, `cuda-abi-v2-force-gpu`, `fallback=false`, `verify-cache=0`):
  - **RTX 5060 Ti `B1/C1`** (local Ultra 7 265K, 3×10s, `CV ≤ 1.7%`): Issue `288.0k`, Base Verify `320.4k`, Derive `193.2k` (`6.16×` vs Orig 5060 Ti), VP Verify `208.5k` (`2.84×` vs Orig 5060 Ti).
  - **RTX 5090 `B1/C1`** (Vast.ai 128T AMD EPYC 9554 + 575W RTX 5090, NUMA-split CPU pinning, 5×30s + 10s warm-up, `CV ≤ 0.43%`, GPU util `77.2%–100.0%`): Issue **`1,064.8k`** (`16ep, c98304`, `1.20×` vs Orig `r24`), Base Verify **`971.3k`** (`8ep, c98304`, `2.14×` vs Orig `r24`), Derive **`498.5k`** (`2ep, c32768`, `6.59×` vs Orig `r24`), VP Verify **`487.8k`** (`2ep, c32768`, `3.08×` vs Orig `r24`).
  - **RTX 5090 `B1/C0` / `B0/C1` / `B0/C0`** (same 128T host, 5×15s, `CV ≤ 2.11%`): `B1/C0` `80.3k / 139.7k / 110.6k / 152.6k`; `B0/C1` `15.1k / 2.75k / 4.24k / 2.22k`; `B0/C0` `14.1k / 2.75k / 4.18k / 2.21k`.
