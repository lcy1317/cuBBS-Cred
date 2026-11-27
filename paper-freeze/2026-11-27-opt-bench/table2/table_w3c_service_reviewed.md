# W3C service capacity

End-to-end saturation capacity of a synchronous W3C VC Data Integrity BBS service, with a dynamic-batching and structural-cache ablation. The frozen workload is x64/disclose31 `bbs-2023`; each numeric cell reports median requests/s at its selected concurrency across five independently started 30-s service lifetimes (zero request failures; sample CV <= 10%). `@ c` records selected client concurrency, and `N/S` marks a public operation not exposed by that implementation. All four RTX 5090 rows use the same r24 service, load generator, and CUDA library on GPU 1: the first three are single-endpoint measurements, while the fully enabled row sums independent HTTP endpoints sharing that GPU.

| Method | Device | Dynamic batching | Structural cache | Issue | Base Verify | Proof Generation | Proof Verification |
|---|---|---|---|---|---|---|---|
| cuBBS-Cred | RTX 5060 Ti | Off | Off | 5,711.37 @ c24 | 1,139.38 @ c64 | 1,696.72 @ c24 | 882.97 @ c24 |
| cuBBS-Cred | RTX 5060 Ti | Off | On | 5,901.96 @ c16 | 1,118.03 @ c16 | 1,668.35 @ c16 | 881.73 @ c16 |
| cuBBS-Cred | RTX 5060 Ti | On | Off | 54,688.67 @ c2048 | 63,666.10 @ c4096 | 44,409.92 @ c1024 | 65,363.67 @ c2048 |
| cuBBS-Cred | RTX 5060 Ti | On | On | 288,022.28 @ c6144 | 320,412.59 @ c14336 | 193,247.83 @ c4096 | 208,537.32 @ c10240 |
| cuBBS-Cred | RTX 5090 | Off | Off | 5,692.26 @ c64 | 1,142.86 @ c128 | 1,794.59 @ c64 | 915.44 @ c64 |
| cuBBS-Cred | RTX 5090 | Off | On | 6,443.69 @ c64 | 1,158.49 @ c64 | 1,808.02 @ c32 | 916.20 @ c32 |
| cuBBS-Cred | RTX 5090 | On | Off | 42,837.98 @ c4096 | 68,684.58 @ c4096 | 48,586.89 @ c2048 | 74,028.77 @ c4096 |
| cuBBS-Cred | RTX 5090 | On | On | 315,666.34 @ c16384 | 326,972.52 @ c16384 | 257,770.62 @ c8192 | 266,400.40 @ c16384 |
| SpruceID | EPYC 9554 (64C/128T) | / | / | 1,488.88 @ c128 | N/S | 1,158.97 @ c128 | 1,466.66 @ c128 |
| SpruceID | EPYC 7763 (64C/128T) | / | / | 1,222.53 @ c128 | N/S | 945.57 @ c128 | 1,201.54 @ c128 |
| Procivis One | EPYC 9554 (64C/128T) | / | / | 8,871.73 @ c128 | N/S | 4,922.37 @ c256 | 7,064.48 @ c128 |
| Procivis One | EPYC 7763 (64C/128T) | / | / | 7,408.41 @ c128 | N/S | 4,091.48 @ c128 | 5,865.48 @ c128 |
| Grotto | EPYC 9554 (64C/128T) | / | / | 154.74 @ c128 | 130.41 @ c128 | 105.10 @ c128 | 130.63 @ c128 |
| Grotto | EPYC 7763 (64C/128T) | / | / | 121.85 @ c64 | 103.30 @ c64 | 83.50 @ c64 | 105.34 @ c64 |
| Digital Bazaar | EPYC 9554 (64C/128T) | / | / | 105.25 @ c128 | N/S | 79.75 @ c128 | 92.42 @ c128 |
| Digital Bazaar | EPYC 7763 (64C/128T) | / | / | 100.50 @ c64 | N/S | 77.94 @ c64 | 90.39 @ c64 |
