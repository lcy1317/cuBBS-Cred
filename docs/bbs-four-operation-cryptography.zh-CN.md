# BBS 四操作密码学分析

本文维护四操作的核心公式；运行与路由细节分别见
[后端设计](gpu-bbs-backend-design.md)和[MSM 路由](msm-route-production-design-v2.zh-CN.md)。

## 1. 范围与记号

本文按 cuBBS-Cred 实际使用的
`BBS_BLS12381G1_XMD:SHA-256_SSWU_RO_` ciphersuite，拆解 Sign、Verify
Signature、Proof Gen 和 Proof Verify 四个操作。公式以项目 vendored
zkryptium 实现为准，而不是以 CUDA kernel 名称反推协议。

- `Fr`：BLS12-381 标量域；
- `G1`、`G2`：BLS12-381 素数阶子群；
- `P1 ∈ G1`、`P2 ∈ G2`：固定基点；
- `SK = x ∈ Fr`，`PK = W = x · P2 ∈ G2`；
- `L`：签名消息总数；
- `R`：Proof 中披露消息数；
- `U = L - R`：未披露消息数；
- `Q1, H1, ..., HL ∈ G1`：由 ciphersuite 和消息数确定的生成元；
- `H2S(data, DST)`：使用 XMD:SHA-256 展开后映射到 `Fr`；
- `e(P, Q)`：`G1 × G2 → GT` 的双线性 pairing。

每条消息先独立映射：

```text
m_i = H2S(message_i, API_ID || "MAP_MSG_TO_SCALAR_AS_HASH_")
```

生成元并非每个 issuer 的秘密参数。它们由 `API_ID`、生成元 seed、序号和
hash-to-curve 确定；实现按 ciphersuite、count 和 `API_ID` 缓存它们。

## 2. Domain 和公共 B 点

四个操作都依赖同一个 domain 规则：

```text
domain = H2S(
    PK ||
    I2OSP(L, 8) ||
    serialize(Q1, H1, ..., HL) ||
    API_ID ||
    I2OSP(len(header), 8) ||
    header,
    API_ID || "H2S_"
)
```

然后构造：

```text
B = P1 + domain · Q1 + Σ(i=1..L) m_i · H_i
```

`domain` 绑定 public key、消息数量、完整生成元序列、ciphersuite/API 和
header。任何一项变化都会改变 `B`，因此缓存必须使用完整密码学 key，不能只按
消息数命中。

`B` 是四操作最重要的加速边界之一。它是一个 `L + 2` 项的 G1 MSM；同
issuer、header 和 shape 的一批请求共享 points，只改变每行 scalar，适合在
GPU 上做 common-points matrix MSM。

## 3. Sign

### 3.1 计算过程

签名 scalar `e_sig` 是确定性的：

```text
e_sig = H2S(
    serialize(SK, domain, m_1, ..., m_L),
    API_ID || "H2S_"
)
```

签名点为：

```text
A = (SK + e_sig)^(-1) · B
signature = (A, e_sig)
```

实现拒绝不可逆的 `SK + e_sig` 和 identity `A`。序列化结果固定为：

```text
compressed(A): 48 bytes || e_sig: 32 bytes = 80 bytes
```

### 3.2 主要成本

一次未命中缓存的 Sign 包含：

1. `L` 次 message-to-scalar；
2. 生成元获取/生成；
3. 1 次 domain hash-to-scalar；
4. 1 次签名 `e_sig` hash-to-scalar；
5. 1 个 `L + 2` 项 G1 MSM；
6. 1 次 `Fr` inversion 和 1 次 G1 scalar multiplication；
7. 48-byte 点压缩和固定宽度拼接。

cuBBS-Cred GPU 路径可承担批量 message-to-scalar、`e_sig` hash 和 `B` MSM，
host 保留 key/header 校验、协议状态与最终 80-byte 组装。对小 `L` 和
batch 1，kernel launch、PCIe/FFI 和同步成本可能大于 MSM 本身；对同 shape
大 batch，共享 points 后 GPU 才容易达到高利用率。

## 4. Verify Signature

### 4.1 计算过程

Verifier 用收到的消息、header 和 public key 重算 `domain` 与 `B`，再检查：

```text
e(A, W + e_sig · P2) · e(B, -P2) == 1_GT
```

等价地：

```text
e(A, W + e_sig · P2) == e(B, P2)
```

这证明 `A` 与同一 public key、header 和消息向量对应，但不泄漏 `SK`。

### 4.2 主要成本

单项 Verify Signature 包含：

1. 解析并验证 48-byte G1 点和 32-byte scalar；
2. `L` 次 message-to-scalar；
3. domain hash-to-scalar；
4. `L + 2` 项 `B` MSM；
5. `e_sig · P2` 的 G2 scalar multiplication 和一次 G2 add；
6. 两个 pairing term 的 Miller loop 与一次 final exponentiation。

MSM 和 pairing 都适合批处理。cuBBS-Cred 大 batch 路径还可用非零安全系数
`α_j` 聚合多个 pairing 等式；只有聚合成功才能整体接受，聚合失败后必须回到
逐项验证以定位坏项。系数不能固定为 1，也不能让一个坏项抵消另一个坏项。

## 5. Proof Gen

### 5.1 目标

Proof Gen 持有有效签名 `(A, e_sig)` 和全部 `L` 条消息，选择披露集合 `D`
及未披露集合 `Uset`。它产生一个随机化的、不可链接的签名知识证明，并把披露
消息、presentation header `ph` 和协议 transcript 绑定到 Fiat–Shamir
challenge。

实现生成 `5 + U` 个 fresh CSPRNG scalar：

```text
r1, r2, e_tilde, r1_tilde, r3_tilde,
m_tilde_j for every undisclosed message j
```

其中 `r1`、`r2` 必须适合协议运算；实现随后要求 `r2` 可逆。

### 5.2 Proof 初始化

```text
D_point = r2 · B
Abar    = (r1 · r2) · A
Bbar    = r1 · D_point - e_sig · Abar

T1 = e_tilde · Abar + r1_tilde · D_point
T2 = r3_tilde · D_point
     + Σ(j in Uset) m_tilde_j · H_j
```

注意这里的 `D_point` 是 G1 点，不是披露集合。

challenge 输入按规范顺序绑定：

```text
R and every (disclosed_index, disclosed_message_scalar),
Abar, Bbar, D_point, T1, T2,
domain,
len(ph), ph
```

并计算：

```text
c = H2S(challenge_input, API_ID || "H2S_")
```

令 `r3 = r2^(-1)`，响应为：

```text
e_cap    = e_tilde  + e_sig · c
r1_cap   = r1_tilde - r1 · c
r3_cap   = r3_tilde - r3 · c
m_cap_j  = m_tilde_j + m_j · c
```

最终 proof：

```text
(Abar, Bbar, D_point, e_cap, r1_cap, r3_cap,
 m_cap_1 ... m_cap_U, c)
```

三个压缩 G1 点各 48 bytes，其余 scalar 各 32 bytes，因此：

```text
proof_size = 272 + 32 · U bytes
```

proof 大小只随未披露消息数增长，不随原始 JSON 字节数增长。

### 5.3 主要成本

未复用预计算时，Proof Gen 包含：

1. `L` 次 message-to-scalar、domain 和 `L + 2` 项 `B` MSM；
2. `5 + U` 个随机 scalar；
3. 生成 `D_point/Abar/Bbar/T1` 的 6 次固定 G1 scalar multiplication
   及相应加减；
4. `T2` 的 `U + 1` 项 G1 MSM；
5. 1 次 `Fr` inversion；
6. transcript 序列化、challenge hash 和 `U + 3` 个响应计算；
7. proof 固定宽度组装。

Proof Gen 没有 pairing，但有两个很适合 GPU 的并行区域：批量 `B` MSM，以及
批量固定乘法/`T2` MSM。cuBBS-Cred 已提供从 GPU 预计算的压缩
`D_point/Abar/Bbar/T1/T2` 继续完成 challenge 和响应的边界，避免 host
重新解压这些点。

随机 scalar、challenge 及响应必须保持协议级正确性。尤其不能为了复用 batch
而重复 nonce/random scalar；跨 proof 重用会破坏零知识和不可链接性。

## 6. Proof Verify

### 6.1 计算过程

Verifier 只收到 `R` 条披露消息。由 proof 中 `U` 个 `m_cap` 得到
`L = R + U`，校验披露索引后重算生成元与 domain。

先构造披露部分：

```text
Bv = P1 + domain · Q1
     + Σ(i in D) m_i · H_i
```

再从响应恢复 transcript commitment：

```text
T1' = c · Bbar + e_cap · Abar + r1_cap · D_point

T2' = c · Bv + r3_cap · D_point
      + Σ(j in Uset) m_cap_j · H_j
```

用 `T1'`、`T2'` 和同一披露数据、domain、`ph` 重算 `c'`，要求：

```text
c' == c
```

最后检查随机化签名 pairing：

```text
e(Abar, W) · e(Bbar, -P2) == 1_GT
```

challenge 检查和 pairing 检查缺一不可：前者证明消息/响应 transcript
一致，后者证明 transcript 确实关联到该 issuer 的有效 BBS 签名。

### 6.2 主要成本

单项 Proof Verify 包含：

1. proof 长度、三个压缩 G1 点、scalar 和披露索引校验；
2. `R` 次 disclosed-message-to-scalar；
3. domain hash-to-scalar；
4. `T1'` 的 3 项 G1 MSM；
5. `Bv` 的 `R + 2` 项 MSM，以及 `T2'` 的 scalar multiplication/MSM；
   代数展开后 `T2'` 可表示为 `L + 3` 个 G1 terms；
6. challenge transcript hash；
7. 两个 pairing term 的 Miller loop 和 final exponentiation。

cuBBS-Cred 将 compatible proof shape 的 `T1/T2` 组成 paired MSM；大 pairing
batch 可走设备端路径，小 batch 可留在 BLST。证明解析、索引语义、challenge
比较和错误回填仍由 host 负责。

## 7. 四操作的 GPU 价值

| 操作 | 算术密集核心 | 最有效的 GPU 粒度 | 容易让 GPU 空闲的情况 |
| --- | --- | --- | --- |
| Sign | message hash、`B` MSM | 同 points、多 scalar rows 的大 batch | batch 1、小 `L`、每项同步 |
| Verify Signature | `B` MSM、pairing | MSM matrix、aggregate pairing batch | HTTP 串行等待、逐项 pairing |
| Proof Gen | `B` MSM、6 个固定乘法、`T2` MSM | 同 proof shape 的批量固定基乘/MSM | random/protocol 阶段间频繁往返 |
| Proof Verify | paired `T1/T2` MSM、pairing | paired MSM 与大 pairing batch | 小 batch、mixed shape 过度碎片化 |

结论不是“BBS 不用 GPU 就无法运行”，而是：

- CPU reference 仍然完整、正确，并且小 batch 可能更低延迟；
- BBS 曲线运算有充分的 GPU 并行性；
- GPU 吞吐依赖足够的 in-flight 请求和兼容 shape；
- W3C/HTTP 每次只向后端提交一个或很小的 batch 时，即使单个 kernel 很快，
  GPU 也会在 canonicalization、JSON、网络等待和同步屏障之间空闲。

因此真实全流程优化的优先级通常是：跨 HTTP 请求微批处理和 backpressure，
保持多个 batch 在途，复用 shared points/template，使用 packed/pinned
buffer 并减少 host/device round trip。把 JSON、base64 或小型
`proofValue` 拼接直接移到 GPU，通常不能解决“喂不满”，反而增加分支、拷贝和
kernel launch；只有点压缩/解压、固定宽度 scalar packing 已与设备流水线融合
时，才值得作为次级优化。

## 8. 安全与实现不变量

- BLS12-381 ciphersuite 的目标安全级别约为 128 bit，不能因 batch/GPU 而
  宣称更高安全强度；
- message、domain、signature challenge 和 proof challenge 使用不同的
  domain-separation 输入；
- Sign 的 `e_sig` 是确定性的；Proof Gen 的随机 scalar 必须 fresh；
- disclosed indexes 必须在范围内、去重并按规范顺序参与 challenge；
- `header` 绑定签名域；`ph` 绑定一次 presentation；
- compressed G1/G2 必须做 canonical decode、curve/subgroup 检查；
- aggregate verification 必须使用安全的非零系数，失败后不能把整批当成同一
  个逐项结果；
- GPU 结果必须与 CPU reference、规范向量、交叉验签/验 proof 和 tamper
  negative cases 一致。

CPU/GPU 的确定性中间值和 Sign 结果应精确一致。ProofGen 使用随机数，
不能要求独立生成的 proof 字节天然相同；应做双向 ProofVerify、篡改负例，
并在固定测试随机数时比较规范向量。测试随机性控制不能进入生产 nonce 复用。

BBS 不自动提供 credential revocation、可信 DID resolution、issuer 授权、
holder binding 或防重放。`ph` 可以绑定 nonce/audience/session，但业务必须
检查其新鲜性和预期上下文。这里也不主张后量子安全。

规范与来源：

- [CFRG BBS signatures](https://identity.foundation/bbs-signature/draft-irtf-cfrg-bbs-signatures.html)
- [W3C BBS Cryptosuite](https://www.w3.org/TR/vc-di-bbs/)
- [RFC 9380: Hashing to Elliptic Curves](https://www.rfc-editor.org/rfc/rfc9380.html)
- 本仓库使用的固定规范向量身份见[正确性验证](testing.md)，不随上游网页更新而改变。

## 9. 源码对应

- ciphersuite 与公共 engine：`crates/cubbs-bbs/src/lib.rs`
- Sign/Verify 公式：`crates/cubbs-bbs-protocol/src/bbsplus/signature.rs`
- Proof Gen/Verify 公式：`crates/cubbs-bbs-protocol/src/bbsplus/proof.rs`
- domain、H2S 和序列化：`crates/cubbs-bbs-protocol/src/utils/util.rs`
- message-to-scalar：`crates/cubbs-bbs-protocol/src/utils/message.rs`
- generator derivation/cache：`crates/cubbs-bbs-protocol/src/bbsplus/generators.rs`
- GPU primitive ABI：`crates/cubbs-bbs/src/gpu_abi.rs`
- CUDA MSM/pairing：`native/cuda/src/prepare_b_msm.cu`、
  `native/cuda/src/kernels/`
