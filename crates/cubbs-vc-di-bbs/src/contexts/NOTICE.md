# Embedded JSON-LD Context Notice

cuBBS-Cred embeds the following context documents with Rust `include_str!` so
that normal credential processing does not depend on live network retrieval.

| Local file | Canonical source | Upstream last modified | Local SHA-256 |
|---|---|---|---|
| `credentials-v1.jsonld` | <https://www.w3.org/2018/credentials/v1> | 2024-07-19 | `00d7dd6d3ad8b920e3e550dd3a3d9090bffcac84cdbf943da6924b3e0a5c8bb8` |
| `credentials-v2.jsonld` | <https://www.w3.org/ns/credentials/v2> | 2025-03-11 | `8a9f494a89ecc51db093e90e84713e07e84d6d9204364a9b3c7868b21751236f` |
| `data-integrity-v2.jsonld` | <https://w3id.org/security/data-integrity/v2> | 2025-03-11 | `0f77743daf5b4e8fc067fc5ba5b21044283053aa717fc6b0219843bed3b00363` |
| `vdl-v1.jsonld` | <https://w3id.org/vdl/v1> | 2024-06-14 | `5e61aa138bc6a1bf92299edd01408df52f20c24057b9dbea72ccaeef5b792010` |

The local copies were compared with the canonical HTTP representations on
2026-09-15. Parsing each pair as JSON and serializing it with sorted keys
produced identical content. The local copies differ, if at all, only in JSON
whitespace and formatting; no key or value was changed.

Verifiable Credentials v1 Context:
<https://www.w3.org/2018/credentials/v1>. Copyright © 2024 World Wide Web
Consortium. All Rights Reserved. This work is distributed under the W3C®
Software and Document License [1] in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.

Verifiable Credentials v2 Context: <https://www.w3.org/ns/credentials/v2>.
Copyright © 2025 World Wide Web Consortium. All Rights Reserved. This work is
distributed under the W3C® Software and Document License [1] in the hope that
it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Data Integrity v2 Context: <https://w3id.org/security/data-integrity/v2>.
Copyright © 2025 World Wide Web Consortium. All Rights Reserved. This work is
distributed under the W3C® Software and Document License [1] in the hope that
it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

W3C Credentials Community Group Verifiable Driver's License Vocabulary v1
Context: <https://w3id.org/vdl/v1>. Copyright © 2024 World Wide Web Consortium
(Massachusetts Institute of Technology, European Research Consortium for
Informatics and Mathematics, Keio University, Beihang). All Rights Reserved.
This work is distributed under the W3C® Software and Document License [2] in
the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. The
upstream repository applies that license to reports and Apache-2.0 to software.
Because a JSON-LD context can be distributed as both a document and software
resource, cuBBS-Cred redistributes both notices without selecting a narrower
classification.

[1]: <https://www.w3.org/copyright/software-license-2023/>
[2]: <https://www.w3.org/copyright/software-license-2015/>

The complete W3C Software and Document License notices are retained in
`LICENSES/W3C-Software-and-Document-2015.txt` and
`LICENSES/W3C-Software-and-Document-2023.txt`. The repository's Apache-2.0
license text is in the root `LICENSE` file. All pre-existing terms remain in
effect; the W3C names and trademarks may not be used for publicity without
permission.
