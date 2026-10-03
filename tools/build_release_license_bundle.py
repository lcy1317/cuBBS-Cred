#!/usr/bin/env python3
"""Build a fail-closed third-party license bundle for a cuBBS-Cred release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN_LICENSE_IDS = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "ISC",
    "LGPL-2.1-or-later",
    "LLVM-exception",
    "MIT",
    "MPL-2.0+",
    "Unicode-3.0",
    "Unlicense",
    "Zlib",
}
NATIVE_INPUTS = {
    "blst": {
        "commit": "e7f90de551e8df682f3cc99067d204d8b90d27ad",
        "tree": "29527bca487b89db54ec66d69d239cc9263e3077",
        "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    },
    "sppark": {
        "commit": "17278d74295392f9813f009300b257a688422b7a",
        "tree": "47f0157dc6285f3a999545788a4a536b82418214",
        "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    },
}
CONTEXT_HASHES = {
    "credentials-v1.jsonld": "00d7dd6d3ad8b920e3e550dd3a3d9090bffcac84cdbf943da6924b3e0a5c8bb8",
    "credentials-v2.jsonld": "8a9f494a89ecc51db093e90e84713e07e84d6d9204364a9b3c7868b21751236f",
    "data-integrity-v2.jsonld": "0f77743daf5b4e8fc067fc5ba5b21044283053aa717fc6b0219843bed3b00363",
    "vdl-v1.jsonld": "5e61aa138bc6a1bf92299edd01408df52f20c24057b9dbea72ccaeef5b792010",
}
DOTENV_LICENSE_SHA256 = "d7366187540d9ec714de6402f3f2adbaa029f5ecb92612f27544e67b99b96739"
LICENSE_NAME_PREFIXES = (
    "authors",
    "copying",
    "copyright",
    "licence",
    "license",
    "notice",
)


class GateError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, cwd=REPO_ROOT):
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GateError("command failed: {}\n{}".format(" ".join(command), detail))
    return completed.stdout.strip()


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def parse_lock_checksums(lock_path):
    checksums = {}
    for block in lock_path.read_text(encoding="utf-8").split("[[package]]")[1:]:
        fields = {}
        for line in block.splitlines():
            match = re.match(r'^([A-Za-z0-9_-]+)\s*=\s*("(?:\\.|[^"])*")\s*$', line)
            if match:
                fields[match.group(1)] = json.loads(match.group(2))
        if all(key in fields for key in ("name", "version", "source", "checksum")):
            key = (fields["name"], fields["version"], fields["source"])
            if key in checksums:
                raise GateError("duplicate Cargo.lock package identity: {}".format(key))
            checksums[key] = fields["checksum"]
    return checksums


def license_ids(expression):
    identifiers = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", expression))
    return identifiers.difference({"AND", "OR", "WITH"})


def validate_license(package):
    expression = package.get("license")
    if not expression:
        raise GateError(
            "{} {} has no license metadata".format(package["name"], package["version"])
        )
    identifiers = license_ids(expression)
    unknown = identifiers.difference(KNOWN_LICENSE_IDS)
    if unknown:
        raise GateError(
            "{} {} introduces unreviewed license identifier(s): {}".format(
                package["name"], package["version"], ", ".join(sorted(unknown))
            )
        )
    if "LGPL-2.1-or-later" in identifiers and not (
        package["name"] == "r-efi"
        and package["version"] == "6.0.0"
        and expression == "MIT OR Apache-2.0 OR LGPL-2.1-or-later"
    ):
        raise GateError(
            "{} {} requires manual LGPL policy review".format(
                package["name"], package["version"]
            )
        )
    return identifiers


def package_license_members(archive, expected_root):
    files = []
    with tarfile.open(str(archive), "r:gz") as crate:
        for member in crate.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != expected_root:
                raise GateError("unexpected path in crate archive {}: {}".format(archive, member.name))
            relative_parts = parts[1:]
            if not relative_parts:
                continue
            top_level_notice = len(relative_parts) == 1 and relative_parts[0].lower().startswith(
                LICENSE_NAME_PREFIXES
            )
            license_directory = relative_parts[0].lower() in {"license", "licenses", "licences"}
            if top_level_notice or license_directory:
                extracted = crate.extractfile(member)
                if extracted is None:
                    raise GateError("cannot read license member: {}".format(member.name))
                files.append(("/".join(relative_parts), extracted.read()))
    names = [name for name, _ in files]
    if len(names) != len(set(names)):
        raise GateError("duplicate license path in crate archive: {}".format(archive))
    return sorted(files, key=lambda item: item[0])


def apache_is_standalone_option(expression):
    normalized = re.sub(r"\s*/\s*", " OR ", expression)
    if "AND" in normalized or "WITH" in normalized:
        return False
    alternatives = [item.strip(" ()") for item in normalized.split(" OR ")]
    return "Apache-2.0" in alternatives


def find_verified_archive(cargo_home, name, version, expected_sha256):
    pattern = "{}-{}.crate".format(name, version)
    candidates = sorted((cargo_home / "registry" / "cache").glob("*/{}".format(pattern)))
    for candidate in candidates:
        if sha256(candidate) == expected_sha256:
            return candidate
    raise GateError(
        "locked source archive missing or checksum-mismatched for {} {} ({})".format(
            name, version, expected_sha256
        )
    )


def verify_native_checkout(name, root, expected):
    root = root.resolve()
    if not (root / ".git").exists():
        raise GateError(
            "{} must be an independent Git checkout: {}".format(name, root)
        )
    top = Path(run(["git", "-C", str(root), "rev-parse", "--show-toplevel"])).resolve()
    if top != root:
        raise GateError("{} checkout resolves to unexpected Git root: {}".format(name, top))
    status = run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"])
    if status:
        raise GateError("{} checkout is dirty:\n{}".format(name, status))
    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    tree = run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"])
    if commit != expected["commit"] or tree != expected["tree"]:
        raise GateError(
            "{} identity mismatch: commit={} tree={}".format(name, commit, tree)
        )
    license_path = root / "LICENSE"
    if not license_path.is_file() or sha256(license_path) != expected["license_sha256"]:
        raise GateError("{} LICENSE is missing or checksum-mismatched".format(name))
    return {
        "name": name,
        "commit": commit,
        "tree": tree,
        "licenseSha256": expected["license_sha256"],
    }


def build_bundle(output, blst_root, sppark_root, cargo_home):
    lock_path = REPO_ROOT / "Cargo.lock"
    metadata = json.loads(
        run(["cargo", "metadata", "--locked", "--offline", "--format-version", "1"])
    )
    checksums = parse_lock_checksums(lock_path)
    external_packages = sorted(
        (package for package in metadata["packages"] if package.get("source")),
        key=lambda package: (package["name"], package["version"], package["id"]),
    )
    if not external_packages:
        raise GateError("cargo metadata returned no external packages")

    native = [
        verify_native_checkout("blst", blst_root, NATIVE_INPUTS["blst"]),
        verify_native_checkout("sppark", sppark_root, NATIVE_INPUTS["sppark"]),
    ]

    output = output.resolve()
    if output.exists():
        raise GateError("output already exists; choose a new directory: {}".format(output))
    if output == REPO_ROOT or output == Path("/"):
        raise GateError("refusing unsafe output directory: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cubbs-license-bundle-", dir=str(output.parent)))

    try:
        for relative in (
            "Cargo.lock",
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY.md",
            "CFRG-IETF-BBS/LICENSE",
            "crates/cubbs-bbs-protocol/LICENSE",
            "crates/cubbs-vc-di-bbs/src/contexts/NOTICE.md",
            "LICENSES/W3C-Software-and-Document-2015.txt",
            "LICENSES/W3C-Software-and-Document-2023.txt",
        ):
            source = REPO_ROOT / relative
            if not source.is_file():
                raise GateError("required notice file missing: {}".format(relative))
            copy_file(source, temporary / "project" / relative)

        context_inventory = []
        context_root = REPO_ROOT / "crates/cubbs-vc-di-bbs/src/contexts"
        for filename, expected_hash in sorted(CONTEXT_HASHES.items()):
            source = context_root / filename
            actual_hash = sha256(source)
            if actual_hash != expected_hash:
                raise GateError(
                    "embedded context changed without notice update: {} ({})".format(
                        filename, actual_hash
                    )
                )
            destination = temporary / "embedded-contexts" / filename
            copy_file(source, destination)
            context_inventory.append(
                {"file": filename, "sha256": actual_hash, "bundlePath": str(destination.relative_to(temporary))}
            )

        dotenv_fallback = REPO_ROOT / "LICENSES/dotenv-0.15.0-MIT.md"
        if sha256(dotenv_fallback) != DOTENV_LICENSE_SHA256:
            raise GateError("pinned dotenv 0.15.0 license text checksum mismatch")

        rust_inventory = []
        seen_destinations = set()
        for package in external_packages:
            if not package["source"].startswith("registry+"):
                raise GateError(
                    "non-registry dependency requires manual source review: {}".format(package["id"])
                )
            identifiers = validate_license(package)
            lock_key = (package["name"], package["version"], package["source"])
            checksum = checksums.get(lock_key)
            if not checksum:
                raise GateError("Cargo.lock checksum missing for {}".format(package["id"]))
            archive = find_verified_archive(
                cargo_home, package["name"], package["version"], checksum
            )

            stem = "{}-{}".format(package["name"], package["version"])
            stem = re.sub(r"[^A-Za-z0-9._+-]", "_", stem)
            if stem in seen_destinations:
                raise GateError("duplicate bundle destination for {}".format(package["id"]))
            seen_destinations.add(stem)
            package_destination = temporary / "rust" / stem
            copied = []
            expected_archive_root = "{}-{}".format(package["name"], package["version"])
            for relative, contents in package_license_members(archive, expected_archive_root):
                destination = package_destination / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents)
                copied.append(str(destination.relative_to(temporary)))

            selection = (
                "Apache-2.0 selected; the LGPL-2.1-or-later alternative is not used"
                if package["name"] == "r-efi" and package["version"] == "6.0.0"
                else None
            )
            if selection:
                destination = package_destination / "LICENSE-Apache-2.0"
                copy_file(REPO_ROOT / "LICENSE", destination)
                copied.append(str(destination.relative_to(temporary)))
            if not copied:
                if package["name"] == "dotenv" and package["version"] == "0.15.0":
                    destination = package_destination / "LICENSE-MIT.md"
                    copy_file(dotenv_fallback, destination)
                    copied.append(str(destination.relative_to(temporary)))
                    selection = "MIT; exact upstream text restored because the crate archive omits it"
                elif apache_is_standalone_option(package["license"]):
                    destination = package_destination / "LICENSE-Apache-2.0"
                    copy_file(REPO_ROOT / "LICENSE", destination)
                    copied.append(str(destination.relative_to(temporary)))
                    selection = "Apache-2.0 selected from an OR expression because the crate archive omits a license file"
                else:
                    raise GateError(
                        "{} {} has no redistributable license text and no reviewed fallback".format(
                            package["name"], package["version"]
                        )
                    )

            source_archive = None
            if "MPL-2.0+" in identifiers:
                destination = temporary / "corresponding-source" / archive.name
                copy_file(archive, destination)
                source_archive = str(destination.relative_to(temporary))

            rust_inventory.append(
                {
                    "name": package["name"],
                    "version": package["version"],
                    "packageId": package["id"],
                    "source": package["source"],
                    "checksum": checksum,
                    "licenseExpression": package["license"],
                    "licenseFiles": copied,
                    "licenseSelection": selection,
                    "correspondingSourceArchive": source_archive,
                }
            )

        for item, root in ((native[0], blst_root), (native[1], sppark_root)):
            destination = temporary / "native" / item["name"] / "LICENSE"
            copy_file(root.resolve() / "LICENSE", destination)
            item["licenseFile"] = str(destination.relative_to(temporary))

        inventory = {
            "schemaVersion": 1,
            "cargoLockSha256": sha256(lock_path),
            "rustPackageCount": len(rust_inventory),
            "rustPackages": rust_inventory,
            "nativeInputs": native,
            "embeddedContexts": context_inventory,
        }
        inventory_path = temporary / "inventory.json"
        inventory_path.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "README.txt").write_text(
            "cuBBS-Cred release license bundle\n\n"
            "This bundle was generated fail-closed from Cargo.lock and clean, pinned BLST/SPPARK "
            "checkouts. inventory.json records package licenses, checksums, native Git identities, "
            "and embedded-context hashes. corresponding-source/ contains the exact locked crate "
            "archives for MPL-2.0+ dependencies.\n",
            encoding="utf-8",
        )
        os.rename(str(temporary), str(output))
    except Exception:
        shutil.rmtree(str(temporary))
        raise

    return len(rust_inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory (must not exist)")
    parser.add_argument(
        "--blst-root",
        type=Path,
        default=REPO_ROOT / "ref/implementations/supranational-blst",
    )
    parser.add_argument(
        "--sppark-root",
        type=Path,
        default=REPO_ROOT / "ref/implementations/supranational-sppark",
    )
    parser.add_argument(
        "--cargo-home",
        type=Path,
        default=Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))),
    )
    args = parser.parse_args()
    try:
        count = build_bundle(args.output, args.blst_root, args.sppark_root, args.cargo_home)
    except (GateError, OSError, ValueError, json.JSONDecodeError) as error:
        print("release-license gate failed: {}".format(error), file=sys.stderr)
        return 1
    print("release-license bundle created: {} ({} Rust packages)".format(args.output, count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
