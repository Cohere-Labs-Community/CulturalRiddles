"""Consumer-visible integrity and extraction boundaries for public releases."""

from __future__ import annotations

import hashlib
import json
import stat
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from scripts import release_support
from scripts.build_release import ROOT_NAME, release_files
from scripts.verify_release import _check_bundle_files, _safe_extract


def _approved_root(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "data/canonical/input.csv"
    path.parent.mkdir(parents=True)
    path.write_text("riddle_id,score\nr1,1\n", encoding="utf-8")
    release_support.freeze(tmp_path, "test")
    return tmp_path, path


def test_validate_data_rejects_missing_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        release_support.validate_data(tmp_path)


@pytest.mark.parametrize("mutation", ["schema", "content", "missing", "unexpected", "duplicate"])
def test_validate_data_rejects_unreviewed_changes(tmp_path: Path, mutation: str) -> None:
    root, path = _approved_root(tmp_path)
    if mutation == "schema":
        path.write_text("riddle_id,score,unexpected\nr1,1,x\n")
    elif mutation == "content":
        path.write_text("riddle_id,score\nr1,0\n")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "unexpected":
        (path.parent / "new.csv").write_text("riddle_id\nr2\n")
    else:
        path.write_text("riddle_id,score\nr1,1\nr1,0\n")
    with pytest.raises(ValueError):
        release_support.validate_data(root)


def test_privacy_checks_all_records_and_nested_keys_before_freezing(tmp_path: Path) -> None:
    path = tmp_path / "data/canonical/input.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"riddle_id": "r1", "score": 1},
        {"riddle_id": "r2", "score": 2, "nested": {"Respondent Email": "private@example.test"}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        release_support.freeze(tmp_path, "test")
    assert not (tmp_path / release_support.CONTRACT).exists()


@pytest.mark.parametrize("mutation", ["missing", "changed", "unexpected"])
def test_file_manifest_rejects_unapproved_artifacts(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "data/derived/tables/coverage.tex"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("expected table\n")
    manifest = {
        "files": [
            {
                "path": "data/derived/tables/coverage.tex",
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
    }
    if mutation == "missing":
        artifact.unlink()
    elif mutation == "changed":
        artifact.write_text("changed table\n")
    else:
        (artifact.parent / "unexpected.tex").write_text("unreviewed\n")
    with pytest.raises(ValueError):
        _check_bundle_files(tmp_path, manifest)


@pytest.mark.parametrize(
    "member",
    [
        "unmanifested.txt",
        "../escape.txt",
        f"{ROOT_NAME}/../escape.txt",
        f"{ROOT_NAME}/data/canonical/../../../escape.txt",
        f"{ROOT_NAME}\\..\\escape.txt",
        "/escape.txt",
    ],
)
def test_archive_rejects_escape_before_writing(tmp_path: Path, member: str) -> None:
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{ROOT_NAME}/README.md", "valid first member")
        archive.writestr(member, "unexpected member")
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError):
        _safe_extract(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_archive_rejects_symlink_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "release.zip"
    link = zipfile.ZipInfo(f"{ROOT_NAME}/link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(link, "../../escape")
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError):
        _safe_extract(archive, destination)
    assert not destination.exists()


@pytest.fixture(
    params=[
        "data/submissions/reviewed/community.xlsx",
        "data/submissions/pending.json",
        "data/submissions/contributor_notes.xlsx",
        "data/derived/submissions/review.json",
        "data/intake/respondents.csv",
        "data/custom-export.csv",
        "data/runs/private-run.json",
    ]
)
def private_data_path(request: pytest.FixtureRequest) -> str:
    return request.param


def test_release_omits_private_data(tmp_path: Path, private_data_path: str) -> None:
    private = tmp_path / private_data_path
    private.parent.mkdir(parents=True)
    private.write_text("private intake")
    public = tmp_path / "data/canonical/public.csv"
    public.parent.mkdir(parents=True)
    public.write_text("riddle_id\nr1\n")
    assert release_files(tmp_path) == [public]


def test_archive_rejects_manifested_private_data_before_writing(
    tmp_path: Path, private_data_path: str
) -> None:
    content = b"private intake"
    manifest = {
        "files": [{"path": private_data_path, "sha256": hashlib.sha256(content).hexdigest()}]
    }
    bundle = tmp_path / "release.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(f"{ROOT_NAME}/release_manifest.json", json.dumps(manifest))
        archive.writestr(f"{ROOT_NAME}/{private_data_path}", content)
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(bundle) as archive, pytest.raises(ValueError, match="Unapproved"):
        _safe_extract(archive, destination)
    assert not destination.exists()


def test_file_manifest_cannot_approve_private_data(tmp_path: Path, private_data_path: str) -> None:
    private = tmp_path / private_data_path
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private intake")
    manifest = {
        "files": [
            {"path": private_data_path, "sha256": hashlib.sha256(private.read_bytes()).hexdigest()}
        ]
    }
    with pytest.raises(ValueError, match="Unapproved"):
        _check_bundle_files(tmp_path, manifest)


def test_public_data_survives_release_inventory_and_extraction(tmp_path: Path) -> None:
    source = tmp_path / "source"
    payloads = {
        "data/canonical/riddles.csv": b"riddle_id\nr1\n",
        "data/reference/topics.json": b'{"topics": []}\n',
        "data/derived/accuracy/model_accuracy_matrix.csv": b"model,community\nm,1\n",
        "data/manifests/input_contract.json": b"{}\n",
        "data/corrections.csv": b"riddle_id,correction\n",
        "data/README.md": b"Public data\n",
        "data/LICENSE": b"License\n",
        "data/submissions/submission_template.xlsx": (
            release_support.ROOT / "data/submissions/submission_template.xlsx"
        ).read_bytes(),
    }
    for relative, content in payloads.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    files = release_files(source)
    assert {path.relative_to(source).as_posix() for path in files} == payloads.keys()
    manifest = {
        "files": [
            {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}
            for relative, content in payloads.items()
        ]
    }
    bundle = tmp_path / "release.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in files:
            archive.write(path, f"{ROOT_NAME}/{path.relative_to(source).as_posix()}")
        archive.writestr(f"{ROOT_NAME}/release_manifest.json", json.dumps(manifest))
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(bundle) as archive:
        _safe_extract(archive, destination)
    root = destination / ROOT_NAME
    _check_bundle_files(root, manifest)
    for relative, content in payloads.items():
        assert (root / relative).read_bytes() == content


def test_shipped_submission_template_has_anonymous_document_properties() -> None:
    template = release_support.ROOT / "data/submissions/submission_template.xlsx"
    with zipfile.ZipFile(template) as archive:
        properties = ET.fromstring(archive.read("docProps/core.xml"))
    for name in (
        "{http://purl.org/dc/elements/1.1/}creator",
        "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy",
    ):
        assert not (properties.findtext(name) or "").strip()
