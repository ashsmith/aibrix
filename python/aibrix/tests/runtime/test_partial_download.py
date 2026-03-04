# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest

import aibrix.runtime.artifact_service as artifact_service_module
from aibrix.runtime.artifact_service import ArtifactDelegationService
from aibrix.runtime.downloaders import ArtifactDownloader


class _TestDownloader(ArtifactDownloader):
    async def download(self, source_url, local_path, credentials=None):
        return local_path


class TestAtomicDownload:
    def test_successful_download_replaces_part_file(self, tmp_path):
        dest = str(tmp_path / "model.bin")
        dl = _TestDownloader()

        def _write(path):
            with open(path, "w") as f:
                f.write("data")

        dl._atomic_download(dest, _write)

        assert os.path.isfile(dest)
        assert not os.path.exists(dest + ".part")
        with open(dest) as f:
            assert f.read() == "data"

    def test_failed_download_cleans_up_part_file(self, tmp_path):
        dest = str(tmp_path / "model.bin")
        dl = _TestDownloader()

        def _fail(path):
            with open(path, "w") as f:
                f.write("partial")
            raise RuntimeError("network error")

        with pytest.raises(RuntimeError, match="network error"):
            dl._atomic_download(dest, _fail)

        assert not os.path.exists(dest)
        assert not os.path.exists(dest + ".part")

    def test_does_not_leave_final_on_failure(self, tmp_path):
        dest = str(tmp_path / "model.bin")
        dl = _TestDownloader()

        # Pre-existing file should remain untouched
        with open(dest, "w") as f:
            f.write("old")

        def _fail(path):
            with open(path, "w") as f:
                f.write("new-partial")
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError):
            dl._atomic_download(dest, _fail)

        # Old file is still intact
        with open(dest) as f:
            assert f.read() == "old"


class TestCompletionMarker:
    def test_write_and_check_marker(self, tmp_path):
        dl = _TestDownloader()
        local = str(tmp_path / "adapter")
        os.makedirs(local)

        assert not ArtifactDownloader.is_download_complete(local)

        dl._write_completion_marker(local)

        assert ArtifactDownloader.is_download_complete(local)
        marker = os.path.join(local, ArtifactDownloader.DOWNLOAD_COMPLETE_MARKER)
        assert os.path.isfile(marker)

    def test_missing_directory_returns_false(self, tmp_path):
        assert not ArtifactDownloader.is_download_complete(
            str(tmp_path / "nonexistent")
        )


class TestArtifactServiceRedownload:
    @pytest.mark.asyncio
    async def test_incomplete_directory_triggers_redownload(
        self, tmp_path, monkeypatch
    ):
        """If a directory exists but lacks the completion marker, it should be
        removed and the artifact re-downloaded."""
        service = ArtifactDelegationService(local_dir=str(tmp_path))
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        # Write a file but NO completion marker -> simulates crash
        (adapter_dir / "partial.bin").write_text("incomplete")

        download_called = False

        class RedownloadFakeDownloader(ArtifactDownloader):
            async def download(self, source_url, local_path, credentials=None):
                nonlocal download_called
                download_called = True
                os.makedirs(local_path, exist_ok=True)
                with open(os.path.join(local_path, "model.bin"), "w") as f:
                    f.write("complete")
                self._write_completion_marker(local_path)
                return local_path

        monkeypatch.setattr(
            artifact_service_module,
            "get_downloader",
            lambda _url: RedownloadFakeDownloader(),
        )

        result = await service.download_artifact(
            "s3://bucket/adapter/", "adapter", credentials={}
        )

        assert download_called
        assert os.path.isfile(os.path.join(result, "model.bin"))
        assert ArtifactDownloader.is_download_complete(result)
        # The partial file from the crashed download should be gone
        assert not os.path.exists(os.path.join(result, "partial.bin"))

    @pytest.mark.asyncio
    async def test_complete_directory_skips_redownload(self, tmp_path, monkeypatch):
        """If a directory exists WITH the completion marker, skip download."""
        service = ArtifactDelegationService(local_dir=str(tmp_path))
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "model.bin").write_text("data")

        dl = _TestDownloader()
        dl._write_completion_marker(str(adapter_dir))

        download_called = False

        class NeverCalledDownloader(ArtifactDownloader):
            async def download(self, source_url, local_path, credentials=None):
                nonlocal download_called
                download_called = True
                return local_path

        monkeypatch.setattr(
            artifact_service_module,
            "get_downloader",
            lambda _url: NeverCalledDownloader(),
        )

        result = await service.download_artifact(
            "s3://bucket/adapter/", "adapter", credentials={}
        )

        assert not download_called
        assert result == str(adapter_dir)

    @pytest.mark.asyncio
    async def test_empty_path_triggers_fresh_download(self, tmp_path, monkeypatch):
        """If the adapter directory does not exist at all, download normally."""
        service = ArtifactDelegationService(local_dir=str(tmp_path))

        download_called = False

        class FreshDownloader(ArtifactDownloader):
            async def download(self, source_url, local_path, credentials=None):
                nonlocal download_called
                download_called = True
                os.makedirs(local_path, exist_ok=True)
                with open(os.path.join(local_path, "model.bin"), "w") as f:
                    f.write("full")
                self._write_completion_marker(local_path)
                return local_path

        monkeypatch.setattr(
            artifact_service_module,
            "get_downloader",
            lambda _url: FreshDownloader(),
        )

        result = await service.download_artifact(
            "s3://bucket/adapter/", "adapter", credentials={}
        )

        assert download_called
        assert ArtifactDownloader.is_download_complete(result)
