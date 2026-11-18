"""Run with python -m unittest discover -s tests -p 'test_download_data.py'."""

import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import requests

from helper_functions import download_data


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous_directory = Path.cwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, previous_directory)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("train/example.txt", "sample data")
        self.archive = archive.getvalue()
        self.source = "https://example.com/dataset.zip"

    def response(self, content):
        response = Mock()
        response.content = content
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    def test_http_failure_can_be_retried(self):
        failed = self.response(b"not found")
        failed.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        with patch("helper_functions.requests.get", return_value=failed):
            with self.assertRaises(requests.HTTPError):
                download_data(self.source, "sample")
        self.assertFalse(Path("data/sample").exists())
        self.assertFalse(Path("data/dataset.zip").exists())
        with patch("helper_functions.requests.get", return_value=self.response(self.archive)) as get:
            result = download_data(self.source, "sample")
        get.assert_called_once_with(self.source, timeout=30)
        self.assertEqual((result / "train/example.txt").read_text(), "sample data")

    def test_invalid_archive_can_be_retried(self):
        with patch("helper_functions.requests.get", return_value=self.response(b"invalid zip")):
            with self.assertRaises(zipfile.BadZipFile):
                download_data(self.source, "sample")
        self.assertFalse(Path("data/sample").exists())
        with patch("helper_functions.requests.get", return_value=self.response(self.archive)):
            result = download_data(self.source, "sample")
        self.assertTrue((result / "train/example.txt").is_file())

    def test_partial_extraction_does_not_publish_incomplete_dataset(self):
        def fail_extract(archive, path):
            Path(path, "partial.txt").write_text("incomplete")
            raise OSError("extraction failed")

        with patch("helper_functions.requests.get", return_value=self.response(self.archive)):
            with patch.object(zipfile.ZipFile, "extractall", fail_extract):
                with self.assertRaises(OSError):
                    download_data(self.source, "sample")
        self.assertFalse(Path("data/sample").exists())
        self.assertEqual(list(Path("data").iterdir()), [Path("data/dataset.zip")])

    def test_success_preserves_return_value_and_source_option(self):
        for remove_source in (True, False):
            with self.subTest(remove_source=remove_source):
                destination = "nested/sample" + str(remove_source)
                with patch("helper_functions.requests.get", return_value=self.response(self.archive)):
                    result = download_data(self.source, destination, remove_source=remove_source)
                self.assertEqual(result, Path("data") / destination)
                self.assertEqual((result / "train/example.txt").read_text(), "sample data")
                self.assertEqual(Path("data/dataset.zip").exists(), not remove_source)
                with patch("helper_functions.requests.get") as get:
                    self.assertEqual(download_data(self.source, destination), result)
                    get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
