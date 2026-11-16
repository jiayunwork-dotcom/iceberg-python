# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""A local ``PyArrowFileIO`` that records which files are opened and created.

The tests use it to prove that data files which the scan planning prunes are
never opened, and that a failed operation does not touch unrelated files.
"""

from typing import List

from pyiceberg.io.pyarrow import PyArrowFileIO, PyArrowLocalFileSystem

OPENED_PATHS: List[str] = []
CREATED_PATHS: List[str] = []


def reset() -> None:
    OPENED_PATHS.clear()
    CREATED_PATHS.clear()


def opened_data_files() -> List[str]:
    return [path for path in OPENED_PATHS if path.endswith(".parquet")]


def created_data_files() -> List[str]:
    return [path for path in CREATED_PATHS if path.endswith(".parquet")]


class RecordingLocalFileSystem(PyArrowLocalFileSystem):
    def open_input_file(self, path: str, *args: object, **kwargs: object) -> object:
        OPENED_PATHS.append(path)
        return super().open_input_file(path, *args, **kwargs)  # type: ignore[arg-type]

    def open_input_stream(self, path: str, *args: object, **kwargs: object) -> object:
        OPENED_PATHS.append(path)
        return super().open_input_stream(path, *args, **kwargs)  # type: ignore[arg-type]

    def open_output_stream(self, path: str, *args: object, **kwargs: object) -> object:
        CREATED_PATHS.append(path)
        return super().open_output_stream(path, *args, **kwargs)  # type: ignore[arg-type]


class RecordingFileIO(PyArrowFileIO):
    def _initialize_local_fs(self) -> PyArrowLocalFileSystem:  # type: ignore[override]
        return RecordingLocalFileSystem()
