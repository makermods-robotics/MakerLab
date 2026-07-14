# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smoke test — confirms the package is installed and importable.

Populated tests will land later. This file exists so the tests/ structure
is in place from day one.
"""

import makerlab
from makerlab.server import app


def test_makerlab_has_version():
    assert isinstance(makerlab.__version__, str)
    assert makerlab.__version__


def test_server_app_has_routes():
    assert len(app.routes) > 0
