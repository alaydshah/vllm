# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native connector entry point; implementation lives in the in-tree extension.

Selectable through KVTransferConfig.kv_connector_module_path. Worker startup is
gated until the mixed attention reader and payload publication are installed.
"""

from vllm.agentic_kv.connector import AgenticKVConnector

__all__ = ["AgenticKVConnector"]
