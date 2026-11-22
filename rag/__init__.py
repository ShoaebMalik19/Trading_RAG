"""Trading RAG package.

ChromaDB imports an OpenTelemetry gRPC exporter at import time, only for its own telemetry.
On some Windows machines (Smart App Control / WDAC) the grpc DLL can be blocked, which would
crash the import. If grpc cannot load, register a no-op exporter so ChromaDB still works.
Hosted Linux environments load grpc normally and never take this path.
"""

import os
import sys
import types

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

try:  # pragma: no cover - environment dependent
    import grpc  # noqa: F401
except ImportError:
    _name = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)

        class OTLPSpanExporter:  # minimal no-op stand-in
            def __init__(self, *args, **kwargs):
                pass

            def export(self, *args, **kwargs):
                return None

            def shutdown(self, *args, **kwargs):
                return None

            def force_flush(self, *args, **kwargs):
                return True

        _stub.OTLPSpanExporter = OTLPSpanExporter
        sys.modules[_name] = _stub
