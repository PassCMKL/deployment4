"""
Manual recovery from a bad deployed model:  python retrain/rollback.py

Restores the previous ONNX file and weights, then asks the running server to
reload (which also flushes the prediction cache). The same function runs
automatically if a fresh deploy fails its post-deploy smoke test.
"""

import json

from deploy_utils import append_history, rollback

if __name__ == "__main__":
    result = rollback()
    append_history({"trigger": "manual rollback", "outcome": "rolled_back" if result["rolled_back"] else "rollback_failed", "detail": result})
    print(json.dumps(result, indent=2))
