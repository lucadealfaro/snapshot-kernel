"""Integration tests for the Bottle REST API — starts a real Cheroot server."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

import bottle
from snapshot_kernel import main as main_module


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

TOKEN = "test123"


def _free_port():
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port, timeout=5):
    """Block until the server is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Server never started on port {port}")


def _request(port, method, path, body=None, token=TOKEN):
    """Send an HTTP request and return (status_code, parsed_json_or_None)."""
    url = f"http://127.0.0.1:{port}{path}"
    if token is not None:
        url += f"?token={token}"

    data = None
    if body is not None:
        data = json.dumps(body).encode()

    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return resp.status, None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return exc.code, None


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def server():
    """Start the Cheroot-backed Bottle server once for the whole module."""
    port = _free_port()
    main_module._token = TOKEN

    def run():
        bottle.run(main_module.app, server="cheroot",
                    host="127.0.0.1", port=port, quiet=True)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    _wait_for_port(port)
    return port


@pytest.fixture(autouse=True)
def _reset_kernel(server):
    """Reset the kernel before every test so tests are independent."""
    _request(server, "POST", "/reset")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_list_states(server):
    """GET /states returns only 'initial' after reset."""
    status, body = _request(server, "GET", "/states")
    assert status == 200
    assert body["states"] == ["initial"]


def test_execute_and_retrieve(server):
    """POST /execute creates a state retrievable via GET /states/<name>."""
    status, body = _request(server, "POST", "/execute", {
        "code": "x = 42",
        "exec_id": "e1",
        "state_name": "initial",
        "new_state_name": "s1",
    })
    assert status == 200
    assert body["state_name"] == "s1"
    assert body["error"] is None

    status, state = _request(server, "GET", "/states/s1")
    assert status == 200
    assert state["variables"]["x"]["repr"] == "42"


def test_chained_execution(server):
    """Two sequential executions where the second reads the first's variable."""
    _request(server, "POST", "/execute", {
        "code": "x = 10",
        "exec_id": "e1",
        "state_name": "initial",
        "new_state_name": "s1",
    })
    status, body = _request(server, "POST", "/execute", {
        "code": "x + 5",
        "exec_id": "e2",
        "state_name": "s1",
    })
    assert status == 200
    expr_items = [o for o in body["output"]
                  if o["output_type"] == "execute_result"]
    assert expr_items[0]["data"]["text/plain"] == "15"


def test_delete_state(server):
    """DELETE /states/<name> removes the state."""
    _request(server, "POST", "/execute", {
        "code": "1",
        "exec_id": "e1",
        "state_name": "initial",
        "new_state_name": "tmp",
    })
    status, _ = _request(server, "DELETE", "/states/tmp")
    assert status == 200

    status, _ = _request(server, "GET", "/states/tmp")
    assert status == 404


def test_reset(server):
    """POST /reset removes all derived states."""
    _request(server, "POST", "/execute", {
        "code": "1",
        "exec_id": "e1",
        "state_name": "initial",
        "new_state_name": "s1",
    })
    _request(server, "POST", "/reset")

    status, body = _request(server, "GET", "/states")
    assert status == 200
    assert body["states"] == ["initial"]


def test_auth_wrong_token(server):
    """A request with the wrong token gets 401."""
    status, _ = _request(server, "GET", "/states", token="wrong")
    assert status == 401


def test_auth_missing_token(server):
    """A request with no token gets 401."""
    status, _ = _request(server, "GET", "/states", token=None)
    assert status == 401


def test_state_not_found(server):
    """GET /states/<name> returns 404 for unknown names."""
    status, _ = _request(server, "GET", "/states/nope")
    assert status == 404


def test_execute_error(server):
    """POST /execute with bad code returns an error payload."""
    status, body = _request(server, "POST", "/execute", {
        "code": "1/0",
        "exec_id": "e1",
        "state_name": "initial",
    })
    assert status == 200
    assert body["error"] is not None
    assert body["error"]["ename"] == "ZeroDivisionError"
    assert body["state_name"] is None


def test_interrupt(server):
    """POST /interrupt stops a long-running execution."""
    result_holder = {}
    code = "import time\nfor _ in range(200):\n    time.sleep(0.1)"

    def run_long():
        _, body = _request(server, "POST", "/execute", {
            "code": code,
            "exec_id": "long_run",
            "state_name": "initial",
        })
        result_holder["body"] = body

    t = threading.Thread(target=run_long)
    t.start()

    # Give the execution a moment to enter the loop.
    time.sleep(1.0)
    _request(server, "POST", "/interrupt", {"exec_id": "long_run"})
    t.join(timeout=5)

    body = result_holder.get("body")
    assert body is not None, "Execution request did not finish"
    assert body["error"] is not None
    assert body["error"]["ename"] == "KeyboardInterrupt"


# ------------------------------------------------------------------
# multistate_execute
# ------------------------------------------------------------------

def test_multistate_execute_basic(server):
    """POST /multistate_execute round trip: create states, access via aliases."""
    _request(server, "POST", "/execute", {
        "code": "x = 10",
        "exec_id": "e1",
        "state_name": "initial",
        "new_state_name": "s1",
    })
    _request(server, "POST", "/execute", {
        "code": "y = 20",
        "exec_id": "e2",
        "state_name": "initial",
        "new_state_name": "s2",
    })

    status, body = _request(server, "POST", "/multistate_execute", {
        "code": "a.x + b.y",
        "exec_id": "me1",
        "state_mapping": {"a": "s1", "b": "s2"},
    })
    assert status == 200
    assert body["error"] is None
    assert body["state_name"] is None
    expr_items = [o for o in body["output"]
                  if o["output_type"] == "execute_result"]
    assert len(expr_items) == 1
    assert expr_items[0]["data"]["text/plain"] == "30"


def test_multistate_execute_missing_state(server):
    """POST /multistate_execute with a nonexistent state returns StateNotFound."""
    status, body = _request(server, "POST", "/multistate_execute", {
        "code": "a.x",
        "exec_id": "me1",
        "state_mapping": {"a": "no_such_state"},
    })
    assert status == 200
    assert body["error"] is not None
    assert body["error"]["ename"] == "StateNotFound"


def test_multistate_execute_missing_fields(server):
    """POST /multistate_execute without required fields returns 400."""
    status, _ = _request(server, "POST", "/multistate_execute", {
        "code": "1",
    })
    assert status == 400

    status, _ = _request(server, "POST", "/multistate_execute", {
        "code": "1",
        "exec_id": "me1",
    })
    assert status == 400
