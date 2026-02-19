"""Bottle REST API server for the Snapshot Kernel."""

import argparse
import json

import bottle

from .kernel import SnapshotKernel

app = bottle.Bottle()
kernel = SnapshotKernel()
_token = None


@app.hook("before_request")
def check_auth():
    """Verify the token URL parameter on every request."""
    if _token is not None:
        if bottle.request.params.get("token") != _token:
            bottle.abort(401, "Invalid or missing token.")


@app.post("/execute")
def execute():
    """Execute code against a state."""
    body = bottle.request.json
    if not body:
        bottle.abort(400, "Request body must be JSON.")
    code = body.get("code", "")
    exec_id = body.get("exec_id")
    state_name = body.get("state_name")
    new_state_name = body.get("new_state_name")
    if exec_id is None or state_name is None:
        bottle.abort(400, "exec_id and state_name are required.")
    result = kernel.execute(code, exec_id, state_name, new_state_name)
    bottle.response.content_type = "application/json"
    return json.dumps(result)


@app.get("/states")
def list_states():
    """Return all stored state names."""
    bottle.response.content_type = "application/json"
    return json.dumps({"states": kernel.list_states()})


@app.get("/states/<name>")
def get_state(name):
    """Return details of a single state."""
    state = kernel.get_state(name)
    if state is None:
        bottle.abort(404, "State not found.")
    bottle.response.content_type = "application/json"
    return json.dumps(state)


@app.delete("/states/<name>")
def delete_state(name):
    """Delete a state."""
    if kernel.delete_state(name):
        return json.dumps({"deleted": name})
    bottle.abort(404, "State not found.")


@app.post("/reset")
def reset():
    """Reset the kernel to its initial state."""
    kernel.reset()
    bottle.response.content_type = "application/json"
    return json.dumps({"status": "ok"})


@app.post("/interrupt")
def interrupt():
    """Interrupt a running execution."""
    body = bottle.request.json
    if not body or "exec_id" not in body:
        bottle.abort(400, "exec_id is required.")
    success = kernel.interrupt(body["exec_id"])
    bottle.response.content_type = "application/json"
    return json.dumps({"interrupted": success})


def main():
    """Entry point: parse arguments and start the Cheroot-backed server."""
    parser = argparse.ArgumentParser(description="Snapshot Kernel Server")
    parser.add_argument(
        "--bind", default="127.0.0.1:8080",
        help="Address to bind to (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--token", required=True,
        help="Secret token for request authentication",
    )
    parser.add_argument(
        "--display-max-rows", type=int, default=200,
        help="Max rows shown for Pandas/Polars DataFrames (default: %(default)s)",
    )
    parser.add_argument(
        "--display-max-columns", type=int, default=100,
        help="Max columns shown for Pandas/Polars DataFrames (default: %(default)s)",
    )
    parser.add_argument(
        "--display-max-colwidth", type=int, default=128,
        help="Max column width for Pandas/Polars DataFrames (default: %(default)s)",
    )
    args = parser.parse_args()

    global _token, kernel
    _token = args.token
    kernel = SnapshotKernel(
        display_max_rows=args.display_max_rows,
        display_max_columns=args.display_max_columns,
        display_max_colwidth=args.display_max_colwidth,
    )

    host, port = args.bind.rsplit(":", 1)
    bottle.run(app, server="cheroot", host=host, port=int(port))


if __name__ == "__main__":
    main()
