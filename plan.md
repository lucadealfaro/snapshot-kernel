# Snapshot Checkpointing Python Kernel - Implementation Plan

## Context

Build a checkpointing Python kernel from scratch (greenfield project — only `CLAUDE.md` exists). The kernel stores immutable execution states (snapshots of variables, modules, timestamps) and creates new states by executing code against existing ones. It exposes a REST API via Bottle + Cheroot for multi-threaded access.

## Package Structure

```
ldakernel/
├── pyproject.toml
└── snapshot_kernel/
    ├── __init__.py
    ├── kernel.py      # Core kernel logic
    └── main.py        # Bottle REST API server
```

## Step 1: Create `pyproject.toml`

Minimal package config with only two external dependencies: `bottle` and `cheroot`.

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "snapshot-kernel"
version = "0.1.0"
dependencies = ["bottle", "cheroot"]
requires-python = ">=3.9"
```

## Step 2: Create `snapshot_kernel/__init__.py`

Simple package marker, exports `SnapshotKernel`.

## Step 3: Implement `snapshot_kernel/kernel.py`

This is the core file. All imports are from the standard library (`ast`, `copy`, `ctypes`, `datetime`, `io`, `sys`, `threading`, `traceback`, `types`, `uuid`).

### State representation

- **`State` class**: Holds `name`, `namespace` (dict), `timestamp` (ISO 8601 string).
- **`_snapshot_namespace(namespace)` helper**: Iterates over namespace dict. Skips dunder keys (`__builtins__`, etc.). Stores module references directly (modules are singletons, can't be deep-copied). Deep-copies everything else via `copy.deepcopy`, falling back to direct reference if that fails.
- On `reset()` and `__init__`, an `"initial"` state is created with an empty namespace.

### Thread-safe output capture

Since cheroot runs multiple requests in parallel, `sys.stdout`/`sys.stderr` are process-global and must not be naively redirected. Solution:

- **`ThreadSafeWriter` class**: Wraps the original `sys.stdout`/`sys.stderr`. Uses `threading.local()` to store per-thread `StringIO` buffers. If a buffer exists for the current thread, writes go there; otherwise, writes go to the original stream.
- Installed once in `SnapshotKernel.__init__()` as replacements for `sys.stdout` and `sys.stderr`.
- Each `execute()` call sets `_thread_local.buffer = StringIO()` before execution and reads it after.

### SnapshotKernel class

Fields:
- `_states`: `dict[str, State]` — state storage
- `_lock`: `threading.Lock` — protects `_states`
- `_executions`: `dict[str, Thread]` — maps `exec_id` to the thread running it
- `_exec_lock`: `threading.Lock` — protects `_executions`

Methods:

- **`reset()`**: Clears `_states`, creates `"initial"` state with empty namespace.
- **`list_states()`**: Returns list of state names (under `_lock`).
- **`get_state(state_name)`**: Returns serializable dict with `name`, `timestamp`, and `variables` (each variable as `{type, repr}`). Returns `None` if not found.
- **`delete_state(state_name)`**: Removes state from dict (under `_lock`).
- **`execute(code, exec_id, state_name, new_state_name=None)`**:
  1. Generate `new_state_name` via `uuid.uuid4().hex` if not provided.
  2. Snapshot the source state's namespace (under `_lock`, briefly).
  3. Register `exec_id -> current_thread` in `_executions`.
  4. Set up per-thread stdout/stderr capture via `ThreadSafeWriter`.
  5. Parse code with `ast.parse()`. If the last statement is an `ast.Expr`, split it off: compile preceding statements as `exec`, compile last expression as `eval` to capture its return value (mimicking Jupyter behavior).
  6. Execute via `exec()`/`eval()` on the copied namespace. Catch `KeyboardInterrupt` (from interrupt) and general `Exception`.
  7. Build Jupyter-style output list: `stream` (stdout/stderr), `execute_result` (last expr value as `text/plain`), `error` (exception info with traceback).
  8. On success, snapshot the modified namespace and store as new state. On error, do **not** store a new state.
  9. Unregister `exec_id`. Return `{output, state_name, error}`.
- **`interrupt(exec_id)`**: Looks up the thread in `_executions`, uses `ctypes.pythonapi.PyThreadState_SetAsyncExc` to raise `KeyboardInterrupt` in that thread. This gets caught by the `except KeyboardInterrupt` in `execute()`.

### Concurrency model

- The `_lock` is only held briefly (to read/write the state dict), so multiple `execute()` calls run in parallel on independent namespace copies.
- Each execution runs in a cheroot worker thread — no need to spawn our own threads.
- `ThreadSafeWriter` ensures stdout/stderr capture is isolated per thread.

## Step 4: Implement `snapshot_kernel/main.py`

### REST API routes

| Method   | Path              | Kernel Method    | Body (JSON)                                          |
|----------|-------------------|------------------|------------------------------------------------------|
| `POST`   | `/execute`        | `execute()`      | `{code, exec_id, state_name, new_state_name?}`      |
| `GET`    | `/states`         | `list_states()`  | —                                                    |
| `GET`    | `/states/<name>`  | `get_state()`    | —                                                    |
| `DELETE` | `/states/<name>`  | `delete_state()` | —                                                    |
| `POST`   | `/reset`          | `reset()`        | —                                                    |
| `POST`   | `/interrupt`      | `interrupt()`    | `{exec_id}`                                          |

### Authentication

A `@app.hook('before_request')` checks the `token` URL parameter against the configured secret. Returns 401 on mismatch.

### Server startup

Custom entry point with `argparse`:

```bash
python -m snapshot_kernel.main --bind 0.0.0.0:8080 --token=SECRET
```

Parses `--bind` (default `127.0.0.1:8080`) and `--token` (required). Runs the Bottle app with `server='cheroot'`.

## Known Limitations

- `copy.deepcopy` will fail on some objects (file handles, generators); fallback is to store a reference (shared between states).
- `PyThreadState_SetAsyncExc` cannot interrupt blocking C extensions (e.g., long numpy ops). This is an inherent Python limitation.
- Each state stores a full namespace copy — memory grows with state count and variable size.

## Verification

1. Install the package: `pip install -e .`
2. Start the server: `python -m snapshot_kernel.main --bind 127.0.0.1:8080 --token=test123`
3. Test basic execution:
   ```bash
   curl -X POST 'http://127.0.0.1:8080/execute?token=test123' \
     -H 'Content-Type: application/json' \
     -d '{"code": "x = 42\nprint(x)", "exec_id": "e1", "state_name": "initial"}'
   ```
4. Test state listing: `curl 'http://127.0.0.1:8080/states?token=test123'`
5. Test state retrieval: `curl 'http://127.0.0.1:8080/states/<name>?token=test123'`
6. Test chained execution (execute against the state produced in step 3):
   ```bash
   curl -X POST 'http://127.0.0.1:8080/execute?token=test123' \
     -H 'Content-Type: application/json' \
     -d '{"code": "x + 1", "exec_id": "e2", "state_name": "<name_from_step_3>"}'
   ```
7. Test interrupt: start a long-running execution, then POST to `/interrupt`.
8. Test auth: make a request without token or with wrong token, verify 401.
9. Test reset: POST to `/reset`, verify states are cleared except new `initial`.
