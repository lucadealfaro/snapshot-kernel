"""LDA Checkpointing Python Kernel.

Stores immutable execution states (snapshots of variables, modules, timestamps)
and creates new states by executing code against existing ones.
"""

import ast
import copy
import ctypes
import datetime
import io
import sys
import threading
import traceback
import types
import uuid


def _snapshot_namespace(namespace):
    """Create a snapshot of a namespace dict.

    Modules are stored by reference (they are singletons).
    Everything else is deep-copied when possible, with a fallback
    to storing a direct reference for non-copyable objects.
    """
    snapshot = {}
    for key, value in namespace.items():
        if key.startswith("__") and key.endswith("__"):
            continue
        if isinstance(value, types.ModuleType):
            snapshot[key] = value
        else:
            try:
                snapshot[key] = copy.deepcopy(value)
            except Exception:
                snapshot[key] = value
    return snapshot


class State:
    """An immutable snapshot of an execution environment."""

    def __init__(self, name, namespace):
        self.name = name
        self.namespace = namespace
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()


class ThreadSafeWriter:
    """A writer that redirects writes to per-thread StringIO buffers.

    When a per-thread buffer is set, writes go there. Otherwise, writes
    go to the original stream. This allows capturing stdout/stderr per
    execution thread without interfering with other threads.
    """

    def __init__(self, original):
        self._original = original
        self._local = threading.local()

    def set_buffer(self, buf):
        """Set a StringIO buffer for the current thread."""
        self._local.buffer = buf

    def clear_buffer(self):
        """Remove the buffer for the current thread."""
        self._local.buffer = None

    def get_buffer(self):
        """Return the current thread's buffer, or None."""
        return getattr(self._local, "buffer", None)

    def write(self, text):
        buf = self.get_buffer()
        if buf is not None:
            buf.write(text)
        else:
            self._original.write(text)

    def flush(self):
        buf = self.get_buffer()
        if buf is not None:
            buf.flush()
        else:
            self._original.flush()

    # Pass through attributes like encoding, isatty, etc.
    def __getattr__(self, name):
        return getattr(self._original, name)


class LDAKernel:
    """Checkpointing Python kernel that stores and forks execution states."""

    def __init__(self):
        self._states = {}
        self._lock = threading.Lock()
        self._executions = {}
        self._exec_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self.reset()

    def _ensure_writers(self):
        """Install ThreadSafeWriter wrappers if they are not already in place.

        This is called at the start of every execution so that
        external code (e.g. pytest) that replaces sys.stdout/stderr
        does not break per-thread capture.
        """
        with self._io_lock:
            if not isinstance(sys.stdout, ThreadSafeWriter):
                sys.stdout = ThreadSafeWriter(sys.stdout)
            if not isinstance(sys.stderr, ThreadSafeWriter):
                sys.stderr = ThreadSafeWriter(sys.stderr)

    def reset(self):
        """Clear all states and create a fresh 'initial' state."""
        with self._lock:
            self._states.clear()
            self._states["initial"] = State("initial", {})

    def list_states(self):
        """Return a list of all stored state names."""
        with self._lock:
            return list(self._states.keys())

    def get_state(self, state_name):
        """Return a serializable dict describing the given state, or None."""
        with self._lock:
            state = self._states.get(state_name)
        if state is None:
            return None
        variables = {}
        for key, value in state.namespace.items():
            variables[key] = {
                "type": type(value).__name__,
                "repr": repr(value),
            }
        return {
            "name": state.name,
            "timestamp": state.timestamp,
            "variables": variables,
        }

    def delete_state(self, state_name):
        """Remove the named state. Returns True if it existed."""
        with self._lock:
            return self._states.pop(state_name, None) is not None

    def execute(self, code, exec_id, state_name, new_state_name=None):
        """Execute code against a state and store the resulting state.

        Returns a dict with keys: output, state_name, error.
        """
        if new_state_name is None:
            new_state_name = uuid.uuid4().hex

        # Snapshot the source state's namespace.
        with self._lock:
            source = self._states.get(state_name)
        if source is None:
            return {
                "output": [],
                "state_name": None,
                "error": {"ename": "StateNotFound", "evalue": state_name, "traceback": []},
            }
        namespace = _snapshot_namespace(source.namespace)
        namespace["__builtins__"] = __builtins__

        # Register this execution for possible interruption.
        with self._exec_lock:
            self._executions[exec_id] = threading.current_thread().ident

        # Ensure ThreadSafeWriter wrappers are in place.
        self._ensure_writers()

        # Set up per-thread output capture.
        stdout_writer = sys.stdout
        stderr_writer = sys.stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        stdout_writer.set_buffer(stdout_buf)
        stderr_writer.set_buffer(stderr_buf)

        output = []
        error = None
        last_expr_value = None

        try:
            # Parse the code and split out the last expression if applicable.
            tree = ast.parse(code)
            last_expr_node = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr_node = tree.body.pop()

            # Execute all statements except the last expression.
            if tree.body:
                exec(compile(tree, "<cell>", "exec"), namespace)

            # Evaluate the last expression to capture its value.
            if last_expr_node is not None:
                expr_code = compile(
                    ast.Expression(body=last_expr_node.value), "<cell>", "eval"
                )
                last_expr_value = eval(expr_code, namespace)

        except KeyboardInterrupt:
            error = {
                "ename": "KeyboardInterrupt",
                "evalue": "",
                "traceback": ["KeyboardInterrupt"],
            }
        except Exception as exc:
            error = {
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            }
        finally:
            # Collect captured output.
            stdout_writer.clear_buffer()
            stderr_writer.clear_buffer()

            stdout_text = stdout_buf.getvalue()
            stderr_text = stderr_buf.getvalue()

            if stdout_text:
                output.append({"output_type": "stream", "name": "stdout", "text": stdout_text})
            if stderr_text:
                output.append({"output_type": "stream", "name": "stderr", "text": stderr_text})

            if last_expr_value is not None:
                output.append({
                    "output_type": "execute_result",
                    "data": {"text/plain": repr(last_expr_value)},
                })

            # Unregister execution.
            with self._exec_lock:
                self._executions.pop(exec_id, None)

        # On success, store the new state; on error, do not.
        result_state_name = None
        if error is None:
            new_ns = _snapshot_namespace(namespace)
            new_state = State(new_state_name, new_ns)
            with self._lock:
                self._states[new_state_name] = new_state
            result_state_name = new_state_name

        return {
            "output": output,
            "state_name": result_state_name,
            "error": error,
        }

    def interrupt(self, exec_id):
        """Interrupt a running execution by raising KeyboardInterrupt in its thread."""
        with self._exec_lock:
            thread_ident = self._executions.get(exec_id)
        if thread_ident is None:
            return False
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_ident),
            ctypes.py_object(KeyboardInterrupt),
        )
        return res == 1
