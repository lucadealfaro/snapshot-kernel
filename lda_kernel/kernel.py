"""LDA Checkpointing Python Kernel.

Stores immutable execution states (snapshots of variables, modules, timestamps)
and creates new states by executing code against existing ones.
"""

import ast
import base64
import copy
import ctypes
import datetime
import io
import os
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


def _format_object(obj):
    """Extract all available rich representations from a Python object.

    Returns (data_dict, metadata_dict) where data_dict maps MIME types
    to their string content.  Binary representations (PNG, JPEG, PDF) are
    base64-encoded.
    """
    data = {}
    metadata = {}

    # Always provide text/plain.
    try:
        data["text/plain"] = repr(obj)
    except Exception:
        data["text/plain"] = "<repr failed>"

    # Prefer _repr_mimebundle_() when available.
    mimebundle_method = getattr(obj, "_repr_mimebundle_", None)
    if callable(mimebundle_method):
        try:
            result = mimebundle_method()
            if isinstance(result, tuple):
                bundle_data, bundle_meta = result
                data.update(bundle_data)
                metadata.update(bundle_meta)
            elif isinstance(result, dict):
                data.update(result)
            return data, metadata
        except Exception:
            pass

    # Fall back to individual _repr_*_() methods.
    _repr_methods = [
        ("_repr_html_", "text/html", False),
        ("_repr_markdown_", "text/markdown", False),
        ("_repr_latex_", "text/latex", False),
        ("_repr_json_", "application/json", False),
        ("_repr_svg_", "image/svg+xml", False),
        ("_repr_png_", "image/png", True),
        ("_repr_jpeg_", "image/jpeg", True),
        ("_repr_pdf_", "application/pdf", True),
    ]
    for method_name, mime_type, is_binary in _repr_methods:
        method = getattr(obj, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
        except Exception:
            continue
        if result is None:
            continue
        if is_binary and isinstance(result, bytes):
            result = base64.b64encode(result).decode("ascii")
        data[mime_type] = result

    return data, metadata


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


class DisplayCollector:
    """Thread-safe collector for display_data outputs.

    Each execution thread gets its own list of outputs so that
    concurrent executions do not interfere with each other.
    """

    def __init__(self):
        self._local = threading.local()

    def set_list(self):
        """Initialise an empty output list for the current thread."""
        self._local.outputs = []

    def clear_list(self):
        """Remove the output list for the current thread."""
        self._local.outputs = None

    def get_outputs(self):
        """Return the current thread's collected outputs."""
        return getattr(self._local, "outputs", None) or []

    def add(self, obj):
        """Format *obj* via _format_object and append a display_data entry."""
        data, metadata = _format_object(obj)
        outputs = getattr(self._local, "outputs", None)
        if outputs is not None:
            outputs.append({
                "output_type": "display_data",
                "data": data,
                "metadata": metadata,
            })

    def add_raw(self, entry):
        """Append a pre-formatted output entry directly."""
        outputs = getattr(self._local, "outputs", None)
        if outputs is not None:
            outputs.append(entry)


def _capture_figures(collector):
    """If matplotlib.pyplot is loaded, save all open figures as PNG and
    append them to *collector*, then close them."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png_b64 = base64.b64encode(buf.read()).decode("ascii")
        collector.add_raw({
            "output_type": "display_data",
            "data": {"image/png": png_b64, "text/plain": "<Figure>"},
            "metadata": {},
        })
    plt.close("all")


class LDAKernel:
    """Checkpointing Python kernel that stores and forks execution states."""

    def __init__(self):
        self._states = {}
        self._lock = threading.Lock()
        self._executions = {}
        self._exec_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._display_collector = DisplayCollector()
        os.environ.setdefault("MPLBACKEND", "Agg")
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

    def _make_display_func(self):
        """Create a ``display(*objs)`` function for use inside executed code.

        The function formats each object and appends it to the
        thread-local DisplayCollector.
        """
        collector = self._display_collector

        def display(*objs):
            for obj in objs:
                ipython_display = getattr(obj, "_ipython_display_", None)
                if callable(ipython_display):
                    ipython_display(display=display)
                else:
                    collector.add(obj)

        return display

    def _install_matplotlib_hook(self):
        """Replace ``plt.show()`` with a wrapper that captures figures.

        Returns a cleanup function that restores the original ``plt.show()``,
        or *None* if matplotlib is not loaded.
        """
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is None:
            return None
        original_show = plt.show
        collector = self._display_collector

        def _hooked_show(*args, **kwargs):
            _capture_figures(collector)

        plt.show = _hooked_show
        return lambda: setattr(plt, "show", original_show)

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

        # Set up display collector and inject display() into namespace.
        self._display_collector.set_list()
        display_func = self._make_display_func()
        namespace["display"] = display_func

        # Hook plt.show() if matplotlib is already imported.
        restore_show = self._install_matplotlib_hook()

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

            # If matplotlib was imported during execution, install hook and
            # capture any remaining open figures.
            if restore_show is None:
                restore_show = self._install_matplotlib_hook()
            _capture_figures(self._display_collector)

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

            # Display data outputs collected via display() and matplotlib.
            collected = self._display_collector.get_outputs()
            prev_count = len(collected)
            output.extend(collected)

            # Last expression result with rich MIME types.
            if last_expr_value is not None:
                ipython_display = getattr(last_expr_value, "_ipython_display_", None)
                if callable(ipython_display):
                    ipython_display(display=display_func)
                    output.extend(self._display_collector.get_outputs()[prev_count:])
                else:
                    data, metadata = _format_object(last_expr_value)
                    output.append({
                        "output_type": "execute_result",
                        "data": data,
                        "metadata": metadata,
                    })

            # Clean up display collector and matplotlib hook.
            self._display_collector.clear_list()
            if restore_show is not None:
                restore_show()

            # Remove display from namespace so it doesn't persist in state.
            namespace.pop("display", None)

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
