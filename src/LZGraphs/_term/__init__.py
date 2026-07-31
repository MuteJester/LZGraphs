"""Terminal rendering layer for the LZGraphs CLI.

``_caps`` is the single place in this package that reads ``os.environ`` or
calls ``isatty``/``shutil.get_terminal_size``; the escape-sequence layer
(``_ansi``), the pure string builders (``_widgets``), and the renderers
(``_plain``, later ``_rich``) all ask a ``Capabilities`` instance rather than
reading the environment themselves.

``Ui`` lives here, rather than in its own ``_protocol.py``, because both
renderers are *implementations of one contract* and this ``__init__`` is
already the module whose own docstring promised to hold the package's public
surface ("later tasks add the rest of the public surface (``ui()``, the
``Ui`` protocol) here"). A dedicated ``_protocol.py`` would just be an extra
hop for no gain: nothing else in the package needs ``Ui`` before ``_plain``
and ``_rich`` do, and both already import from this package's namespace.
``ui()`` itself (the facade that resolves capabilities and picks a renderer)
is Task 6's addition, not this one's.

## ``ui()``, the facade

:func:`ui` is the one call site every command (well, so far only
``cmd_build``; see the plan's Task 6) needs: it detects capabilities,
resolves the mode (``"rich"``, ``"plain"``, or ``"quiet"``), and returns a
ready-to-use renderer satisfying :class:`Ui`. A caller never branches on
which renderer it got back, including for ``quiet``: :class:`NullRenderer`
implements every :class:`Ui` method as a no-op, so ``ui(requested="quiet")``
is exactly as safe to call ``start``/``progress``/``update``/``warn``/
``error``/``done`` on as either real renderer, never a special ``None`` or
``if term: ...`` check.

``stream`` and ``env`` are threaded straight through to
:func:`~LZGraphs._term._caps.detect`, so a caller (or a test) can inject a
fake stream/environment exactly as that function already allows; omitting
either defaults to ``sys.stderr``/``os.environ`` there, matching every other
default in this layer.

``ui()`` deliberately does not grow a ``quiet``/``no_color`` parameter of its
own beyond ``requested``: a caller that wants ``-q/--quiet`` to force quiet
regardless of ``--ui``, or ``--no-color``/``NO_COLOR`` to strip colour,
folds that decision into ``requested`` and ``env`` *before* calling this
function, exactly the way ``cli.py``'s ``cmd_build`` does it (see its module
docstring/the Task 6 report for the exact precedence chosen between
``--ui``, ``-q``, ``--log-level``, and ``NO_COLOR``). Keeping the extra
policy out of ``ui()`` itself is what keeps ``_caps`` the only place in this
package that interprets the environment: ``cmd_build`` merges ``--no-color``
into a plain ``dict`` and hands it to ``env=``, it never reads or mutates
``os.environ`` itself, and this function never re-derives quiet-ness from
anything but the ``requested`` string it was given.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ._caps import Capabilities, detect, resolve_mode
from ._plain import PlainRenderer
from ._rich import RichRenderer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO


@runtime_checkable
class Ui(Protocol):
    """The rendering contract every ``_term`` renderer satisfies.

    Decorated ``@runtime_checkable`` so ``isinstance(renderer, Ui)`` is a
    real, enforceable check (exercised by both
    ``tests/test_term_plain.py::test_plain_renderer_satisfies_ui_protocol``
    and its ``RichRenderer`` counterpart) rather than a docstring's word
    alone: with a second renderer now landed, "both satisfy this surface" is
    worth checking mechanically, not just by construction. Two caveats
    worth stating plainly, since ``runtime_checkable`` is easy to
    over-trust: (1) it only checks that a same-named attribute/method
    *exists* on the candidate, not that its signature matches, so a
    renderer with the right method names but the wrong parameters would
    still pass ``isinstance``; the signature-shape comparison test
    alongside it (via :func:`inspect.signature`) is what actually catches
    that kind of drift. (2) ``Ui`` declares only methods, so it stays valid
    for ``issubclass()`` too (a runtime-checkable ``Protocol`` with any
    non-method member cannot be used with ``issubclass``); the tests use
    ``isinstance`` regardless, since that is what a caller holding an actual
    renderer instance would check.

    Both the plain (CI/pipe, scrolling) renderer and the rich (TTY,
    redrawing) renderer implement exactly this surface, so calling code
    (``cmd_build`` and friends, in Task 6) is written once against ``Ui`` and
    never branches on which renderer it was actually handed. The six methods
    mirror the lifecycle of one CLI command:

    * :meth:`start` opens the command's reporting session with a title (used
      to derive a stable line prefix, e.g. ``"lzg build"`` -> ``[build]``)
      and an optional first batch of descriptive facts (source file, format,
      engine, ...).
    * :meth:`progress` reports fractional completion of one named,
      potentially very hot, sub-step (e.g. ``"ingest"``). Implementations
      MUST throttle this: a caller reporting progress 100,000 times must not
      produce 100,000 lines or redraws.
    * :meth:`update` reports a batch of discrete facts that changed
      (node/edge counts, a rate) as keyword arguments, not part of a
      fractional progress stream.
    * :meth:`warn` reports a single human-readable, non-fatal message.
    * :meth:`error` reports a fatal failure: a short ``title`` plus zero or
      more supporting ``lines`` of detail (available columns, a suggested
      fix, ...). Implementations MUST make this legible without colour.
    * :meth:`done` closes the session with a final title and a table of
      result facts (output path, counts, elapsed time, ...).

    No method returns a value and none may raise on the shape of its input;
    a renderer degrades (e.g. an empty ``fields`` mapping) rather than
    raising, since a rendering failure must never take down the command it
    is merely reporting on.
    """

    def start(self, title: str, fields: Mapping[str, Any] | None = None) -> None:
        """Open a reporting session titled ``title`` with initial ``fields``."""
        ...

    def progress(self, label: str, fraction: float, detail: str | None = None) -> None:
        """Report ``fraction`` (0.0-1.0) complete for the named ``label`` step.

        Implementations throttle their own output; callers may invoke this
        as often as they like, including once per record processed.
        """
        ...

    def update(self, **fields: Any) -> None:
        """Report a batch of named facts that changed."""
        ...

    def warn(self, message: str) -> None:
        """Report a single non-fatal warning."""
        ...

    def error(self, title: str, lines: Sequence[str] | None = None) -> None:
        """Report a fatal error: a short ``title`` plus supporting ``lines``."""
        ...

    def done(self, title: str, rows: Mapping[str, Any] | None = None) -> None:
        """Close the session with a final ``title`` and result ``rows``."""
        ...


class NullRenderer:
    """The ``"quiet"`` renderer: satisfies :class:`Ui`, does nothing at all.

    Exists so a caller resolving a mode via :func:`ui` never has to special-
    case ``quiet`` with an ``if term is not None`` or ``if mode != "quiet"``
    guard: every :class:`Ui` method is implemented here as a plain no-op, so
    ``term.start(...)``, ``term.progress(...)``, and so on are always safe to
    call, quiet or not. This is also exactly what makes ``-q/--quiet``
    "suppress everything but errors" trivially true for whatever calls
    through this layer: nothing this class does can ever write anything, and
    a genuine error still reaches the user because it is raised as an
    exception, not reported through :class:`Ui` (see ``cli.py``'s
    ``cmd_build``, which lets a fatal error propagate to ``main()``'s own
    unconditional ``Error: ...`` printer rather than routing it through
    ``term.error()`` alone).
    """

    def start(self, title: str, fields: Mapping[str, Any] | None = None) -> None:
        pass

    def progress(self, label: str, fraction: float, detail: str | None = None) -> None:
        pass

    def update(self, **fields: Any) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def error(self, title: str, lines: Sequence[str] | None = None) -> None:
        pass

    def done(self, title: str, rows: Mapping[str, Any] | None = None) -> None:
        pass


def ui(
    requested: str | None = None,
    stream: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> Ui:
    """Resolve capabilities and mode, and return the matching renderer.

    ``requested`` is one of ``"auto"`` (or ``None``), ``"rich"``, ``"plain"``,
    or ``"quiet"``; see :func:`~LZGraphs._term._caps.resolve_mode` for the
    full precedence this delegates to (an explicit request wins except that
    ``"rich"`` downgrades on a stream that cannot support cursor control).
    ``"auto"`` is accepted as a synonym for ``None`` (the CLI's
    ``--ui {auto,rich,plain,quiet}`` choice list spells the default option
    out explicitly rather than leaving it unnamed) since ``resolve_mode``
    itself only recognises ``None`` for "no explicit request".

    ``stream`` and ``env`` are forwarded verbatim to
    :func:`~LZGraphs._term._caps.detect`, so both default to ``sys.stderr``/
    ``os.environ`` exactly as that function already does, and both remain
    injectable for tests without touching a real terminal or the process
    environment.

    Returns a :class:`NullRenderer` for ``"quiet"``, a
    :class:`~LZGraphs._term._rich.RichRenderer` for ``"rich"``, or a
    :class:`~LZGraphs._term._plain.PlainRenderer` for ``"plain"``; all three
    satisfy :class:`Ui`, so calling code never needs to know which one it
    received.
    """
    caps = detect(stream=stream, env=env)
    mode = resolve_mode(caps, None if requested == "auto" else requested)

    if mode == "quiet":
        return NullRenderer()
    if mode == "rich":
        return RichRenderer(caps, stream=stream)
    return PlainRenderer(caps, stream=stream)


__all__ = [
    "Capabilities",
    "detect",
    "resolve_mode",
    "Ui",
    "NullRenderer",
    "PlainRenderer",
    "RichRenderer",
    "ui",
]
