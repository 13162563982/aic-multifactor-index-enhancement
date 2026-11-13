"""DAG parsing, validation and deterministic topological ordering.

``config/workflow.yaml`` is the single source of truth for the stage graph. This
module turns the ``workflow`` section of the merged configuration bundle into an
immutable :class:`Workflow` that the runner can walk.

Validation performed here (all raise :class:`aicwf.contracts.ConfigError`):

* every stage name is a known :data:`aicwf.contracts.STAGES` name, unique, and a
  lower-case identifier;
* every ``handler`` is a non-empty string (the runner resolves it against the
  pipeline handler registry);
* every entry in ``needs`` references a declared stage, never itself, and the
  resulting graph is acyclic;
* ``timeout_s`` is a positive number and ``retries`` a non-negative integer;
* ``produces`` is a list of strings and ``agent`` an optional mapping.

The topological order is deterministic: Kahn's algorithm with a priority queue
keyed by the stage's declaration index, so independent stages keep the order in
which they appear in ``workflow.yaml`` on every run and every platform.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from aicwf.contracts import STAGES, ConfigError
from aicwf.util.hashing import hash_json

__all__ = [
    "DEFAULT_STAGE_TIMEOUT_S",
    "StageSpec",
    "Workflow",
    "load_workflow",
    "topological_order",
    "handler_map",
    "stage_order",
]

DEFAULT_STAGE_TIMEOUT_S = 600.0
_SUPPORTED_EXECUTION_MODES = frozenset({"deterministic_topological"})
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_str_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{label} must be a list of strings, got {type(value).__name__}")
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigError(f"{label} entries must be non-empty strings, got {entry!r}")
        items.append(entry.strip())
    if len(set(items)) != len(items):
        raise ConfigError(f"{label} contains duplicate entries: {items}")
    return tuple(items)


# ---------------------------------------------------------------------------
# Stage specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """One validated node of the workflow DAG."""

    name: str
    handler: str
    needs: tuple[str, ...] = ()
    timeout_s: float = DEFAULT_STAGE_TIMEOUT_S
    retries: int = 0
    produces: tuple[str, ...] = ()
    agent: Mapping[str, Any] = field(default_factory=dict, compare=False)
    human_gate: bool = False
    approval_action: str | None = None
    order_index: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def max_attempts_hint(self) -> int:
        """How many attempts a stage would use with an unbounded retry budget."""

        return 1 + int(self.retries)

    def as_dict(self) -> dict[str, Any]:
        """Canonical, JSON-serialisable description (used in input hashes)."""

        return {
            "name": self.name,
            "handler": self.handler,
            "needs": list(self.needs),
            "timeout_s": self.timeout_s,
            "retries": self.retries,
            "produces": list(self.produces),
            "agent": dict(self.agent),
            "human_gate": self.human_gate,
            "approval_action": self.approval_action,
        }


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workflow:
    """A validated stage DAG with deterministic execution order."""

    name: str
    version: str
    order: tuple[StageSpec, ...]
    execution: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    source: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    _index: dict[str, StageSpec] = field(init=False, repr=False, compare=False)
    _children: dict[str, tuple[str, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = {spec.name: spec for spec in self.order}
        if len(index) != len(self.order):  # defence in depth; load_workflow validates first
            raise ConfigError("duplicate stage names in workflow")
        children: dict[str, list[tuple[int, str]]] = {name: [] for name in index}
        for spec in self.order:
            for need in spec.needs:
                children[need].append((spec.order_index, spec.name))
        ordered_children = {
            name: tuple(name for _, name in sorted(entries)) for name, entries in children.items()
        }
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_children", ordered_children)

    # -- mapping-ish access -------------------------------------------
    @property
    def stages(self) -> Mapping[str, StageSpec]:
        return MappingProxyType(self._index)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.order)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def __iter__(self) -> Iterable[StageSpec]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)

    def stage(self, name: str) -> StageSpec:
        try:
            return self._index[name]
        except KeyError as exc:
            raise ConfigError(f"unknown stage {name!r}") from exc

    def needs(self, name: str) -> tuple[str, ...]:
        return self.stage(name).needs

    # -- graph queries -------------------------------------------------
    def children(self, name: str) -> tuple[str, ...]:
        """Stages that directly depend on ``name``."""

        self.stage(name)
        return self._children[name]

    def descendants(self, name: str) -> tuple[str, ...]:
        """All transitive dependants of ``name`` in deterministic order."""

        return self.downstream_closure((name,), include_self=False)

    def downstream_closure(self, names: Sequence[str], *, include_self: bool = True) -> tuple[str, ...]:
        """Transitive dependants of several stages, in workflow order."""

        seen: set[str] = set()
        stack = list(names)
        if include_self:
            seen.update(names)
        while stack:
            current = stack.pop()
            for child in self._children.get(current, ()):  # unknown names are ignored
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return tuple(spec.name for spec in self.order if spec.name in seen)

    def ancestors(self, name: str) -> tuple[str, ...]:
        """All transitive prerequisites of ``name`` in deterministic order."""

        required = self.stage(name).needs
        seen: set[str] = set()
        stack = list(required)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.stage(current).needs)
        return tuple(spec.name for spec in self.order if spec.name in seen)

    def human_gates(self) -> tuple[StageSpec, ...]:
        return tuple(spec for spec in self.order if spec.human_gate)

    def handler_map(self) -> dict[str, str]:
        """Map handler name -> stage name (inverse of ``StageSpec.handler``)."""

        return {spec.handler: spec.name for spec in self.order}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "execution": dict(self.execution),
            "stages": [spec.as_dict() for spec in self.order],
        }

    def fingerprint(self) -> str:
        """Deterministic hash of the whole DAG definition."""

        return hash_json(self.as_dict())


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse_stage(raw: Any, index: int, execution: Mapping[str, Any], *, workflow_version: str) -> StageSpec:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"workflow.stages[{index}] must be a mapping, got {type(raw).__name__}")
    label = f"workflow.stages[{index}]"

    name_value = raw.get("name")
    if not isinstance(name_value, str) or not name_value.strip():
        raise ConfigError(f"{label}.name must be a non-empty string")
    name = name_value.strip()
    if not _NAME_RE.match(name):
        raise ConfigError(f"{label}.name {name!r} must match {_NAME_RE.pattern}")
    if name not in STAGES:
        raise ConfigError(
            f"{label}.name {name!r} is not a known stage; expected one of {list(STAGES)}"
        )

    handler_value = raw.get("handler")
    if not isinstance(handler_value, str) or not handler_value.strip():
        raise ConfigError(f"{label}.handler must be a non-empty string")
    handler = handler_value.strip()

    needs = _as_str_tuple(raw.get("needs"), f"{label}.needs")
    if name in needs:
        raise ConfigError(f"{label}.needs contains the stage itself ({name!r})")

    timeout_default = execution.get("stage_default_timeout_s", DEFAULT_STAGE_TIMEOUT_S)
    timeout_value = raw.get("timeout_s", timeout_default)
    if not _is_number(timeout_value) or float(timeout_value) <= 0.0:
        raise ConfigError(f"{label}.timeout_s must be a positive number, got {timeout_value!r}")
    timeout_s = float(timeout_value)

    retries_value = raw.get("retries", 0)
    if not isinstance(retries_value, int) or isinstance(retries_value, bool) or retries_value < 0:
        raise ConfigError(f"{label}.retries must be a non-negative integer, got {retries_value!r}")
    retries = int(retries_value)

    produces = _as_str_tuple(raw.get("produces"), f"{label}.produces")

    agent_value = raw.get("agent")
    if agent_value is None:
        agent: dict[str, Any] = {}
    elif isinstance(agent_value, Mapping):
        agent = dict(agent_value)
        for key in ("prompt", "skill", "schema"):
            if key in agent and agent[key] is not None and not isinstance(agent[key], str):
                raise ConfigError(f"{label}.agent.{key} must be a string path or null")
    else:
        raise ConfigError(f"{label}.agent must be a mapping, got {type(agent_value).__name__}")

    human_gate = raw.get("human_gate", False)
    if not isinstance(human_gate, bool):
        raise ConfigError(f"{label}.human_gate must be a boolean, got {human_gate!r}")

    approval_action = raw.get("approval_action")
    if approval_action is not None and (not isinstance(approval_action, str) or not approval_action.strip()):
        raise ConfigError(f"{label}.approval_action must be a non-empty string or null")
    if human_gate and not approval_action:
        approval_action = None  # the approval gate derives a fallback action

    return StageSpec(
        name=name,
        handler=handler,
        needs=needs,
        timeout_s=timeout_s,
        retries=retries,
        produces=produces,
        agent=agent,
        human_gate=human_gate,
        approval_action=approval_action.strip() if approval_action else None,
        order_index=index,
        raw=dict(raw),
    )


def _find_cycle(specs: Mapping[str, StageSpec]) -> list[str]:
    """Return one cycle as a list of stage names (for a helpful error message)."""

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(name: str) -> list[str] | None:
        state[name] = 1
        stack.append(name)
        for need in specs[name].needs:
            marker = state.get(need, 0)
            if marker == 1:
                start = stack.index(need)
                return stack[start:] + [need]
            if marker == 0:
                found = visit(need)
                if found:
                    return found
        stack.pop()
        state[name] = 2
        return None

    for stage_name in specs:
        if state.get(stage_name, 0) == 0:
            found = visit(stage_name)
            if found:
                return found
    return []


def topological_order(specs: Iterable[StageSpec]) -> tuple[StageSpec, ...]:
    """Deterministic topological order of validated stage specs.

    Ties between ready stages are broken by declaration order, so the result is
    stable across runs and platforms.
    """

    by_name: dict[str, StageSpec] = {}
    for spec in specs:
        if spec.name in by_name:
            raise ConfigError(f"duplicate stage name {spec.name!r}")
        by_name[spec.name] = spec

    indegree: dict[str, int] = {}
    dependants: dict[str, list[StageSpec]] = {name: [] for name in by_name}
    for name, spec in by_name.items():
        for need in spec.needs:
            if need not in by_name:
                raise ConfigError(f"stage {name!r} needs unknown stage {need!r}")
            if need == name:
                raise ConfigError(f"stage {name!r} cannot depend on itself")
            dependants[need].append(spec)
        indegree[name] = len(spec.needs)
    for name in dependants:
        dependants[name].sort(key=lambda item: item.order_index)

    ready: list[tuple[int, str]] = [(spec.order_index, spec.name) for spec in by_name.values() if indegree[spec.name] == 0]
    heapq.heapify(ready)
    ordered: list[StageSpec] = []
    while ready:
        _, name = heapq.heappop(ready)
        ordered.append(by_name[name])
        for child in dependants[name]:
            indegree[child.name] -= 1
            if indegree[child.name] == 0:
                heapq.heappush(ready, (child.order_index, child.name))

    if len(ordered) != len(by_name):
        cycle = _find_cycle(by_name)
        rendered = " -> ".join(cycle) if cycle else "unknown cycle"
        raise ConfigError(f"workflow DAG contains a cycle: {rendered}")
    return tuple(ordered)


def load_workflow(config: Mapping[str, Any]) -> Workflow:
    """Parse and validate the workflow DAG from a merged config bundle.

    Accepts either the full bundle (``{"workflow": {...}, ...}``) or the bare
    ``workflow`` mapping itself.
    """

    if not isinstance(config, Mapping):
        raise ConfigError(f"workflow config must be a mapping, got {type(config).__name__}")
    workflow_raw = config.get("workflow")
    if isinstance(workflow_raw, Mapping) and "stages" not in workflow_raw:
        # Tolerate bundles produced by aio-style loaders that keep the YAML root
        # key (``workflow:``) nested inside ``bundle["workflow"]``.
        nested = workflow_raw.get("workflow")
        if isinstance(nested, Mapping) and "stages" in nested:
            workflow_raw = nested
    if isinstance(workflow_raw, Mapping):
        raw = workflow_raw
    elif "stages" in config:
        raw = config
    else:
        raise ConfigError("config is missing the 'workflow' section")

    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ConfigError("workflow.stages must be a non-empty list")

    execution_value = raw.get("execution") or {}
    if not isinstance(execution_value, Mapping):
        raise ConfigError("workflow.execution must be a mapping")
    execution = dict(execution_value)
    mode = execution.get("mode", "deterministic_topological")
    if mode not in _SUPPORTED_EXECUTION_MODES:
        raise ConfigError(
            f"unsupported workflow.execution.mode {mode!r}; supported: {sorted(_SUPPORTED_EXECUTION_MODES)}"
        )
    if execution.get("stop_on_first_failure", True) is not True:
        raise ConfigError("workflow.execution.stop_on_first_failure must be true (fail-fast runner)")

    version = raw.get("version", "0.0.0")
    if not isinstance(version, str):
        raise ConfigError("workflow.version must be a string")
    name_value = raw.get("name", "workflow")
    if not isinstance(name_value, str) or not name_value.strip():
        raise ConfigError("workflow.name must be a non-empty string")

    specs = [
        _parse_stage(stage, index, execution, workflow_version=version)
        for index, stage in enumerate(stages_raw)
    ]
    ordered = topological_order(specs)
    return Workflow(
        name=name_value.strip(),
        version=version,
        order=ordered,
        execution=execution,
        source=dict(raw),
    )


# ---------------------------------------------------------------------------
# Small helpers used by the runner and tests
# ---------------------------------------------------------------------------


def handler_map(workflow: Workflow) -> dict[str, str]:
    return workflow.handler_map()


def stage_order(config: Mapping[str, Any]) -> tuple[str, ...]:
    return load_workflow(config).stage_names
