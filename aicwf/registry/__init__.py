"""Version registry: append-only release history, promotion and rollback.

Public entry points:
    aicwf.registry.stage.run_registry(ctx)   -- pipeline registry stage handler
    aicwf.registry.promote.promote(...)      -- append one release (approval-gated)
    aicwf.registry.promote.promote_run(...)  -- promote from a run directory
    aicwf.registry.rollback.rollback(...)    -- restore a release bundle + smoke eval
    aicwf.registry.store.load_history(...)   -- read the append-only history
"""

from aicwf.registry.promote import promote, promote_run  # noqa: F401
from aicwf.registry.rollback import rollback, rollback_run, rollback_to, rollback_to_version  # noqa: F401
from aicwf.registry.stage import (  # noqa: F401
    load_registry_paths,
    registry_status,
    resolve_approval_artifact,
    run_registry,
)
from aicwf.registry.store import (  # noqa: F401
    RegistryConflict,
    RegistryError,
    RegistryIntegrityError,
    append_entry,
    create_bundle,
    find_entry,
    history_summary,
    latest_release,
    load_history,
    release_versions,
    restore_bundle,
    validate_history,
    verify_bundle,
)

__all__ = [
    "RegistryConflict",
    "RegistryError",
    "RegistryIntegrityError",
    "append_entry",
    "create_bundle",
    "find_entry",
    "history_summary",
    "latest_release",
    "load_history",
    "load_registry_paths",
    "promote",
    "promote_run",
    "registry_status",
    "release_versions",
    "resolve_approval_artifact",
    "restore_bundle",
    "rollback",
    "rollback_run",
    "rollback_to",
    "rollback_to_version",
    "run_registry",
    "validate_history",
    "verify_bundle",
]
