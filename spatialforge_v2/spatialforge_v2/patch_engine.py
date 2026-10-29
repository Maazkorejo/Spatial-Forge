"""
SpatialForge Programmatic JSON-Patch Engine with Immutability Lock Guard.

Implements RFC 6902 JSON-patch operations on Assemblies (replace, add, remove).
Enforces programmatic safety invariants:
  - Immutable locked parts (parts with silhouette IoU > 0.85 or manually anchored) CANNOT be mutated.
  - Profile self-intersections or non-manifold topology patches are rejected.
  - Dimension values must remain within physical bounds.
"""
from typing import List, Dict, Any, Tuple, Optional
import copy
from schema import Assembly, Part, Feature


def apply_patch(
    asm: Assembly,
    patches: List[Dict[str, Any]],
    locked_parts: Optional[List[str]] = None
) -> Tuple[Assembly, List[str]]:
    """
    Applies a list of RFC 6902 patches to an Assembly.
    Returns:
        (updated_assembly, log_messages)
    Raises:
        ValueError if a patch attempts to modify a locked part or breaks schema rules.
    """
    locked_parts = set(locked_parts or [])
    data = asm.model_dump()
    logs = []

    for patch in patches:
        op = patch.get("op")
        path = patch.get("path", "")
        val = patch.get("value")

        parts = [seg for seg in path.split("/") if seg]
        if not parts:
            continue

        # Lock Guard: check if modifying a locked part
        if parts[0] == "parts" and len(parts) > 1:
            part_id_or_idx = parts[1]
            target_id = None
            if part_id_or_idx.isdigit():
                idx = int(part_id_or_idx)
                if idx < len(data["parts"]):
                    target_id = data["parts"][idx].get("id")
            else:
                target_id = part_id_or_idx

            if target_id and target_id in locked_parts:
                raise ValueError(
                    f"LockGuardViolation: Part '{target_id}' is locked (high IoU or user anchor). "
                    f"Patch to path '{path}' rejected."
                )

        # Apply standard operations
        if op == "replace":
            _apply_replace(data, parts, val)
            logs.append(f"Replaced {path} -> {val}")
        elif op == "add":
            _apply_add(data, parts, val)
            logs.append(f"Added {val.get('id') if isinstance(val, dict) else val} at {path}")
        elif op == "remove":
            _apply_remove(data, parts)
            logs.append(f"Removed {path}")
        else:
            raise ValueError(f"Unsupported patch operation: {op}")

    # Validate against schema invariants
    updated_asm = Assembly.model_validate(data)
    return updated_asm, logs


def _apply_replace(root: Any, path_segments: List[str], value: Any):
    cur = root
    for seg in path_segments[:-1]:
        if isinstance(cur, list):
            # Check by index or part id
            if seg.isdigit():
                cur = cur[int(seg)]
            else:
                found = next((item for item in cur if isinstance(item, dict) and item.get("id") == seg), None)
                if found is not None:
                    cur = found
                else:
                    raise KeyError(f"Item with id '{seg}' not found in list")
        elif isinstance(cur, dict):
            # If seg is a part id in parts list
            if seg in cur:
                cur = cur[seg]
            else:
                raise KeyError(f"Key '{seg}' not found in dict")
        else:
            raise TypeError(f"Cannot traverse path segment '{seg}' on type {type(cur)}")

    last = path_segments[-1]
    if isinstance(cur, dict):
        cur[last] = value
    elif isinstance(cur, list) and last.isdigit():
        cur[int(last)] = value
    else:
        raise KeyError(f"Failed to replace at {last}")


def _apply_add(root: Any, path_segments: List[str], value: Any):
    cur = root
    for seg in path_segments[:-1]:
        if isinstance(cur, list):
            if seg.isdigit():
                cur = cur[int(seg)]
            else:
                found = next((item for item in cur if isinstance(item, dict) and item.get("id") == seg), None)
                if found is not None:
                    cur = found
        elif isinstance(cur, dict):
            cur = cur[seg]

    last = path_segments[-1]
    if last == "-" and isinstance(cur, list):
        cur.append(value)
    elif isinstance(cur, list) and last.isdigit():
        cur.insert(int(last), value)
    elif isinstance(cur, dict):
        cur[last] = value


def _apply_remove(root: Any, path_segments: List[str]):
    cur = root
    for seg in path_segments[:-1]:
        if isinstance(cur, list):
            if seg.isdigit():
                cur = cur[int(seg)]
            else:
                found = next((item for item in cur if isinstance(item, dict) and item.get("id") == seg), None)
                if found is not None:
                    cur = found
        elif isinstance(cur, dict):
            cur = cur[seg]

    last = path_segments[-1]
    if isinstance(cur, list):
        if last.isdigit():
            cur.pop(int(last))
        else:
            cur[:] = [item for item in cur if isinstance(item, dict) and item.get("id") != last]
    elif isinstance(cur, dict):
        cur.pop(last, None)
