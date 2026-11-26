"""Apply only Central-selected tracked retirements, with exact Git object verification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess


def git(*args):
    return subprocess.check_output(["git", "--literal-pathspecs", *args])


def records(raw):
    out = {}
    for row in raw.split(b"\0"):
        if not row:
            continue
        meta, path = row.split(b"\t", 1)
        mode, typ, sha = meta.decode("ascii").split()
        name = path.decode("utf-8")
        if name in out:
            raise RuntimeError("duplicate path")
        out[name] = {"mode": mode, "type": typ, "sha": sha}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expected-head", required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if root != Path.cwd().resolve():
        raise RuntimeError("Run from isolated worktree root")
    if args.receipt.resolve().is_relative_to(root):
        raise RuntimeError("Receipt must be outside the candidate worktree")
    if git("rev-parse", "HEAD").decode().strip() != args.expected_head:
        raise RuntimeError("Unexpected HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Worktree must be clean")
    doc = json.loads(Path("staging/native-dialogue-retirements.json").read_text())
    if doc["schema"] != "nemovox-retirement-manifest/v1":
        raise RuntimeError("Unexpected manifest")
    if git("rev-parse", doc["inventory_commit"]+"^{tree}").decode().strip() != doc["inventory_tree"]:
        raise RuntimeError("Inventory tree mismatch")
    selected = {}
    for item in doc["entries"]:
        path = item["path"]
        expect = {k:item[k] for k in ("mode","type","sha")}
        for rev in (doc["inventory_commit"], "HEAD"):
            found = records(git("ls-tree", "-z", rev, "--", path))
            if found != {path: expect}:
                raise RuntimeError(f"Object mismatch: {path} at {rev}")
        expanded = records(git("ls-tree", "-r", "-z", "HEAD", "--", path))
        if not expanded or set(expanded).intersection(selected):
            raise RuntimeError(f"Empty or overlapping selection: {path}")
        selected.update(expanded)
    # The carrier is itself temporary; its exact objects are bound by the expected HEAD.
    for path in ("staging/native-dialogue-retirements.json", "staging/apply_native_dialogue_retirements.py"):
        found = records(git("ls-tree", "-z", "HEAD", "--", path))
        if set(found) != {path} or found[path]["type"] != "blob":
            raise RuntimeError("Missing retirement carrier")
        selected.update(found)
    receipt = {"head": args.expected_head, "inventory_commit": doc["inventory_commit"],
               "entries": [{"path": p, **selected[p]} for p in sorted(selected)],
               "applied": False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    if args.apply:
        # Exact paths only, never remove untracked/user assets or use broad shell deletion.
        for offset in range(0, len(selected), 100):
            subprocess.run(["git","--literal-pathspecs","rm","--",*sorted(selected)[offset:offset+100]],check=True)
        changed = git("diff","--cached","--name-status","-z").split(b"\0")
        actual = {changed[i+1].decode():changed[i].decode() for i in range(0,len(changed)-1,2)}
        if actual != {p:"D" for p in selected}:
            raise RuntimeError("Unexpected staged delta")
        receipt["applied"] = True
        args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    print(f"{len(selected)} exact tracked files; applied={receipt['applied']}")


if __name__ == "__main__":
    main()
