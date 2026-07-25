# OpenASR fork maintenance

This repository separates the upstream mirror from the source revision used by
OpenASR.

## Branches and pins

- `master` mirrors `ggml-org/ggml:master`. Do not add OpenASR patches there.
- `oasr/pin-<upstream-sha>` starts at one reviewed upstream commit and carries
  the small, linear OpenASR patch stack for that baseline.
- Feature and fix branches target the active `oasr/pin-*` branch.
- The OpenASR superproject records an exact submodule commit. It never depends
  on a floating branch reference.

Once a pin commit is referenced by the superproject, do not rebase or
force-push its branch. Keep old pin branches reachable so historical OpenASR
revisions remain reproducible.

## Updating upstream

1. Choose and record an exact `ggml-org/ggml` commit.
2. Create a new `oasr/pin-<upstream-sha>` branch at that commit.
3. Replay the OpenASR patch stack in order, dropping patches already supplied
   by upstream.
4. Open feature or fix pull requests against the new pin branch.
5. Require the `OpenASR pin / required` check and the OpenASR superproject's
   full regression before updating its submodule gitlink.

The superproject helper `tooling/ggml-sync/sync.sh` performs the local replay.
It intentionally leaves review, pushing, and the gitlink update as separate
steps.
