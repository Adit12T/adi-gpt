# Deployment runbook

## Strategy
Releases go out with a blue-green switch on the ridgeline-prod cluster. Smoke tests must pass on the green stack before traffic moves.

## Rollback
To roll back, run `./ops/rollback.sh --release <id>` from the ops repository. A rollback takes about four minutes.

## Change freeze
No production deployments on Fridays after 14:00 Pacific, and none during the last three business days of a quarter.

## Approvals
A production release needs sign-off from the release manager and one engineer from the owning team.
