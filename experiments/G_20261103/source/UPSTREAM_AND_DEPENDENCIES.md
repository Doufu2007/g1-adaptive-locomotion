# Source provenance and environment limitations

The `scripts/` and `src/` snapshots are byte-identical to the frozen private G
archive, identified by its SHA256 in the publication manifests. Existing source
headers and third-party notices are retained. These are experimental snapshots
with local additions and modifications, not an unmodified upstream release.

Upstream project: https://github.com/unitreerobotics/unitree_rl_mjlab
Local checkout HEAD when preparing publication:
`1425b15f73bd4095f0df53709d7c389c3eb9e790`.
This identifies the local base only; it is NOT evidence that every cloud
dependency was pinned to that commit. The accompanying upstream `LICENCE` is
Apache-2.0. Dependency/data terms are separate; no blanket relicensing is made.

The cloud backup does not contain a complete environment or dependency lock.
Known missing runtime dependencies include `scripts/system_check.py`,
`scripts/cloud_training_monitor.py`, and G1 XML/mesh assets. The locally observed
setup.py specifies mjlab==1.2.0, mujoco==3.5.0, mujoco-warp==3.5.0,
warp-lang==1.12.0, scipy>=1.11,<2.0; this local configuration is not a verified
complete cloud package inventory. Do not install/run a training campaign merely
from this evidence snapshot. Obtain and separately verify the full environment,
source dependencies, data permission and reference hash before reproduction.

For existing-result analysis only, use Python with numpy/matplotlib:
`python analysis/analyze_g_stage.py --help` gives the analysis arguments.
The exact input is `evidence/final_evaluation_envelope.json`.

Motion arrays and the machine-specific QA orchestration source are intentionally
omitted. Historical original manifests are retained as provenance, and name
private members that are absent here. Public manifests are authoritative for
the curated deliverable. Logs/configurations retain historical infrastructure
paths so their bytes and training receipts remain independently checkable.
