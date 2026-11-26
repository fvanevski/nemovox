# Source robustness: targeted evidence comparison

Use this driver only for private, bounded experiments following the completed
Studio intake for issue #24. It is not a new user-facing mode or a ground-truth
annotation tool. The current production scene path retains rejected responses,
identifies invalid decision predicates and permits at most one corrective
request per case; all source-edit admissibility remains unchanged.

CPU: run the focused scene/occurrence tests and the replay phase on the
authenticated archive. The sequence detector only nominates source-order
correspondence with disjoint provisional CTC times. Evaluate all nominations
at the three screening lengths, including false positives and existing task
coverage; no chosen threshold or speech retirement is authorized by this run.

GPU: use the acoustic phase with the Central-provided private JSON crop plan.
The primary matched arm recreates the archived WAV exactly and compares against
the same right edge and core with left context removed. Local crops use the
same primary generation settings and reconstruct absolute source times.
The independent CUDA phase follows release of the owned native stage. Compare
raw recognition, CTC, bounded independent hypotheses, and retained canonical
rows. Missing recognition is not proof of silence. Record differences in
model/runtime identities, actual costs, unresolved speech and speaker truth.
The scene phase uses reconstructed failed inputs and successful initial
controls, the real native Messages client, and the same source-edit safeguards.
A corrected structure is not evidence that a source change is warranted.

Use the normal resource lifecycle and existing cached models; do not start a
concurrent local GPU coding model, evict another workload, download weights,
or change production configuration. Use a fresh private output directory for
each phase. Keep archive, probe plans, media, raw responses and derived text
off GitHub. Return a single private handback with immutable source and input
identities, observed positive/negative results, costs, and resource restoration.
Remove this diagnostic driver and obsolete controls when successor mechanisms
have been selected and qualified, unless a current useful purpose remains.
