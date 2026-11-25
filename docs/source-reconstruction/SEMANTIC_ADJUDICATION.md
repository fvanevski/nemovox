# Evidence-directed semantic adjudication and contextual MT

The normal current-media source quality path runs recording context once before
the shared structural/acoustic scheduler. The former per-boundary text vote
did not authorize source edits; structural boundary requests are retained,
and the scheduler uses their evidence without the extra model requests.
The low-level historical boundary-policy diagnostic remains separate.

After the bounded native and independent CUDA recognition phases, scene
adjudication consumes the same context nominations, revalidating literal
source row/unit references. Unlocalized or retired references remain explicit;
no guessed mapping to a different occurrence. Scene text and optional vision
run in one managed phase. Vision is requested only for relevant unresolved
entity nominations or an explicit visual evidence request. Visible titles and
entity labels are separated from dialogue captions, bound to actual frame
hashes and requested seek positions. They are model observations, not human
verification or proof of who spoke. A literal label matching current source
can be carried as spelling evidence; no automatic name/identity merge.

A scene wording change requires one uniquely supported cross-model acoustic
alternative at the existing occurrence; two competing supported wordings
remain unresolved. A caption patch additionally requires a complete resulting
wording supported by that acoustic alternative. Visual-only text cannot
establish new spoken words, event timing or speakers. Prior source hypotheses,
stable IDs, support observations and event timing remain distinct; revised
word-level timing is not asserted. The private quality report retains raw
scene/vision receipts; the canonical contract carries compact case outcomes
and row-bound semantic context. Discovery, corroboration and actual accepted
source changes are separate metrics. Missing-dialogue/hallucination proposals
are not resolved merely because nearby existing dialogue was corroborated.

## Dedicated MT and English review

Riva receives current canonical source. Source-only evidence does not supply
an English spelling or translation. Explicit operator bilingual glossary
examples can additionally be provided using the documented alternating
user/assistant Riva v2 chat template (language-pair system tag stays intact):

    {"translation_examples":[{"source":"porte","target":"door"}]}

Pairs are literal-scoped to target text or nearby corroborated current-source
rows, bounded to four examples per request, and included in MT cache identity.
The worker checks actual tokenizer context capacity (8192 including generation)
and returns example IDs and native input-token counts. No model-generated
English example or automatic glossary from context hypotheses. Without
operator bilingual examples Riva translates the accepted canonical source
without extra few-shot messages; do not claim arbitrary source-only context
is a supported Riva instruction interface.

English review receives source-bound acoustic/visual evidence and the same
literal operator examples in proposal and comparison. Original dedicated MT
is retained even on rereview. Source-first candidate generation and independent
comparison remain required for changed proposals; no blanket bypass. A mixed
accept/reject result within an unfinished same-speaker source passage, including
across subdivision or separately compared batches, cannot splice a new
unsupported English passage. The existing French source/primary
negation guard also remains in force. Twelve target rows is the draft candidate
batch default (previously six). Explicit native completion exhaustion can
subdivide a batch without accepting incomplete output; whitespace-loop compact
comparison recovery and preserve-primary deferral remain in place. Native
attempt receipts and cache keys distinguish policy and source/MT identity.

## Qualification

Run one bounded real current-media CLI job on authenticated video (target
60–110 seconds, no full episode). Use an already current manifest-bound
recording only if media identity matches; otherwise generate a single fresh
bounded recording in the normal public CLI. Use the qualified CUDA challenger
config copy, actual Riva and native NInfer resources, explicit no condensation,
and no unrelated service eviction. Preserve original recording, source-quality,
canonical, refined, translated, render and checkpoint receipts.

With that actual job's translated.contract.json, use
`python -m scripts.evaluate_semantic_translation --translated PATH --model-resources PATH --output-dir NEW_PATH`
to evaluate six-row review on exactly its original primary MT; compare with
the job's twelve-row review, not a historical full-episode denominator.
Supply the same `--glossary` and `--lifecycle-config` where used.
`--controls` runs bounded known French/Mandarin text, actual Riva few-shot
and native English review, and known rendered text through real visual
inspection. These controls are synthetic text/pixels, not ASR evidence. No
mandatory source correction or hypothesis count. Inspect comparative changes,
polarity, idioms, continuity, unsupported source and visual suggestions;
report provider failures, actual accepted work and timings. Quality and
efficiency require adjudication, not simply fewer API calls. Current-version
resume and final matched geometry/full episode remain C3.

Reference for Riva chat template: https://huggingface.co/nvidia/Riva-Translate-4B-Instruct-v2#chat-template-structure
