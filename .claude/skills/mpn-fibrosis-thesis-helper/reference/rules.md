# Working rules and writing rules (the user's standing preferences)

## 1. Working with the user

- Reply in Thai; keep technical terms in English; manuscript text in English.
- Give one recommendation, not a menu of options.
- Do not run anything unless the user says to run it (including inference-only checks).
- Experiments are closed: do not propose new models, variants, baselines, feature re-extraction, backbone fusion, loss changes, external data or a second rater. If asked "what else", say the in-scope space is covered.
- Never state "nothing is left" as a fact; say what was tried.
- Check before claiming something is decided — a question from the user is not a decision.
- Verify a paper by reading it before describing it.

## 2. Reporting

- One fixed patient-level split (seed 2); do not discuss other seeds or cross-validation unless asked.
- ROI-level metrics only; never patient-level metrics.
- All three encoders, always; no primary encoder.
- Report validation next to test; state that selection used validation.
- Formulation headline = errors ≥ 2 grades (7 vs 54, 12/12), not QWK or G0 recall.
- Multi-class = cross-entropy + label smoothing with uniform class weights.
- No significance claims; claim consistency across encoders.

## 3. Words

- Never: first, novel mechanism, state of the art, significantly better, attention highlights/attends to fibrosis, module codes (a445, a381, a215) in text.
- Use: to our knowledge; we show that; class token ([CLS]); whole-ROI embedding (not "global"); ABMIL (not "simple"); WR-TransMIL; "TransMIL + whole-ROI token" for the comparator.

## 4. Corrections already made — do not repeat the old version

- "Multi-class collapses G0 to 0%" came from an older 42-patient cohort; current runs keep G0 at 89–93%.
- Multi-class used uniform class weights, not weighted CE.
- The whole-ROI view is not cropped like the patches.
- SMILE does not use sparsemax; MINN-SA does.
- "The whole-ROI token avoids competing for attention" is an unsupported explanation — do not use it.
- Feature norm is grade-uninformative (it is not "bone"); patch features do not form tissue-type clusters.
- ASGAP is less robust to added foreign patches than ABMIL, not more.

## 5. Decisions (dates)

| date | decision |
|---|---|
| 2026-06 | local-Mac runs only; no zero-shot concept labels; ASGAP reported with learnable-α wording |
| 2026-07-30 | augmentation is a control variable; image augmentation scoped out |
| 2026-08 | ROI-level only; single split only; stop running experiments; no primary encoder |
| 2026-09 | the large model search is not presented as a thesis section; journal shortlist from the advisor, BSPC first |
| 2026-09-27 | one paper, one method (WR-TransMIL, name confirmed); ASGAP thesis-only; 5-model main table (CLAM-SB/DSMIL/DTFD-MIL cited, not tabulated, results kept in reserve); no fibrosis-axis analysis in the paper; bootstrap CIs only if a reviewer asks |
