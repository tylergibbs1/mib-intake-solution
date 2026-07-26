# MIB Doc Challenge — Solution

Offline document-processing pipeline for 8090's MIB Doc Challenge: reads a
directory of PDF case packets, extracts applicant records, and adjudicates
each case as `APPROVED`, `DENIED`, or `NEEDS_REVIEW`.

## Run

```bash
docker build -t mib-submission .
docker run --rm --network none \
  --mount type=bind,src=/path/to/pdfs,dst=/input,readonly \
  --mount type=bind,src=/path/to/out,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

No network, no GPU, no API keys. Runtime dependencies: Python 3.12, PyMuPDF,
Pillow, NumPy, tesseract-ocr. No ML model artifacts — the pipeline is
classical CV + OCR + a rule engine derived from the public field manual and
the public training labels.

## Architecture

1. **Text-layer extraction** (PyMuPDF spans): pure-white spans are quarantined
   as prompt-injection material (never used as evidence); dark spans become
   visible lines; colored spans become stamps/notices.
2. **OCR for scanned pages**: 250 dpi grayscale render → Sauvola local
   adaptive binarization (two parameterizations) + a global threshold →
   projection-profile deskew → tesseract (sparse-text PSM). Pages that still
   read as noise get an erosion escalation pass that un-fuses ink-bled
   strokes.
3. **Parsing**: label:value parsing with fuzzy label matching, value-driven
   scanning (formats + closed vocabularies), and positional template parsing
   keyed on each document type's fixed field order.
4. **Evidence resolution**: cross-page weighted voting per field, with the
   field-manual precedence as tie-break weight and signed manual corrections
   ranked highest; forged-sponsor reconciliation between intake form and
   attestation letter.
5. **Adjudication**: manual-note findings and visible stamps first, then
   policy rules (disqualifying flags, embargo worlds, revoked sponsors,
   transit class, fee rules, stale-arrival with DIP-1 exemption, damage
   routing for unreadable packets).
6. **Confidence**: per-decision-path accuracies measured on a held-out
   training split (smoothed), which directly minimizes the Brier calibration
   penalty.

See `MEMO.md` in the submission folder for design rationale and failure-mode
analysis.
