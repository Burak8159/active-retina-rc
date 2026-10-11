# ActiveRetina Pilot Outreach Package Checklist

Status: PREPARED_STRUCTURE_NOT_FINAL

This checklist defines what must be ready before contacting medical schools or eye hospitals.

## Must Have Before Sending

- One-page research brief.
- Professor-reviewed or professor-endorsed manuscript package.
- arXiv manuscript draft.
- IEEE-style manuscript draft.
- Product demo screenshots or short video.
- Validation summary with exact dataset names and metrics.
- Safety and limitations statement.
- Pilot protocol outline.
- Data governance statement.
- GitHub repository link with restricted data excluded.

## One-Page Brief Structure

1. Project title: ActiveRetina.
2. Problem: diabetic retinopathy screening access and imaging burden.
3. Proposed system: portable imaging workflow plus local risk-controlled inference.
4. Current status: product software/hardware prototype and validation package.
5. Evidence summary: exact metrics from public and credentialed datasets.
6. Pilot ask: retrospective de-identified validation first.
7. Safety boundary: no autonomous clinical decision-making during initial pilot.
8. What institution provides: governed images or evaluation pathway.
9. What we provide: software, protocol, reports, model outputs, manuscript.
10. Contact details.

## Pilot Protocol Skeleton

### Phase 1: Retrospective Validation

- Input: de-identified retinal images and labels under institutional rules.
- Output: model predictions, quality flags, uncertainty/triage decisions.
- Evaluation: sensitivity, specificity, missed referable DR, coverage, calibration, subgroup checks if metadata permits.
- Safety: no patient-care use.

### Phase 2: Clinician-Reviewed Workflow Study

- Input: new or archived cases reviewed by clinicians.
- Output: clinician feedback, time/workflow metrics, failure modes.
- Safety: outputs are advisory only.

### Phase 3: Prospective Pilot

- Only after Phase 1/2 evidence and institutional approval.
- Clinician remains decision-maker.
- Autonomous mode is only considered if validated and approved by the institution.

## Data Governance Commitments

- No raw institutional images in public GitHub.
- No restricted data uploaded to LLMs or public cloud tools.
- Local encrypted or access-controlled storage where required.
- Aggregate-only reporting unless institution approves otherwise.
- Any human-subject or patient-data workflow follows the institution's IRB or ethics process.

## Readiness Gate

Do not send outreach until this line is true:

`OUTREACH_READY = professor_reviewed AND manuscript_ready AND validation_summary_ready AND demo_ready AND pilot_protocol_ready`
