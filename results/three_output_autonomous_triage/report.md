# Three-Output Autonomous Triage

Status: `GO_THREE_OUTPUT_AUTONOMOUS_TRIAGE`

Every image receives one autonomous product output: nonreferable, referable, or insufficient evidence/reacquire. The insufficient-evidence output is a completed product action, not a diagnostic clearance.

Candidate score policy: `median_all_ranknorm` with low `0.050000` and high `0.587500`.

## Test Result

- False-negative clearances: `0`
- Specificity among diagnostic decisions: `94.7%`
- Diagnostic decision rate: `43.8%`
- Insufficient-evidence/reacquire rate: `56.2%`
- Negative clearance rate: `14.7%`

## Interpretation

This converts the successful selective policy into a product-complete output contract. It does not make every image receive a binary diagnosis; it makes every image receive a safe autonomous workflow action.
