# ActiveRetina-RC Public Dataset Evidence Matrix

Created: `2026-10-03`

The flagship paper no longer depends on MFIDDR. Public datasets are now organized around the ActiveRetina-RC claim: risk-controlled adaptive retinal imaging under uncertainty and dataset shift.

DeepDRiD is the current primary acquisition-depth dataset because it provides paired same-eye macula/disc color fundus views. Other public datasets support external robustness, calibration stress, lesion/failure analysis, or specificity checks. MFIDDR, if obtained later, becomes a four-field extension rather than a dependency.

| Dataset | Access | Local State | Role | Supports | Cannot Support |
|---|---|---|---|---|---|
| DeepDRiD | `public_download` | `DOWNLOAD_OR_REQUEST_NEEDED` | two_view_acquisition_generalization | A0_schema, A1_1_vs_2_view_value, external_validation | four_field_subsets, nonmyopic_planning_after_initial_field |
| DRTiD | `application_gated_research_only` | `UPSTREAM_APPLICATION_NEEDED` | two_field_model_comparison_when_access_is_granted | A0_schema_after_access, A1_1_vs_2_view_value | four_field_subsets, nonmyopic_planning_after_initial_field |
| MMRDR | `public_download` | `DOWNLOAD_OR_REQUEST_NEEDED` | modality_and_lesion_label_robustness | external_validation, encoder_pretraining_or_stress_test, lesion_label_analysis | standardized_same_eye_field_selection, four_field_subsets |
| BRSET | `physionet_open_account` | `PHYSIONET_LOGIN_NEEDED` | patient_level_external_validation_and_demographic_robustness | external_validation, patient_level_split, domain_robustness | field_subset_acquisition, nonmyopic_planning |
| mBRSET | `physionet_credentialed_dua` | `PHYSIONET_CREDENTIALING_AND_DUA_NEEDED` | portable_screening_generalization | A0_schema_after_access, A1_1_vs_2_view_value_if_pairs_verified, domain_robustness | four_field_subsets, nonmyopic_planning_after_initial_field |
| EyePACS_Kaggle_DR | `kaggle_rules` | `KAGGLE_TERMS_AND_CLI_AUTH_NEEDED` | large_scale_encoder_training_and_external_validation | encoder_pretraining, single_image_external_validation | same_eye_field_selection, nonmyopic_planning |
| APTOS2019 | `kaggle_rules` | `KAGGLE_TERMS_AND_CLI_AUTH_NEEDED` | single_image_external_validation | single_image_external_validation | field_subset_acquisition, nonmyopic_planning |
| DDR_OIA | `public_download` | `DOWNLOAD_OR_REQUEST_NEEDED` | lesion_aware_external_validation | single_image_external_validation, lesion_label_analysis | same_eye_field_selection, nonmyopic_planning |
| IDRiD | `public_or_challenge_download` | `DOWNLOAD_OR_REQUEST_NEEDED` | small_high_quality_lesion_validation | single_image_external_validation, lesion_label_analysis | same_eye_field_selection, nonmyopic_planning |
| Messidor2 | `public_request_or_download` | `DOWNLOAD_OR_REQUEST_NEEDED` | classic_external_screening_validation | single_image_external_validation | same_eye_field_selection, nonmyopic_planning |
| ODIR5K | `public_download` | `DOWNLOAD_OR_REQUEST_NEEDED` | disease_confounding_and_specificity_check | external_specificity_check | referable_dr_grading_primary_endpoint, same_eye_field_selection, nonmyopic_planning |
| RFMiD | `public_or_research_download` | `UNKNOWN_ACCESS_ROUTE` | disease_confounding_and_specificity_check | external_specificity_check | referable_dr_grading_primary_endpoint_unless_grade_mapping_verified, same_eye_field_selection, nonmyopic_planning |

## Evidence Roles

- Acquisition evidence: same-eye paired or multi-field images with DR grades.
- Risk-control evidence: patient-level calibration, sealed-test risk, confidence intervals, and defer behavior.
- External robustness evidence: graded single-image or paired datasets from different populations/devices.
- Failure-analysis evidence: image quality, lesion annotations, grade strata, and domain-shift patterns.

## Decision Rule

- DeepDRiD can support the main ActiveRetina-RC claim if a calibrated stop/acquire/defer policy reduces image burden while maintaining the target missed-referable-DR risk on sealed patients.
- Public two-view datasets can strengthen the acquisition-depth argument if they independently show value from a second image.
- Single-image datasets can support encoder and external-validity claims only.
- Only MFIDDR or another true multi-step same-eye dataset can support four-field acquisition and non-myopic planning claims.
