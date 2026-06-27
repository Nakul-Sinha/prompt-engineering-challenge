Overview
This challenge is an anonymized metadata repair task. Each example has been transformed from real experimental perturbation records into a row-local token puzzle: direct compound identifiers, original category names, structure strings, vendor ids, plate ids, and external URLs are removed before release.

Each row describes one corrupted perturbation card. The card contains derived metadata from a real small-molecule treatment record: structure-derived feature bins, a vendor-family token, a plate-QC context, and three corrupted categorical token fields. In machine-learning terms, the task is to choose the correct row-local source_token, name_type_token, and library_token from the row-specific candidate sets.

The public data do not expose compound names, structure strings, original ids, original category names, vendor catalog ids, plate ids, or URLs. Tokens are salted and opaque. Candidate tokens are local to each row, so a token value seen in one row should not be treated as the same category in another row. Solving the task should rely on weak consistency between the corrupted card and field-specific candidate evidence cards.

Dataset Files
train.csv

Description: Labeled perturbation repair examples.
Row count: 600.
Columns: id, prompt, corrupted_card, support_cards, source_options, name_type_options, library_options, repair_fields, answer_json.
test.csv

Description: Evaluation examples with the same public fields as training but without answer_json.
Row count: 300.
Columns: id, prompt, corrupted_card, support_cards, source_options, name_type_options, library_options, repair_fields.
sample_submission.csv

Description: Valid submission template.
Columns: id, answer_json.
Column Descriptions
id (string): unique repair task id.

prompt (string): task instruction.

corrupted_card (JSON object): anonymized perturbation card with corrupted source_token, name_type_token, and library_token values. The card also includes vendor_family_token, smiles_features, and qc_context.

support_cards (JSON list): field-specific candidate evidence cards. Each card has repair_field, candidate_token, vendor_family_token, smiles_features, and evidence_rank_hint. A support card describes one candidate token for one repair field; it does not expose the other true repair fields.

source_options (JSON list): row-local candidate source tokens. The true source_token is one of these values. Most rows contain 32 source candidates.

name_type_options (JSON list): row-local candidate name-type tokens. The true name_type_token is one of these values. Rows include all available name-type candidates.

library_options (JSON list): row-local candidate library tokens. The true library_token is one of these values. Rows include all available library candidates.

repair_fields (JSON list): fields to repair. Current value is always ["source_token", "name_type_token", "library_token"].

answer_json (JSON object): target repair object. It must contain exactly three string fields: source_token, name_type_token, and library_token. Each value should be selected from the corresponding candidate list in the same row.

Submission Format
Submit a CSV with these required columns, in any order. Extra columns are ignored by the grader:

```text
id,answer_json  
cpr_00001_ab12cd34,"{""source_token"":""src_1234abcd"",""name_type_token"":""name_5678efab"",""library_token"":""lib_90ab12cd""}"  

Every id from test.csv should appear once. Missing ids, duplicate ids, invalid JSON, or missing JSON fields receive zero credit for the affected row instead of failing the whole submission.

The answer_json value must be a JSON object with string token values:

source_token: selected from source_options.
name_type_token: selected from name_type_options.
library_token: selected from library_options.
Evaluation
Higher is better. A perfect submission scores 1.0.

Each row is scored by a strict three-field repair metric:

field_accuracy = (  
    correct(source_token) + correct(name_type_token) + correct(library_token)  
) / 3  
  
row_score = 0.97 * all_three_fields_exact + 0.03 * field_accuracy  

The final score is:

score = mean(row_score)  

Invalid JSON or missing fields score 0 for the affected fields. The all_three_fields_exact term is 1 only when all three repaired tokens are exactly correct.

What Not To Use
Use only the public CSV files supplied with this challenge.

Do not use external lookup resources corresponding to this dataset, compound-name lookup, structure-string lookup, original ids, original category names, vendor catalog ids, raw files, private answer files, or hardcoded mappings from task ids to answers.

 