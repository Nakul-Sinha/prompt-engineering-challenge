# Perturbation Metadata Repair

## The problem

Each row is a corrupted metadata card describing one small-molecule perturbation
experiment. Three categorical fields have been damaged, `source_token`,
`name_type_token` and `library_token`, and I have to restore each one by choosing
from that row's own candidate list.

Everything identifying has been stripped before release: no compound names, no
structure strings, no vendor catalog ids, no plate ids, no URLs. What is left is
structure-derived feature bins, a vendor family token and a plate QC context. The
tokens are salted per row, so a token in one row has nothing to do with the same
token in another. There is no global mapping to memorize, which is the entire
point of the task.

The metric is 0.97 times all-three-exact plus 0.03 times field accuracy, so it is
almost entirely all or nothing per row.

## What I did

Because the metric is dominated by getting all three fields right at once, and
the fields are close to independent given the available evidence, maximizing each
field's top-1 accuracy separately is also what maximizes the expected all-three
score. So I optimize the three fields independently rather than trying to model
the joint.

With no cross-row token identity to lean on, the signal has to come from weak
consistency between the row-local evidence and each candidate, which is where the
work went.

## Layout

`solution.py` is the entry point, `approach.md` is the write up, and `research/`
holds the probes. Datasets are not committed.
