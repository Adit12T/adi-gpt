# EDI comparison notes

## Goal
Compare a client's EDI 837 files between the UAT and PROD environments and report real differences only.

## Segments to ignore
Control numbers always differ between environments, so ignore ISA13 and GS06. Also ignore the ISA09 and ISA10 date and time fields.

## Tooling
The comparison is done with an internal tool called SegDiff. It writes its report to `reports/segdiff-<client>.html`.

## Known issue
SegDiff reports false differences when a file uses a tilde followed by a newline as the segment terminator; normalise terminators first.
