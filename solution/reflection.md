# Reflection

## Hardest faults

Subtle data quality faults were hardest to catch because each value could remain inside its normal range while several values drifted together

Feature drift also required careful threshold selection because normal serving data can vary more than the published baseline suggests

## Cost and coverage

Used one toolkit call for each event to keep coverage high and avoid missed faults

Direct threshold checks catch clear violations while combined signals help detect smaller data shifts

Practice achieved full coverage with zero false positives

Public still kept zero false positives and reached a high band across all pillars despite moderate tool overage

## Next improvement

Store recent clean measurements in `ctx.state` and update rolling mean, standard deviation, and signal correlations for each event type, supporting adaptive thresholds and improve detection of subtle faults without raising false alarms too much
