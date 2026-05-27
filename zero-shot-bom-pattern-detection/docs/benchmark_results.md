# Benchmark Results

This document provides a template for recording evaluation runs. The project does not claim any fixed accuracy; results should be reported per drawing and pattern type.

## Benchmark Table Template

| test_case | pattern_type | drawing_size | num_instances | confidence_threshold | num_detections | runtime_seconds | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| example_01 | resistor | 2500x1800 | 8 | 0.50 | 7 | 12.4 | misses 1 faint symbol |
| example_02 | fuse | 2500x1800 | 2 | 0.55 | 2 | 10.9 | clean results |
| example_03 | diode | 2500x1800 | 6 | 0.50 | 8 | 14.2 | 2 false positives in table |

## Notes
- Record the hardware (CPU model, RAM) and OS.
- Report both detections and failure modes (false positives/negatives).
- Use consistent confidence thresholds when comparing runs.
