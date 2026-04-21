# 16-Bit Carry Select Adder

This project now includes the full final report in `assets/final-report.pdf`.

Recommendation for the website:

- Do not embed the full PDF directly into the homepage or a card view. The report is large and better used as a supporting document.
- Pull the key engineering story and metrics into the site itself.
- Offer the PDF as a download or "Full Report" link for deeper review.

## Suggested Portfolio Outline

### Overview

Designed and validated a custom 16-bit carry-select adder in Cadence Virtuoso for Purdue VLSI coursework. The project covered transistor-level gate design, hierarchical schematic development, layout creation, and post-layout performance analysis.

### Design Constraints

- 16-bit adder width with 4-bit subdivision blocks
- Worst-case propagation delay at or below 2 ns
- Energy consumption at or below 700 fJ
- DRC-clean and LVS-clean layout required
- Post-layout parasitic extraction required for validation

### Architecture

- First 4 bits implemented as a 4-bit ripple-carry adder
- Remaining 12 bits divided into three 4-bit carry-select blocks
- Each carry-select block computes carry-in 0 and carry-in 1 cases in parallel
- 2-bit and 4-bit multiplexers select the correct carry and sum outputs

### Implementation Notes

- Logic gates were sized for balanced rise and fall propagation delays
- Full adder used mirror topology after a gate-level implementation proved too slow
- Final system was composed hierarchically from gates, adders, MUXes, and RCA/MUX selector blocks
- Layout was built in Cadence Virtuoso XL using a hierarchical block approach and two metal layers

### Key Results

- Schematic worst-case delay: 410.4 ps average
- Schematic worst-case energy: 121.1 fJ
- Layout worst-case delay: 1.245 ns
- Layout energy estimate: 366.9 fJ
- Final layout passed DRC and LVS checks

### Interesting Detail

The layout energy calculation returned an eval error, so the layout energy was estimated by scaling from the schematic result using the measured increase in delay. That kept the analysis grounded in the collected data while acknowledging the limitation in the direct layout energy computation.

### Lessons Learned

- Mirror-topology full adders offered a major performance advantage over a gate-composed implementation
- Hierarchical layout improved manageability but increased area and interconnect length
- Post-layout validation changed the performance picture enough to make extracted analysis essential
- Clean documentation and staged verification made it easier to justify design tradeoffs

Use `assets/` for future screenshots, layout exports, waveforms, and diagrams pulled from the report.
