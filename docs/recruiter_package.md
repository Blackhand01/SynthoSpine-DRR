# Recruiter Package - X23D

## 1. Interview Positioning

I am not presenting this as a finished medical model. I am presenting it as evidence that I understand the system X23D needs: data contracts, projection geometry, multi-view reconstruction, validation metrics, and reproducible artifacts.

Core message:

> I can translate a medical imaging reconstruction paper into a working, testable prototype with explicit geometry and measurable failure modes.

## 2. Five-Slide Outline

### Slide 1 - Why X23D

- Sparse fluoroscopy has clinical value because it can reduce dependency on intraoperative 3D imaging.
- The hard part is preserving patient-specific geometry from few views.
- Projection matrices are the differentiator, not a detail.

### Slide 2 - What I Built

- Python MVP under `src/`.
- Multi-view sample schema.
- Crop-adjusted projection matrices.
- CTSpine1K micro-dataset ingestion and CT-derived DRR proxy generation.
- Isolated L3 target extraction from CTSpine label `22`.
- Patient-level train/validation split for the first non-overfit check.
- X23D-like model: 2D encoder, back-projection, mean fusion, 3D refiner.
- Training loops for real-sample overfit and 20-sample subset execution.
- Evaluation CLI with JSON/CSV/Markdown/PNG/GIF/PLY and smooth mesh artifacts.

### Slide 3 - Technical Fit

- PyTorch model implementation.
- Computer vision preprocessing.
- 3D geometry and projection handling.
- Metrics-driven validation.
- Production mindset from AI systems work.

### Slide 4 - Known Limits

- CT-derived proxy DRRs, not clinical fluoroscopy.
- Real-sample overfit succeeds, but subset generalization is still weak.
- The real-split Kaggle path is designed to expose that weakness instead of hiding it.
- Smooth mesh export exists, but it is a PoC visualization artifact, not a physically calibrated surgical mesh.
- Clinical validity is not claimed.

### Slide 5 - Next Contributions

- More physical DRR generation.
- Patient-level train/validation/test splits.
- GPU training with stronger capacity.
- Calibration sensitivity analysis.
- Domain gap analysis from DRR to real X-ray.

## 3. What I Would Improve At X23D

- Add systematic perturbation tests on projection matrices.
- Compare mean fusion with attention-based view fusion.
- Track uncertainty near pedicles, canal, and low-confidence surfaces.
- Separate data leakage checks from model metrics.
- Report failure cases by anatomy level and view configuration.

## 4. Questions For The Team

- Is the current work focused more on hip, thoracic spine, or a reusable 2D-to-3D platform?
- Are projection matrices treated as reliable inputs or uncertain measurements?
- What is the internal acceptance threshold for ASD/HD95 in early R&D?
- How much real X-ray data is available versus synthetic DRR?
- Which module would benefit most from an intern: data, geometry, model fusion, or validation?

## 5. Closing Pitch

This prototype shows my working style: reduce the research idea to interfaces, preserve the physical assumptions, build a small end-to-end path, and validate it before scaling.
