## [0.2.0]

> Introduces modular model uploads, richer logging, and a more resilient `submit()` workflow.

### Added
- **Model‑store abstraction** with `get_model_store()` and back‑ends for **Hugging Face** and **GitHub**.
- **`ensure_package_installed()`** helper for on‑demand installation of optional dependencies (`torch`, `tensorflow`, `huggingface_hub`, `PyGithub`).
- **`PatraIDGenerationError`** for precise error propagation during PID creation, credential retrieval, and server interactions.
- **Citation support** (`citation` field) in `ModelCard` for easy academic referencing.
- **Inference‑label uploads** via new `inference_labels` argument in `submit()` and corresponding `AIModel.inference_labels` field.
- **Flexible model serialization** — supports **PyTorch** (`pt`, `onnx`) and **TensorFlow** (`h5`) models.
- **Rich logging** throughout the submission workflow (`logging.INFO` default) replacing print statements.
- `_extract_repository_link()` utility for generating clean repository URLs from upload locations.

### Changed
- `populate_requirements()` now excludes `shap` and `fairlearn` by package *name*, not prefix.
- Author field in `ModelCard` auto‑updates to match the PID prefix after a successful submission.
- All network calls use explicit timeouts and clearer error messages.

### Fixed
- Robust schema validation messages with detailed logging of JSONSchema errors.

### Removed / Deprecated
- None.

---

## [0.1.1]

> **Note:** Version `0.1.1` is the **first public release** of the Patra Toolkit.  

### Added
- **Structured schema** for capturing essential model metadata (purpose, datasets, performance metrics).
- **Semi‑automated scanners**: Fairness, Explainability, and Requirements.
- **JSON Model Card** generation and schema validation.
- **Integration** with the Patra Knowledge Base for graph‑based storage and provenance tracking.
- **Command‑line utilities** for authentication, submission, and conflict handling.
