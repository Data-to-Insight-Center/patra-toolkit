## [0.2.0]

> Introduces modular model uploads, richer logging, and a more resilient `submit()` workflow.

### Added
- Models in Patra are accessed by querying their model cards, which provide metadata and references. The actual model files are stored and managed externally.
- Patra includes an interoperability layer that enables seamless integration with external platforms such as Hugging Face and GitHub, allowing users to upload, retrieve, or link models across these services.

### Changed
- Upgraded schema validation and error messages for clarity.

### Fixed
- Robust schema validation messages with detailed logging of JSONSchema errors.

### Removed / Deprecated
- None.

---

## [0.1.1]

- First public release featuring structured metadata, automated scanners for fairness/explainability, schema validation, Patra Knowledge Base integration, and command-line tools for model submission and management.
