# Changelog

All notable changes to the project are documented in this file.

The format follows [Common Changelog](https://common-changelog.org/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add support for model finetuning ([#21](https://github.com/microsoft/retrochimera/pull/21)) ([@kmaziarz])

## [1.2.0] - 2026-06-17

### Changed

- Speed up localization model loading and inference ([#15](https://github.com/microsoft/retrochimera/pull/15), [#16](https://github.com/microsoft/retrochimera/pull/16)) ([@kmaziarz])
- Speed up SMILES Transformer model inference ([#17](https://github.com/microsoft/retrochimera/pull/17)) ([@kmaziarz])
- Run submodel inference in parallel ([#18](https://github.com/microsoft/retrochimera/pull/18)) ([@kmaziarz])
- Drop the explicit TensorBoard dependency ([#12](https://github.com/microsoft/retrochimera/pull/12)) ([@kmaziarz])

### Added

- Expose setting `num_processes` for template-based models ([#13](https://github.com/microsoft/retrochimera/pull/13)) ([@kmaziarz])

### Fixed

- Fix `weakref` bug preventing garbage collection of template-based models ([#14](https://github.com/microsoft/retrochimera/pull/14)) ([@kmaziarz])

## [1.1.0] - 2026-03-12

### Changed

- Avoid circular imports that would arise during integration into `syntheseus` ([#6](https://github.com/microsoft/retrochimera/pull/6)) ([@kmaziarz])
- Base submodel classes on `ExternalBackwardReactionModel` from `syntheseus` ([#7](https://github.com/microsoft/retrochimera/pull/7)) ([@kmaziarz])
- Expose submodel classes under externally-facing names ([#9](https://github.com/microsoft/retrochimera/pull/9)) ([@kmaziarz])

## [1.0.0] - 2025-11-30

:seedling: Initial public release.

[Unreleased]: https://github.com/microsoft/retrochimera/compare/v1.2.0...HEAD
[1.0.0]: https://github.com/microsoft/retrochimera/releases/tag/v1.0.0
[1.1.0]: https://github.com/microsoft/retrochimera/releases/tag/v1.1.0
[1.2.0]: https://github.com/microsoft/retrochimera/releases/tag/v1.2.0

[@kmaziarz]: https://github.com/kmaziarz
