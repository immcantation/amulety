# Changelog

## 2.0dev

### Added

- Allow calling Python API as well as CLI.
- Allow choosing sequence column for translation.

### Fixed

- Refactored code to allow scalability to TCR embeddings.
- Added PROTT5 and TCRbert embeddings.

## 1.1 - Blue Tui

### Added

- Added explicit command for BALM-paired embedding
- Added Tutorial
- Added better docs on included embeddings

### Fixed

- When multiple heavy or light chains are present, pick the one with highest duplicate count, not consensus count.
- Consider IGH, TRB and TRD as heavy chains.

## 1.0 - Green kea

- First release on PyPI of amulety
