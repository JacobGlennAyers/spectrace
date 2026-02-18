"""
ML-ready whale vocalization dataset preparation (v2 — consolidated format).

This module handles:
1. Converting XCF annotations to a consolidated HDF5 format
   (one file per audio clip, multiple annotation sets indexed within)
2. Loading data efficiently for ML training
3. Handling multi-class binary masks
4. Managing evolving class vocabularies via ClassRegistry

HDF5 Schema v2.0 layout:
    clip_basename.hdf5
    ├── audio_wav              (opaque dataset — raw .wav bytes)
    │   ├── @filename
    │   └── @size_bytes
    ├── spectrogram            (uint8, HxW, gzip)
    ├── metadata/              (group — shared audio/spectrogram params)
    │   ├── @sample_rate
    │   ├── @nfft
    │   ├── @noverlap
    │   ├── @duration_sec
    │   ├── @max_freq_hz
    │   ├── @time_per_pixel
    │   ├── @freq_per_pixel
    │   └── @audio_path
    ├── @class_names           (JSON string)
    ├── @num_classes
    ├── @num_annotations
    ├── @schema_version        ("2.0")
    ├── @registry_version      (int — registry version at write time)
    └── annotations/
        ├── 0/
        │   ├── masks          (uint8, CxHxW, gzip)
        │   ├── @notes         (str, default "")
        │   └── @timing_drift  (bool, default False)
        ├── 1/
        │   ├── masks
        │   ├── @notes
        │   └── @timing_drift
        └── ...
"""

import os
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
from dataclasses import dataclass, asdict
from collections import defaultdict

from class_registry import ClassRegistry


SCHEMA_VERSION = "2.0"


# =============================================================================
# Exceptions
# =============================================================================

class LayerValidationError(Exception):
    """Raised when project XCF layers don't match the template."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpectrogramMetadata:
    """Metadata for the shared spectrogram / audio parameters."""
    sample_rate: int
    nfft: int
    noverlap: int
    duration_sec: float
    max_freq_hz: float
    time_per_pixel: float
    freq_per_pixel: float
    audio_path: str


@dataclass
class AnnotationInfo:
    """Info for a single annotation set within a clip."""
    index: int
    notes: str
    timing_drift: bool
    non_empty_classes: List[str]
    class_pixel_counts: Dict[str, int]


# =============================================================================
# HDF5 Loader (v2 — consolidated format)
# =============================================================================

class HDF5SpectrogramLoader:
    """
    Loader for consolidated HDF5 spectrogram data (schema v2).

    Each file contains one audio clip with one spectrogram and one or more
    annotation sets (indexed mask arrays).

    Usage:
        loader = HDF5SpectrogramLoader("path/to/clip.hdf5")

        # Basics
        meta = loader.get_metadata()
        class_names = loader.get_class_names()
        spec = loader.load_spectrogram()

        # List annotation indices
        indices = loader.get_annotation_indices()  # e.g. [0, 1, 2]

        # Load masks for a specific annotation set
        masks = loader.load_masks(annotation_index=0)

        # Load a specific class mask from a specific annotation
        mask = loader.get_class_mask("f0_LFC", annotation_index=0)

        # Extract embedded WAV to disk
        loader.extract_wav("output_dir/")

        # Context manager supported
        with HDF5SpectrogramLoader("clip.hdf5") as loader:
            spec = loader.load_spectrogram()
    """

    def __init__(self, hdf5_path: str):
        self.path = Path(hdf5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self._file: Optional[h5py.File] = None
        self._class_names: Optional[List[str]] = None
        self._metadata: Optional[SpectrogramMetadata] = None

    def __enter__(self):
        self._file = h5py.File(self.path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    # -- Schema info ----------------------------------------------------------

    def get_schema_version(self) -> str:
        self._ensure_open()
        return str(self._file.attrs.get('schema_version', '1.0'))

    def get_registry_version(self) -> int:
        self._ensure_open()
        return int(self._file.attrs.get('registry_version', 0))

    # -- Class names ----------------------------------------------------------

    def get_class_names(self) -> List[str]:
        if self._class_names is None:
            self._ensure_open()
            self._class_names = json.loads(self._file.attrs['class_names'])
        return self._class_names

    # -- Metadata -------------------------------------------------------------

    def get_metadata(self) -> SpectrogramMetadata:
        if self._metadata is None:
            self._ensure_open()
            mg = self._file['metadata']
            self._metadata = SpectrogramMetadata(
                sample_rate=int(mg.attrs['sample_rate']),
                nfft=int(mg.attrs['nfft']),
                noverlap=int(mg.attrs['noverlap']),
                duration_sec=float(mg.attrs['duration_sec']),
                max_freq_hz=float(mg.attrs['max_freq_hz']),
                time_per_pixel=float(mg.attrs['time_per_pixel']),
                freq_per_pixel=float(mg.attrs['freq_per_pixel']),
                audio_path=str(mg.attrs['audio_path']),
            )
        return self._metadata

    # -- Spectrogram ----------------------------------------------------------

    def load_spectrogram(self) -> np.ndarray:
        """Load the spectrogram array (H, W)."""
        self._ensure_open()
        return self._file['spectrogram'][:]

    # -- Annotations ----------------------------------------------------------

    def get_annotation_indices(self) -> List[int]:
        """Return sorted list of annotation indices present in the file."""
        self._ensure_open()
        ann = self._file['annotations']
        return sorted(int(k) for k in ann.keys())

    def get_num_annotations(self) -> int:
        self._ensure_open()
        return int(self._file.attrs['num_annotations'])

    def load_masks(self, annotation_index: int = 0) -> np.ndarray:
        """
        Load masks for a specific annotation set.

        Args:
            annotation_index: Which annotation set (0, 1, 2, …)

        Returns:
            Masks array (C, H, W)
        """
        self._ensure_open()
        key = f'annotations/{annotation_index}/masks'
        if key not in self._file:
            available = self.get_annotation_indices()
            raise KeyError(
                f"Annotation index {annotation_index} not found. "
                f"Available: {available}"
            )
        return self._file[key][:]

    def get_class_mask(
        self, class_name: str, annotation_index: int = 0
    ) -> np.ndarray:
        """
        Get mask for a specific class from a specific annotation set.

        Returns:
            Binary mask (H, W)
        """
        class_names = self.get_class_names()
        if class_name not in class_names:
            raise ValueError(
                f"Class '{class_name}' not found. Available: {class_names}"
            )
        idx = class_names.index(class_name)
        self._ensure_open()
        return self._file[f'annotations/{annotation_index}/masks'][idx, :, :]

    def get_annotation_info(self, annotation_index: int = 0) -> AnnotationInfo:
        """Get metadata for a specific annotation set."""
        self._ensure_open()
        grp = self._file[f'annotations/{annotation_index}']
        masks = grp['masks'][:]
        class_names = self.get_class_names()

        non_empty = []
        pixel_counts = {}
        for i, name in enumerate(class_names):
            count = int(masks[i].sum())
            pixel_counts[name] = count
            if count > 0:
                non_empty.append(name)

        return AnnotationInfo(
            index=annotation_index,
            notes=str(grp.attrs.get('notes', '')),
            timing_drift=bool(grp.attrs.get('timing_drift', False)),
            non_empty_classes=non_empty,
            class_pixel_counts=pixel_counts,
        )

    def get_non_empty_classes(self, annotation_index: int = 0) -> List[str]:
        """Get class names with non-zero masks for a given annotation."""
        return self.get_annotation_info(annotation_index).non_empty_classes

    # -- Audio ----------------------------------------------------------------

    def has_audio(self) -> bool:
        self._ensure_open()
        return 'audio_wav' in self._file

    def extract_wav(self, output_dir: str = ".") -> Path:
        """
        Extract the embedded WAV file to disk.

        Returns:
            Path to the extracted file.
        """
        self._ensure_open()
        if 'audio_wav' not in self._file:
            raise KeyError("No embedded WAV found in this file.")
        wav_bytes = bytes(self._file['audio_wav'][()])
        filename = str(self._file['audio_wav'].attrs['filename'])
        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(wav_bytes)
        return out_path

    # -- Convenience ----------------------------------------------------------

    def load(self, annotation_index: int = 0) -> Tuple[
        np.ndarray, np.ndarray, SpectrogramMetadata
    ]:
        """
        Load spectrogram, masks, and metadata in one call.

        Returns:
            (spectrogram, masks, metadata)
        """
        return (
            self.load_spectrogram(),
            self.load_masks(annotation_index),
            self.get_metadata(),
        )

    def get_summary(self) -> Dict:
        """Get a full summary of the file contents."""
        self._ensure_open()
        metadata = self.get_metadata()
        class_names = self.get_class_names()
        indices = self.get_annotation_indices()

        summary = {
            'file_path': str(self.path),
            'schema_version': self.get_schema_version(),
            'registry_version': self.get_registry_version(),
            'clip_basename': self.path.stem,
            'spectrogram_shape': self._file['spectrogram'].shape,
            'num_classes': len(class_names),
            'class_names': class_names,
            'num_annotations': len(indices),
            'annotation_indices': indices,
            'has_audio': self.has_audio(),
            'duration_sec': metadata.duration_sec,
            'sample_rate': metadata.sample_rate,
            'max_freq_hz': metadata.max_freq_hz,
        }

        annotation_details = {}
        for idx in indices:
            info = self.get_annotation_info(idx)
            annotation_details[idx] = {
                'notes': info.notes,
                'timing_drift': info.timing_drift,
                'non_empty_classes': info.non_empty_classes,
                'class_pixel_counts': info.class_pixel_counts,
            }
        summary['annotations'] = annotation_details

        return summary


# =============================================================================
# Converter: XCF project folders → consolidated HDF5
# =============================================================================

class XCFToHDF5Converter:
    """
    Converts raw XCF project folders into consolidated HDF5 files.

    Groups projects by clip_basename so that all annotation passes for
    the same audio clip end up in a single HDF5 file.

    Class ordering is managed by a ClassRegistry, seeded from the
    template XCF. This ensures stable channel indices even as the
    template gains new layers over time.

    Usage:
        converter = XCFToHDF5Converter(
            project_folder="path/to/projects",
            output_folder="path/to/ml_data",
            template_xcf="templates/orca_template.xcf",
        )
        converter.convert_all()
    """

    def __init__(
        self,
        project_folder: str,
        output_folder: str,
        template_xcf: str,
        layer_group_name: str = "OrcinusOrca_FrequencyContours",
        registry_path: Optional[str] = None,
    ):
        """
        Args:
            project_folder: Root folder containing per-project subdirectories.
            output_folder: Where to write HDF5 files and index.
            template_xcf: Path to the master XCF template. This is the
                single source of truth for which classes exist.
            layer_group_name: Name of the layer group in the XCF files.
            registry_path: Path to the class registry JSON. Defaults to
                <output_folder>/class_registry.json.
        """
        self.project_folder = Path(project_folder)
        self.output_folder = Path(output_folder)
        self.template_xcf = Path(template_xcf)
        self.layer_group_name = layer_group_name

        if not self.template_xcf.exists():
            raise FileNotFoundError(
                f"Template XCF not found: {self.template_xcf}"
            )

        self.output_folder.mkdir(parents=True, exist_ok=True)

        if registry_path is None:
            registry_path = str(self.output_folder / "class_registry.json")
        self.registry = ClassRegistry(registry_path)

    def _extract_template_classes(self) -> Set[str]:
        """
        Extract layer names from the template XCF using the same function
        used for project XCFs. This guarantees name format consistency
        (e.g. path-qualified names like 'Heterodynes/0' match between
        template and projects).

        Returns:
            Set of layer name strings from the template.
        """
        from utils import extract_layers_from_xcf

        template_data = extract_layers_from_xcf(
            str(self.template_xcf),
            self.layer_group_name,
            verbose=False,
        )
        return set(template_data['layers'].keys())

    def convert_all(self) -> pd.DataFrame:
        """
        Convert all projects to consolidated HDF5 files.

        Runs a full validation pass on all project XCF files before
        writing anything. If any project contains layers not present
        in the template, conversion is aborted with a detailed report.

        Returns:
            DataFrame index of all converted samples.
        """
        from utils import extract_layers_from_xcf

        # -----------------------------------------------------------------
        # 1. Extract class names directly from the template XCF
        #    Using the SAME function as for project XCFs ensures
        #    identical name formatting (path-qualified or not).
        # -----------------------------------------------------------------
        template_classes = self._extract_template_classes()

        print(f"Template: {self.template_xcf.name}")
        print(f"Template classes ({len(template_classes)}):")
        for name in sorted(template_classes):
            print(f"    {name}")
        print()

        # -----------------------------------------------------------------
        # 2. VALIDATION PASS — check every project XCF before writing
        # -----------------------------------------------------------------
        print("=" * 60)
        print("VALIDATION PASS — checking all project XCF files...")
        print("=" * 60)

        validation_errors: Dict[str, Dict] = {}
        projects_checked = 0

        for project_dir in sorted(self.project_folder.iterdir()):
            if not project_dir.is_dir():
                continue

            xcf_files = list(project_dir.glob("*.xcf"))
            if not xcf_files:
                continue

            projects_checked += 1
            xcf_path = xcf_files[0]

            try:
                layer_data = extract_layers_from_xcf(
                    str(xcf_path),
                    self.layer_group_name,
                    verbose=False,
                )
                project_layers = set(layer_data['layers'].keys())
                extra_layers = project_layers - template_classes
                missing_layers = template_classes - project_layers

                if extra_layers:
                    validation_errors[project_dir.name] = {
                        'xcf_file': xcf_path.name,
                        'extra_layers': sorted(extra_layers),
                        'missing_layers': sorted(missing_layers),
                        'error_type': 'extra_layers',
                    }

            except Exception as e:
                validation_errors[project_dir.name] = {
                    'xcf_file': xcf_path.name,
                    'extra_layers': [],
                    'missing_layers': [],
                    'error_type': 'read_error',
                    'error_message': str(e),
                }

        # Report results
        if validation_errors:
            print(f"\n❌ VALIDATION FAILED — {len(validation_errors)} "
                  f"problem(s) found in {projects_checked} projects:\n")

            extra_layer_issues = {
                k: v for k, v in validation_errors.items()
                if v['error_type'] == 'extra_layers'
            }
            read_errors = {
                k: v for k, v in validation_errors.items()
                if v['error_type'] == 'read_error'
            }

            if extra_layer_issues:
                all_extra: Set[str] = set()
                for v in extra_layer_issues.values():
                    all_extra.update(v['extra_layers'])

                print(f"  LAYER MISMATCH — {len(extra_layer_issues)} "
                      f"project(s) contain layers not in the template:")
                print(f"  Unexpected layers found: {sorted(all_extra)}\n")

                for project_name, info in sorted(extra_layer_issues.items()):
                    print(f"    {project_name}")
                    print(f"      XCF: {info['xcf_file']}")
                    print(f"      Extra layers:   {info['extra_layers']}")
                    if info['missing_layers']:
                        print(f"      Missing layers: {info['missing_layers']}")
                    print()

                print("  Possible causes:")
                print("    - Layer was renamed in the project but not the template")
                print("    - Layer was added to a project manually "
                      "without updating the template")
                print("    - Typo in a layer name")
                print()
                print("  To fix:")
                print("    - If the layer is intentional: "
                      "add it to the template XCF")
                print("    - If it's a mistake: fix/remove it in GIMP "
                      "and re-save the project XCF")

            if read_errors:
                print(f"\n  READ ERRORS — {len(read_errors)} project(s) "
                      f"could not be parsed:")
                for project_name, info in sorted(read_errors.items()):
                    print(f"    {project_name}: "
                          f"{info.get('error_message', 'unknown error')}")

            print(f"\n⛔ Conversion aborted. No files were written.")
            print(f"   Fix the issues above and re-run.")
            raise LayerValidationError(
                f"{len(validation_errors)} project(s) failed validation. "
                f"See report above for details."
            )
        else:
            print(f"\n✅ All {projects_checked} projects validated — "
                  f"every project's layers are a subset of the template.\n")

        # -----------------------------------------------------------------
        # 3. Sync registry with template classes
        #    Build a simple name→name mapping (registry needs a dict)
        # -----------------------------------------------------------------
        template_class_mapping = {name: name for name in template_classes}
        new_classes = self.registry.sync_with_color_mapping(
            template_class_mapping
        )
        class_names = self.registry.get_class_names()
        print(f"Using {len(class_names)} classes from registry "
              f"v{self.registry.get_version()}")

        # -----------------------------------------------------------------
        # 4. If the registry grew, migrate any existing HDF5 files
        # -----------------------------------------------------------------
        if new_classes:
            existing_hdf5 = list(self.output_folder.glob("*.hdf5"))
            if existing_hdf5:
                print(f"\n🔄 Registry grew — migrating {len(existing_hdf5)} "
                      f"existing HDF5 files...")
                self.registry.migrate_all_in_directory(
                    str(self.output_folder)
                )
                print()

        # -----------------------------------------------------------------
        # 5. Discover and group project folders by clip_basename
        # -----------------------------------------------------------------
        clip_groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)

        for project_dir in sorted(self.project_folder.iterdir()):
            if not project_dir.is_dir():
                continue
            clip_basename, project_index = self._parse_project_name(
                project_dir.name
            )
            clip_groups[clip_basename].append((project_index, project_dir))

        for clip_basename in clip_groups:
            clip_groups[clip_basename].sort(key=lambda t: t[0])

        print(f"Found {len(clip_groups)} unique clips across "
              f"{sum(len(v) for v in clip_groups.values())} project folders.\n")

        # -----------------------------------------------------------------
        # 6. Convert each clip group into one HDF5 file
        # -----------------------------------------------------------------
        index_rows = []

        for clip_basename, projects in sorted(clip_groups.items()):
            try:
                rows = self._convert_clip_group(
                    clip_basename, projects, class_names
                )
                index_rows.extend(rows)
                idx_str = ', '.join(str(idx) for idx, _ in projects)
                print(f"✅ {clip_basename}  (indices: {idx_str})")
            except Exception as e:
                print(f"❌ Error converting {clip_basename}: {e}")

        # -----------------------------------------------------------------
        # 7. Save index
        # -----------------------------------------------------------------
        index_df = pd.DataFrame(index_rows)
        index_path = self.output_folder / "dataset_index.csv"
        index_df.to_csv(index_path, index=False)
        print(f"\n📊 Index with {len(index_df)} annotation sets across "
              f"{len(clip_groups)} clips: {index_path}")

        return index_df

    # -----------------------------------------------------------------
    # Standalone validation (can be called without converting)
    # -----------------------------------------------------------------

    def validate(self) -> bool:
        """
        Run the validation pass without converting.

        Checks that every project XCF's layers are a subset of the
        template's layers. Both are extracted with the same function
        (extract_layers_from_xcf) to ensure consistent name formatting.

        Returns:
            True if all projects pass, False otherwise.
        """
        from utils import extract_layers_from_xcf

        template_classes = self._extract_template_classes()

        print(f"Template: {self.template_xcf.name}")
        print(f"Template classes ({len(template_classes)}):")
        for name in sorted(template_classes):
            print(f"    {name}")
        print()

        errors = {}
        checked = 0

        for project_dir in sorted(self.project_folder.iterdir()):
            if not project_dir.is_dir():
                continue
            xcf_files = list(project_dir.glob("*.xcf"))
            if not xcf_files:
                continue

            checked += 1
            xcf_path = xcf_files[0]

            try:
                layer_data = extract_layers_from_xcf(
                    str(xcf_path), self.layer_group_name, verbose=False
                )
                project_layers = set(layer_data['layers'].keys())
                extra = project_layers - template_classes

                if extra:
                    errors[project_dir.name] = sorted(extra)
                    print(f"  ❌ {project_dir.name}: "
                          f"extra layers {sorted(extra)}")
                else:
                    print(f"  ✅ {project_dir.name}")

            except Exception as e:
                errors[project_dir.name] = [f"READ ERROR: {e}"]
                print(f"  ❌ {project_dir.name}: could not read — {e}")

        print(f"\n{'✅ All' if not errors else '❌ ' + str(len(errors))} "
              f"of {checked} projects "
              f"{'passed' if not errors else 'failed'} validation.")

        return len(errors) == 0

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_project_name(name: str) -> Tuple[str, int]:
        """Parse 'clip_basename_INDEX' → (clip_basename, index)."""
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return name, 0

    def _convert_clip_group(
        self,
        clip_basename: str,
        projects: List[Tuple[int, Path]],
        class_names: List[str],
    ) -> List[Dict]:
        """Build one consolidated HDF5 file for a clip."""
        from utils import extract_layers_from_xcf, load_metadata
        from PIL import Image

        first_index, first_dir = projects[0]

        # -- shared spectrogram ------------------------------------------------
        spectrogram_files = list(first_dir.glob("*_spectrogram.png"))
        if not spectrogram_files:
            raise FileNotFoundError(f"No spectrogram PNG in {first_dir}")
        spectrogram = self._to_grayscale_array(Image.open(spectrogram_files[0]))
        print(f"  Spectrogram: shape={spectrogram.shape}, "
              f"range=[{spectrogram.min()}, {spectrogram.max()}]")

        # -- shared metadata ---------------------------------------------------
        raw_meta = load_metadata(str(first_dir))
        spec_metadata = SpectrogramMetadata(
            sample_rate=raw_meta.get('sample_rate', 44100),
            nfft=raw_meta.get('nfft', 2048),
            noverlap=raw_meta.get('noverlap', 1024),
            duration_sec=(
                spectrogram.shape[1] * raw_meta.get('time_per_pixel', 0.01)
            ),
            max_freq_hz=raw_meta.get('sample_rate', 44100) / 2,
            time_per_pixel=raw_meta.get('time_per_pixel', 0.01),
            freq_per_pixel=raw_meta.get('frequency_spacing', 10.0),
            audio_path=raw_meta.get('copied_audio_path', ''),
        )

        # -- shared WAV --------------------------------------------------------
        wav_files = list(first_dir.glob("*.wav"))
        wav_path = wav_files[0] if wav_files else None

        # -- per-annotation masks ---------------------------------------------
        annotation_sets: Dict[int, np.ndarray] = {}

        for proj_index, proj_dir in projects:
            xcf_files = list(proj_dir.glob("*.xcf"))
            if not xcf_files:
                print(f"  ⚠️  No XCF in {proj_dir.name}, skipping")
                continue

            layer_data = extract_layers_from_xcf(
                str(xcf_files[0]),
                self.layer_group_name,
                verbose=False,
            )

            masks_dict = {}
            for class_name in class_names:
                if class_name in layer_data['layers']:
                    mask = layer_data['layers'][class_name]['mask']
                    if mask.shape != spectrogram.shape[:2]:
                        from scipy.ndimage import zoom
                        sy = spectrogram.shape[0] / mask.shape[0]
                        sx = spectrogram.shape[1] / mask.shape[1]
                        mask = zoom(mask, (sy, sx), order=0) > 0.5
                    masks_dict[class_name] = mask.astype(np.uint8)
                else:
                    masks_dict[class_name] = np.zeros(
                        spectrogram.shape[:2], dtype=np.uint8
                    )

            masks_array = np.stack(
                [masks_dict[n] for n in class_names], axis=0
            )
            annotation_sets[proj_index] = masks_array

        # -- write HDF5 -------------------------------------------------------
        output_path = self.output_folder / f"{clip_basename}.hdf5"

        self._write_hdf5(
            path=output_path,
            spectrogram=spectrogram,
            annotation_sets=annotation_sets,
            metadata=spec_metadata,
            class_names=class_names,
            wav_path=wav_path,
            registry_version=self.registry.get_version(),
        )

        # -- build index rows -------------------------------------------------
        rows = []
        for proj_index in sorted(annotation_sets.keys()):
            masks = annotation_sets[proj_index]
            rows.append({
                'clip_basename': clip_basename,
                'annotation_index': proj_index,
                'hdf5_file': f"{clip_basename}.hdf5",
                'num_classes': len(class_names),
                'spectrogram_shape': str(spectrogram.shape),
                'has_annotations': bool(masks.sum() > 0),
                'duration_sec': spec_metadata.duration_sec,
            })
        return rows

    @staticmethod
    def _to_grayscale_array(img) -> np.ndarray:
        if img.mode == 'L':
            return np.array(img)
        elif img.mode == 'RGB':
            return np.array(img)[:, :, 0]
        else:
            return np.array(img.convert('L'))

    @staticmethod
    def _write_hdf5(
        path: Path,
        spectrogram: np.ndarray,
        annotation_sets: Dict[int, np.ndarray],
        metadata: SpectrogramMetadata,
        class_names: List[str],
        wav_path: Optional[Path],
        registry_version: int,
    ):
        """Write the consolidated HDF5 file."""
        with h5py.File(path, 'w') as f:

            # -- root attributes -----------------------------------------------
            f.attrs['schema_version'] = SCHEMA_VERSION
            f.attrs['registry_version'] = registry_version
            f.attrs['class_names'] = json.dumps(class_names)
            f.attrs['num_classes'] = len(class_names)
            f.attrs['num_annotations'] = len(annotation_sets)

            # -- spectrogram (stored once) ------------------------------------
            f.create_dataset(
                'spectrogram', data=spectrogram, compression='gzip'
            )

            # -- metadata (stored once) ---------------------------------------
            meta_grp = f.create_group('metadata')
            for key, value in asdict(metadata).items():
                meta_grp.attrs[key] = value

            # -- embedded WAV (stored once) -----------------------------------
            if wav_path and Path(wav_path).exists():
                with open(wav_path, 'rb') as wf:
                    wav_bytes = wf.read()
                ds = f.create_dataset('audio_wav', data=np.void(wav_bytes))
                ds.attrs['filename'] = Path(wav_path).name
                ds.attrs['size_bytes'] = len(wav_bytes)

            # -- annotation sets (one group per index) ------------------------
            ann_grp = f.create_group('annotations')
            for idx in sorted(annotation_sets.keys()):
                idx_grp = ann_grp.create_group(str(idx))
                idx_grp.create_dataset(
                    'masks',
                    data=annotation_sets[idx],
                    compression='gzip',
                )
                idx_grp.attrs['notes'] = ""
                idx_grp.attrs['timing_drift'] = False


# =============================================================================
# Utility: update annotation attributes after the fact
# =============================================================================

def set_annotation_attrs(
    hdf5_path: str,
    annotation_index: int,
    notes: Optional[str] = None,
    timing_drift: Optional[bool] = None,
):
    """
    Update the notes / timing_drift attributes on an existing annotation.

    Usage:
        set_annotation_attrs(
            "ml_data/clip.hdf5",
            annotation_index=2,
            notes="Annotator was unsure about HFC harmonics",
            timing_drift=True,
        )
    """
    with h5py.File(hdf5_path, 'a') as f:
        key = f'annotations/{annotation_index}'
        if key not in f:
            available = sorted(int(k) for k in f['annotations'].keys())
            raise KeyError(
                f"Annotation index {annotation_index} not found. "
                f"Available: {available}"
            )
        grp = f[key]
        if notes is not None:
            grp.attrs['notes'] = notes
        if timing_drift is not None:
            grp.attrs['timing_drift'] = timing_drift


def batch_set_timing_drift(
    hdf5_path: str,
    drift_map: Dict[int, bool],
):
    """
    Set timing_drift for multiple annotation indices at once.

    Usage:
        batch_set_timing_drift("clip.hdf5", {0: False, 1: True, 2: True})
    """
    with h5py.File(hdf5_path, 'a') as f:
        for idx, drift in drift_map.items():
            key = f'annotations/{idx}'
            if key in f:
                f[key].attrs['timing_drift'] = drift
            else:
                print(f"⚠️  Annotation index {idx} not found, skipping.")


# =============================================================================
# Utility functions
# =============================================================================

def list_hdf5_files(directory: str) -> List[Path]:
    """List all HDF5 files in a directory."""
    return sorted(Path(directory).glob("*.hdf5"))


def load_dataset_index(ml_data_folder: str) -> pd.DataFrame:
    """Load the dataset index CSV."""
    index_path = Path(ml_data_folder) / "dataset_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    return pd.read_csv(index_path)