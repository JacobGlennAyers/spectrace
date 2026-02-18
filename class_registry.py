"""
Master class registry for managing an evolving set of annotation classes.

As new layers are added to XCF templates over time, this registry ensures
that all HDF5 files share a consistent class vocabulary and channel ordering.

Key principles:
    - New classes are always APPENDED (existing indices never change)
    - Old HDF5 files can be migrated to include new classes (zero-filled)
    - The registry file is the single source of truth

Usage:
    from class_registry import ClassRegistry

    # Initialize (creates registry file if it doesn't exist)
    registry = ClassRegistry("ml_data/class_registry.json")

    # Sync with current template — discovers new layers, appends them
    registry.sync_with_color_mapping(color_mapping)

    # Use during conversion (instead of sorted(color_mapping.keys()))
    class_names = registry.get_class_names()

    # After adding new classes, migrate old HDF5 files
    registry.migrate_hdf5("ml_data/old_clip.hdf5")
    registry.migrate_all_in_directory("ml_data/")
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class ClassRegistry:
    """
    Maintains an ordered, append-only list of annotation class names.

    The registry file (JSON) looks like:
    {
        "version": 3,
        "classes": [
            {"name": "f0_LFC", "added_version": 1, "added_date": "2025-01-15"},
            {"name": "f0_HFC", "added_version": 1, "added_date": "2025-01-15"},
            ...
            {"name": "new_layer", "added_version": 3, "added_date": "2025-07-20"}
        ]
    }

    The index of each entry in "classes" is its permanent channel index.
    New classes are only ever appended — existing indices never shift.
    """

    def __init__(self, registry_path: str):
        self.path = Path(registry_path)
        self._data = None
        self._load_or_create()

    def _load_or_create(self):
        if self.path.exists():
            with open(self.path, 'r') as f:
                self._data = json.load(f)
        else:
            self._data = {"version": 0, "classes": []}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2)

    # -- Read API -------------------------------------------------------------

    def get_class_names(self) -> List[str]:
        """
        Get the canonical ordered list of class names.

        This is what should be used for the C-axis of mask arrays
        and for the class_names attribute in HDF5 files.
        """
        return [entry['name'] for entry in self._data['classes']]

    def get_version(self) -> int:
        return self._data['version']

    def get_index(self, class_name: str) -> int:
        """Get the permanent channel index for a class name."""
        names = self.get_class_names()
        if class_name not in names:
            raise ValueError(
                f"Class '{class_name}' not in registry. "
                f"Known classes: {names}"
            )
        return names.index(class_name)

    def __len__(self) -> int:
        return len(self._data['classes'])

    def __contains__(self, class_name: str) -> bool:
        return class_name in self.get_class_names()

    # -- Write API ------------------------------------------------------------

    def sync_with_color_mapping(self, color_mapping: Dict[str, str]) -> List[str]:
        """
        Sync the registry with a color mapping (e.g., from the current template).

        Any class names in color_mapping that are NOT already in the registry
        are appended. Existing classes are never removed or reordered.

        Args:
            color_mapping: Dict of {class_name: color_hex} from your template.

        Returns:
            List of newly added class names (empty if nothing changed).
        """
        existing = set(self.get_class_names())
        incoming = set(color_mapping.keys())
        new_classes = sorted(incoming - existing)  # sorted for determinism

        if new_classes:
            new_version = self._data['version'] + 1
            today = datetime.now().strftime("%Y-%m-%d")

            for name in new_classes:
                self._data['classes'].append({
                    'name': name,
                    'added_version': new_version,
                    'added_date': today,
                })

            self._data['version'] = new_version
            self._save()

            print(f"📋 Registry updated to v{new_version}: "
                  f"added {len(new_classes)} new class(es): {new_classes}")
        else:
            print(f"📋 Registry v{self._data['version']} is up to date "
                  f"({len(existing)} classes).")

        return new_classes

    def sync_with_xcf_template(
        self,
        template_path: str,
        layer_group_name: str = "OrcinusOrca_FrequencyContours",
    ) -> List[str]:
        """
        Sync directly with an XCF template file.

        Convenience wrapper that extracts layer names from the template
        and passes them to sync_with_color_mapping.
        """
        from utils import get_or_create_color_mapping
        import tempfile

        # get_or_create_color_mapping expects a project folder; we just
        # need the layer names, so we use it with the template directory.
        template_dir = str(Path(template_path).parent)
        color_mapping = get_or_create_color_mapping(
            template_dir, layer_group_name
        )
        return self.sync_with_color_mapping(color_mapping)

    # -- Migration API --------------------------------------------------------

    def migrate_hdf5(self, hdf5_path: str, dry_run: bool = False) -> bool:
        """
        Migrate an HDF5 file to match the current registry.

        If the file's class_names list is a subset of (or equal to) the
        registry, new zero-filled channels are appended to each annotation's
        mask array so the C-axis matches the registry ordering.

        Args:
            hdf5_path: Path to the HDF5 file.
            dry_run: If True, report what would change without modifying.

        Returns:
            True if the file was modified (or would be), False if already
            up to date.
        """
        registry_names = self.get_class_names()

        with h5py.File(hdf5_path, 'r') as f:
            file_names = json.loads(f.attrs['class_names'])

        # Check if already current
        if file_names == registry_names:
            return False

        # Validate: file's classes must be a prefix or subset we can map
        file_set = set(file_names)
        registry_set = set(registry_names)

        if not file_set.issubset(registry_set):
            removed = file_set - registry_set
            raise ValueError(
                f"File {hdf5_path} contains classes not in registry: "
                f"{removed}. Classes should never be removed from the "
                f"registry — only appended."
            )

        if dry_run:
            new_classes = [n for n in registry_names if n not in file_set]
            print(f"  [dry run] {Path(hdf5_path).name}: would add "
                  f"{len(new_classes)} channels: {new_classes}")
            return True

        # Build mapping: for each registry index, which file channel has it?
        # If the class didn't exist in the file, we'll zero-fill.
        file_index_map = {name: i for i, name in enumerate(file_names)}

        with h5py.File(hdf5_path, 'a') as f:
            # Migrate each annotation set
            ann_grp = f['annotations']
            for idx_key in sorted(ann_grp.keys(), key=int):
                old_masks = ann_grp[idx_key]['masks'][:]  # (C_old, H, W)
                _, H, W = old_masks.shape

                new_masks = np.zeros(
                    (len(registry_names), H, W), dtype=old_masks.dtype
                )

                for new_idx, name in enumerate(registry_names):
                    if name in file_index_map:
                        old_idx = file_index_map[name]
                        new_masks[new_idx] = old_masks[old_idx]
                    # else: stays zero

                # Replace the dataset
                del ann_grp[idx_key]['masks']
                ann_grp[idx_key].create_dataset(
                    'masks', data=new_masks, compression='gzip'
                )

            # Update root attributes
            f.attrs['class_names'] = json.dumps(registry_names)
            f.attrs['num_classes'] = len(registry_names)

        print(f"  ✅ Migrated {Path(hdf5_path).name}: "
              f"{len(file_names)} → {len(registry_names)} classes")
        return True

    def migrate_all_in_directory(
        self,
        directory: str,
        dry_run: bool = False,
    ) -> int:
        """
        Migrate all HDF5 files in a directory to the current registry.

        Args:
            directory: Path to directory containing HDF5 files.
            dry_run: If True, report without modifying.

        Returns:
            Number of files that were (or would be) modified.
        """
        hdf5_files = sorted(Path(directory).glob("*.hdf5"))
        if not hdf5_files:
            print("No HDF5 files found.")
            return 0

        print(f"{'[DRY RUN] ' if dry_run else ''}"
              f"Checking {len(hdf5_files)} files against registry "
              f"v{self.get_version()} ({len(self)} classes)...\n")

        modified = 0
        for path in hdf5_files:
            try:
                if self.migrate_hdf5(str(path), dry_run=dry_run):
                    modified += 1
            except Exception as e:
                print(f"  ❌ {path.name}: {e}")

        action = "would migrate" if dry_run else "migrated"
        print(f"\n📊 {action} {modified}/{len(hdf5_files)} files.")
        return modified
