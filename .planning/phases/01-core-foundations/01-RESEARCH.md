# Phase 1: Core Foundations - Research

**Researched:** 2026-03-04
**Domain:** YAML template externalization, Python config loading, GIMP plugin refactoring
**Confidence:** HIGH

## Summary

Phase 1 replaces four hardcoded Python constants (`ROOT_GROUP_NAME`, `LAYER_STRUCTURE`, `LAYER_SECTIONS`, `LAYER_COLORS`) in `spectrace_annotator.py` with a YAML template file that defines the complete species layer schema. A new shared module (`spectrace/core/templates.py`) loads and validates these templates. All downstream consumers (GIMP plugin, `utils.py`, `hdf5_utils.py`) must read from the same template source.

The codebase currently has zero test infrastructure -- no test framework, no test files, no test config. The hardcoded constants in `spectrace_annotator.py` (lines 44-148) define 26 layers in 4 groups with exact colors and UI sections. The existing `templates/orca_template.yaml` is documentation-only (descriptions for biologists) and must be rewritten into a machine-readable format that also serves the GIMP plugin layer creation, color assignment, and UI section organization.

**Primary recommendation:** Use PyYAML `safe_load` for parsing with hand-written Python validation (no schema library needed -- the template structure is simple and fixed). Create `spectrace/core/templates.py` as the single entry point for all template access. Rewrite `templates/orca_template.yaml` to include layer hierarchy, colors, and UI sections in one file.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- YAML format for all template files -- consistent with existing orca_template.yaml convention
- Templates include inline comments explaining each layer's scientific purpose (self-documenting for biologists)
- Colors merged into the YAML template alongside layer structure -- one file = one species definition
- Hex color strings (#FF0000) for readability; code converts to RGB tuples as needed
- Nested YAML hierarchy -- groups contain their children as nested keys, mirroring the GIMP layer tree
- All-in-one template: spectrogram params + layer hierarchy + colors + UI sections in a single file
- UI sections (LAYER_SECTIONS) included in template -- full species customization, not derived from hierarchy
- Version field included -- stored as GIMP parasite for template-change detection by export pipeline
- Templates live in `templates/` directory, shipped with spectrace
- Plugin functions accept template name as parameter (default: 'orca') -- plumbing for species selection exists from Phase 1
- Auto-discovery: any YAML file in templates/ with the right structure is a valid template
- Malformed templates fail loud with clear error messages -- no silent fallback
- YAML replaces XCF as the single source of truth for layer definitions
- Rewrite existing `orca_template.yaml` into the new machine-readable format (same path, same git history, descriptions become inline comments)
- Full refactor -- all consumers (GIMP plugin, utils.py, hdf5_utils.py) read from YAML template. One source of truth, no split behavior.
- Shared Python module (`spectrace/core/templates.py`) handles all template loading/validation -- used by both GIMP plugin and CLI tools. Starts the `spectrace/core/` directory.

### Claude's Discretion
- Exact YAML schema design (key names, nesting depth)
- Template validation implementation details
- How to handle the `layer_color_mapping.json` migration (may become unnecessary)
- Whether to keep `orca_template.xcf` as a visual reference or remove it

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-01 | Layer schema externalized from hardcoded orca constants to template config files, enabling species-agnostic annotation | All research sections below: YAML schema design, templates.py module, refactoring of all four consumers |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyYAML | 6.x (ships with most Python) | Parse YAML template files | De facto Python YAML parser, already implicit dependency via conda ecosystem |
| Python stdlib | 3.11 | Validation, path handling, hex-to-RGB conversion | No external deps needed for simple schema validation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| collections.OrderedDict | stdlib | Preserve YAML key order if needed | Only if insertion order matters beyond Python 3.7 dict guarantees (it doesn't -- Python 3.11 dicts are ordered) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-written validation | Pydantic, Yamale, pykwalify | Overkill -- template schema is ~5 keys deep, adding a dependency for 20 lines of validation code is wrong. Keep dependencies minimal for a GIMP plugin environment. |
| PyYAML | StrictYAML, ruamel.yaml | StrictYAML is safer but adds a dependency. ruamel.yaml preserves comments but is unnecessary for read-only parsing. PyYAML + safe_load is sufficient and widely available. |

**Installation:**
```bash
pip install pyyaml
# Or add to environment.yml:
#   - pyyaml
```

Note: PyYAML is likely already available in the conda environment as a transitive dependency. Verify with `python -c "import yaml; print(yaml.__version__)"`.

## Architecture Patterns

### Recommended Project Structure
```
spectrace/
├── spectrace/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       └── templates.py       # Template loading, validation, data classes
├── templates/
│   └── orca_template.yaml     # Rewritten machine-readable template
├── gimp_plugin/
│   └── spectrace_annotator.py # Refactored to import from spectrace.core.templates
├── utils.py                   # Refactored: template-aware functions
├── hdf5_utils.py              # Refactored: reads layer names from template
└── tests/
    ├── __init__.py
    └── test_templates.py      # Template loading/validation tests
```

### Pattern 1: Template Data Model
**What:** A plain Python class (dataclass or named tuple) that represents a loaded, validated template. All consumers receive this object -- never raw YAML dicts.
**When to use:** Always -- the template module returns structured objects, not dicts.
**Example:**
```python
# spectrace/core/templates.py
import os
import yaml
from collections import OrderedDict

class TemplateError(Exception):
    """Raised when a template file is malformed or missing."""
    pass

class SpeciesTemplate:
    """Loaded and validated species template."""

    def __init__(self, name, version, root_group_name, layer_structure,
                 layer_colors, layer_sections, spectrogram_params=None):
        self.name = name
        self.version = version
        self.root_group_name = root_group_name
        self.layer_structure = layer_structure  # list of (name, parent, type) tuples
        self.layer_colors = layer_colors        # dict of path-qualified name -> (r, g, b)
        self.layer_sections = layer_sections    # list of (section_name, [layer_paths])
        self.spectrogram_params = spectrogram_params or {}

    def get_layer_names(self):
        """Return set of path-qualified layer names (excluding groups)."""
        return {
            (parent + "/" + name if parent else name)
            for name, parent, ltype in self.layer_structure
            if ltype == "layer"
        }


def hex_to_rgb(hex_str):
    """Convert '#FF0000' to (255, 0, 0)."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        raise TemplateError(f"Invalid hex color: '#{hex_str}'")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def load_template(template_name='orca', templates_dir=None):
    """
    Load and validate a species template by name.

    Args:
        template_name: Name without extension (e.g. 'orca')
        templates_dir: Override templates/ directory path

    Returns:
        SpeciesTemplate object

    Raises:
        TemplateError: If template is missing or malformed
    """
    if templates_dir is None:
        # Default: templates/ relative to project root
        templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'templates'
        )

    path = os.path.join(templates_dir, f"{template_name}_template.yaml")
    if not os.path.isfile(path):
        raise TemplateError(f"Template not found: {path}")

    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    return _validate_and_build(raw, template_name, path)
```

### Pattern 2: GIMP Plugin Import Strategy
**What:** The GIMP plugin (Python-Fu / Python 2.7 environment in GIMP 2.10) cannot use `spectrace.core.templates` via normal imports because the GIMP plugin directory is separate from the project. The plugin must add the spectrace project root to `sys.path`.
**When to use:** In `spectrace_annotator.py` at module level.
**Example:**
```python
# At top of spectrace_annotator.py, after standard imports
import sys
import os

# Add spectrace project root to sys.path so we can import spectrace.core
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PLUGIN_DIR)  # gimp_plugin/../
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spectrace.core.templates import load_template, TemplateError
```

**CRITICAL CAVEAT:** GIMP 2.10 uses Python 2.7 internally. The `spectrace/core/templates.py` module MUST be Python 2.7 compatible OR the plugin must have a compatibility shim. Key concerns:
- No f-strings (use `format()` or `%` formatting)
- No type hints in the module itself (or guard with `if False: ...` for IDE support)
- `yaml.safe_load()` works identically in Python 2.7 PyYAML
- `os.path` operations are identical

**Recommendation:** Write `templates.py` in Python 2/3 compatible style. This is the safest approach since it runs in GIMP's Python 2.7 AND in the external Python 3.11 environment.

### Pattern 3: Template Auto-Discovery
**What:** Scan `templates/` for valid YAML files matching the naming convention `*_template.yaml`.
**When to use:** For listing available templates (future wizard dropdown).
**Example:**
```python
def discover_templates(templates_dir=None):
    """Return list of available template names."""
    if templates_dir is None:
        templates_dir = _default_templates_dir()

    templates = []
    for fname in os.listdir(templates_dir):
        if fname.endswith('_template.yaml'):
            name = fname.replace('_template.yaml', '')
            templates.append(name)
    return sorted(templates)
```

### Anti-Patterns to Avoid
- **Partial migration:** Do NOT keep some constants in Python and some in YAML. All four constants (ROOT_GROUP_NAME, LAYER_STRUCTURE, LAYER_SECTIONS, LAYER_COLORS) must come from the template. One source of truth.
- **Loading YAML in multiple places:** Do NOT call `yaml.safe_load()` directly in `spectrace_annotator.py`, `utils.py`, or `hdf5_utils.py`. Always go through `templates.py`.
- **Lazy validation:** Do NOT silently skip malformed fields. If a layer has no color, raise `TemplateError` immediately. The user decision explicitly says "fail loud."
- **Deep nesting in data model:** The YAML can be nested (groups contain children), but the `SpeciesTemplate.layer_structure` output should be the flat list-of-tuples format that `create_template_layers()` already expects. Transform on load, not on use.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML parsing | Custom parser | `yaml.safe_load()` | YAML spec is deceptively complex (anchors, tags, multiline strings) |
| Hex color conversion | Complex color library | 3-line function (see above) | Only need `#RRGGBB` -> `(R, G, B)` tuple. No alpha, no named colors, no HSV. |
| Template file watching | inotify/fsevents watcher | Reload on each plugin invocation | Templates change rarely. No need for hot-reload. |

**Key insight:** The template loading is straightforward: read YAML, validate required keys exist, transform nested structure to flat layer list, convert hex colors. No library beyond PyYAML is needed.

## Common Pitfalls

### Pitfall 1: GIMP 2.10 Python 2.7 Compatibility
**What goes wrong:** The `templates.py` module is written with Python 3 features (f-strings, type hints, walrus operator) and fails when loaded inside GIMP's Python 2.7.
**Why it happens:** The external Python environment is 3.11, but GIMP 2.10 ships its own Python 2.7 interpreter.
**How to avoid:** Write `templates.py` in polyglot Python 2/3 style. Test by running `python2.7 -c "from spectrace.core.templates import load_template"` if possible.
**Warning signs:** `SyntaxError` in GIMP's error console mentioning f-strings or type hints.

### Pitfall 2: YAML Key Ordering
**What goes wrong:** Layer creation order in GIMP matters (top of layer stack = first created). If YAML dict order doesn't match intended stack order, layers appear in wrong order.
**Why it happens:** YAML mappings are technically unordered, though PyYAML 5.1+ preserves insertion order by default.
**How to avoid:** Use a YAML list for layer ordering (explicit sequence), or document that the YAML key order IS the stack order and rely on PyYAML's preservation behavior. The recommended approach: use an explicit `order` field or a YAML sequence for the hierarchy, not relying on dict key order.
**Warning signs:** Layers appear in alphabetical order instead of the intended scientific order.

### Pitfall 3: Path-Qualified Layer Name Consistency
**What goes wrong:** The template produces layer names like `Heterodynes/0` but some code path produces `Heterodynes/0 ` (trailing space) or `heterodynes/0` (wrong case).
**Why it happens:** Path-qualified names are constructed from group names and layer names. If the YAML has inconsistent whitespace or the code path differs between GIMP plugin and `utils.py`, mismatches occur.
**How to avoid:** The `SpeciesTemplate` object should pre-compute all path-qualified names during loading. All consumers use `template.get_layer_names()` instead of constructing paths themselves.
**Warning signs:** `hdf5_utils.py` validation fails with "extra layers" or "missing layers" after refactoring.

### Pitfall 4: sys.path Pollution in GIMP Plugin
**What goes wrong:** Adding project root to `sys.path` in the GIMP plugin causes import conflicts with GIMP's bundled Python packages.
**Why it happens:** GIMP has its own `os`, `sys`, etc. Adding a project root that contains a `utils.py` (which this project does) can shadow GIMP internals or vice versa.
**How to avoid:** Use `sys.path.insert(0, ...)` to put project root FIRST. Name the package `spectrace/core/` not just `core/`. The import path `spectrace.core.templates` is sufficiently namespaced to avoid conflicts. The existing `utils.py` at project root is NOT in a package, so it won't conflict as long as imports use `spectrace.core.templates`.
**Warning signs:** `ImportError` or wrong module loaded in GIMP.

### Pitfall 5: Breaking the Existing XCF-Based Export Pipeline
**What goes wrong:** After refactoring, `hdf5_utils.py:XCFToHDF5Converter` still expects `template_xcf` parameter and calls `extract_layers_from_xcf()` on the template. But the template is now YAML.
**Why it happens:** Incomplete refactoring -- some code paths still reference the XCF template.
**How to avoid:** Refactor `XCFToHDF5Converter.__init__()` to accept either a template name (YAML) or template XCF path (backward compat). The `_extract_template_classes()` method should be updated to read from YAML via `templates.py`. A deprecation warning can bridge the transition.
**Warning signs:** `FileNotFoundError` on template XCF path, or validation mismatches because XCF and YAML define different layer sets.

### Pitfall 6: GIMP Plugin YAML Dependency
**What goes wrong:** PyYAML is not installed in GIMP 2.10's bundled Python 2.7 environment.
**Why it happens:** GIMP ships a minimal Python 2.7 with limited packages. PyYAML may not be included.
**How to avoid:** Check if PyYAML is available in GIMP's Python. If not, the install script (`gimp_plugin/install_mac.sh`) must install it into GIMP's Python environment, or bundle a vendored copy of PyYAML alongside the plugin.
**Warning signs:** `ImportError: No module named yaml` when running the plugin in GIMP.

## Code Examples

### Recommended YAML Template Schema
```yaml
# templates/orca_template.yaml
# Orca (Orcinus orca) frequency contour annotation template
# Author: Pascale Hatt
# Each species template defines the complete annotation workspace.

template:
  name: orca
  version: "1.0.0"
  species: "Orcinus orca"
  description: "Frequency contour annotation for killer whale vocalizations"

spectrogram:
  nfft: 2048
  grayscale: true

root_group: "OrcinusOrca_FrequencyContours"

# Layer hierarchy: order defines GIMP layer stack (top to bottom).
# Groups contain children. Layers are leaf nodes with a color.
layers:
  - group: Heterodynes
    # Heterodynes occur above and below the HFC of biphonic calls.
    children:
      - layer: unsure
        color: "#2D66E5"
        # Catch-all when affiliated harmonic is unknown
      - layer: "12"
        color: "#74E52D"
      - layer: "11"
        color: "#9EE52D"
      - layer: "10"
        color: "#C9E52D"
      - layer: "9"
        color: "#2D90E5"
      - layer: "8"
        color: "#2DBBE5"
      - layer: "7"
        color: "#2DE5E5"
      - layer: "6"
        color: "#2DE5BB"
      - layer: "5"
        color: "#2DE590"
      - layer: "4"
        color: "#2DE566"
      - layer: "3"
        color: "#2DE53C"
      - layer: "2"
        color: "#4AE52D"
      - layer: "1"
        color: "#E5D72D"
      - layer: "0"
        color: "#E5AD2D"

  - group: Subharmonics
    # Subharmonics at f0/N below fundamentals and harmonics.
    children:
      - layer: subharmonics_HFC
        color: "#2D3CE5"
      - layer: subharmonics_LFC
        color: "#4A2DE5"

  - layer: heterodyne_or_subharmonic_or_other
    color: "#E52DAD"
    # Unsure whether contour is subharmonic, heterodyne, or other

  - group: Cetacean_AdditionalContours
    # Non-orca cetacean vocalizations or ambiguous source
    children:
      - layer: unsure_CetaceanAdditionalContours
        color: "#E5822D"
      - layer: harmonics_CetaceanAdditionalContours
        color: "#E5582D"
      - layer: f0_CetaceanAdditionalContours
        color: "#E52D2D"

  - layer: harmonics_HFC
    color: "#C92DE5"
  - layer: f0_HFC
    color: "#0000FF"
  - layer: unsure_HFC
    color: "#E52D82"
  - layer: harmonics_LFC
    color: "#E52DD7"
  - layer: f0_LFC
    color: "#FF0000"
  - layer: unsure_LFC
    color: "#E52D58"

# UI sections for the annotation control panel.
# Defines radio button groupings in the GIMP plugin.
ui_sections:
  - name: "Main Layers"
    layers:
      - f0_LFC
      - f0_HFC
      - harmonics_LFC
      - harmonics_HFC
      - unsure_LFC
      - unsure_HFC
      - heterodyne_or_subharmonic_or_other

  - name: "Heterodynes"
    layers:
      - Heterodynes/unsure
      - Heterodynes/0
      - Heterodynes/1
      - Heterodynes/2
      - Heterodynes/3
      - Heterodynes/4
      - Heterodynes/5
      - Heterodynes/6
      - Heterodynes/7
      - Heterodynes/8
      - Heterodynes/9
      - Heterodynes/10
      - Heterodynes/11
      - Heterodynes/12

  - name: "Subharmonics"
    layers:
      - Subharmonics/subharmonics_HFC
      - Subharmonics/subharmonics_LFC

  - name: "Cetacean Additional"
    layers:
      - Cetacean_AdditionalContours/f0_CetaceanAdditionalContours
      - Cetacean_AdditionalContours/harmonics_CetaceanAdditionalContours
      - Cetacean_AdditionalContours/unsure_CetaceanAdditionalContours
```

### Template Validation Logic
```python
def _validate_and_build(raw, template_name, path):
    """Validate raw YAML dict and build SpeciesTemplate."""
    if not isinstance(raw, dict):
        raise TemplateError("Template must be a YAML mapping, got: %s" % type(raw).__name__)

    # Required top-level keys
    for key in ('template', 'root_group', 'layers', 'ui_sections'):
        if key not in raw:
            raise TemplateError("Missing required key '%s' in %s" % (key, path))

    tmpl = raw['template']
    version = str(tmpl.get('version', '0.0.0'))
    root_group = raw['root_group']

    # Parse layer hierarchy into flat structure
    layer_structure = []
    layer_colors = {}
    _parse_layers(raw['layers'], None, layer_structure, layer_colors)

    # Parse UI sections
    layer_sections = []
    all_layer_names = {
        (parent + "/" + name if parent else name)
        for name, parent, ltype in layer_structure
        if ltype == "layer"
    }

    for section in raw['ui_sections']:
        section_name = section['name']
        section_layers = section['layers']
        # Validate all referenced layers exist
        for lname in section_layers:
            if lname not in all_layer_names:
                raise TemplateError(
                    "UI section '%s' references unknown layer '%s'" % (section_name, lname)
                )
        layer_sections.append((section_name, section_layers))

    spec_params = raw.get('spectrogram', {})

    return SpeciesTemplate(
        name=template_name,
        version=version,
        root_group_name=root_group,
        layer_structure=layer_structure,
        layer_colors=layer_colors,
        layer_sections=layer_sections,
        spectrogram_params=spec_params,
    )


def _parse_layers(items, parent_name, structure, colors):
    """Recursively parse layer hierarchy from YAML list."""
    for item in items:
        if 'group' in item:
            group_name = item['group']
            structure.append((group_name, parent_name, 'group'))
            if 'children' in item:
                _parse_layers(item['children'], group_name, structure, colors)
        elif 'layer' in item:
            layer_name = str(item['layer'])  # str() handles numeric YAML keys like "0"
            structure.append((layer_name, parent_name, 'layer'))

            # Build path-qualified name for color mapping
            path_name = (parent_name + "/" + layer_name) if parent_name else layer_name

            if 'color' not in item:
                raise TemplateError("Layer '%s' missing required 'color' field" % path_name)
            colors[path_name] = hex_to_rgb(item['color'])
        else:
            raise TemplateError("Layer item must have 'group' or 'layer' key, got: %s" % list(item.keys()))
```

### Refactored GIMP Plugin Constants Replacement
```python
# In spectrace_annotator.py -- BEFORE (hardcoded):
# ROOT_GROUP_NAME = "OrcinusOrca_FrequencyContours"
# LAYER_STRUCTURE = [...]
# LAYER_SECTIONS = [...]
# LAYER_COLORS = {...}

# AFTER (template-driven):
try:
    _TEMPLATE = load_template('orca')
    ROOT_GROUP_NAME = _TEMPLATE.root_group_name
    LAYER_STRUCTURE = _TEMPLATE.layer_structure
    LAYER_SECTIONS = _TEMPLATE.layer_sections
    LAYER_COLORS = _TEMPLATE.layer_colors
except TemplateError as e:
    gimp.message("Spectrace: Failed to load template: %s" % str(e))
    raise
```

### Refactored utils.py Functions
```python
# In utils.py -- functions that currently hardcode "OrcinusOrca_FrequencyContours"
# should accept a SpeciesTemplate or at minimum read from templates.py

def discover_layer_names_from_template(template_name='orca'):
    """Discover all layer names from a YAML template (replaces XCF-based discovery)."""
    from spectrace.core.templates import load_template
    template = load_template(template_name)
    return template.get_layer_names()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded constants in plugin .py | YAML template files | This phase | All layer definitions externalized, species-agnostic |
| XCF template as source of truth | YAML template as source of truth | This phase | No binary file dependency for layer definitions |
| `layer_color_mapping.json` generated from XCF | Colors embedded in YAML template | This phase | One fewer generated artifact, colors are authoritative from template |
| `extract_layers_from_xcf()` for template reading | `load_template()` from YAML | This phase | Simpler, no gimpformats dependency for template reading |

**Deprecated/outdated after this phase:**
- `templates/orca_template.xcf` as source of truth (keep as visual reference only, or remove)
- `discover_layer_names_from_template()` XCF-based version in `utils.py` (replaced by YAML-based version)
- `layer_color_mapping.json` generation for template colors (colors now come from YAML)
- Direct use of `gimpformats` for template reading (still needed for project XCF reading in export pipeline)

## Open Questions

1. **GIMP 2.10 Python 2.7 PyYAML availability**
   - What we know: GIMP 2.10 bundles Python 2.7 with limited packages. PyYAML may or may not be included.
   - What's unclear: Whether the macOS GIMP 2.10 install includes PyYAML. The `install_mac.sh` script does not install PyYAML.
   - Recommendation: Test on target machine. If missing, add `pip install pyyaml` to install script, or vendor a single-file YAML parser as fallback. Alternatively, the plugin could read a pre-converted JSON file that `templates.py` generates from YAML.

2. **Python 2/3 compatibility of templates.py**
   - What we know: GIMP 2.10 = Python 2.7, external tools = Python 3.11
   - What's unclear: How much of the template module logic can be shared vs. duplicated
   - Recommendation: Write `templates.py` in Python 2/3 compatible syntax. Avoid f-strings, use `%` formatting. Avoid type hints (or guard them). This is achievable since the module is pure logic (YAML parsing + validation + data classes).

3. **Backward compatibility of hdf5_utils.py**
   - What we know: `XCFToHDF5Converter` currently takes `template_xcf` path and calls `extract_layers_from_xcf()` to get template layer names.
   - What's unclear: Whether existing HDF5 files need re-validation after the refactor
   - Recommendation: Update `XCFToHDF5Converter` to accept `template_name` parameter alongside (or instead of) `template_xcf`. The `_extract_template_classes()` method reads from YAML. Keep `template_xcf` parameter as deprecated alias that logs a warning.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (to be installed) |
| Config file | none -- see Wave 0 |
| Quick run command | `pytest tests/test_templates.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01a | Template loads layer structure from YAML | unit | `pytest tests/test_templates.py::test_load_orca_template -x` | No -- Wave 0 |
| FOUND-01b | Changing template file changes created layers | unit | `pytest tests/test_templates.py::test_modified_template_produces_different_layers -x` | No -- Wave 0 |
| FOUND-01c | Malformed template raises TemplateError | unit | `pytest tests/test_templates.py::test_malformed_template_fails_loud -x` | No -- Wave 0 |
| FOUND-01d | Loaded template matches hardcoded constants exactly | unit | `pytest tests/test_templates.py::test_orca_template_matches_legacy_constants -x` | No -- Wave 0 |
| FOUND-01e | Template auto-discovery finds templates/ YAML files | unit | `pytest tests/test_templates.py::test_discover_templates -x` | No -- Wave 0 |
| FOUND-01f | Hex color conversion correctness | unit | `pytest tests/test_templates.py::test_hex_to_rgb -x` | No -- Wave 0 |
| FOUND-01g | End-to-end: template -> layer structure -> HDF5 export | integration | `pytest tests/test_templates.py::test_template_to_hdf5_pipeline -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_templates.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/__init__.py` -- empty init for test package
- [ ] `tests/test_templates.py` -- covers FOUND-01a through FOUND-01g
- [ ] `pytest.ini` or `pyproject.toml` [tool.pytest.ini_options] -- pytest config
- [ ] Framework install: `pip install pytest` (add to environment.yml)
- [ ] `spectrace/__init__.py` and `spectrace/core/__init__.py` -- package init files

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis of `spectrace_annotator.py` lines 44-148 (hardcoded constants)
- Direct codebase analysis of `templates/orca_template.yaml` (existing documentation-only YAML)
- Direct codebase analysis of `utils.py` (XCF-based template discovery functions)
- Direct codebase analysis of `hdf5_utils.py` (XCFToHDF5Converter template dependency)
- [PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation) - safe_load API

### Secondary (MEDIUM confidence)
- [Better Stack - YAML files in Python](https://betterstack.com/community/guides/scaling-python/yaml-files-in-python/) - best practices
- [Validate YAML in Python](https://www.andrewvillazon.com/validate-yaml-python-schema/) - validation approaches

### Tertiary (LOW confidence)
- GIMP 2.10 Python 2.7 PyYAML availability (needs hands-on verification on target machine)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyYAML is the de facto Python YAML library, verified through documentation
- Architecture: HIGH - Based on direct analysis of existing codebase; transformation patterns are straightforward
- Pitfalls: HIGH - Python 2/3 compatibility concern is based on known GIMP 2.10 architecture; path-qualified naming concern is based on direct code analysis showing this pattern throughout
- YAML schema design: MEDIUM - Schema is author's recommendation under "Claude's Discretion"; exact key names may need adjustment during implementation

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (stable domain -- YAML parsing and GIMP 2.10 are not changing)
