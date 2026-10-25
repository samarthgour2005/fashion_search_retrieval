# 🤝 Contributing Guide — Fashion Search Studio

Guidelines for contributors to the Fashion Search Studio codebase.

---

## Code Style

- **Python 3.13+** with type hints on all public functions and methods
- Docstrings on all public functions (NumPy docstring style is preferred)
- Line length: 100 characters max
- All configuration values in config.py — no magic numbers in module code

## Git Workflow

`ash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add .
git commit -m "feat: add new embedding fusion mode"

# Run tests before pushing
python -m pytest tests/ -v

# Push
git push origin feature/my-new-feature
`

## Commit Message Convention

`
feat:     New feature
fix:      Bug fix
docs:     Documentation changes
refactor: Code restructuring (no behavior change)
test:     Add or update tests
config:   Config changes
perf:     Performance improvements
`

## Adding a New Search Mode

1. **Engine method** — Add a new method to FashionSearchEngine in etriever/rerank_search.py
   - Accept 	op_k and candidate_pool as parameters
   - Return (results, metadata_dict) tuple
   - Document parameters with NumPy-style docstring

2. **Config** — Add new hyperparameters to config.py with descriptive comments

3. **UI tab** — Add a new tab in pp.py
   - Follow the existing tab structure pattern
   - Add info box explaining the mode
   - Show latency metrics
   - Support HTML export via uild_export_html()
   - Add a _add_history() call

4. **Tests** — Add unit tests in 	ests/

5. **Docs** — Update docs/SEARCH_MODES.md with a description
   - Update the overview table
   - Add a "How It Works" section with ASCII flow diagram

## Adding a New Garment Type

`python
# 1. config.py — add to CATEGORY_TO_GARMENT if it's a Fashionpedia category
CATEGORY_TO_GARMENT = {
    ...
    "tracksuit": "tracksuit",   # <-- new entry
}

# 2. config.py — add to SEGFORMER_LABEL_TO_GARMENT if SegFormer supports it
SEGFORMER_LABEL_TO_GARMENT = {
    ...
    # Only add if SegFormer's label space covers this garment
}

# 3. config.py — add to KNOWN_COLORS if adding a new color too
KNOWN_COLORS = {
    ...
    "beige",  # <-- new color
}

# 4. indexer/color_labeling.py — update hsv_to_color_name() buckets if needed

# 5. Rebuild both indices:
python -m indexer.build_index
python -m indexer.build_garment_index
`

## Testing Guidelines

- Unit tests go in 	ests/test_*.py
- Use unittest.mock.patch to avoid loading actual models in unit tests
- Test both success paths and failure modes (invalid inputs, missing files, etc.)
- Test vocabulary validation (hallucinated garments/colors should be dropped)

## Documentation Standards

When adding new features:
- Update docs/API_REFERENCE.md with function signatures
- Update docs/ARCHITECTURE.md if the system diagram changes
- Update docs/SEARCH_MODES.md if a new search mode is added
- Update README.md if the public-facing features change
- Add inline comments for non-obvious algorithms (especially math)

---

*See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for setup instructions.*
