# Contributing

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                  # Python 3.12 with every extra and pytest
uv run pytest tests -m "not gpu"         # CPU tests
```

The calculus (`ecalc/calculus`, `ecalc/formula`, `ecalc/command_line.py`) must keep working with
the standard library alone. To check that, run the tests in an environment without torch:

```bash
python -m venv /tmp/plain && /tmp/plain/bin/pip install pytest
/tmp/plain/bin/python -m pytest tests -m "not gpu"
```

Tests that need torch are skipped there. Tests marked `gpu` need a CUDA GPU.

## Rules of the code base

- **Names are spelled out.** No abbreviations in file names or interface names (`sequential`,
  not `seq`; `KernelSpecification`, not `KernelSpec`). `ops` is the one accepted short form.
- **Measured data is not committed.** The store (`data/ecalc.sqlite`) and the exported signature
  database (`data/signature_database.jsonl`) stay local. Test fixtures under `tests/fixtures`
  hold model structure only; regenerate them with `tests/fixtures/regenerate.sh`.
- **Keys are stable.** A change to `ecalc/signature.py` or `ecalc/calculus/kernel.py` must not
  change the key of an existing kernel. New fields are written only when they differ from their
  default. If a key must change, bump `SIG_VERSION` and say so in the pull request.
- **Never clamp, never guess silently.** A missing kernel is reported as missing. An estimate
  that is not a measurement carries its provenance.
- **Laws are tested exactly.** New algebra goes with a test in `tests/test_laws.py` on the exact
  grid from `tests/calculus_helpers.py`.

## Reporting results

Say which GPU, driver, library versions and static power definition the numbers come from, and
whether persistence mode was on. Report composition error against a whole-model measurement
taken with the same protocol.
