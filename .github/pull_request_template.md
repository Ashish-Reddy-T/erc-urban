## What changed

<!-- One or two sentences: what this PR does and why. -->

## Which stack(s)

- [ ] erc_indoor
- [ ] erc_urban
- [ ] erc_urban_test
- [ ] erc_offroad
- [ ] earth-rovers-sdk
- [ ] other (docs, CI, tooling)

## Verification checklist

- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test --ignore=test/test_live_sdk.py` passes
- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest erc_urban_test/tests` passes
- [ ] If this touches `erc_urban`, the same fix was mirrored into `erc_urban_test` (or the code differs and mirroring doesn't apply — say why below)
- [ ] No `.env` file, API token, or other secret is included in the diff
- [ ] Branch name and commit messages follow the Contributing section of `README.md`

## Risk / rollout

<!-- What could this break at the venue? Anything that needs a live-rover check before relying on it? -->
