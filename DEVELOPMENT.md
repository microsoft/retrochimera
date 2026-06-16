To release a new stable version of `retrochimera` one needs to complete the following steps:

1. Create a PR editing the `CHANGELOG.md` (note it has to be modified in three places).
2. Run the ["Release Stable Version" workflow](https://github.com/microsoft/retrochimera/actions/workflows/release.yml), providing the version number as argument (use the `x.y.z` format _without_ the leading "v") and setting branch to `main`. The workflow pushes the tag, builds the package, and publishes to PyPI via [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/).
3. Ask one of the eligible approvers (defined in the `pypi` GitHub environment) to approve the push to PyPI. Among other things, they should double-check the package version.
4. Create a GitHub Release from the [newly created tag](https://github.com/microsoft/retrochimera/tags). Set the name to `retrochimera x.y.z`. The description should be the list of changes copied from the changelog. Consider including a short description before the list of changes to describe the main gist of the release.
