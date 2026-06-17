To release a new stable version of `retrochimera` one needs to complete the following steps:

1. Create a PR editing the `CHANGELOG.md` (note it has to be modified in three places).
2. Run the ["Release Stable Version" workflow](https://github.com/microsoft/retrochimera/actions/workflows/release.yml), providing the version number as argument (use the `x.y.z` format _without_ the leading "v") and setting branch to `main`.
3. Ask one of the eligible approvers (anyone in the [@microsoft/msrai4s-dd](https://github.com/orgs/microsoft/teams/msrai4s-dd) team) to approve the workflow. Among other things, they should verify step (1) was completed, and double-check the package version.
4. After approval, the release workflow will push the tag, build the package, and publish to PyPI via [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/).
5. Create a GitHub Release from the [newly created tag](https://github.com/microsoft/retrochimera/tags). Set the name to `retrochimera x.y.z`. The description should be the list of changes copied from the changelog. Consider including a short description before the list of changes to describe the main gist of the release.
