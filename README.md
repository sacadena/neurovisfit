# coockie-test

[![Release](https://img.shields.io/github/v/release/sacadena/coockie-test)](https://img.shields.io/github/v/release/sacadena/coockie-test)
[![Build status](https://img.shields.io/github/actions/workflow/status/sacadena/coockie-test/main.yml?branch=main)](https://github.com/sacadena/coockie-test/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sacadena/coockie-test/branch/main/graph/badge.svg)](https://codecov.io/gh/sacadena/coockie-test)
[![Commit activity](https://img.shields.io/github/commit-activity/m/sacadena/coockie-test)](https://img.shields.io/github/commit-activity/m/sacadena/coockie-test)
[![License](https://img.shields.io/github/license/sacadena/coockie-test)](https://img.shields.io/github/license/sacadena/coockie-test)

This is a template repository for Python projects that use Poetry for their dependency management.

- **Github repository**: <https://github.com/sacadena/coockie-test/>
- **Documentation** <https://sacadena.github.io/coockie-test/>

## Getting started with your project

First, create a repository on GitHub with the same name as this project, and then run the following commands:

``` bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:sacadena/coockie-test.git
git push -u origin main
```

Finally, install the environment and the pre-commit hooks with 

```bash
make install
```

You are now ready to start development on your project! The CI/CD
pipeline will be triggered when you open a pull request, merge to main,
or when you create a new release.

To finalize the set-up for publishing to PyPi or Artifactory, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).

## Releasing a new version



---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).