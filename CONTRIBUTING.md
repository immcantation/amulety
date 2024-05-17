# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/immcantation/amulet/issues](https://github.com/immcantation/amulet/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

AMULET could always use more documentation, whether as part of the official AMULET docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/immcantation/amulet/issues](https://github.com/immcantation/amulet/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome

## Get Started!

Ready to contribute? Here's how to set up `AMULET` for local development.

1. Fork the `AMULET` repo on GitHub.
2. Clone your fork locally:

```
$ git clone git@github.com:<your_name_here>/amulet.git
```

3. Install your local copy into a virtualenv or conda. Assuming you have conda installed, this is how you set up your fork for local development:

```
$ conda create -n amulet python=3.11
$ cd amulet/
$ pip install -e .
```

4. Create a branch for local development:

```
$ git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

5. When you're done making changes, check that the linting tests pass:

```
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

Now pre-commit will run automatically on `git commit`!.
If you want to run it manually run

```
$ pre-commit .
```

6. Check that the tests pass locally, and add tests if necessary.

```
pytest .
```

7. Commit your changes and push your branch to GitHub:

```
$ git add .
$ git commit -m "Your detailed description of your changes."
$ git push origin name-of-your-bugfix-or-feature
```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.rst.
3. The pull request should work for Python 3.8, 3.9, 3.10 and 3.11, and for PyPy. Check [https://github.com/immcantation/amulet/actions](https://github.com/immcantation/amulet/actions) and make sure that the tests pass for all supported Python versions.
4. Automatic GitHub actions CI tests will also run the tests, all the tests must pass before merging the PR.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run:

```
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

GitHub actions will then deploy to PyPI if tests pass.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.
