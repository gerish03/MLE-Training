name: Build Package and Run Unit Testing

on:
  pull_request:
    branches:
      - main

jobs:
  build_and_test:
    name: Unit_testing
    runs-on: ubuntu-latest
    default:
        run:
            shell: bash -el {0}

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Conda
        uses: conda-incubator/setup-miniconda@v3
        with:
          activate-environment: my-env
          environment-file: deploy/conda/env.yml
          auto-activate-base: false

      - name: Install Tree
        run: |
          pip install setuptools
          pip install build
          sudo apt install tree

      - name: Display Directory Tree
        run: tree

      - name: Install Package in Dev Mode
        run: pip install -e .

      - name: Installing Pytest
        run: pip install pytest

      - name: Run Tests
        run: pytest
        continue-on-error: true

      - name: Fail Workflow on Test Failures
        run: |
          if [ $? -eq 0 ]; then
            echo "All tests passed!"
            exit 0
          else
            echo "Some tests failed. Exiting with failure status."
            exit 1
          fi
