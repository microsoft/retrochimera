Code in this directory comes from the [`dgllife` library](https://github.com/awslabs/dgl-lifesci) (v0.3.0).

As we only need a small fraction of the library (part of `utils/featurizers.py`), we directly copy the relevant code
instead of adding it as a formal dependency.
This avoids pulling in heavy dependencies of `dgllife` (most notably `dgl`).