<div align="center">

# RetroChimera

<p align="center">
  Backed by <a href="https://github.com/microsoft/syntheseus">Syntheseus</a> •
  <a href="https://arxiv.org/abs/2412.05269">Paper</a>
</p>

[![CI](https://github.com/microsoft/syntheseus/actions/retrochimera/ci.yml/badge.svg?branch=main)](https://github.com/microsoft/retrochimera/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![pypi](https://img.shields.io/pypi/v/retrochimera.svg)](https://pypi.org/project/retrochimera/)
[![code style](https://img.shields.io/badge/code%20style-black-202020.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/microsoft/syntheseus/blob/main/LICENSE)

</div>

#

RetroChimera is a frontier retrosynthesis model, built upon ensembling two novel components with complementary inductive biases.
It outperforms existing models by a large margin, can learn from a very small number of examples per reaction class, and is preferred by industrial organic chemists over the reactions it was trained on in blind tests.

## Using RetroChimera via Syntheseus

Easiest way to use RetroChimera will be via Syntheseus. We are currently preparing the integration - stay tuned!

## Using RetroChimera directly

To install `retrochimera` locally with all the extras, run

```bash
conda env create -f environment.yml
conda activate retrochimera

pip install "retrochimera[all]"
```

After installation, you can run inference via

```python
from retrochimera.inference import EnsembleModel
from syntheseus import Molecule

model = EnsembleModel(model_dir="/model/checkpoint/dir/")
mol = Molecule("Oc1ccc(OCc2ccccc2)c(Br)c1")

[predictions] = model([mol], num_results=3)

for p in predictions:
    print(p, f"({100. * p.metadata['probability']:.2f}%)")
```

The `all` extra includes two dependency groups: `dev` for running tests, and `graphium` for building the model architecture we used for USPTO-50K.
If you do not need to run tests nor specifically care about the exact USPTO-50K model, `pip install retrochimera` is enough.

If you want to train your own checkpoint, please follow the instructions in [`retrochimera/README.md`](retrochimera/README.md).

## Checkpoints

The main (and most powerful) checkpoint we release is trained on [Pistachio](https://figshare.com/ndownloader/files/59468882).
For benchmarking, we also provide (weaker) checkpoints trained on [USPTO-50K](https://figshare.com/ndownloader/files/59511926) and [USPTO-FULL](https://figshare.com/ndownloader/files/59494598).

If you care about reproducing the USPTO-* results from our paper _exactly_, make sure to use the inference hyperparameters listed in Extended Data Tables 3 and 4.
By default, these parameters are set to values optimal for the Pistachio checkpoint.

## Citation

If you use RetroChimera in your work, please consider citing our
[arXiV preprint](https://arxiv.org/abs/2412.05269)
(bibtex below).

```
@article{maziarz2025chemist,
  title={Chemist-aligned retrosynthesis by ensembling diverse inductive bias models},
  author={Maziarz, Krzysztof and Liu, Guoqing and Misztela, Hubert and Tripp, Austin and Li, Junren and Kornev, Aleksei and Gai{\'n}ski, Piotr and Hoefling, Holger and Fortunato, Mike and Gupta, Rishi and Segler, Marwin},
  journal={arXiv preprint arXiv:2412.05269},
  year={2025}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
