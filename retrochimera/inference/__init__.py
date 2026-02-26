from retrochimera.inference.retrochimera import RetroChimeraModel
from retrochimera.inference.smiles_transformer import SmilesTransformerModel
from retrochimera.inference.template_classification import TemplateClassificationModel
from retrochimera.inference.template_localization import TemplateLocalizationModel


class NeuralSymModel(TemplateClassificationModel):
    pass


class RetroChimeraDeNovoModel(SmilesTransformerModel):
    pass


class RetroChimeraEditModel(TemplateLocalizationModel):
    pass


__all__ = [
    "NeuralSymModel",
    "RetroChimeraDeNovoModel",
    "RetroChimeraEditModel",
    "RetroChimeraModel",
    "SmilesTransformerModel",
    "TemplateClassificationModel",
    "TemplateLocalizationModel",
]
