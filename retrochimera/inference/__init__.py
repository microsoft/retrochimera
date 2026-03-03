from retrochimera.inference.retrochimera import RetroChimeraModel
from retrochimera.inference.smiles_transformer import SmilesTransformerModel
from retrochimera.inference.template_classification import TemplateClassificationModel
from retrochimera.inference.template_localization import TemplateLocalizationModel


class BasicTemplateClassificationModel(TemplateClassificationModel):
    pass


class RetroChimeraDeNovoModel(SmilesTransformerModel):
    pass


class RetroChimeraEditModel(TemplateLocalizationModel):
    pass


__all__ = [
    "BasicTemplateClassificationModel",
    "RetroChimeraDeNovoModel",
    "RetroChimeraEditModel",
    "RetroChimeraModel",
    "SmilesTransformerModel",
    "TemplateClassificationModel",
    "TemplateLocalizationModel",
]
