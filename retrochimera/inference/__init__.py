from retrochimera.inference.retrochimera import RetroChimeraModel
from retrochimera.inference.smiles_transformer import SmilesTransformerModel
from retrochimera.inference.smiles_transformer_forward import SmilesTransformerForwardModel
from retrochimera.inference.template_classification import TemplateClassificationModel
from retrochimera.inference.template_localization import TemplateLocalizationModel


class BasicTemplateClassificationModel(TemplateClassificationModel):
    pass


class ForwardChimeraDeNovoModel(SmilesTransformerForwardModel):
    pass


class RetroChimeraDeNovoModel(SmilesTransformerModel):
    pass


class RetroChimeraEditModel(TemplateLocalizationModel):
    pass


__all__ = [
    "BasicTemplateClassificationModel",
    "ForwardChimeraDeNovoModel",
    "RetroChimeraDeNovoModel",
    "RetroChimeraEditModel",
    "RetroChimeraModel",
    "SmilesTransformerModel",
    "SmilesTransformerForwardModel",
    "TemplateClassificationModel",
    "TemplateLocalizationModel",
]
