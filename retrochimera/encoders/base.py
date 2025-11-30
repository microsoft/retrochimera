from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Generic, Optional, TypeVar

from torch import Tensor, nn

from retrochimera.utils.misc import lookup_by_name


@dataclass
class EncoderOutput:
    """Encoder outputs for a batch of chemical objects (e.g. molecules).

    Attributes:
        atom_outputs: Encodings of individual atoms, as a concatenated tensor of shape
            `[total_num_atoms, atom_out_channels]`. Atoms from each molecule appear as consecutive
            segments, within each segment ordered as returned by `GetAtoms` from `rdkit`.
        bond_outputs: Encodings of individual bonds, as a concatenated tensor of shape
            `[2 * total_num_bonds, bond_out_channels]`. Bonds from each molecule appear as
            consecutive segments, within each segment ordered as returned by `GetBond` from `rdkit`.
            Each bond appears twice, in two directions (first the direction returned by `rdkit` and
            then the complementary one).
        mol_outputs: Encodings of entire molecules, as a tensor of shape
            `[batch_size, mol_out_channels]`.
    """

    atom_outputs: Optional[Tensor]
    bond_outputs: Optional[Tensor]
    mol_outputs: Optional[Tensor]


InputType = TypeVar("InputType")
DataType = TypeVar("DataType")
BatchType = TypeVar("BatchType")


class Encoder(nn.Module, Generic[InputType, DataType, BatchType], metaclass=ABCMeta):
    """Abstract interface for classes capable of encoding chemical objects.

    Generics:
        InputType: Type of the raw chemical entity to encode; should either be `Molecule` or another
            type that can also be understood in terms of atoms/bonds (also see `EncoderOutput`).
            Other examples include templates (graph rewrites) and reactions.
        DataType: Type of a single processed datapoint, for example a raw featurized molecular graph.
        BatchType: Type of a batch of processed datapoints, ready for processing by learned
            components of the encoder (if any).
    """

    @abstractproperty
    def preprocess(self) -> Callable[[InputType], DataType]:
        """Preprocess a raw input.

        Note: This function is a property returning a callable instead of a method to avoid the
        entirety of `self` being pickled when preprocessing is wrapped with multiprocessing. In this
        way, only the parts of `self` accessed in the implementation of this property will get
        attached to the function and sent over to the workers.
        """
        pass

    @abstractmethod
    def collate(self, inputs: list[DataType]) -> BatchType:
        pass

    @abstractmethod
    def forward(self, batch: BatchType) -> EncoderOutput:
        pass

    @abstractmethod
    def move_batch_to_device(self, batch: BatchType, device: str) -> BatchType:
        pass

    def forward_raw(self, inputs: list[InputType], device: str = "cpu") -> EncoderOutput:
        """Map raw inputs to final encoder outputs; useful for running inference."""
        batch = self.collate([self.preprocess(input) for input in inputs])
        return self.forward(self.move_batch_to_device(batch, device))

    @abstractproperty
    def atom_out_channels(self) -> Optional[int]:
        pass

    @abstractproperty
    def bond_out_channels(self) -> Optional[int]:
        pass

    @abstractproperty
    def mol_out_channels(self) -> Optional[int]:
        pass


def get_encoder_by_name(encoder_name: str) -> type[Encoder]:
    module = import_module("retrochimera.encoders." + encoder_name.removesuffix("Encoder").lower())
    return lookup_by_name(module, encoder_name)
