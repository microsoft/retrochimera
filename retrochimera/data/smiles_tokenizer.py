import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from syntheseus.reaction_prediction.data.dataset import DataFold, DiskReactionDataset

from retrochimera.data.smiles_reaction_sample import SmilesReactionSample
from retrochimera.utils.logging import get_logger

# Special tokens
DEFAULT_BEGIN_TOKEN = "<BOS>"
DEFAULT_END_TOKEN = "<EOS>"
DEFAULT_PAD_TOKEN = "<PAD>"
DEFAULT_UNK_TOKEN = "<UNK>"
PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
REGEX = re.compile(PATTERN)

logger = get_logger(__name__)


class Tokenizer:
    """Tokenizer class for SMILES strings.

    To use it, first create the tokenizer from an existing vocab file or a SMILES file.
    """

    def __init__(
        self,
        vocab: list[str],
        regex: re.Pattern = REGEX,
        begin_token: str = DEFAULT_BEGIN_TOKEN,
        end_token: str = DEFAULT_END_TOKEN,
        pad_token: str = DEFAULT_PAD_TOKEN,
        unk_token: str = DEFAULT_UNK_TOKEN,
    ):
        """Initialize a tokenizer.
        Args:
            vocab: Vocabulary for tokenizer, typically should include the above four special tokens
            regex: Regex object for tokenizing
            begin_token: Token to use at start of each sequence
            end_token: Token to use at end of each sequence
            pad_token: Token to use when padding batches of sequences
            unk_token: Token to use for tokens which are not in the vocabulary
        """
        self.begin_token = begin_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        vocab_dict = {
            begin_token: 0,
            end_token: 1,
            pad_token: 2,
            unk_token: 3,
        }
        for token in vocab:
            vocab_dict.setdefault(token, len(vocab_dict))

        self.vocab_dict = vocab_dict  # token -> id
        self.decode_vocab_dict = {i: t for t, i in self.vocab_dict.items()}  # id -> token
        self.regex = regex
        self.begin_token_id = self.vocab_dict[begin_token]
        self.end_token_id = self.vocab_dict[end_token]
        self.pad_token_id = self.vocab_dict[pad_token]
        self.unk_token_id = self.vocab_dict[unk_token]
        self.met_unk_tokens_dict: dict[str, int] = defaultdict(int)

        self.log_vocab_info()

    def log_vocab_info(self) -> None:
        logger.info(f"Vocabulary size: {len(self.vocab_dict)}")
        logger.info(f"Vocabulary: {self.vocab_dict}")

    @staticmethod
    def from_vocab_file(
        vocab_path: str,
        regex: re.Pattern = REGEX,
        begin_token: str = DEFAULT_BEGIN_TOKEN,
        end_token: str = DEFAULT_END_TOKEN,
        pad_token: str = DEFAULT_PAD_TOKEN,
        unk_token: str = DEFAULT_UNK_TOKEN,
    ):
        """Build a tokenizer from a vocab file and regex.
        Args:
            vocab_path: Path to vocab file
            regex: Regex object for tokenizing
        Returns:
            Tokenizer object
        """
        text = Path(vocab_path).read_text()
        tokens = text.split("\n")
        tokens = [t for t in tokens if t is not None and t != ""]

        tokenizer = Tokenizer(
            tokens,
            regex,
            begin_token=begin_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        return tokenizer

    @staticmethod
    def from_smiles(
        smiles: list[str],
        regex: re.Pattern = REGEX,
        begin_token: str = DEFAULT_BEGIN_TOKEN,
        end_token: str = DEFAULT_END_TOKEN,
        pad_token: str = DEFAULT_PAD_TOKEN,
        unk_token: str = DEFAULT_UNK_TOKEN,
    ):
        """Build a tokenizer from smiles strings and regex.
        Args:
            smiles: SMILES strings to use to build vocabulary
            regex: Regex object for tokenizing
        """
        vocab = set()
        for smi in smiles:
            for token in regex.findall(smi):
                vocab.add(token)

        tokenizer = Tokenizer(
            list(vocab),
            regex,
            begin_token=begin_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        return tokenizer

    @staticmethod
    def from_reactions(
        reaction_data_path: str,
        regex: re.Pattern = REGEX,
        begin_token: str = DEFAULT_BEGIN_TOKEN,
        end_token: str = DEFAULT_END_TOKEN,
        pad_token: str = DEFAULT_PAD_TOKEN,
        unk_token: str = DEFAULT_UNK_TOKEN,
    ):
        """Build the tokenizer from reaction data and a pattern.
        Args:
            reaction_data_path: Path to reaction data
            regex: Regex object for tokenizing
        """
        reaction_dataset = DiskReactionDataset(reaction_data_path, sample_cls=SmilesReactionSample)
        training_dataset = reaction_dataset[DataFold.TRAIN]

        smiles = []
        for sample in training_dataset:
            smiles.append(sample.raw_reactants_smiles)
            smiles.append(sample.raw_products_smiles)

        tokenizer = Tokenizer.from_smiles(
            smiles,
            regex,
            begin_token=begin_token,
            end_token=end_token,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        return tokenizer

    def save_vocab(self, vocab_path: str) -> None:
        """Save the vocabulary to a file."""
        tokens = [key for key, val in sorted(self.vocab_dict.items(), key=lambda x: x[1])]

        tokens_str = ""
        for token in tokens:
            tokens_str += f"{token}\n"

        p = Path(vocab_path)
        p.write_text(tokens_str)

    def __len__(self) -> int:
        return len(self.vocab_dict)

    def encode(
        self,
        smiles: list[str],
        pad: bool = False,
        right_padding: bool = True,
        add_begin_token: bool = False,
        add_end_token: bool = False,
    ) -> tuple[list[list[int]], Optional[list[list[int]]]]:
        """Convert a list of smiles to a list of token ids."""
        tokens_list = [self.regex.findall(smi) for smi in smiles]
        if add_begin_token:
            tokens_list = [[self.begin_token] + tokens for tokens in tokens_list]
        if add_end_token:
            tokens_list = [tokens + [self.end_token] for tokens in tokens_list]

        token_ids_list = self._convert_tokens_to_ids(tokens_list)

        if pad:
            token_ids_list, masks = self.pad_token_ids_list(
                token_ids_list, right_padding=right_padding
            )
            return token_ids_list, masks

        return token_ids_list, None

    def _convert_tokens_to_ids(self, token_data: list[list[str]]) -> list[list[int]]:
        """Convert a list of tokenized sequences to a list of token ids."""
        token_ids_list = []
        for tokens in token_data:
            for token in tokens:
                token_id = self.vocab_dict.get(token)
                if token_id is None:
                    self.met_unk_tokens_dict[token] += 1

            ids = [self.vocab_dict.get(token, self.unk_token_id) for token in tokens]
            token_ids_list.append(ids)
        return token_ids_list

    def pad_token_ids_list(
        self, token_ids_list: list[list[int]], right_padding: bool = True
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Pad sequences to the same length and create masks for padding."""
        pad_length = max([len(token_ids) for token_ids in token_ids_list])
        if right_padding:
            padded_token_ids_list = [
                token_ids + [self.pad_token_id] * (pad_length - len(token_ids))
                for token_ids in token_ids_list
            ]
            masks = [
                [0] * len(token_ids) + [1] * (pad_length - len(token_ids))
                for token_ids in token_ids_list
            ]
        else:
            padded_token_ids_list = [
                [self.pad_token_id] * (pad_length - len(token_ids)) + token_ids
                for token_ids in token_ids_list
            ]
            masks = [
                [1] * (pad_length - len(token_ids)) + [0] * len(token_ids)
                for token_ids in token_ids_list
            ]
        return padded_token_ids_list, masks

    def decode(self, token_ids_list: list[list[int]]) -> list[str]:
        """Convert a list of token ids to a list of sequences."""
        processed_token_ids_list = []
        for token_ids in token_ids_list:
            if token_ids[0] == self.begin_token_id:
                token_ids = token_ids[1:]

            # Remove any tokens after the end token (and end token) if it's there
            if self.end_token_id in token_ids:
                end_token_pos = token_ids.index(self.end_token_id)
                token_ids = token_ids[:end_token_pos]

            processed_token_ids_list.append(token_ids)

        processed_tokens_list = self._convert_ids_to_tokens(processed_token_ids_list)
        smiles = ["".join(tokens) for tokens in processed_tokens_list]
        return smiles

    def _convert_ids_to_tokens(self, token_ids: list[list[int]]) -> list[list[str]]:
        """Convert a list of token ids to a list of tokenized sequences."""
        tokens_list = []
        for ids in token_ids:
            for token_id in ids:
                token = self.decode_vocab_dict.get(token_id)
                if token is None:
                    raise ValueError(f"Token id {token_id} is not recognised")

            tokens = [str(self.decode_vocab_dict.get(token_id)) for token_id in ids]
            tokens_list.append(tokens)
        return tokens_list

    def print_unknown_tokens(self) -> None:
        """Used when training on a new dataset."""
        logger.info(f"{'Token':<10}Count")
        for token, count in self.met_unk_tokens_dict.items():
            logger.info(f"{token:<10}{count}")
