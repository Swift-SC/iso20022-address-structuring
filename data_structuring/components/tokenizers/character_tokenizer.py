"""
Module providing a character tokenizer.
"""
from typing import Sequence


class CharacterTokenizer:

    def __init__(
        self,
        vocabulary: Sequence[str],
        unk_token: str = "\ue003",  # Special unicode character
        pad_token: str = "\x00",  # Special unicode character
    ):

        assert all(
            len(char) == 1 for char in vocabulary
        ), "All characters must be of length 1"

        self._unk_token = unk_token
        self._pad_token = pad_token

        self._vocabulary = [unk_token, pad_token] + vocabulary
        self._char_to_idx = {char: idx for idx, char in enumerate(self._vocabulary)}
        self._idx_to_char = dict(enumerate(self._vocabulary))

    @property
    def vocab_size(self):
        return len(self._vocabulary)

    @property
    def pad_idx(self):
        return self._char_to_idx[self._pad_token]

    def encode(self, string: str):
        return [
            self._char_to_idx.get(char, self._char_to_idx[self._unk_token])
            for char in string
        ]

    def decode(self, indices: Sequence[int]):
        return (
            "".join([self._idx_to_char.get(idx, self._unk_token) for idx in indices])
            .replace(self._unk_token, "<UNKNOWN>")
            .replace(self._pad_token, "<PAD>")
        )
