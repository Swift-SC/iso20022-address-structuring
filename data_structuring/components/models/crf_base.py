"""
This module provides a base class for Conditional Random Field (CRF) models.

It includes the implementation of the forward-backward algorithm for computing marginal probabilities
and the Viterbi algorithm for decoding the most likely sequence of tags.
"""
from abc import abstractmethod
import torch


UNLABELED_INDEX = -1
IMPOSSIBLE_SCORE = -100


def log_sum_exp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


class BaseCRF(torch.nn.Module):
    """BaseCRF"""

    def __init__(self, num_tags: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = torch.nn.Parameter(torch.randn(num_tags))
        self.end_transitions = torch.nn.Parameter(torch.randn(num_tags))
        init_transition = torch.randn(num_tags, num_tags)
        if padding_idx is not None:
            init_transition[:, padding_idx] = IMPOSSIBLE_SCORE
            init_transition[padding_idx, :] = IMPOSSIBLE_SCORE
        self.transitions = torch.nn.Parameter(init_transition)

    def get_transitions(self):
        # Replace every call to self.transitions with this
        return self.transitions

    @abstractmethod
    def forward(self,
                emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: torch.ByteTensor | None = None
                ) -> torch.Tensor:
        raise NotImplementedError()

    def marginal_probabilities(self,
                               emissions: torch.Tensor,
                               mask: torch.ByteTensor | None = None
                               ) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            marginal_probabilities: (sequence_length, sequence_length, num_tags)
        """
        if mask is None:
            batch_size, sequence_length, _ = emissions.data.shape
            mask = torch.ones([batch_size, sequence_length], dtype=torch.uint8, device=emissions.device)

        alpha = self._forward_algorithm(emissions, mask, reverse_direction=False)
        beta = self._forward_algorithm(emissions, mask, reverse_direction=True)
        z = log_sum_exp(alpha[alpha.size(0) - 1] + self.end_transitions, dim=1)

        proba = alpha + beta - z.view(1, -1, 1)
        return torch.exp(proba)

    def _forward_algorithm(self,
                           emissions: torch.Tensor,
                           mask: torch.ByteTensor,
                           reverse_direction: bool = False
                           ) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
            reverse_direction: This parameter decide algorithm direction.
        Returns:
            log_probabilities: (sequence_length, batch_size, num_tags)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        # (sequence_length, batch_size, 1, num_tags)
        broadcast_emissions = emissions.transpose(0, 1).unsqueeze(2).contiguous()
        # (sequence_length, batch_size)
        mask = mask.float().transpose(0, 1).contiguous()
        # (1, num_tags, num_tags)
        broadcast_transitions = self.get_transitions().unsqueeze(0)

        # backward algorithm
        if reverse_direction:
            # Transpose transitions matrix and emissions
            # (1, num_tags, num_tags)
            broadcast_transitions = broadcast_transitions.transpose(1, 2)
            # (sequence_length, batch_size, num_tags, 1)
            broadcast_emissions = broadcast_emissions.transpose(2, 3)
            sequence_iter = range(sequence_length - 1, 0, -1)

            # It is beta
            log_proba = [self.end_transitions.expand(batch_size, num_tags)]
        # forward algorithm
        else:
            # It is alpha
            log_proba = [emissions.transpose(0, 1)[0] + self.start_transitions.view(1, -1)]
            sequence_iter = range(1, sequence_length)

        for i in sequence_iter:
            # Broadcast log probability
            broadcast_log_proba = log_proba[-1].unsqueeze(2)  # (batch_size, num_tags, 1)

            # Add all scores
            # inner: (batch_size, num_tags, num_tags)
            # broadcast_log_proba:   (batch_size, num_tags, 1)
            # broadcast_transitions: (1, num_tags, num_tags)
            # broadcast_emissions:   (batch_size, 1, num_tags)
            inner = broadcast_log_proba + broadcast_transitions + broadcast_emissions[i]

            # Append log proba
            log_proba.append(log_sum_exp(inner, 1) * mask[i].view(batch_size, 1)
                             + log_proba[-1] * (1 - mask[i]).view(batch_size, 1))

        if reverse_direction:
            log_proba.reverse()

        return torch.stack(log_proba)

    def viterbi_decode(self,
                       emissions: torch.Tensor,
                       mask: torch.Tensor | None = None
                       ) -> list[list[int | float]]:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            tags: (batch_size)
        """
        batch_size, sequence_length, _ = emissions.shape
        if mask is None:
            mask = torch.ones([batch_size, sequence_length], dtype=torch.uint8, device=emissions.device)

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        # Start transition and first emission score
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, sequence_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.get_transitions() + broadcast_emissions
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transition score
        score += self.end_transitions

        # Compute the best path
        seq_ends = mask.long().sum(dim=0) - 1

        best_tags_list = []
        for i in range(batch_size):
            _, best_last_tag = score[i].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[i]]):
                best_last_tag = hist[i][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    def decode(self, emissions, mask):
        # Alias
        return self.viterbi_decode(emissions, mask)
