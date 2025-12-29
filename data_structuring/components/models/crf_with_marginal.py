"""
Based on https://github.com/kajyuuen/pytorch-partial-crf
"""
import torch
from torch import nn
from data_structuring.components.models.crf_base import BaseCRF, log_sum_exp


class CRF(BaseCRF):
    """Conditional random field."""

    def forward(self,
                emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: torch.ByteTensor | None = None
                ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        gold_score = self._numerator_score(emissions, tags, mask)
        forward_score = self._denominator_score(emissions, mask)
        return torch.sum(forward_score - gold_score)

    def _denominator_score(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask: Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first emissions score
        alpha = self.start_transitions.view(1, num_tags) + emissions[0]

        for i in range(1, sequence_length):
            # Emissions scores
            emissions_score = emissions[i].view(batch_size, 1, num_tags)  # (batch_size, 1, num_tags)
            # Transition scores
            transition_scores = self.get_transitions().view(1, num_tags, num_tags)  # (1, num_tags, num_tags)
            # Broadcast alpha
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)  # (batch_size, num_tags, 1)

            # Add all scores
            inner = broadcast_alpha + emissions_score + transition_scores  # (batch_size, num_tags, num_tags)
            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1)
                     + alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        stops = alpha + self.end_transitions.view(1, num_tags)

        return log_sum_exp(stops)  # (batch_size,)

    def individual_transition_score(self, current_tag, next_tag, ante_tag):
        return self.transitions[current_tag.view(-1), next_tag.view(-1)]

    def _numerator_score(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            tags:  (batch_size, sequence_length)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """

        batch_size, sequence_length, _ = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first emission
        score = self.start_transitions.index_select(0, tags[0])

        for i in range(sequence_length - 1):
            current_tag, next_tag, ante_tag = tags[i], tags[i + 1], tags[max(0, i - 1)]
            # Emissions score for next tag
            emissions_score = (emissions[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1))
            # Transition score from current_tag to next_tag
            transition_score = self.individual_transition_score(current_tag, next_tag, ante_tag)

            # Add all score
            score += transition_score * mask[i + 1] + emissions_score * mask[i]

        # Add end transition score
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to STOP_TAG from each LAST_TAG
        last_transition_score = self.end_transitions.index_select(0, last_tags)

        last_inputs = emissions[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score


class CRFSecondOrder(CRF):
    """
    CRF with order-2 transitions
    """

    def __init__(self, num_tags: int, padding_idx: int | None = None, batch_first: bool = True) -> None:
        super().__init__(num_tags, padding_idx)

        if not batch_first:  # For now only support batch first
            raise NotImplementedError("batch_first must be True")

        # Also intialize an order-2 transition matrix
        self.transitions_order_2 = nn.Parameter(torch.randn(num_tags, num_tags))

    def individual_transition_score(self, current_tag, next_tag, ante_tag):
        # Transition score now depends on both orders
        transition_order_1 = self.transitions[current_tag.view(-1), next_tag.view(-1)]
        transition_order_2 = self.transitions_order_2[ante_tag.view(-1), next_tag.view(-1)]

        return transition_order_1 + transition_order_2

    def get_transitions(self):
        # When getting the full transition matrix (used when calculating normalizer)
        # get the sum of both orders
        return self.transitions + self.transitions_order_2

    def decode(self, emissions, mask):
        # Alias
        return self.viterbi_decode(emissions, mask)

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor | None = None):
        loss = super().forward(emissions, tags, mask)
        # Respect our original convention : return MINUS the log likelihood
        return -1 * loss


if __name__ == "__main__":
    NUMBER_OF_TAGS = 6

    BATCH_SIZE, SEQUENCE_LENGTH = 1, 5

    main_emissions = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, NUMBER_OF_TAGS)

    all_tags = torch.LongTensor([[1, 2, 3, 3, 5]])

    # First order
    model = CRF(NUMBER_OF_TAGS)
    llh = model(main_emissions, all_tags)
    mp_llh = model.marginal_probabilities(main_emissions)

    # Second order
    model2 = CRFSecondOrder(num_tags=NUMBER_OF_TAGS, batch_first=True)

    model2.transitions = torch.nn.Parameter(torch.ones_like(model2.transitions))
    model2.transitions_order_2 = torch.nn.Parameter(
        torch.ones_like(model2.transitions_order_2)
    )
    mp_llh = model2.marginal_probabilities(main_emissions)

    llh = model2(main_emissions, torch.LongTensor([[0, 0, 0, 0, 0]]))
    print(llh)

    llh = model2(main_emissions, all_tags)
    print(llh)

    tagsest = model2.viterbi_decode(
        main_emissions, torch.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.bool)
    )
    print(tagsest)
    llh = model2(main_emissions, torch.LongTensor(tagsest))
    print(llh)
