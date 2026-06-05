import unittest

import torch

from src.training.airl_segment_utils import (
    broadcast_segment_values_to_tokens,
    fixed_interval_boundary_mask,
    normalize_segments_by_group,
    segment_layout_from_boundaries,
    sum_tokens_by_segment,
)


class AIRLSegmentUtilsTest(unittest.TestCase):
    def test_fixed_interval_boundaries_include_last_completion_token(self):
        completion_mask = torch.tensor(
            [
                [False, True, True, True, True, True, False],
                [False, False, True, True, False, False, False],
            ]
        )

        boundary = fixed_interval_boundary_mask(completion_mask, segment_tokens=2)

        expected = torch.tensor(
            [
                [False, False, True, False, True, True, False],
                [False, False, False, True, False, False, False],
            ]
        )
        self.assertTrue(torch.equal(boundary, expected))

    def test_segment_layout_uses_prompt_prefix_and_segment_ends(self):
        completion_mask = torch.tensor([[False, False, True, True, True, True]])
        boundary = torch.tensor([[False, False, False, True, False, True]])

        layout = segment_layout_from_boundaries(boundary, completion_mask)

        self.assertEqual(layout.starts.tolist(), [[2, 4]])
        self.assertEqual(layout.ends.tolist(), [[3, 5]])
        self.assertEqual(layout.prev_indices.tolist(), [[1, 3]])
        self.assertEqual(layout.next_indices.tolist(), [[3, 5]])
        self.assertEqual(layout.valid_mask.tolist(), [[True, True]])

    def test_sum_and_broadcast_segment_values(self):
        completion_mask = torch.tensor([[False, True, True, True, True, False]])
        boundary = torch.tensor([[False, False, True, False, True, False]])
        layout = segment_layout_from_boundaries(boundary, completion_mask)
        token_values = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 0.0]])

        segment_sums = sum_tokens_by_segment(token_values, layout)
        broadcast = broadcast_segment_values_to_tokens(segment_sums, layout, seq_len=6)

        self.assertEqual(segment_sums.tolist(), [[3.0, 7.0]])
        self.assertEqual(broadcast.tolist(), [[0.0, 3.0, 3.0, 7.0, 7.0, 0.0]])

    def test_normalize_segments_by_group_ignores_padding(self):
        values = torch.tensor(
            [
                [1.0, 3.0, 0.0],
                [5.0, 7.0, 9.0],
                [10.0, 0.0, 0.0],
            ]
        )
        valid = torch.tensor(
            [
                [True, True, False],
                [True, True, True],
                [True, False, False],
            ]
        )

        normalized = normalize_segments_by_group(values, valid, group_size=2)

        group_values = normalized[:2][valid[:2]]
        self.assertAlmostEqual(float(group_values.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(group_values.std(unbiased=False)), 1.0, places=6)
        self.assertEqual(float(normalized[0, 2]), 0.0)
        self.assertEqual(float(normalized[2, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
