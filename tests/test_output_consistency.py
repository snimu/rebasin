"""
The purpose of this test is to find out how to properly permute the weights such that
the output of the model stays constant.

This is for different Module-types.
"""

from __future__ import annotations

import torch
from torch import nn

from rebasin.util import reverse_permutation


def test_model_output_consistency_tensors() -> None:
    """
    Demonstrate that the output of a model is invariant to the
    permutation of the weight, iff the input is also permuted accordingly.
    """
    W = torch.randn(10)
    x = torch.randn(10)
    y_orig = x @ W

    perm = torch.randperm(10)
    W_new = W[perm]
    x_new = x[perm]
    y_new = x_new @ W_new

    assert torch.allclose(y_orig, y_new)


class TestConsistencyLinear:

    @staticmethod
    def test_model_output_consistency_lin_only_input() -> None:
        """
        Demonstrate that, as long as the output dimension is
        untouched, permuting the input and the input-dimension of the weights
        yields the same output as the unpermuted input and weight.
        """
        model = nn.Linear(10, 10)
        x = torch.randn(10)
        y_orig = model(x)

        perm = torch.randperm(10)
        model.weight.data = model.weight.data[:, perm]
        y_xfail = model(x)
        assert not torch.allclose(y_orig, y_xfail)

        x_new = x[perm]
        y_xpass = model(x_new)
        assert torch.allclose(y_orig, y_xpass)

    @staticmethod
    def test_permuted_output_produces_same_digits() -> None:
        model = nn.Linear(10, 10, bias=False)
        x = torch.randn(10)
        y_orig = model(x)

        perm_out = torch.randperm(10)
        model.weight.data = model.weight.data[perm_out]
        y_new = model(x)

        for digit in y_new:
            assert digit in y_orig

    @staticmethod
    def test_perm_output_and_next_input_with_same_permutation() -> None:
        for _ in range(100):
            lin1, lin2 = nn.Linear(8, 10, bias=False), nn.Linear(10, 12, bias=False)
            model = nn.Sequential(lin1, lin2)
            x = torch.randn(8)
            y_orig = model(x)

            perm = torch.randperm(10)

            lin1.weight.data = lin1.weight.data[perm]
            lin2.weight.data = lin2.weight.data[:, perm]

            y_new = model(x)
            # Floating point operations lead to enough error that we need to
            # relax the tolerance a bit.
            assert torch.allclose(y_orig, y_new, atol=1e-7, rtol=1e-4)

    @staticmethod
    def test_perm_out_same_perm_in_multi_layer() -> None:
        for _ in range(100):
            lin1 = nn.Linear(8, 10, bias=False)
            lin2 = nn.Linear(10, 12, bias=False)
            lin3 = nn.Linear(12, 14, bias=False)

            model = nn.Sequential(lin1, lin2, lin3)
            x = torch.randn(8)
            y_orig = model(x)

            perm1 = torch.randperm(10)
            perm2 = torch.randperm(12)

            lin1.weight.data = lin1.weight.data[perm1]  # permute output dim of lin1
            lin2.weight.data = lin2.weight.data[:, perm1]  # reverse effects of perm1
            lin2.weight.data = lin2.weight.data[perm2]  # permute output dim of lin2
            lin3.weight.data = lin3.weight.data[:, perm2]  # reverse effects of perm2

            y_new = model(x)

            assert torch.allclose(y_orig, y_new, atol=1e-7, rtol=1e-4)


class TestConsistencyConv2d:

    @staticmethod
    def test_output_dim_perm_is_output_perm() -> None:
        conv = nn.Conv2d(3, 2, (3, 3), bias=False)
        x = torch.randn(1, 3, 9, 9)
        y_orig = conv(x)

        perm = torch.tensor([1, 0])
        conv.weight.data = conv.weight.data[perm]
        y_new = conv(x)

        assert torch.allclose(y_orig, y_new[:, perm])

    @staticmethod
    def test_output_perm_multi_layer() -> None:
        for _ in range(10):
            # conv1.weight.shape == [10, 3, 3, 3]
            conv1 = nn.Conv2d(3, 10, (3, 3), bias=False)
            # conv2.weight.shape == [12, 10, 3, 3]
            conv2 = nn.Conv2d(10, 12, (3, 3), bias=False)
            relu = nn.ReLU()

            model = nn.Sequential(conv1, relu, conv2)
            x = torch.randn(4, 3, 9, 9)
            y_orig = model(x)

            perm = torch.randperm(10)
            conv1.weight.data = conv1.weight.data[perm]
            conv2.weight.data = conv2.weight.data[:, perm]

            y_new = model(x)

            assert torch.allclose(y_orig, y_new, atol=1e-7, rtol=1e-4)


class TestLayerNorm:
    @staticmethod
    def test_multiplication_function() -> None:
        ln_norm = nn.LayerNorm(10)
        ln_weight = nn.LayerNorm(10)
        ln_weight.weight.data *= torch.randn(10)

        x = torch.randn(10)

        y_norm = ln_norm(x)
        y_weight = ln_weight(x)

        # LayerNorm normalizes, and then multiplies **element-wise**!
        assert torch.allclose(y_norm * ln_weight.weight.data, y_weight)

    @staticmethod
    def test_layernorm_after_linear() -> None:
        """If a model ends in a LayerNorm, then output has to be permuted
        by the inverse of the permutation that was applied to the weights
        of the LayerNorm."""
        lin = nn.Linear(10, 10, bias=False)
        ln = nn.LayerNorm(10)
        # Adjust LayerNorm weight to be non-identity
        ln.weight.data *= torch.randn(10)
        model = nn.Sequential(lin, ln)

        x = torch.randn(10)
        y_orig = model(x)

        perm = torch.randperm(10)
        lin.weight.data = lin.weight.data[perm]
        ln.weight.data = ln.weight.data[perm]
        y_new = model(x)
        rev_perm = reverse_permutation(perm)
        y_new = y_new[rev_perm]

        assert torch.allclose(y_orig, y_new, atol=1e-7, rtol=1e-4)

    @staticmethod
    def test_layernorm_between_linears() -> None:
        lin1 = nn.Linear(10, 10, bias=False)
        ln = nn.LayerNorm(10)
        lin2 = nn.Linear(10, 10, bias=False)
        # Adjust LayerNorm weight to be non-identity
        ln.weight.data *= torch.randn(10)
        model = nn.Sequential(lin1, ln, lin2)

        x = torch.randn(10)
        y_orig = model(x)

        perm = torch.randperm(10)
        lin1.weight.data = lin1.weight.data[perm]
        ln.weight.data = ln.weight.data[perm]
        lin2.weight.data = lin2.weight.data[:, perm]
        y_new = model(x)

        assert torch.allclose(y_orig, y_new, atol=1e-7, rtol=1e-4)
