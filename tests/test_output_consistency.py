"""
The purpose of this test is to find out how to properly permute the weights such that
the output of the model stays constant.

This is for different Module-types.
"""

from __future__ import annotations

import copy
import itertools

import pytest
import torch
from torch import nn

from tests.fixtures.utils import allclose, model_change_percent


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

    assert allclose(y_orig, y_new)


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
        assert torch.allclose(y_orig, y_xpass, atol=1e-7, rtol=1e-4)

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
            assert allclose(y_orig, y_new)

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

            assert allclose(y_orig, y_new)

    @staticmethod
    def test_two_layer_control() -> None:
        lin1 = nn.Linear(8, 10, bias=False)
        lin2 = nn.Linear(10, 8, bias=False)

        model = nn.Sequential(lin1, lin2)
        x = torch.randn(8)
        y_orig = model(x)

        perm1 = torch.tensor([1, 2, 5, 3, 8, 9, 0, 6, 7, 4])
        perm2 = torch.tensor([4, 7, 2, 6, 0, 3, 1, 9, 5, 8])

        lin1.weight.data = lin1.weight.data[perm1]
        lin2.weight.data = lin2.weight.data[:, perm2]

        # This shouldn't work if lin1-output_perm != lin2-input_perm
        y_new = model(x)
        assert not allclose(y_orig, y_new)


class TestConsistencyConv1d:

    @staticmethod
    def test_output_dim_perm_is_output_perm() -> None:
        conv = nn.Conv1d(3, 2, 3, bias=False)
        x = torch.randn(1, 3, 9)
        y_orig = conv(x)

        perm = torch.tensor([1, 0])
        conv.weight.data = conv.weight.data[perm]
        y_new = conv(x)

        assert torch.allclose(y_orig, y_new[:, perm])

    @staticmethod
    def test_output_perm_multi_layer() -> None:
        for _ in range(10):
            conv1 = nn.Conv1d(3, 10, 3, bias=False)
            conv2 = nn.Conv1d(10, 12, 3, bias=False)
            relu = nn.ReLU()

            model = nn.Sequential(conv1, relu, conv2)
            x = torch.randn(4, 3, 9)
            y_orig = model(x)

            perm = torch.randperm(10)
            conv1.weight.data = conv1.weight.data[perm]
            conv2.weight.data = conv2.weight.data[:, perm]

            y_new = model(x)

            assert allclose(y_orig, y_new)


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

            assert allclose(y_orig, y_new)


class TestConsistencyConv3d:

    @staticmethod
    def test_output_dim_perm_is_output_perm() -> None:
        for in_channels, out_channels, kernel_size in itertools.product(
                [1, 3], [1, 2], [2, 3]
        ):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
            x = torch.randn(1, in_channels, 5, 5, 5)
            y_orig = conv(x)

            perm = torch.randperm(out_channels)
            conv.weight.data = conv.weight.data[perm]
            y_new = conv(x)

            assert torch.allclose(y_orig, y_new[:, perm])

    @staticmethod
    def test_output_perm_multi_layer() -> None:
        for in_channels, _out_channels, kernel_size in itertools.product(
                [1, 3], [1, 2], [2, 3]
        ):
            conv1 = nn.Conv3d(in_channels, 6, kernel_size, bias=False)
            conv2 = nn.Conv3d(6, 4, kernel_size, bias=False)
            relu = nn.ReLU()
            model = nn.Sequential(conv1, relu, conv2)
            x = torch.randn(4, in_channels, 5, 5, 5)
            y_orig = model(x)

            perm = torch.randperm(6)
            conv1.weight.data = conv1.weight.data[perm]
            conv2.weight.data = conv2.weight.data[:, perm]

            y_new = model(x)

            assert allclose(y_orig, y_new)


class TestLayerNorm:

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
        rev_perm = torch.argsort(perm)
        y_new = y_new[rev_perm]

        assert allclose(y_orig, y_new)

    @staticmethod
    def test_layernorm_before_linear() -> None:
        ln = nn.LayerNorm(10)
        # Adjust LayerNorm weight to be non-identity
        ln.weight.data *= torch.randn(10)
        lin = nn.Linear(10, 10, bias=False)
        model = nn.Sequential(ln, lin)

        x = torch.randn(10)
        y_orig = model(x)

        perm = torch.randperm(10)
        ln.weight.data = ln.weight.data[perm]
        lin.weight.data = lin.weight.data[:, perm]

        y_new = model(x[perm])  # permute input to match permutation of weights

        assert allclose(y_orig, y_new)

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

        assert allclose(y_orig, y_new)

    @staticmethod
    def test_layernorm_before_conv() -> None:
        for _ in range(10):
            x = torch.randn(2, 3, 18, 18)

            ln = nn.LayerNorm([2, 3, 18, 18])
            ln.weight.data *= torch.randn(2, 3, 18, 18)
            conv = nn.Conv2d(3, 4, (3, 3), bias=False)

            model = nn.Sequential(ln, conv)
            y_orig = model(x)

            perm_in = torch.randperm(3)
            ln.weight.data = ln.weight.data[:, perm_in]
            conv.weight.data = conv.weight.data[:, perm_in]
            y_new = model(x[:, perm_in])

            assert allclose(y_orig, y_new)

    @staticmethod
    def test_layernorm_after_conv() -> None:
        for _ in range(10):
            x = torch.randn(2, 3, 4, 4)

            conv = nn.Conv2d(3, 4, (3, 3), bias=False)
            ln = nn.LayerNorm([2, 4, 2, 2])
            ln.weight.data *= torch.randn(2, 4, 2, 2)

            model = nn.Sequential(conv, ln)
            y_orig = model(x)

            perm_out = torch.randperm(4)
            conv.weight.data = conv.weight.data[perm_out]
            ln.weight.data = ln.weight.data[:, perm_out]
            y_new = model(x)

            assert allclose(y_orig, y_new[:, torch.argsort(perm_out)])

    @staticmethod
    def test_layernorm_between_conv() -> None:
        for _ in range(10):
            x = torch.randn(2, 3, 6, 6)

            conv1 = nn.Conv2d(3, 4, (3, 3), bias=False)
            ln = nn.LayerNorm([2, 4, 4, 4])
            ln.weight.data *= torch.randn(2, 4, 4, 4)
            conv2 = nn.Conv2d(4, 5, (3, 3), bias=False)

            model = nn.Sequential(conv1, ln, conv2)
            y_orig = model(x)

            perm = torch.randperm(4)
            conv1.weight.data = conv1.weight.data[perm]
            ln.weight.data = ln.weight.data[:, perm]
            conv2.weight.data = conv2.weight.data[:, perm]
            y_new = model(x)

            assert allclose(y_orig, y_new)


class TestBatchNorm1d:

    @staticmethod
    def test_batchnorm_after_linear() -> None:
        """If a model ends in a BatchNorm1d, then output has to be permuted
        by the inverse of the permutation that was applied to the weights
        of the BatchNorm1d.

        However, in BatchNorm1d, that inverse has to be applied to the
        input-dimension of the output!
        """
        lin = nn.Linear(10, 10, bias=False)
        bn = nn.BatchNorm1d(10)
        # Adjust BatchNorm weight to be non-identity
        bn.weight.data *= torch.randn(10)
        model = nn.Sequential(lin, bn)

        x = torch.randn(3, 10)
        y_orig = model(x)

        perm = torch.randperm(10)
        lin.weight.data = lin.weight.data[perm]
        bn.weight.data = bn.weight.data[perm]
        bn.reset_running_stats()
        y_new = model(x)
        rev_perm = torch.argsort(perm)
        y_new = y_new[:, rev_perm]

        assert allclose(y_orig, y_new)

    @staticmethod
    def test_batchnorm_before_linear() -> None:
        bn = nn.BatchNorm1d(10)
        # Adjust BatchNorm weight to be non-identity
        bn.weight.data *= torch.randn(10)
        lin = nn.Linear(10, 10, bias=False)
        model = nn.Sequential(bn, lin)

        x = torch.randn(3, 10)
        y_orig = model(x)

        bn.reset_running_stats()
        perm = torch.randperm(10)
        bn.weight.data = bn.weight.data[perm]
        lin.weight.data = lin.weight.data[:, perm]

        y_new = model(x[:, perm])

        assert allclose(y_orig, y_new)

    @staticmethod
    def test_batchnorm_between_linears() -> None:
        lin1 = nn.Linear(10, 10, bias=False)
        bn = nn.BatchNorm1d(10)
        lin2 = nn.Linear(10, 10, bias=False)
        # Adjust BatchNorm weight to be non-identity
        bn.weight.data *= torch.randn(10)
        model = nn.Sequential(lin1, bn, lin2)

        x = torch.randn(3, 10)
        y_orig = model(x)

        perm = torch.randperm(10)
        lin1.weight.data = lin1.weight.data[perm]
        bn.weight.data = bn.weight.data[perm]
        lin2.weight.data = lin2.weight.data[:, perm]
        bn.reset_running_stats()
        y_new = model(x)

        assert allclose(y_orig, y_new)


class TestBatchNorm2d:
    @staticmethod
    def test_batchnorm_after_conv2d() -> None:
        conv = nn.Conv2d(3, 10, 3, bias=False)
        bn = nn.BatchNorm2d(10)
        model = nn.Sequential(conv, bn)

        x = torch.randn(4, 3, 9, 9)
        y_orig = model(x)

        perm = torch.randperm(10)
        conv.weight.data = conv.weight.data[perm]
        bn.weight.data = bn.weight.data[perm]
        bn.reset_running_stats()
        y_new = model(x)
        rev_perm = torch.argsort(perm)
        y_new = y_new[:, rev_perm]

        assert allclose(y_orig, y_new)

    @staticmethod
    def test_batchnorm_before_conv2d() -> None:
        bn = nn.BatchNorm2d(3)
        conv = nn.Conv2d(3, 10, 3, bias=False)
        model = nn.Sequential(bn, conv)

        x = torch.randn(4, 3, 9, 9)
        y_orig = model(x)

        bn.reset_running_stats()
        perm = torch.randperm(3)
        bn.weight.data = bn.weight.data[perm]
        conv.weight.data = conv.weight.data[:, perm]

        y_new = model(x[:, perm])

        assert allclose(y_orig, y_new)


class TestConsistencyBatchNorm3d:

    @staticmethod
    def test_batchnorm_after_conv3d() -> None:
        for in_channels, out_channels, kernel_size in itertools.product(
                [1, 3], [1, 2], [2, 3]
        ):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
            bn = nn.BatchNorm3d(out_channels)
            # Adjust BatchNorm weight to be non-identity
            bn.weight.data *= torch.randn(out_channels)
            model = nn.Sequential(conv, bn)

            x = torch.randn(4, in_channels, 5, 5, 5)
            y_orig = model(x)

            perm = torch.randperm(out_channels)
            conv.weight.data = conv.weight.data[perm]
            bn.weight.data = bn.weight.data[perm]
            bn.reset_running_stats()
            y_new = model(x)
            rev_perm = torch.argsort(perm)
            y_new = y_new[:, rev_perm]

            assert allclose(y_orig, y_new)

    @staticmethod
    def test_batchnorm_before_conv3d() -> None:
        for in_channels, out_channels, kernel_size in itertools.product(
                [1, 3], [1, 2], [2, 3]
        ):
            bn = nn.BatchNorm3d(in_channels)
            # Adjust BatchNorm weight to be non-identity
            bn.weight.data *= torch.randn(in_channels)
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
            model = nn.Sequential(bn, conv)

            x = torch.randn(4, in_channels, 5, 5, 5)
            y_orig = model(x)

            bn.reset_running_stats()
            perm = torch.randperm(in_channels)
            bn.weight.data = bn.weight.data[perm]
            conv.weight.data = conv.weight.data[:, perm]

            y_new = model(x[:, perm])

            assert allclose(y_orig, y_new)

    @staticmethod
    def test_batchnorm_between_conv3d() -> None:
        for in_channels, mid_channels, out_channels, kernel_size in itertools.product(
                [1, 3], [4], [1, 2], [2, 3]
        ):
            conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size, bias=False)
            bn = nn.BatchNorm3d(mid_channels)
            conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size, bias=False)
            # Adjust BatchNorm weight to be non-identity
            bn.weight.data *= torch.randn_like(bn.weight.data)
            model = nn.Sequential(conv1, bn, conv2)

            x = torch.randn(4, in_channels, 5, 5, 5)
            y_orig = model(x)

            bn.reset_running_stats()
            perm = torch.randperm(mid_channels)
            bn.weight.data = bn.weight.data[perm]
            conv2.weight.data = conv2.weight.data[:, perm]
            conv1.weight.data = conv1.weight.data[perm]

            y_new = model(x)

            assert allclose(y_orig, y_new)


class TestMultiheadAttention:
    @staticmethod
    def test_io() -> None:
        embed_dim = 6
        num_heads = 3

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=False)
                self.lin1 = nn.Linear(embed_dim, embed_dim, bias=False)
                self.lin2 = nn.Linear(embed_dim, embed_dim, bias=False)
                self.relu = nn.ReLU()

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                input_tensor = self.relu(self.lin1(input_tensor))
                input_tensor = self.relu(
                    self.mha(input_tensor, input_tensor, input_tensor)[0]
                )
                input_tensor = self.relu(self.lin2(input_tensor))
                return input_tensor

        model = Model()
        model_orig = copy.deepcopy(model)

        x = torch.randn(embed_dim, embed_dim)
        y_orig = model(x)

        perm1 = torch.tensor([0, 2, 4, 3, 5, 1], dtype=torch.long)
        perm2 = torch.tensor([2, 4, 5, 3, 0, 1], dtype=torch.long)

        model.lin1.weight.data = model.lin1.weight.data[perm1]
        model.mha.in_proj_weight.data = model.mha.in_proj_weight.data[:, perm1]
        model.mha.out_proj.weight.data = model.mha.out_proj.weight.data[perm2]
        model.lin2.weight.data = model.lin2.weight.data[:, perm2]

        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)

        assert allclose(y_orig, y_new)


class TestEmbedding:
    @staticmethod
    def test_io() -> None:
        num_embeddings = 10
        embedding_dim = 6

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(num_embeddings, embedding_dim)
                self.lin = nn.Linear(embedding_dim, embedding_dim, bias=False)
                self.relu = nn.ReLU()

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                input_tensor = self.relu(self.emb(input_tensor))
                input_tensor = self.relu(self.lin(input_tensor))
                return input_tensor

        model = Model()
        model_orig = copy.deepcopy(model)

        x = torch.randint(0, num_embeddings, (embedding_dim,))
        y_orig = model(x)

        perm = torch.randperm(embedding_dim)
        model.emb.weight.data = model.emb.weight.data[:, perm]
        model.lin.weight.data = model.lin.weight.data[:, perm]

        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)

        assert allclose(y_orig, y_new)


class TestConsistencyPath:
    @staticmethod
    def test_path() -> None:
        size = 4
        relu = nn.ReLU()
        lin1 = nn.Linear(size, size, bias=False)

        lin_res1 = nn.Linear(size, size, bias=False)
        lin_res2 = nn.Linear(size, size, bias=False)

        residual_path = nn.Sequential(lin_res1, nn.ReLU(), lin_res2, nn.ReLU())

        lin2 = nn.Linear(size, size, bias=False)

        class ResBlock(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + residual_path(x)  # type: ignore[no-any-return]

        model = nn.Sequential(lin1, relu, ResBlock(), lin2, relu)
        model_orig = copy.deepcopy(model)
        x = torch.randn(size)
        y_orig = model(x)

        perm1 = [1, 3, 0, 2]
        perm2 = [1, 2, 3, 0]

        lin1.weight.data = lin1.weight.data[perm1]
        lin_res1.weight.data = lin_res1.weight.data[:, perm1]
        lin_res1.weight.data = lin_res1.weight.data[perm2]
        lin_res2.weight.data = lin_res2.weight.data[:, perm2]
        lin_res2.weight.data = lin_res2.weight.data[perm1]
        lin2.weight.data = lin2.weight.data[:, perm1]

        assert model_change_percent(model, model_orig) > 0.1

        y_new = model(x)

        assert allclose(y_orig, y_new)
