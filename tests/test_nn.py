from typing import Generator

import pytest

import micrograd_pp as mpp

np = mpp.numpy

BATCH_SZ = 64
NUM_FEATURES = 10
SEQ_LEN = 5


@pytest.fixture(autouse=True)
def run_before_and_after_tests() -> Generator[None, None, None]:
    np.random.seed(0)
    yield


@pytest.mark.parametrize("momentum", [0.1, None])
def test_batch_norm_1d_track_running_stats(momentum: float) -> None:
    num_iters = 1_000
    shift = np.random.randn(10)
    scale = np.random.randn(10)
    bn = mpp.BatchNorm1d(NUM_FEATURES, affine=False, momentum=momentum)
    for _ in range(num_iters):
        x = scale * np.random.randn(BATCH_SZ, NUM_FEATURES) + shift
        x_ = mpp.Constant(x)
        bn(x_)
    assert bn._running_mean is not None
    assert bn._running_var is not None
    np.testing.assert_allclose(bn._running_mean, shift, atol=0.1, rtol=0.0)
    np.testing.assert_allclose(bn._running_var, scale * scale, atol=0.1, rtol=0.0)


def test_batch_norm_1d_standardize() -> None:
    shift = np.random.randn(10)
    scale = np.random.randn(10)
    bn = mpp.BatchNorm1d(NUM_FEATURES, affine=False)
    x = scale * np.random.randn(BATCH_SZ, NUM_FEATURES) + shift
    x_ = mpp.Constant(x)
    y_ = bn(x_)
    np.testing.assert_allclose(y_.value.mean(axis=0), 0.0, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(y_.value.var(axis=0), 1.0, atol=1e-3, rtol=0.0)


def test_batch_norm_1d_eval() -> None:
    shift = np.random.randn(10)
    scale = np.random.randn(10)
    bn = mpp.BatchNorm1d(NUM_FEATURES, affine=False)
    x = scale * np.random.randn(BATCH_SZ, NUM_FEATURES) + shift
    x_ = mpp.Constant(x)
    with mpp.eval():
        y_ = bn(x_)
        # The input should be close to the output since the batch norm scale and shift are 1 and 0 at initialization
        np.testing.assert_allclose(x_.value, y_.value, atol=1e-4, rtol=0.0)


@pytest.mark.skipif(not pytest.importorskip("torch"), reason="Unable to import torch")
def test_conv2d() -> None:  # Test against PyTorch implementation
    import torch

    torch_x = torch.randn((2, 3, 7, 8), dtype=torch.float64, requires_grad=True)
    torch_conv = torch.nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 2),
        dilation=(1, 2),
        bias=True,
        dtype=torch.float64,
    )

    torch_y = torch_conv(torch_x)
    torch_loss = (torch_y * torch_y).sum()
    torch_loss.backward()

    assert torch_conv.bias is not None
    assert torch_x.grad is not None
    assert torch_conv.weight.grad is not None
    assert torch_conv.bias.grad is not None

    mpp_conv = mpp.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 2),
        dilation=(1, 2),
        bias=True,
    )
    mpp_conv._a._value = torch_conv.weight.detach().numpy()
    assert mpp_conv._b is not None
    mpp_conv._b._value = torch_conv.bias.detach().numpy()

    mpp_x = mpp.Parameter(torch_x.detach().numpy())
    mpp_y = mpp_conv(mpp_x)
    mpp_loss = (mpp_y * mpp_y).sum()
    mpp_loss.backward()

    np.testing.assert_allclose(mpp_y.value, torch_y.detach().numpy(), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(mpp_x.grad, torch_x.grad.detach().numpy(), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(mpp_conv._a.grad, torch_conv.weight.grad.detach().numpy(), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(mpp_conv._b.grad, torch_conv.bias.grad.detach().numpy(), atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("p", [-1.0, 2.0])
def test_dropout_bad_probabilities(p: float) -> None:
    with pytest.raises(ValueError):
        mpp.Dropout(p)


def test_dropout_eval() -> None:
    x = mpp.Constant(np.random.randn(BATCH_SZ, NUM_FEATURES))
    dropout = mpp.Dropout(0.5)
    with mpp.eval():
        y = dropout(x)
    np.testing.assert_equal(x.value, y.value)


def test_embedding() -> None:
    num_embeddings = 10
    embedding = mpp.Embedding(num_embeddings=num_embeddings, embedding_dim=NUM_FEATURES)
    x = np.random.randint(low=0, high=num_embeddings, size=(1, 2, 3))
    y = embedding(x)
    assert y.shape == x.shape + (NUM_FEATURES,)


def test_layer_norm() -> None:
    normalized_shape = (4, 3)
    x = mpp.Constant(np.random.randn(BATCH_SZ, *normalized_shape))
    ln = mpp.LayerNorm(normalized_shape, eps=0.0)
    y = ln(x)
    np.testing.assert_allclose(y.mean((-1, -2)).value, 0.0, atol=1e-12)
    np.testing.assert_allclose(y.var((-1, -2)).value, 1.0)


def test_maxpool2d() -> None:
    x = np.array(
        [
            [
                [
                    [1.0, -2.0, 3.0, 0.0, -1.0],
                    [4.0, 5.0, -6.0, 2.0, 8.0],
                    [-3.0, 7.0, 1.0, -4.0, 6.0],
                    [9.0, -8.0, 0.0, 3.0, -2.0],
                ],
                [
                    [-1.0, 2.0, -3.0, 4.0, -5.0],
                    [6.0, -7.0, 8.0, -9.0, 10.0],
                    [-11.0, 12.0, -13.0, 14.0, -15.0],
                    [16.0, -17.0, 18.0, -19.0, 20.0],
                ],
            ]
        ],
        dtype=np.float64,
    )

    expected = np.array(
        [
            [
                [
                    [5.0, 5.0, 5.0, 8.0, 8.0],
                    [9.0, 9.0, 5.0, 8.0, 8.0],
                ],
                [
                    [6.0, 8.0, 8.0, 10.0, 10.0],
                    [16.0, 18.0, 18.0, 20.0, 20.0],
                ],
            ]
        ],
        dtype=np.float64,
    )

    y = mpp.MaxPool2d(
        kernel_size=(2, 3),
        stride=(2, 1),
        padding=(1, 1),
        dilation=(2, 1),
    )(mpp.Constant(x))

    np.testing.assert_equal(y.value, expected)


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.skipif(not pytest.importorskip("torch"), reason="Unable to import torch")
def test_multihead_attention(is_causal: bool) -> None:  # Test against PyTorch implementation
    import torch

    torch_attn_mask = None
    mpp_attn_mask = None
    if is_causal:
        torch_attn_mask = torch.zeros((SEQ_LEN, SEQ_LEN))
        torch_attn_mask[torch.triu_indices(SEQ_LEN, SEQ_LEN, offset=1)] = -np.inf
        mpp_attn_mask = mpp.Constant(torch_attn_mask.numpy())

    torch_attn = torch.nn.MultiheadAttention(embed_dim=10, num_heads=2, kdim=20, vdim=30, batch_first=True)
    named_parameters = dict(torch_attn.named_parameters())
    torch_wq = named_parameters["q_proj_weight"]
    torch_wk = named_parameters["k_proj_weight"]
    torch_wv = named_parameters["v_proj_weight"]
    torch_wo = named_parameters["out_proj.weight"]
    torch_q = torch.randn(BATCH_SZ, SEQ_LEN, 10)
    torch_k = torch.randn(BATCH_SZ, SEQ_LEN, 20)
    torch_v = torch.randn(BATCH_SZ, SEQ_LEN, 30)
    torch_attn_output, torch_attn_output_weights = torch_attn(torch_q, torch_k, torch_v, attn_mask=torch_attn_mask)

    mpp_attn = mpp.MultiheadAttention(embed_dim=10, num_heads=2, kdim=20, vdim=30, batch_first=True)
    mpp_attn._wq._a._value = torch_wq.detach().numpy()
    mpp_attn._wk._a._value = torch_wk.detach().numpy()
    mpp_attn._wv._a._value = torch_wv.detach().numpy()
    mpp_attn._wo._a._value = torch_wo.detach().numpy()
    mpp_q = mpp.Constant(torch_q.numpy())
    mpp_k = mpp.Constant(torch_k.numpy())
    mpp_v = mpp.Constant(torch_v.numpy())
    mpp_attn_output, mpp_attn_output_weights = mpp_attn(mpp_q, mpp_k, mpp_v, attn_mask=mpp_attn_mask)

    np.testing.assert_allclose(torch_attn_output.detach().numpy(), mpp_attn_output.value, atol=1e-6)
    np.testing.assert_allclose(torch_attn_output_weights.detach().numpy(), mpp_attn_output_weights.value, atol=1e-6)
