import torch

from .. import signal


def test_signal_normalize():
    x = torch.rand(2, 100)
    xminl, xmaxl = torch.argmin(x), torch.argmax(x)
    r = signal.signal_minmax(x)
    assert r[0] == x.min()
    assert r[1] == x.max()

    xn = signal.signal_normalize(x, r, (0.0, 1.0))
    assert xn.min() == 0.0
    assert xn.max() == 1.0
    assert xn.view(-1)[xminl] == 0.0
    assert xn.view(-1)[xmaxl] == 1.0

    xn = signal.signal_normalize(x, r, (-1.0, 1.0))
    assert xn.min() == -1.0
    assert xn.max() == 1.0
    assert xn.view(-1)[xminl] == -1.0
    assert xn.view(-1)[xmaxl] == 1.0


def test_signal_quantize():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q, k = signal.signal_quantize_midtread(x, 0.5)
    assert torch.allclose(q, x)
    assert torch.allclose(k, torch.tensor([-2, -1, 0, 1, 2]))


def test_signal_encoder():
    ed = signal.EncoderDecoder(
        signal.EncoderParams(num_levels=5, input_range=(-2.0, 2.0))
    )
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    k = ed.encode(x * 2)
    q = ed.decode(k)
    assert torch.allclose(k, torch.tensor([0, 1, 2, 3, 4]))
    assert torch.allclose(q, x * 2)

    k = ed.encode(x)
    q = ed.decode(k)
    assert torch.allclose(k, torch.tensor([1, 2, 2, 3, 3]))
    assert torch.allclose(q, torch.tensor([-1.0, 0.0, 0.0, 1.0, 1.0]))
