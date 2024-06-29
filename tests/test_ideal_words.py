import torch
import torch.nn as nn

from ideal_words import FactorEmbedding, IdealWords


def test_toy_embeddings():
    # toy example with predefined embeddings that are already compositional
    Z1 = ['blue', 'red']
    Z2 = ['car', 'bike']

    # embeddings have 2 dimensions per factor and use one-hot encoding
    embeddings = {
        'blue car': torch.Tensor([1, 0, 1, 0]),
        'red car': torch.Tensor([0, 1, 1, 0]),
        'blue bike': torch.Tensor([1, 0, 0, 1]),
        'red bike': torch.Tensor([0, 1, 0, 1]),
    }

    # we use the predefined embeddings by returning them from the tokenizer and using nn.Identity() as text encoder
    txt_encoder = nn.Identity()

    def tokenizer(text: list[str]) -> torch.Tensor:
        return torch.stack([embeddings[z] for z in text])

    # create factor embeddings and compute ideal words
    fe = FactorEmbedding(txt_encoder, tokenizer, normalize=False, device='cpu')
    iw = IdealWords(fe, [Z1, Z2])

    assert torch.allclose(iw.u_zero, torch.Tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.allclose(iw.get_iw('blue'), torch.Tensor([0.5, -0.5, 0, 0]))
    assert torch.allclose(iw.get_iw('red'), torch.Tensor([-0.5, 0.5, 0, 0]))
    assert torch.allclose(iw.get_iw('car'), torch.Tensor([0, 0, 0.5, -0.5]))
    assert torch.allclose(iw.get_iw('bike'), torch.Tensor([0, 0, -0.5, 0.5]))

    for caption, embedding in embeddings.items():
        z = caption.split(' ')
        assert torch.allclose(iw.get_uz(z), embedding)  # u_z = u_zero + u_color + u_object

    # approximations are perfect because embeddings were already compositional, thus distance is 0
    assert iw.iw_score == (0.0, 0.0)


def test_random_embeddings():
    # toy example with predefined embeddings that are already compositional
    Z1 = ['A', 'B', 'C', 'D', 'E']
    Z2 = ['1', '2', '3', '4', '5']
    Z3 = ['.', '?', '!', ',', ';']

    # we have 5 x 5 x 5 = 125 different combinations and an embedding dimension of 64
    torch.manual_seed(42)
    embeddings = torch.randn(125, 64)

    # we use the predefined embeddings by returning them from the tokenizer and using nn.Identity() as text encoder
    txt_encoder = nn.Identity()

    def tokenizer(text: list[str]) -> torch.Tensor:
        _ = text
        return embeddings

    # create factor embeddings and compute ideal words
    fe = FactorEmbedding(txt_encoder, tokenizer, normalize=True, device='cpu')
    iw = IdealWords(fe, [Z1, Z2, Z3])

    # mean over normally distributed random vectors is close to zero
    assert iw.u_zero.norm() <= 0.1

    for iw_per_factor in iw.ideal_words:
        # ideal words belonging to a factor Z_i should sum to zero
        assert torch.allclose(iw_per_factor.sum(dim=0), torch.zeros(64))

    # approximations are not perfect because embeddings are random
    assert iw.iw_score[0] >= 0.0
    assert iw.iw_score[1] >= 0.0
