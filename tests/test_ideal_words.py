import pytest
import torch
import torch.nn as nn

from ideal_words import FactorEmbedding, IdealWords


def test_auxillary_setup():
    # toy example to test the auxillary attributes used during ideal word computation
    Z1 = ['X1', 'X2']
    Z2 = ['Y1', 'Y2', 'Y3', 'Y4']
    Z3 = ['Z1', 'Z2', 'Z3']

    # dummy embeddings
    torch.manual_seed(42)
    embeddings = torch.randn(len(Z1) * len(Z2) * len(Z3), 4)

    # we use random embeddings here as we only want to verify auxillary attributes used during ideal word computation
    txt_encoder = nn.Identity()

    def tokenizer(text: list[str]) -> torch.Tensor:
        _ = text
        return embeddings

    # when device is not specified, it defaults to cuda if available
    fe = FactorEmbedding(txt_encoder, tokenizer, device=None)
    assert fe.device == 'cuda' if torch.cuda.is_available() else 'cpu'

    # create factor embeddings and compute ideal words
    fe = FactorEmbedding(txt_encoder, tokenizer, device='cpu')
    iw = IdealWords(fe, [Z1, Z2, Z3])

    # factor combinations
    assert iw.pairs == [
        ('X1', 'Y1', 'Z1'),
        ('X1', 'Y1', 'Z2'),
        ('X1', 'Y1', 'Z3'),
        ('X1', 'Y2', 'Z1'),
        ('X1', 'Y2', 'Z2'),
        ('X1', 'Y2', 'Z3'),
        ('X1', 'Y3', 'Z1'),
        ('X1', 'Y3', 'Z2'),
        ('X1', 'Y3', 'Z3'),
        ('X1', 'Y4', 'Z1'),
        ('X1', 'Y4', 'Z2'),
        ('X1', 'Y4', 'Z3'),
        ('X2', 'Y1', 'Z1'),
        ('X2', 'Y1', 'Z2'),
        ('X2', 'Y1', 'Z3'),
        ('X2', 'Y2', 'Z1'),
        ('X2', 'Y2', 'Z2'),
        ('X2', 'Y2', 'Z3'),
        ('X2', 'Y3', 'Z1'),
        ('X2', 'Y3', 'Z2'),
        ('X2', 'Y3', 'Z3'),
        ('X2', 'Y4', 'Z1'),
        ('X2', 'Y4', 'Z2'),
        ('X2', 'Y4', 'Z3'),
    ]

    # factor to index mapping
    assert iw.factor2idx == {
        'X1': (0, 0),
        'X2': (0, 1),
        'Y1': (1, 0),
        'Y2': (1, 1),
        'Y3': (1, 2),
        'Y4': (1, 3),
        'Z1': (2, 0),
        'Z2': (2, 1),
        'Z3': (2, 2),
    }

    for zi in iw.factor2idx:
        i, j = iw.factor2idx[zi]
        assert iw.factors[i][j] == zi

    # automatic uniform weights
    assert len(iw.weights) == 3
    assert iw.weights[0] == [1 / 2] * 2
    assert iw.weights[1] == [1 / 4] * 4
    assert iw.weights[2] == [1 / 3] * 3

    assert iw.factor_indices == [
        {
            'X1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'X2': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        },
        {
            'Y1': [0, 1, 2, 12, 13, 14],
            'Y2': [3, 4, 5, 15, 16, 17],
            'Y3': [6, 7, 8, 18, 19, 20],
            'Y4': [9, 10, 11, 21, 22, 23],
        },
        {
            'Z1': [0, 3, 6, 9, 12, 15, 18, 21],
            'Z2': [1, 4, 7, 10, 13, 16, 19, 22],
            'Z3': [2, 5, 8, 11, 14, 17, 20, 23],
        },
    ]

    for i, factor_index in enumerate(iw.factor_indices):
        for zi, inds in factor_index.items():
            # verify that the returned indices actually map back to the expected zi
            for ind in inds:
                assert iw.pairs[ind][i] == zi

            # compare mean over indexed embedding vectors to ideal word
            if not torch.allclose(iw.get_iw(zi), iw.embeddings[inds].mean(dim=0) - iw.u_zero):
                print((iw.get_iw(zi) - (iw.embeddings[inds].mean(dim=0) - iw.u_zero)).abs().max())
            assert torch.allclose(iw.get_iw(zi), iw.embeddings[inds].mean(dim=0) - iw.u_zero)

    # u_zero is just the mean over all embeddings
    if not torch.allclose(iw.u_zero, iw.embeddings.mean(dim=0)):
        print((iw.u_zero - iw.embeddings.mean(dim=0)).abs().max())
    assert torch.allclose(iw.u_zero, iw.embeddings.mean(dim=0))


def test_compositional_toy_embeddings():
    # toy example with predefined embeddings that are already compositional
    Z1 = ['blue', 'red']
    Z2 = ['car', 'bike']

    # joint representations for ideal words
    joint_embeddings = {
        'blue car': torch.Tensor([1, 0, 1, 0]),
        'red car': torch.Tensor([0, 1, 1, 0]),
        'blue bike': torch.Tensor([1, 0, 0, 1]),
        'red bike': torch.Tensor([0, 1, 0, 1]),
    }

    # single representations for real words
    single_embeddings = {
        'blue': torch.Tensor([1, 0, 0, 0]),
        'red': torch.Tensor([0, 1, 0, 0]),
        'car': torch.Tensor([0, 0, 1, 0]),
        'bike': torch.Tensor([0, 0, 0, 1]),
    }

    # embeddings have 2 dimensions per factor and use one-hot encoding
    embeddings = {**joint_embeddings, **single_embeddings}

    # we use the predefined embeddings by returning them from the tokenizer and using nn.Identity() as text encoder
    txt_encoder = nn.Identity()

    def tokenizer(text: list[str]) -> torch.Tensor:
        return torch.stack([embeddings[z] for z in text])

    # create factor embeddings and compute ideal words
    fe = FactorEmbedding(txt_encoder, tokenizer, normalize=False, device='cpu')
    iw = IdealWords(fe, [Z1, Z2])

    # test ideal word computation
    assert torch.allclose(iw.u_zero, torch.Tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.allclose(iw.get_iw('blue'), torch.Tensor([0.5, -0.5, 0, 0]))
    assert torch.allclose(iw.get_iw('red'), torch.Tensor([-0.5, 0.5, 0, 0]))
    assert torch.allclose(iw.get_iw('car'), torch.Tensor([0, 0, 0.5, -0.5]))
    assert torch.allclose(iw.get_iw('bike'), torch.Tensor([0, 0, -0.5, 0.5]))

    # approximations are perfect because embeddings were already compositional, thus distance is 0
    assert iw.iw_score == (0.0, 0.0)
    assert iw.iw_accuracy == 1.0

    # ideal word approximations exactly equal the original embeddings
    for caption, embedding in joint_embeddings.items():
        z = caption.split(' ')
        assert torch.allclose(iw.get_uz(z), embedding)  # u_z = u_zero + u_color + u_object

    # real words are just the embeddings themselves
    for caption, embedding in single_embeddings.items():
        assert torch.allclose(iw.get_rw(caption), embedding)

    # real word approximations are also close to the original embeddings except scaled by 0.5
    assert torch.allclose(iw.get_uz(['blue', 'car'], approx='real'), torch.Tensor([0.5, 0, 0.5, 0]))
    assert torch.allclose(iw.get_uz(['red', 'car'], approx='real'), torch.Tensor([0, 0.5, 0.5, 0]))
    assert torch.allclose(iw.get_uz(['blue', 'bike'], approx='real'), torch.Tensor([0.5, 0, 0, 0.5]))
    assert torch.allclose(iw.get_uz(['red', 'bike'], approx='real'), torch.Tensor([0, 0.5, 0, 0.5]))

    # real word approximations are not perfect even when embeddings are compositional, thus distance greater than 0
    assert iw.rw_score == (pytest.approx(0.5), 0.0)
    assert iw.rw_accuracy == 1.0

    # requesting approximation modes other than 'ideal' or 'real' raises a value error
    with pytest.raises(ValueError):
        iw.get_uz(['blue', 'car'], approx='foobar')


def test_noncompositional_toy_embeddings():
    # toy example with predefined embeddings that are not compositional
    Z1 = ['blue', 'red']
    Z2 = ['car', 'bike']

    # embeddings use one-hot encoding
    embeddings = {
        'blue car': torch.Tensor([1, 0, 0, 0]),
        'red car': torch.Tensor([0, 1, 0, 0]),
        'blue bike': torch.Tensor([0, 0, 1, 0]),
        'red bike': torch.Tensor([0, 0, 0, 1]),
    }

    # we use the predefined embeddings by returning them from the tokenizer and using nn.Identity() as text encoder
    txt_encoder = nn.Identity()

    def tokenizer(text: list[str]) -> torch.Tensor:
        return torch.stack([embeddings[z] for z in text])

    # create factor embeddings and compute ideal words
    fe = FactorEmbedding(txt_encoder, tokenizer, normalize=False, device='cpu')
    iw = IdealWords(fe, [Z1, Z2])

    assert torch.allclose(iw.u_zero, torch.Tensor([0.25, 0.25, 0.25, 0.25]))
    assert torch.allclose(iw.get_iw('blue'), torch.Tensor([0.25, -0.25, 0.25, -0.25]))
    assert torch.allclose(iw.get_iw('red'), torch.Tensor([-0.25, 0.25, -0.25, 0.25]))
    assert torch.allclose(iw.get_iw('car'), torch.Tensor([0.25, 0.25, -0.25, -0.25]))
    assert torch.allclose(iw.get_iw('bike'), torch.Tensor([-0.25, -0.25, 0.25, 0.25]))

    # in this case the ideal word approximations do not match the original embeddings
    assert torch.allclose(iw.get_uz(('blue', 'car')), torch.Tensor([0.75, 0.25, 0.25, -0.25]))
    assert torch.allclose(iw.get_uz(('red', 'car')), torch.Tensor([0.25, 0.75, -0.25, 0.25]))
    assert torch.allclose(iw.get_uz(('blue', 'bike')), torch.Tensor([0.25, -0.25, 0.75, 0.25]))
    assert torch.allclose(iw.get_uz(('red', 'bike')), torch.Tensor([-0.25, 0.25, 0.25, 0.75]))

    # approximations are imperfect but distances are always the same so we have score > 0 but std = 0
    assert iw.iw_score == (0.25, 0.0)
    assert iw.iw_accuracy == 1.0  # approximations are still close enough


def test_random_embeddings():
    # toy example with randomly initialized embeddings
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
        if not torch.allclose(iw_per_factor.sum(dim=0), torch.zeros(64)):
            print(iw_per_factor.sum(dim=0).abs().max())
        assert torch.allclose(iw_per_factor.sum(dim=0), torch.zeros(64))

    # approximations are not perfect because embeddings are random
    assert iw.iw_score[0] >= 0.0
    assert iw.iw_score[1] >= 0.0

    assert iw.avg_score[0] >= 0.0
    assert iw.avg_score[1] >= 0.0
