# Ideal Words

This package provides a PyTorch implementation of ideal word computation which was proposed by Trager et al. in the paper [Linear Spaces of Meanings: Compositional Structures in Vision-Language Models](https://arxiv.org/abs/2302.14383). Ideal words can be seen as a compositional approximation to a given set of embedding vectors. This package allows computing these ideal words given a factored set of concepts $\mathcal{Z} = \mathcal{Z}_1 \times \dots \times \mathcal{Z}_k$ (e.g., $\\{\mathrm{blue}, \mathrm{red}\\} \times \\{\mathrm{car}, \mathrm{bike}\\}$) and a embedding function $f : \mathcal{Z} \to \mathbb{R}^n$. Additionally, it allows to  quantify compositionality using the ideal word, real word, and average scores from the paper (see Table 6 and 7 for details).

## Usage

You can install the package using:
```
pip install ideal_words
```

Consider you have a text encoder, a tokenizer, and a set of factors. You can then compute ideal words as follows:
```python
from ideal_words import FactorEmbedding, IdealWords

# tokenizer and encoder whose embeddings we want to approximate with ideal words
txt_encoder = MyTextEncoder()
tokenizer = MyTokenizer()

# the factors we want to consider
Z1 = ['blue', 'red']
Z2 = ['car', 'bike']

# factor embedding is a embedding function with some additional logic
fe = FactorEmbedding(txt_encoder, tokenizer)
# compute ideal words from factor embedding and factors
iw = IdealWords(fe, [Z1, Z2])

# retrieve ideal word representation for a specific element of a factor
print(f'Ideal word for "blue": {iw.get_iw("blue")}')
# retrieve ideal word approximation for a combination of factor elements
print(f'Ideal word approximation for "red car": {iw.get_uz(("red", "car"))}')
# directly access the ideal word representation of a certain factor element
i, j = 1, 0  # freely adjustable, as long as i <= num_factors, j <= len_factor_i
print(f'Ideal word for the {j}-th element of the {i}-th factor: {iw.ideal_words[i][j]}')
```

If you have a CUDA-capable GPU, it will be automatically used. If you prefer to use the CPU, you can pass `device='cpu'` when creating the `FactorEmbedding` object.

## Advanced example

You can also customize the behaviour of the `FactorEmbedding` class if your use-case is different (e.g., you are not using a plain text encoder but a CLIP model). [This example](examples/clip_vit_large_14.py) shows how you can compute ideal words and the scores from the paper for the factors from the MIT-States and the UT Zappos datasets using a CLIP model (compare Table 6 and 7 from the paper).

You can run this example locally by using:
```
git clone https://github.com/eliaskempf/ideal_words.git
cd ideal_words
pip install .[demo]  # it is recommended to do this in a virtual environment
python examples/clip_vit_large_14.py
```

## Scalability

For small numbers of factors and/or small datasets, computing ideal words is really fast. The [example](examples/clip_vit_large_14.py) from the previous section computes ideal words using a CLIP ViT-L-14 model on two datasets and runs in less than a minute on a reasonably modern GPU.

However, the approach does not scale well with an increasing number of factors. The computational complexity is at least exponential in the number of factors $\mathcal{\Omega}(\vert\mathcal{Z_1}\vert \times \dots \times \vert\mathcal{Z_k}\vert)$.

## Contributing

The code is roughly tested but there still might be some bugs and/or inefficiencies. If you find anything, feel free to create an issue or to submit a pull request. If you want to contribute to this package, you should install it with the additional development dependencies:
```
git clone https://github.com/eliaskempf/ideal_words.git
cd ideal_words
pip install -e .[dev]  # it is recommended to do this in a virtual environment
```

## Acknowledgement

The ideal word approach was proposed by Trager et al. in https://arxiv.org/abs/2302.14383. Please make sure to appropriately credit their idea by citing their paper if you use this code in research.
