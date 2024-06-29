# Ideal Words

This small package provides a PyTorch implementation of ideal word computation which was proposed by Trager et al. in the paper [Linear Spaces of Meanings: Compositional Structures in Vision-Language Models](https://arxiv.org/abs/2302.14383). Ideal words can be seen as a compositional approximation to a given set of embedding vectors. This package allows computing these ideal words given a factored set of concepts $\mathcal{Z} = \mathcal{Z}_1 \times \dots \times \mathcal{Z}_k$ (e.g., $\{\mathrm{blue}, \mathrm{red}\} \times \{\mathrm{car}, \mathrm{bike}\}$) and a embedding function $f : \mathcal{Z} \to \mathbb{R}^n$. Additionally, it allows to  quantify compositionality using the ideal word, real word, and average score from the paper (see Table 6 and 7 for details).

## Usage

You can install the package using:
```
pip install git+https://github.com/icetube23/ideal_words.git
```
