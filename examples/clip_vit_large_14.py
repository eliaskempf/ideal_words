import os
import time

import torch
from open_clip import create_model_and_transforms, get_tokenizer

from ideal_words import FactorEmbedding, IdealWords


class AttributeObjectFactorEmbedding(FactorEmbedding):
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        # CLIP is not only a text encoder, so we need to specify how to use it for encoding text

        return self.txt_encoder.encode_text(text)

    def joint_repr(self, pair: tuple[str, ...]) -> str:
        attr, obj = pair
        # classic zero shot type caption of attribute-object dataset
        return f'an image of a {attr} {obj}'

    def single_repr(self, zi: str, factor_idx: int) -> str:
        # for the real word score, we also need to encode factors separately
        if factor_idx == 0:
            # zi is an attribute
            return f'image of a {zi} object'
        elif factor_idx == 1:
            # zi is an object type
            return f'image of a {zi}'
        else:
            raise IndexError(f'Invalid factor index: {factor_idx}')


if __name__ == '__main__':
    # keep track of execution time
    start = time.time()

    # in the paper, they used a ViT-L-14 based CLIP model from OpenAI
    clip, _, preprocess_val = create_model_and_transforms('ViT-L-14', precision='fp16', pretrained='openai')
    tokenizer = get_tokenizer('ViT-L-14')
    print('Loaded CLIP model and tokenizer.')

    # load factors from mit-states
    dirname = os.path.dirname(os.path.abspath(__file__))
    factors = {}
    with open(os.path.join(dirname, 'mit-states.csv'), 'r') as f:
        Z1, Z2 = [line.strip().split(',') for line in f.readlines()]
        factors['mit-states'] = Z1, Z2
    with open(os.path.join(dirname, 'ut-zappos.csv'), 'r') as f:
        Z1, Z2 = [line.strip().split(',') for line in f.readlines()]
        factors['ut-zappos'] = Z1, Z2
    print('Loaded factors for MIT-States and UT Zappos.')

    fe = AttributeObjectFactorEmbedding(clip, tokenizer, normalize=True)

    # compute ideal words and score for mit-states
    mit_iw = IdealWords(fe, factors['mit-states'], weights=None)  # weights=None uses uniform weights for all factors
    mit_iw_score, mit_iw_std = mit_iw.iw_score
    mit_rw_score, mit_rw_std = mit_iw.rw_score
    mit_avg_score, mit_avg_std = mit_iw.avg_score
    print('Computed ideal words and scores for MIT-States.')

    # compute ideal words and score for ut-zappos
    ut_iw = IdealWords(fe, factors['ut-zappos'], weights=None)  # weights=None uses uniform weights for all factors
    ut_iw_score, ut_iw_std = ut_iw.iw_score
    ut_rw_score, ut_rw_std = ut_iw.rw_score
    ut_avg_score, ut_avg_std = ut_iw.avg_score
    print('Computed ideal words and scores for UT Zappos.')

    # print table
    print()
    print('                 IW            RW            Avg    ')
    print('----------------------------------------------------')
    print(
        f'MIT-States   {mit_iw_score:.2f} ± {mit_iw_std:.2f}   {mit_rw_score:.2f} ± {mit_rw_std:.2f}   '
        f'{mit_avg_score:.2f} ± {mit_avg_std:.2f}'
    )
    print(
        f'UT Zappos    {ut_iw_score:.2f} ± {ut_iw_std:.2f}   {ut_rw_score:.2f} ± {ut_rw_std:.2f}   '
        f'{ut_avg_score:.2f} ± {ut_avg_std:.2f}'
    )
    print(f'\nTook {time.time() - start:.2f} seconds')
