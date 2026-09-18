from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from perpetual.config import Config as LedgerConfig, ROOT, TEST_START, TRAIN_START, write_json

OUTPUT = ROOT / 'results/perpetual_v5'
CACHE = ROOT / 'results/perpetual_v4/cache'
SEEDS = (11, 29, 47, 71, 101, 137, 179)
YEARS = (2023, 2024, 2025)


@dataclass(frozen=True)
class Config(LedgerConfig):
    cell: str = 'gru'
    normalization: str = 'layer'
    dropout: float = 0.1
    attention: bool = False
    attention_heads: int = 4
    position_hidden: int = 16
    head_width: int = 64
    environments: int = 8
    sequence: int = 128
    burn_in: int = 128
    minibatch_sequences: int = 8
    updates: int = 400
    entropy_end: float = 0.005
    initial_persistence: float = 0.9
    counterfactual_stride: int = 128
    counterfactual_horizon: int = 48
    counterfactual_actor_weight: float = 0.5
    counterfactual_critic_weight: float = 0.5
    counterfactual_warmup: int = 10
    counterfactual_enabled: bool = True
    validation_every: int = 100
    threads: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.cell not in ('gru', 'lstm') or self.normalization not in ('none', 'layer', 'batch'):
            raise ValueError('unsupported recurrent cell or normalization')
        if not 0 <= self.dropout < 1 or not 0 < self.initial_persistence < 1:
            raise ValueError('invalid dropout or initial persistence')
        if self.local_width % self.attention_heads:
            raise ValueError('attention width must divide into heads')
        if self.counterfactual_enabled and (self.rollout % self.counterfactual_stride or
                not 0 < self.counterfactual_horizon < self.counterfactual_stride // 2):
            raise ValueError('counterfactual paths must fit inside each rollout interval')


def variants(base=Config()):
    # Matched changes plus combined models, with both recurrent cell types.
    arms = (
        ('plain', 'none', 0., False),
        ('dropout', 'none', .1, False),
        ('layernorm', 'layer', 0., False),
        ('batchnorm', 'batch', 0., False),
        ('attention', 'none', 0., True),
        ('layernorm_dropout', 'layer', .1, False),
        ('layernorm_dropout_attention', 'layer', .1, True),
        ('batchnorm_dropout_attention', 'batch', .1, True),
    )
    result = {f'{cell}_{label}': replace(base, cell=cell, normalization=norm, dropout=drop, attention=attn)
              for label, norm, drop, attn in arms for cell in ('gru', 'lstm')}
    # Same architecture and exploration prior; isolates alternative-action learning.
    for cell in ('gru', 'lstm'):
        result[f'{cell}_layernorm_dropout_ppo_only'] = replace(base, cell=cell, normalization='layer',
                                                              dropout=.1, attention=False, counterfactual_enabled=False)
    return result


def source_hash():
    digest = hashlib.sha256()
    for folder in (ROOT / 'src/perpetual', Path(__file__).parent):
        for path in sorted(folder.glob('*.py')):
            digest.update(f'{folder.name}/{path.name}'.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol(base=Config()):
    return {'version': 'perpetual_v5', 'source_sha256': source_hash(),
            'instrument': 'single BitMEX XBTUSD inverse perpetual; BTC collateral',
            'objective': 'net BTC equity log growth, including commission/slippage/funding',
            'algorithm': 'recurrent PPO plus clipped enumerated counterfactual policy gradient and Q regression',
            'seeds': list(SEEDS), 'validation_years': list(YEARS),
            'variants': {k: v.as_dict() for k, v in variants(base).items()},
            'training_start': str(TRAIN_START), 'test_start': str(TEST_START),
            'test_locked': True, 'test_performance_accessed': False,
            'dropout': 'explicit head masks, fixed per environment/rollout and replayed exactly; identity at evaluation',
            'batchnorm': 'training-only initial calibration; fixed running moments throughout PPO and evaluation; affine parameters learned',
            'attention': 'four-head final-query attention over the 60 past recurrent minute states, with a final-state residual',
            'exploration': 'trainable continuation gate initialized to 0.9; no forced holding time or market trading rule',
            'initial_training_states': '128-step persistent random-action prefix creates valid flat/long/short account histories including costs; evaluation always starts flat',
            'counterfactual': 'all three first actions on copied ledgers, then sampled current-policy continuation for 48 decisions and value bootstrap; no fake on-policy samples',
            'selection': 'one common budget per architecture across all seven seeds and three years; actual seven-member mixture evaluated; no automatic test or final retraining',
            'cache': str(CACHE), 'base_ledger': 'frozen perpetual_v4 accounting and causal features'}
