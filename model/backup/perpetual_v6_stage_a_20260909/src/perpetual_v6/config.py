from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from perpetual.config import Config as LedgerConfig, ROOT, TEST_START, TRAIN_START, write_json

OUTPUT = ROOT / 'results/perpetual_v6'
CACHE = ROOT / 'results/perpetual_v4/cache'
SEEDS = (11, 29, 47, 71, 101, 137, 179)
YEARS = (2023, 2024, 2025)

# Stage A asks one question: with a credit window longer than the cost
# break-even, does the agent hold positions for an economically viable time and
# does its behaviour depend on the market at all? Nothing else changes from v5.
#
# A 12 bp round trip is 0.81x the 14.8 bp five-minute return standard deviation,
# so a signal worth an annualised Sharpe of 1.0 needs about 22 hours in the
# position to pay for one entry and one exit. v5 discounted with gamma*lambda
# = 0.9716, a 2.9 hour window: every profitable trade looked like a loss, and
# every run collapsed to a single side held for months.
GAMMA = 0.99825
GAE_LAMBDA = 0.99825          # gamma*lambda = 0.99650 -> 286 steps -> 23.8 h
ROLLOUT = 1024                # >= 3x the credit window; GAE cannot see past it


@dataclass(frozen=True)
class Config(LedgerConfig):
    cell: str = 'lstm'
    normalization: str = 'none'
    dropout: float = 0.0
    attention: bool = False
    attention_heads: int = 4
    position_hidden: int = 16
    head_width: int = 64
    environments: int = 8
    rollout: int = ROLLOUT
    sequence: int = 128
    burn_in: int = 128
    minibatch_sequences: int = 8
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    updates: int = 400
    entropy_end: float = 0.005
    # Mean holding prior of ~24 h and a flat prior of ~17 h, against v5's 50 and
    # 10 minutes. Both stay trainable; this only sets the starting belief and
    # the random position histories training episodes begin from.
    initial_persistence: float = 0.9965
    initial_flat_persistence: float = 0.995
    flat_start_fraction: float = 1/3
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
        if not 0 < self.initial_flat_persistence < 1 or not 0 <= self.flat_start_fraction <= 1:
            raise ValueError('invalid flat-state exploration setting')
        if self.local_width % self.attention_heads:
            raise ValueError('attention width must divide into heads')
        if self.counterfactual_enabled and (self.rollout % self.counterfactual_stride or
                not 0 < self.counterfactual_horizon < self.counterfactual_stride // 2):
            raise ValueError('counterfactual paths must fit inside each rollout interval')
        # Self-consistency only, so reduced test configurations stay legal: GAE
        # cannot see past the end of a rollout, so a rollout shorter than the
        # window it is meant to carry silently truncates it. The absolute
        # requirement that the window clear the cost break-even is a property of
        # the experiment, not of the dataclass, and is enforced in experiment.run.
        if self.rollout < 3 * self.credit_window:
            raise ValueError('rollout truncates the credit window it is supposed to carry')

    @property
    def credit_window(self):
        """Effective GAE credit-assignment horizon, in decision steps."""
        return 1 / (1 - self.gamma * self.gae_lambda)

    @property
    def credit_window_hours(self):
        return self.credit_window * self.decision_minutes / 60


def variants(base=Config()):
    # v5 ran 18 architectures over 78 completed runs. Variant ranking did not
    # reproduce across seeds (rank correlation -0.18), so architecture is not
    # resolvable here and the budget goes to seeds instead. LSTM over GRU purely
    # for speed: 3723 s vs 6521 s mean per run on this host, same measured result.
    return {'lstm_plain': replace(base, cell='lstm', normalization='none', dropout=0., attention=False)}


def source_hash():
    digest = hashlib.sha256()
    for folder in (ROOT / 'src/perpetual', Path(__file__).parent):
        for path in sorted(folder.glob('*.py')):
            digest.update(f'{folder.name}/{path.name}'.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol(base=Config()):
    return {'version': 'perpetual_v6_stage_a', 'source_sha256': source_hash(),
            'instrument': 'single BitMEX XBTUSD inverse perpetual; BTC collateral',
            'objective': 'net BTC equity log growth, including commission/slippage/funding',
            'algorithm': 'recurrent PPO plus clipped enumerated counterfactual policy gradient and Q regression',
            'stage': 'A of three; isolates the credit-assignment horizon and the holding-time prior',
            'stage_a_hypothesis': 'v5 collapsed because gamma*lambda gave a 2.9 h advantage window while a '
                                  '12 bp round trip needs ~22 h in position to break even, so entering was '
                                  'negative in expectation everywhere the gradient could see',
            'stage_a_success_test': 'median in-market holding spell within 12-48 h and '
                                    'state_action_mutual_information > 0 at update 400; validation return is '
                                    'NOT the stage A criterion',
            'unchanged_from_v5': ['reward (100x net log BTC equity change)', 'fee 5 bp and slippage 1 bp per leg',
                                  'entropy coefficient 0.005', 'per-batch advantage normalization',
                                  'entry notional 1x equity', 'annual validation folds', 'learning rates'],
            'credit_window_steps': base.credit_window, 'credit_window_hours': base.credit_window_hours,
            'cost_break_even_hours': 22.0,
            'seeds': list(SEEDS), 'validation_years': list(YEARS),
            'variants': {k: v.as_dict() for k, v in variants(base).items()},
            'training_start': str(TRAIN_START), 'test_start': str(TEST_START),
            'test_locked': True, 'test_performance_accessed': False,
            'exploration': 'trainable side-specific continuation bias initialized to 0.995 when flat and 0.9965 '
                           'when holding; no forced holding time or market trading rule',
            'initial_training_states': 'one-third untouched flat histories, otherwise 128-step persistent '
                                       'random-action prefix creates valid flat/long/short histories including '
                                       'costs; evaluation always starts flat',
            'counterfactual': 'all three first actions on copied ledgers, then sampled current-policy '
                              'continuation for 48 decisions and value bootstrap; no fake on-policy samples',
            'diagnostics': 'state_action_mutual_information (zero exactly when the policy ignores the market), '
                           'the same restricted to flat states (entry-direction dependence), and the mean and '
                           'standard deviation of the switch probability across states',
            'selection': 'one common budget across all seven seeds and three years; actual seven-member '
                         'mixture evaluated; no automatic test or final retraining',
            'cache': str(CACHE), 'base_ledger': 'frozen perpetual_v4 accounting and causal features'}
