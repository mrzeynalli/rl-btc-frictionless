"""Single specification for the agent: paths, locks and every constant.

Each constant's provenance - cited, derived, measured, a user decision, or an
explicitly unverified design choice - is tabulated in ARCHITECTURE.md §12. The
comments here say what a reader of the code needs; the table says why.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/bitmex_xbtusd_1m_2016_2026_08_01'
OUTPUT = ROOT / 'results/agent'
CACHE = OUTPUT / 'cache'
TEST_START = pd.Timestamp('2026-01-01', tz='UTC')
TRAIN_START = pd.Timestamp('2017-01-01', tz='UTC')
SEEDS = (11, 29, 47)
# Expanding walk-forward folds, the protocol of Zhang, Zohren & Roberts (2020),
# who retrain on all data available up to each point: validate 2024 after
# training on 2017-2023, and 2025 after 2017-2024. Everything from 2026 is locked.
YEARS = (2024, 2025)

# MEASURED on the cached 2023-2026 closes: the five-minute log-return standard
# deviation is 14.8 bp, so a 12 bp round trip costs 0.81 of one bar's typical
# move. ASSUMED, not measured: a signal worth an annualised Sharpe ratio of 1.0,
# which then needs about 22 hours in position to pay for one entry and one exit.
# The discount horizon below is anchored to that break-even.
FIVE_MINUTE_RETURN_SD = 0.00148
COST_BREAK_EVEN_HOURS = 22.0


class TestLockedError(ValueError):
    pass


def development_range(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError('UTC-aware timestamps required')
    start, end = start.tz_convert('UTC'), end.tz_convert('UTC')
    if start >= TEST_START or end > TEST_START:
        raise TestLockedError('Test data from 2026-01-01 onward are LOCKED; no test entrypoint exists')
    if start >= end:
        raise ValueError('empty development interval')
    return start, end


@dataclass(frozen=True)
class Config:
    # --- market and ledger ---------------------------------------------------
    # User decisions: five-minute decisions, 5 bp commission and 1 bp adverse
    # slippage per leg with no cost curriculum, and 1x entry notional.
    decision_minutes: int = 5
    fee: float = 0.0005
    slippage: float = 0.0001
    entry_leverage: float = 1.0
    # BitMEX XBTUSD base maintenance margin: 0.50% until 17 Apr 2020, 0.45% from
    # then (BitMEX announcement of 13 Apr 2020). The higher value is used for the
    # whole sample, which is conservative by 0.05% of notional.
    maintenance: float = 0.005

    # --- network -------------------------------------------------------------
    # 128 is CleanRL ppo_atari_lstm's LSTM width. The critic is twice as wide:
    # Andrychowicz et al. (ICLR 2021) find no downside to a wider value network.
    # Both widths passed the planted-edge test (ARCHITECTURE.md §11).
    hidden: int = 128
    head_width: int = 128
    critic_width_multiple: int = 2

    # --- rollout shape ---------------------------------------------------------
    # Eight parallel environments: Andrychowicz et al. and CleanRL ppo_atari_lstm.
    environments: int = 8
    rollout: int = 1024
    sequence: int = 256
    # R2D2 (Kapturowski et al. 2019) stores 80-step sequences behind a 40-step
    # burn-in, a ratio of one half.
    burn_in: int = 128
    # One quarter: 26 rollouts of 1,024 five-minute steps is 92.4 days, the
    # window FinRL's ensemble strategy validates and trades on (Yang et al. 2020)
    # and the length of every validation block here.
    episode_steps: int = 26 * 1024
    minibatch_sequences: int = 4
    # CleanRL ppo_atari_lstm uses 4 epochs (the PPO paper's Atari table uses 3).
    epochs: int = 4
    # Andrychowicz et al.: recomputing advantages each pass "performs best among
    # all variants".
    recompute_advantages_per_pass: bool = True

    # --- horizon ---------------------------------------------------------------
    # User decision. gamma*lambda = 0.99650 is a 286-step, 23.8 h credit window,
    # just above the 22 h cost break-even.
    gamma: float = 0.99825
    gae_lambda: float = 0.99825

    # --- optimisation ------------------------------------------------------------
    # Andrychowicz et al.: Adam at 3e-4 "is a safe default", linear decay,
    # gradient clipping "of secondary importance".
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    anneal_lr: bool = True
    gradient_clip: float = 0.5
    # PPO paper (Atari, Table 5) and CleanRL: 0.01. Entropy regularisation turns
    # softmax policy gradient's O(1/t) convergence linear (Mei et al., ICML 2020).
    entropy_coefficient: float = 0.01
    # The policy improvement step is MPO's E-step (Abdolmaleki et al., ICLR 2018):
    # the target is proportional to pi_old * exp(advantage / temperature), the
    # temperature solved so that the mean KL(target || pi_old) equals this bound.
    # PPO's ratio clip cannot serve here: its gradient on an action's logit is
    # multiplied by that action's own probability, so a policy that has gone
    # deterministic stops responding to its critic altogether (MEASURED, §11.6).
    # 0.1 is MPO's own value (their Table 2) and V-MPO's for DMLab single-task.
    policy_kl_step: float = 0.1

    # --- starting states ---------------------------------------------------------
    # Exploring starts (Sutton & Barto 2018, §5.3): each episode starts flat,
    # short or long with equal probability, so exits are practised from update one.
    flat_start_fraction: float = 1/3

    # --- reward: exact differential Sharpe increment (Moody & Saffell 2001) ------
    # A five-day window. DESIGN CHOICE inside derived bounds - slower than the
    # credit window, faster than an episode - not pinned by any study.
    dsr_eta: float = 1/1440
    # Measured: a fresh account starts with the asset's own five-minute variance.
    dsr_initial_second_moment: float = FIVE_MINUTE_RETURN_SD ** 2

    # --- observation ---------------------------------------------------------------
    # OpenAI baselines PPO clips normalised observations to [-10, 10]
    # (Engstrom et al., ICLR 2020, §3 and Table 4).
    observation_clip: float = 10.0

    # --- schedule and evaluation ---------------------------------------------------
    # 5,000 x 4 epochs x 4 chunks x 2 groups = 160,000 actor gradient steps, the
    # budget of CleanRL ppo_atari_lstm (10M / 1,024 x 4 x 4 = 156,240).
    updates: int = 5000
    validation_every: int = 500
    threads: int = 1
    # Actions are sampled, never taken by argmax, so every evaluation runs several
    # seeds. The count is inherited; its adequacy is measured in ARCHITECTURE.md §10.
    evaluation_seeds: int = 4
    # rliable defaults (Agarwal et al., NeurIPS 2021): 50,000 stratified bootstrap
    # replications for a 95% percentile interval.
    bootstrap_replications: int = 50_000
    confidence_level: float = 0.95
    # Sensitivity only, never used for selection: live slippage and 3x and 5x it.
    stress_slippage: tuple = (0.0001, 0.0003, 0.0005)

    def __post_init__(self):
        if self.decision_minutes != 5:
            raise ValueError('this study fixes a five-minute decision interval')
        if self.rollout % self.sequence or self.episode_steps % self.rollout:
            raise ValueError('sequence must divide rollout and rollout must divide episode length')
        if min(self.environments, self.burn_in, self.sequence, self.updates, self.epochs) < 1:
            raise ValueError('positive training sizes required')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid discount')
        if not 0 <= self.fee < 0.01 or not 0 <= self.slippage < 0.01:
            raise ValueError('invalid execution cost')
        if not 0 < self.entry_leverage <= 1:
            raise ValueError('this study permits at most 1x entry contract notional')
        if not 0 <= self.flat_start_fraction <= 1:
            raise ValueError('invalid flat starting share')
        if self.critic_width_multiple < 1:
            raise ValueError('the value network may not be narrower than the policy')
        if self.environments % self.minibatch_sequences:
            raise ValueError('minibatch_sequences must divide the environment count')
        if self.entropy_coefficient < 0 or self.observation_clip <= 0:
            raise ValueError('invalid entropy coefficient or observation clip')
        if self.policy_kl_step <= 0:
            raise ValueError('the policy improvement step must be a positive KL bound')
        # GAE bootstraps at the end of every rollout. The average step sits half
        # a rollout before that bootstrap and must still see a whole credit window.
        if self.rollout / 2 < self.credit_window:
            raise ValueError('rollout truncates the credit window it is supposed to carry')
        # By the end of a training sequence the recurrent state has integrated
        # burn-in plus sequence steps; that span must cover one credit window.
        if self.burn_in + self.sequence < self.credit_window:
            raise ValueError('recurrent span is too short for the credit window')
        # The Sharpe statistics must move slower than the advantage window, or
        # the reward's own drift is read as signal, and settle inside an episode.
        if not self.credit_window < 1 / self.dsr_eta < self.episode_steps:
            raise ValueError('differential Sharpe window must lie between the credit window and the episode')
        if self.dsr_initial_second_moment <= 0:
            raise ValueError('invalid differential Sharpe prior')
        if min(self.evaluation_seeds, self.bootstrap_replications) < 1 or not 0 < self.confidence_level < 1:
            raise ValueError('invalid evaluation settings')

    @property
    def credit_window(self):
        """Effective GAE credit-assignment horizon, in decision steps."""
        return 1 / (1 - self.gamma * self.gae_lambda)

    @property
    def credit_window_hours(self):
        return self.credit_window * self.decision_minutes / 60

    @property
    def round_trip_cost(self):
        return 2 * (self.fee + self.slippage)

    def branch_widths(self, actor):
        scale = 1 if actor else self.critic_width_multiple
        return {'hidden': self.hidden * scale, 'head_width': self.head_width * scale}

    @property
    def reward_scale(self):
        """Puts the discounted return back at roughly unit scale.

        Derived, not tuned. The differential Sharpe ratio is approximately a
        z-score, and the discounted sum of an uncorrelated unit-variance sequence
        has standard deviation 1/sqrt(1 - gamma^2), about 17 here. A critic head
        that starts near zero cannot travel that far at Adam's step size, so the
        reward is scaled by sqrt(1 - gamma^2); the measured target standard
        deviation in the previous run was about 0.6.
        """
        return math.sqrt(1 - self.gamma ** 2)

    def as_dict(self):
        # Normalised through JSON so tuple fields compare equal to the lists read
        # back from protocol.json.
        return json.loads(json.dumps(asdict(self)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


REFERENCES = {
    'exogenous market, endogenous account': 'Dietterich, Trimponias & Chen, ICML 2018',
    'Markov-sufficient account state': 'Subramanian, Sinha, Seraj & Mahajan, JMLR 2022',
    'few, dimensionless features': 'Song, Jiang, Tu, Du & Neyshabur, Observational Overfitting in RL, ICLR 2020',
    'volatility-scaled momentum': 'Moskowitz, Ooi & Pedersen, JFE 2012; Lim, Zohren & Roberts 2019; '
                                  'Zhang, Zohren & Roberts 2020',
    'momentum horizons and attention construction': 'Liu & Tsyvinski, Risks and Returns of Cryptocurrency, RFS 2021',
    'realised variance from five-minute returns over a day': 'Andersen & Bollerslev, JEF 1997; '
                                                              'Liu, Patton & Sheppard, JoE 2015',
    'high-low range': 'Parkinson, Journal of Business 1980',
    'volume as attention': 'Barber & Odean, RFS 2008',
    'perpetual funding': 'He, Manela, Ross & von Wachter, Fundamentals of Perpetual Futures, 2024; '
                         'Koijen, Moskowitz, Pedersen & Vrugt, Carry, JFE 2018',
    'cyclical time encoding': 'Huang 2018; Carapuco, Neves & Horta 2018',
    'target-position actions': 'Moody & Saffell 2001; Huang 2018; Zhang, Zohren & Roberts 2020',
    'reward': 'Moody & Saffell, IEEE TNN 2001 - exact increment of the ratio whose first-order term is the DSR',
    'recurrent memory': 'Hausknecht & Stone 2015',
    'layer normalisation': 'Lyle et al., Understanding Plasticity in Neural Networks, ICML 2023; '
                           'Ba, Kiros & Hinton 2016',
    'PPO settings': 'Schulman et al. 2017; Andrychowicz et al., ICLR 2021; CleanRL ppo_atari_lstm',
    'entropy regularisation': 'Mei, Xiao, Szepesvari & Schuurmans, ICML 2020',
    'policy improvement step': 'Abdolmaleki et al., MPO, ICLR 2018 - E-step with a KL bound of 0.1; '
                               'Song et al., V-MPO, ICLR 2020',
    'all-action values on the same bar': 'Huang 2018 (action augmentation); Ciosek & Whiteson, AAAI 2018 '
                                         '(expected policy gradients); Allen et al. 2017 (mean actor-critic); '
                                         'Sinclair et al., ICML 2023 (exogenous inputs)',
    'forget-gate bias': 'Jozefowicz, Zaremba & Sutskever, ICML 2015',
    'burn-in': 'Kapturowski et al., R2D2, ICLR 2019',
    'time limits versus terminations': 'Pardo, Tavakoli, Levdik & Kormushev, ICML 2018',
    'exploring starts': 'Sutton & Barto 2018, section 5.3',
    'decorrelated parallel actors': 'Mnih et al., A3C, ICML 2016',
    'walk-forward folds': 'Zhang, Zohren & Roberts 2020',
    'quarterly windows': 'Yang, Liu, Zhong & Walid, ICAIF 2020',
    'aggregate statistics': 'Agarwal et al., NeurIPS 2021 (IQM, stratified bootstrap, rliable defaults)',
    'checkpoint-selection overfitting': 'Bailey, Borwein, Lopez de Prado & Zhu, J. Computational Finance 2017',
    'observation clipping': 'Engstrom et al., ICLR 2020',
}


def protocol(base=Config()):
    return {
        'version': 'agent',
        'source_sha256': source_hash(),
        'instrument': 'single BitMEX XBTUSD inverse perpetual; BTC collateral',
        'objective': 'exact per-step increment of an exponentially weighted Sharpe ratio of net BTC '
                     'equity log growth, including commission, slippage and funding',
        'algorithm': 'all-action recurrent PPO: every action evaluated on the same bar by exact '
                     'counterfactual branches of the account (the market is exogenous); clipped surrogate '
                     'summed over actions at all three accounts reachable over the previous bar; critic '
                     'on GAE lambda-returns for the visited account and expected one-step targets for the '
                     'others; fixed entropy bonus; separate LSTM actor and critic with layer normalisation '
                     'after every hidden layer; categorical target position SHORT/FLAT/LONG',
        'seeds': list(SEEDS), 'validation_years': list(YEARS),
        'config': base.as_dict(),
        'credit_window_steps': base.credit_window,
        'credit_window_hours': base.credit_window_hours,
        'cost_break_even_hours': COST_BREAK_EVEN_HOURS,
        'round_trip_cost_bp': base.round_trip_cost * 1e4,
        'reward_scale': base.reward_scale,
        'actor_widths': base.branch_widths(True),
        'critic_widths': base.branch_widths(False),
        'actor_gradient_steps': base.updates * base.epochs * (base.rollout // base.sequence)
                                * (base.environments // base.minibatch_sequences),
        'references': REFERENCES,
        'cost_policy': 'fixed at the realistic 5 bp commission plus 1 bp adverse slippage per leg '
                       'throughout training; no curriculum; sensitivity reported at higher slippage',
        'evaluation': 'each calendar quarter of the validation year from a flat account, sampled actions '
                      'over several seeds; argmax is never used; IQM with a stratified bootstrap interval '
                      'and the probability of backtest overfitting across checkpoints',
        'training_episodes': 'one quarter each, starting flat, short or long with equal probability, '
                             'at uniformly random points of the training span; first episodes staggered',
        'training_start': str(TRAIN_START), 'test_start': str(TEST_START),
        'test_locked': True, 'test_performance_accessed': False,
        'data': str(DATA), 'cache': str(CACHE),
    }
