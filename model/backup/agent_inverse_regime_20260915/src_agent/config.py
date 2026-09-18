"""Single specification for the agent: paths, locks and every constant.

Each constant's provenance - cited, derived, measured, a user decision, or an
explicitly unverified design choice - is tabulated in ARCHITECTURE.md §12. The
comments here say what a reader of the code needs; the table says why.

Two quantities are measured per fold, on that fold's training span only, and are
never constants: the five-minute return volatility, which sets the reward's prior
risk and the cost break-even, and the length of the training span, which turns the
budget in passes over it into a number of updates (fold_config).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
# USER: five training seeds; the values themselves are arbitrary identifiers.
SEEDS = (11, 29, 47, 61, 83)
# Expanding walk-forward folds, the protocol of Zhang, Zohren & Roberts (2020),
# who retrain on all data available up to each point: validate 2024 after
# training on 2017-2023, and 2025 after 2017-2024. Everything from 2026 is locked.
YEARS = (2024, 2025)

REWARDS = ('sharpe', 'sortino')
EVALUATION_POLICIES = ('argmax', 'sampled')
BAR_MINUTES = 5
BARS_PER_YEAR = 365 * 24 * 60 // BAR_MINUTES
# The quarterly window FinRL's ensemble strategy retrains and validates on (Yang et
# al. 2020): a quarter of a Julian year, in five-minute bars.
QUARTER_BARS = round(365.25 / 4 * 24 * 60 / BAR_MINUTES)
# ASSUMED, not measured: the annualised Sharpe ratio of a signal worth trading. With
# the volatility measured on a fold's training span and the fixed cost it sets the
# holding period a trade needs to pay for itself (cost_break_even_hours).
SIGNAL_SHARPE_ASSUMPTION = 1.0


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
    # User decisions: 5 bp commission and 1 bp adverse slippage per leg with no cost
    # curriculum, and 1x entry notional.
    # Bar length: the resolution of the features and the reward statistics; the ledger
    # and its margin checks run every minute.
    decision_minutes: int = BAR_MINUTES
    # USER (2026-09-11): hourly decisions on five-minute bars, adopted on the smoke tests'
    # evidence that five-minute decisions lose entry exploration (ARCHITECTURE.md §4.1).
    # Holding an action for several steps is frame skip (Mnih et al. 2015; Andrychowicz
    # et al. 2021). gamma and lambda stay per bar and are raised to this power per decision.
    decision_bars: int = 12
    fee: float = 0.0005
    slippage: float = 0.0001
    entry_leverage: float = 1.0
    # BitMEX XBTUSD base maintenance margin: 0.50% until 17 Apr 2020, 0.45% from
    # then (BitMEX announcement of 13 Apr 2020). The higher value is used for the
    # whole sample, which is conservative by 0.05% of notional.
    maintenance: float = 0.005

    # --- network -------------------------------------------------------------
    # 128 is CleanRL ppo_atari_lstm's LSTM width. Two hidden layers after it: "Two hidden
    # layers appear to work well for policy and value networks in all tested environments"
    # (Andrychowicz et al., ICLR 2021). The critic is twice as wide: "for the value function
    # there seems to be no downside in using wider networks" (ibid.).
    hidden: int = 128
    head_width: int = 128
    critic_width_multiple: int = 2

    # --- rollout shape, in decisions ------------------------------------------------
    # The discrete-action settings of PPO's Atari experiments (Schulman et al. 2017,
    # Table 5: 8 actors, horizon 128, minibatch 32 x 8) as CleanRL ppo_atari_lstm runs
    # them with a recurrent policy. The hourly credit window (24.3 decisions) admits them:
    # rollout / 2 and burn_in + sequence both clear it (validated below).
    environments: int = 8
    rollout: int = 128
    # CleanRL trains each environment's whole rollout as one sequence.
    sequence: int = 128
    # CleanRL starts that sequence from the recurrent state stored during collection,
    # which is stale after the first gradient step. R2D2 (Kapturowski et al. 2019)
    # rebuilds it over a burn-in of half the sequence (40 steps before 80).
    burn_in: int = 64
    # A quarter to the nearest whole rollout (quarter_episode): 17 x 128 hourly
    # decisions is 90.7 days, inside the user's 2-3 month range.
    episode_steps: int = 17 * 128
    # 8 environments in 4 minibatches of 2, CleanRL's num_minibatches = 4: 256 samples
    # each, the PPO paper's Atari minibatch of 32 x 8.
    minibatch_sequences: int = 2
    # CleanRL ppo_atari_lstm update_epochs = 4 (the PPO paper's Atari table uses 3).
    epochs: int = 4
    # Andrychowicz et al.: "recompute advantages once per data pass".
    recompute_advantages_per_pass: bool = True

    # --- horizon ---------------------------------------------------------------
    # User decision, per five-minute bar. gamma*lambda = 0.99650 per bar is a 24.3 h
    # credit window at hourly decisions, about twice the Sharpe-1 cost break-even
    # measured on the training spans (12.4 h and 12.9 h), and checked per fold.
    gamma: float = 0.99825
    gae_lambda: float = 0.99825

    # --- optimisation ------------------------------------------------------------
    # PPO's Atari settings (Table 5) and CleanRL ppo_atari_lstm: Adam at 2.5e-4 decayed
    # linearly to zero, clip 0.1, entropy bonus 0.01, and CleanRL's global gradient clip
    # of 0.5. Andrychowicz et al.'s 0.25 clip and 3e-4 step come from continuous control.
    actor_lr: float = 2.5e-4
    critic_lr: float = 2.5e-4
    anneal_lr: bool = True
    gradient_clip: float = 0.5
    entropy_coefficient: float = 0.01
    clip: float = 0.1

    # --- starting states ---------------------------------------------------------
    # Exploring starts (Sutton & Barto 2018, §5.3): each episode starts flat,
    # short or long with equal probability, so exits are practised from update one.
    flat_start_fraction: float = 1/3

    # --- reward (Moody & Saffell 2001) -------------------------------------------
    # 'sharpe': exact per-bar increment of an exponentially weighted Sharpe ratio.
    # 'sortino': the same with downside deviation as its risk - Moody & Saffell's
    # downside deviation ratio, which has "no penalty for large positive returns".
    reward: str = 'sharpe'
    # A five-day window, per bar. DESIGN CHOICE inside derived bounds - slower than
    # the credit window, faster than an episode - not pinned by any study.
    dsr_eta: float = 1/1440
    # MEASURED per fold: the variance of five-minute log returns over the fold's
    # training span (fold_config). The downside second moment starts at half of it.
    dsr_initial_second_moment: float | None = None

    # --- observation ---------------------------------------------------------------
    # OpenAI baselines PPO clips normalised observations to [-10, 10]
    # (Engstrom et al., ICLR 2020, §3 and Table 4).
    observation_clip: float = 10.0

    # --- schedule and evaluation ---------------------------------------------------
    # The budget counts passes over the fold's training history, not simulator steps:
    # Moody & Saffell (2001) and Deng et al. (2017) train for 100 epochs and stop by
    # validation. One pass collects as many decisions as the training span holds; the
    # number of updates follows from the span's length (fold_config).
    training_passes: int = 100
    # A validated checkpoint every 5 passes, 20 over the run, evenly spaced so that the
    # checkpoints of the two folds align (checkpoint_updates).
    validation_passes: int = 5
    # Derived per fold by fold_config; set directly only by tests.
    updates: int | None = None
    threads: int = 1
    # USER (2026-09-15): evaluation takes the most probable action ('argmax'), so one
    # evaluation is one deterministic path. Training still samples, as on-policy PPO
    # requires. 'sampled' keeps the previous protocol, under which every result up to
    # 2026-09-13 was produced.
    evaluation_policy: str = 'argmax'
    # Used only when evaluation_policy is 'sampled'. MEASURED at hourly decisions: one
    # evaluation's standard deviation across sampling seeds was 0.23 log over 51 weeks,
    # so a 4-seed mean has a standard error of 0.12, below the 0.26-0.63 differences
    # between training seeds.
    evaluation_seeds: int = 4
    # rliable defaults (Agarwal et al., NeurIPS 2021): 50,000 stratified bootstrap
    # replications for a 95% percentile interval.
    bootstrap_replications: int = 50_000
    confidence_level: float = 0.95

    def __post_init__(self):
        if self.decision_minutes != BAR_MINUTES:
            raise ValueError('this study fixes five-minute bars')
        if self.decision_bars < 1:
            raise ValueError('a decision spans at least one bar')
        if self.reward not in REWARDS:
            raise ValueError(f'reward must be one of {REWARDS}')
        if self.rollout % self.sequence or self.episode_steps % self.rollout:
            raise ValueError('sequence must divide rollout and rollout must divide episode length')
        if min(self.environments, self.burn_in, self.sequence, self.epochs, self.training_passes,
               self.validation_passes) < 1:
            raise ValueError('positive training sizes required')
        if self.updates is not None and self.updates < 1:
            raise ValueError('at least one update')
        if self.validation_passes > self.training_passes:
            raise ValueError('a checkpoint needs at least one validation interval of training')
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
        if not 0 < self.clip < 1:
            raise ValueError('the PPO clip must lie between zero and one')
        # GAE bootstraps at the end of every rollout. The average step sits half
        # a rollout before that bootstrap and must still see a whole credit window.
        if self.rollout / 2 < self.credit_window:
            raise ValueError('rollout truncates the credit window it is supposed to carry')
        # By the end of a training sequence the recurrent state has integrated
        # burn-in plus sequence steps; that span must cover one credit window.
        if self.burn_in + self.sequence < self.credit_window:
            raise ValueError('recurrent span is too short for the credit window')
        # The ratio's statistics must move slower than the advantage window, or
        # the reward's own drift is read as signal, and settle inside an episode.
        if not self.credit_window * self.decision_bars < 1 / self.dsr_eta < self.episode_steps * self.decision_bars:
            raise ValueError('differential Sharpe window must lie between the credit window and the episode')
        if self.dsr_initial_second_moment is not None and self.dsr_initial_second_moment <= 0:
            raise ValueError('invalid differential Sharpe prior')
        if min(self.evaluation_seeds, self.bootstrap_replications) < 1 or not 0 < self.confidence_level < 1:
            raise ValueError('invalid evaluation settings')
        if self.evaluation_policy not in EVALUATION_POLICIES:
            raise ValueError(f'evaluation_policy must be one of {EVALUATION_POLICIES}')

    @property
    def decision_gamma(self):
        """Discount per decision: gamma is specified per bar."""
        return self.gamma ** self.decision_bars

    @property
    def decision_lambda(self):
        return self.gae_lambda ** self.decision_bars

    @property
    def credit_window(self):
        """Effective GAE credit-assignment horizon, in decisions."""
        return 1 / (1 - self.decision_gamma * self.decision_lambda)

    @property
    def credit_window_hours(self):
        return self.credit_window * self.decision_bars * self.decision_minutes / 60

    @property
    def initial_reward_second_moment(self):
        """Prior second moment of the reward's risk term: the variance, or half of it downside."""
        if self.dsr_initial_second_moment is None:
            raise ValueError('the reward prior is measured on the training span: use fold_config')
        return self.dsr_initial_second_moment * (0.5 if self.reward == 'sortino' else 1.0)

    @property
    def round_trip_cost(self):
        return 2 * (self.fee + self.slippage)

    @property
    def samples_per_update(self):
        return self.environments * self.rollout

    @property
    def gradient_steps_per_update(self):
        return self.epochs * (self.rollout // self.sequence) * (self.environments // self.minibatch_sequences)

    def branch_widths(self, actor):
        scale = 1 if actor else self.critic_width_multiple
        return {'hidden': self.hidden * scale, 'head_width': self.head_width * scale}

    @property
    def reward_scale(self):
        """Puts the discounted return back at roughly unit scale.

        Derived, not tuned. The differential Sharpe ratio is approximately a
        z-score, and the discounted sum of an uncorrelated unit-variance sequence
        has standard deviation 1/sqrt(1 - gamma^2). A critic head that starts near
        zero cannot travel that far at Adam's step size, so the reward is scaled by
        sqrt(1 - gamma^2). A decision spanning k bars sums k increments, whose
        standard deviation grows as sqrt(k), so the per-decision discount is used and
        the sum is divided by sqrt(k). The downside ratio's increment is sqrt(2) times
        the Sharpe ratio's for symmetric returns, since its risk is sigma / sqrt(2), so
        it is divided by sqrt(2) as well.
        """
        downside = math.sqrt(0.5) if self.reward == 'sortino' else 1.0
        return downside * math.sqrt((1 - self.decision_gamma ** 2) / self.decision_bars)

    def as_dict(self):
        # Normalised through JSON so tuple fields compare equal to the lists read
        # back from protocol.json.
        return json.loads(json.dumps(asdict(self)))


def quarter_episode(decision_bars, rollout=Config.rollout):
    """Episode length in decisions: the whole number of rollouts nearest one quarter at this interval."""
    return max(1, round(QUARTER_BARS / decision_bars / rollout)) * rollout


def cost_break_even_hours(bar_return_sd, config):
    """Hours in position for a signal of the assumed Sharpe ratio to pay one round trip."""
    drift_per_bar = SIGNAL_SHARPE_ASSUMPTION * bar_return_sd / math.sqrt(BARS_PER_YEAR)
    return config.round_trip_cost / drift_per_bar * config.decision_minutes / 60


def check_horizon(config, bar_return_sd):
    """Refuse a credit window below the cost break-even measured on the fold; return the break-even."""
    break_even = cost_break_even_hours(bar_return_sd, config)
    if config.credit_window_hours < break_even:
        raise ValueError(f'credit window {config.credit_window_hours:.1f} h is below the cost break-even '
                         f'holding period of {break_even:.1f} h measured on the training span')
    return break_even


def fold_config(base, bar_return_sd, training_decisions):
    """One fold's configuration: the reward prior and the update budget, both measured on its training span."""
    if bar_return_sd is None or not bar_return_sd > 0:
        raise ValueError('the bar return volatility must be measured on the training span')
    updates = math.ceil(base.training_passes * training_decisions / base.samples_per_update)
    return replace(base, dsr_initial_second_moment=bar_return_sd ** 2, updates=updates)


def checkpoint_updates(config):
    """The updates after which a checkpoint is validated: one per validation interval, evenly spaced."""
    count = config.training_passes // config.validation_passes
    return tuple(sorted({max(1, round(k * config.updates / count)) for k in range(1, count + 1)}))


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
    'state representation reference': 'Deng, Bao, Kong, Ren & Dai, Deep Direct Reinforcement Learning for '
                                      'Financial Signal Representation and Trading, IEEE TNNLS 2017',
    'market regimes': 'Hamilton, Econometrica 1989; Ardia, Bluteau & Ruede, Finance Research Letters 2019',
    'trend efficiency': 'Kaufman, Smarter Trading, 1995',
    'bull and bear phases from peaks and their duration dependence': 'Pagan & Sossounov, JAE 2003; '
                                                                     'Lunde & Timmermann, JBES 2004',
    'daily and monthly realised volatility': 'Corsi, JFEC 2009; Andersen & Bollerslev, JEF 1997; '
                                             'Liu, Patton & Sheppard, JoE 2015',
    'volatility conditions trend payoffs': 'Barroso & Santa-Clara, JFE 2015; Moreira & Muir, JF 2017',
    'volume as attention': 'Barber & Odean, RFS 2008; Liu & Tsyvinski, RFS 2021',
    'perpetual funding': 'He, Manela, Ross & von Wachter, Fundamentals of Perpetual Futures, 2024; '
                         'Koijen, Moskowitz, Pedersen & Vrugt, Carry, JFE 2018',
    'variance ratio (tested, excluded)': 'Lo & MacKinlay, RFS 1988',
    'target-position actions': 'Moody & Saffell 2001; Huang 2018; Zhang, Zohren & Roberts 2020',
    'reward': 'Moody & Saffell, IEEE TNN 2001 - exact increments of the ratios whose first-order terms are the '
              'differential Sharpe and downside deviation ratios; Sortino & van der Meer, JPM 1991',
    'recurrent memory': 'Hausknecht & Stone 2015',
    'layer normalisation': 'Lyle et al., Understanding Plasticity in Neural Networks, ICML 2023; '
                           'Ba, Kiros & Hinton 2016',
    'PPO': 'Schulman, Wolfe, Dhariwal, Radford & Klimov 2017; Schulman, Moritz, Levine, Jordan & Abbeel, '
           'GAE, ICLR 2016',
    'PPO discrete-action settings': 'Schulman et al. 2017, Table 5 (Atari); CleanRL ppo_atari_lstm '
                                    '(Huang et al., JMLR 2022)',
    'PPO implementation study': 'Andrychowicz et al., What Matters in On-Policy RL, ICLR 2021',
    'PPO discrete-action failure mode': 'Hsu, Mendler-Dunner & Hardt, Revisiting Design Choices in PPO, 2020',
    'training budget in passes, stopping by validation': 'Moody & Saffell, IEEE TNN 2001; Deng et al., IEEE TNNLS '
                                                         '2017; Theate & Ernst, ESWA 2021; Prechelt, Early '
                                                         'Stopping - But When?, 1998',
    'overfitting to a repeated history': 'Sato 2019; Zhang, Vinyals, Munos & Bengio 2018',
    'entropy regularisation': 'Mei, Xiao, Szepesvari & Schuurmans, ICML 2020',
    'frame skip': 'Mnih et al., Nature 2015; Andrychowicz et al., ICLR 2021; Kalyanakrishnan et al. 2021',
    'forget-gate bias': 'Jozefowicz, Zaremba & Sutskever, ICML 2015',
    'burn-in': 'Kapturowski et al., R2D2, ICLR 2019',
    'time limits versus terminations': 'Pardo, Tavakoli, Levdik & Kormushev, ICML 2018',
    'exploring starts': 'Sutton & Barto 2018, section 5.3',
    'decorrelated parallel actors': 'Mnih et al., A3C, ICML 2016',
    'walk-forward folds': 'Zhang, Zohren & Roberts 2020',
    'quarterly windows': 'Yang, Liu, Zhong & Walid, ICAIF 2020',
    'aggregate statistics': 'Agarwal et al., NeurIPS 2021 (IQM, stratified bootstrap, rliable defaults)',
    'checkpoint-selection overfitting': 'Bailey, Borwein, Lopez de Prado & Zhu, J. Computational Finance 2017; '
                                        'Gort et al., backtest overfitting in DRL cryptocurrency trading, 2022',
    'observation clipping': 'Engstrom et al., ICLR 2020',
}


def protocol(base=Config()):
    # Imported here: both modules import this one.
    from .data import FEATURE_NAMES
    from .ledger import PORTFOLIO_NAMES
    ratio = 'Sharpe ratio' if base.reward == 'sharpe' else 'downside deviation (Sortino) ratio'
    return {
        'version': 'agent',
        'source_sha256': source_hash(),
        'instrument': 'single BitMEX XBTUSD inverse perpetual; BTC collateral',
        'objective': f'exact per-bar increment of an exponentially weighted {ratio} of net BTC equity log '
                     'growth, including commission, slippage and funding',
        'algorithm': 'recurrent PPO with the discrete-action settings of PPO\'s Atari experiments as CleanRL '
                     'ppo_atari_lstm runs them: clipped surrogate on GAE advantages normalised per minibatch '
                     'and recomputed before every pass; fixed entropy bonus; separate LSTM actor and critic with '
                     'layer normalisation after every hidden layer and recurrent states rebuilt over a burn-in '
                     'prefix; categorical target position SHORT/FLAT/LONG',
        'state': {'market': list(FEATURE_NAMES), 'account': list(PORTFOLIO_NAMES)},
        'decision_hours': base.decision_bars * base.decision_minutes / 60,
        'seeds': list(SEEDS), 'validation_years': list(YEARS),
        'config': base.as_dict(),
        'credit_window_steps': base.credit_window,
        'credit_window_hours': base.credit_window_hours,
        'signal_sharpe_assumption': SIGNAL_SHARPE_ASSUMPTION,
        'round_trip_cost_bp': base.round_trip_cost * 1e4,
        'reward_scale': base.reward_scale,
        'actor_widths': base.branch_widths(True),
        'critic_widths': base.branch_widths(False),
        'samples_per_update': base.samples_per_update,
        'gradient_steps_per_update': base.gradient_steps_per_update,
        'budget': f'{base.training_passes} passes over each fold\'s training span, a validated checkpoint every '
                  f'{base.validation_passes}; updates, the reward prior and the cost break-even measured per fold',
        'references': REFERENCES,
        'cost_policy': 'fixed at the realistic 5 bp commission plus 1 bp adverse slippage per leg '
                       'throughout training; no curriculum; the extra slippage that would erase the result is '
                       'reported from turnover',
        'evaluation': 'each calendar quarter of the validation year from a flat account; '
                      + ('the most probable action (argmax), one deterministic path per evaluation; '
                         'training still samples'
                         if base.evaluation_policy == 'argmax' else
                         'sampled actions over several seeds; argmax is not used')
                      + '; IQM with a stratified bootstrap interval and the probability of backtest '
                        'overfitting across checkpoints',
        'training_episodes': f'{base.episode_steps} decisions '
                             f'({base.episode_steps * base.decision_bars * base.decision_minutes / 1440:.1f} days) each, '
                             'starting flat, short or long with equal probability, at uniformly random points of '
                             'the training span; first episodes staggered',
        'training_start': str(TRAIN_START), 'test_start': str(TEST_START),
        'test_locked': True, 'test_performance_accessed': False,
        'data': str(DATA), 'cache': str(CACHE),
    }
