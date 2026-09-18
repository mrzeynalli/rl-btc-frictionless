"""Single specification for the agent: paths, locks, and every tuned constant.

Self-contained. Nothing here imports from an earlier generation, and the market
cache lives under this agent's own output directory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import math
import json
import os

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/bitmex_xbtusd_1m_2016_2026_08_01'
OUTPUT = ROOT / 'results/agent'
CACHE = OUTPUT / 'cache'
TEST_START = pd.Timestamp('2026-01-01', tz='UTC')
TRAIN_START = pd.Timestamp('2017-01-01', tz='UTC')
# Three seeds over three folds: nine runs. The predecessor's seven seeds existed
# when a run cost 90 minutes; at 5,000 updates a run is roughly two days, and
# depth of training now matters more than breadth of seeds. Three is the minimum
# that can show whether an outcome reproduces at all.
SEEDS = (11, 29, 47)
YEARS = (2023, 2024, 2025)

# Measured on the cached 2023-2026 closes: the five-minute log-return standard
# deviation is 14.8 bp, so a 12 bp round trip costs 0.81 of one bar's typical
# move and a signal worth an annualised Sharpe of 1.0 needs about 22 hours in
# position before it has paid for one entry and one exit. Every horizon,
# holding-prior and cost constant below is anchored to that number.
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
    # --- market and ledger -------------------------------------------------
    decision_minutes: int = 5
    fast_minutes: int = 60
    # Realistic and fixed. No curriculum: an edge discovered at 2 bp does not
    # survive contact with 6 bp, and cost is a turnover regulariser rather than
    # a constant, so training below live cost selects a trading frequency that
    # cannot pay for itself. Robustness is reported by re-scoring the selected
    # policy at 8 and 10 bp, never by training cheaply.
    fee: float = 0.0005
    slippage: float = 0.0001
    entry_leverage: float = 1.0
    maintenance: float = 0.005

    # --- network -----------------------------------------------------------
    # Widths below are the ACTOR's. Andrychowicz et al. (ICLR 2021, 250k agents)
    # find separate policy and value networks better on 4 of 5 environments, that
    # the optimal policy width is environment-dependent with both too-small and
    # too-large hurting, and that there is "no downside in using wider networks"
    # for the value function - on HalfCheetah their best was 16-32 units in the
    # policy against 256 in the value network. The critic therefore scales up.
    # 128 matches CleanRL's ppo_atari_lstm recurrent baseline; Tummon, Raja and
    # Ryan (2020) reached 256 by Optuna search on a BitMEX crypto task, so 64 was
    # very likely undersized. Which of 128 or 256 suits THIS problem is settled
    # by the planted-edge test, not by citation.
    cell: str = 'lstm'
    normalization: str = 'none'
    dropout: float = 0.0
    attention: bool = False
    attention_heads: int = 4
    hidden: int = 128
    local_width: int = 64
    position_hidden: int = 32
    head_width: int = 128
    critic_width_multiple: int = 2

    # --- rollout shape -----------------------------------------------------
    # 8 parallel environments: Andrychowicz find performance DECREASES sharply as
    # parallel environments rise, because it shortens the experience chunks a
    # recurrent policy needs; CleanRL's ppo_atari_lstm also uses 8.
    environments: int = 8
    rollout: int = 1024
    # The truncated-BPTT window must span the credit window it assigns credit
    # over; 128 steps was 10.7 h against a 23.8 h window. Halving the chunk count
    # is compensated by halving minibatch_sequences, which keeps the minibatch at
    # 1024 samples and the gradient-step budget at 32 per update.
    sequence: int = 256
    # R2D2 (Kapturowski et al. 2019) stores 80-step sequences behind a 40-step
    # burn-in: a ratio of one half, which 128 against 256 reproduces exactly.
    burn_in: int = 128
    episode_steps: int = 4096
    minibatch_sequences: int = 4
    # "Going over the experience multiple times appears to be crucial for good
    # sample complexity" - Andrychowicz. Four passes is the PPO paper's Atari
    # setting (Table 5) and CleanRL's default.
    epochs: int = 4
    critic_epochs: int = 2
    # Andrychowicz propose and measure recomputing advantages at the start of
    # each pass rather than once per iteration: stale advantages "can in fact
    # hurt performance" and per-pass recomputation "performs best among all
    # variants".
    recompute_advantages_per_pass: bool = True

    # --- horizon -----------------------------------------------------------
    # gamma*lambda = 0.99650 -> 286 steps -> 23.8 h, above the 22 h break-even.
    # The predecessor used 0.9716, a 2.9 h window in which entering collected
    # 1.6 bp against 12 bp of cost, so every profitable trade looked like a loss
    # and entropy fell monotonically in 78 of 78 runs.
    gamma: float = 0.99825
    gae_lambda: float = 0.99825

    # --- optimisation ------------------------------------------------------
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # "Start with the clipping threshold set to 0.25" - Andrychowicz; 0.2 and 0.3
    # both perform reasonably in their sweep.
    clip: float = 0.25
    target_kl: float = 0.01
    # Off by default, which is CleanRL's ppo_atari_lstm setting (target_kl=None)
    # and follows Andrychowicz: the PPO clip "already enforces the trust region
    # which makes KL penalties or constraints redundant". As a sticky flag this
    # guard cancelled every remaining actor minibatch on 8-16% of updates while
    # the median KL sat 14x BELOW target, so it was deleting gradient steps on
    # outliers rather than constraining drift. 'update' reproduces CleanRL's
    # behaviour when target_kl is set; 'pass' releases it each pass. KL is logged
    # every update either way, so real drift stays visible.
    kl_stop_scope: str = 'off'
    # "Linearly decaying the learning rate may slightly improve performance" -
    # Andrychowicz; CleanRL anneals by default.
    anneal_lr: bool = True
    gradient_clip: float = 0.5
    # Per-minibatch standardisation forces unit-scale gradients even when a
    # batch holds no signal, which in a domain with a per-sample SNR near 0.05
    # means confidently learning noise. A slow global scale keeps relative
    # magnitudes across updates instead.
    advantage_scale_decay: float = 0.05

    # --- exploration -------------------------------------------------------
    # The target is on the DIRECTION choice P(side | switch), whose maximum is
    # ln 2 = 0.693, and never on the gate. Maximum entropy on the gate means
    # switching every other bar, a ten-minute holding period, which is the worst
    # possible policy under these costs. The coefficient tunes itself to hold
    # this target, as in Haarnoja et al. 2018 (arXiv:1812.05905) section 5.
    # TWO targets, one per factor of the policy. The direction target guards
    # P(side | switch); the GATE target guards P(switch) and is the one that was
    # missing. Every collapse observed so far - v4 to permanent flat, v5 to one
    # side held for months, and both runs of the previous agent - was a gate
    # collapse, while direction entropy never left 0.64-0.68 and so never
    # triggered its dual. Andrychowicz find entropy regularisation unnecessary,
    # but explicitly because "careful policy initialization is enough to
    # guarantee good exploration" for Gaussian policies in continuous control;
    # in a discrete action space CleanRL's Atari configs carry a fixed
    # ent_coef = 0.01, and our own logs show the collapse directly.
    entropy_target: float = 0.6
    entropy_alpha_init: float = 0.005
    # Matches CleanRL's SAC exactly: a_optimizer = Adam([log_alpha], lr=q_lr)
    # with q_lr = 1e-3. Optimising log_alpha rather than alpha is also theirs.
    entropy_alpha_lr: float = 1e-3
    # Effectively zero: the floor is only reached when entropy is already above
    # target and no pressure is wanted. Verified harmless - the direction dual
    # sat on it for an entire run with no ill effect.
    entropy_alpha_min: float = 1e-4
    # SAC leaves the temperature unbounded and can afford to, because for a
    # Gaussian policy dH/d(log sigma) = 1 exactly: the dual can always move the
    # entropy, so it never has to wind up. A Bernoulli gate saturates instead.
    # With z = logit(P(keep)), dH/dz = -z*p*(1-p), which is 0.0009 at a 833 h
    # hold, 0.0119 at the 44 h target and 0.217 near p = 0.78 - a restoring force
    # 13x too weak to escape a collapse and 18x too strong once escaped. A spring
    # whose stiffness rises with displacement. Alpha grows large enough to break
    # out, then that same alpha slams the policy past the target.
    #
    # The bound is empirical, from our own two experiments. Peak alpha where the
    # mechanism WORKED: 0.0089, 0.0090, 0.0122, 0.0346, 0.0474. Peak alpha where
    # the gate overshot its target by 19-36x: 0.0926, 0.1903. 0.05 sits above
    # every value in the first set and below every value in the second.
    entropy_alpha_max: float = 0.05

    # --- holding priors (trainable starting beliefs, not constraints) -------
    # 0.9965 per five-minute bar is a ~24 h mean hold; 0.995 is ~17 h flat.
    # The predecessor started at 50 minutes and 10 minutes, so every trade in
    # its formative experience was too short to cover costs and the only lesson
    # available was that trading loses money. Both remain learnable.
    initial_persistence: float = 0.9965
    # Equal to the holding prior, and that is a derivation rather than symmetry
    # for its own sake. In the jump chain over {short, flat, long} every state is
    # entered equally often, so time in each state is proportional to its dwell
    # time; equal dwell times therefore put one third of the time flat and two
    # thirds in the market. The predecessor used 0.9 against 0.5 - a 5x bias
    # toward being in the market - and when I replaced it with 24 h against 17 h
    # I removed that protection without noticing, and one run collapsed to
    # permanent flat. The protection now comes from the gate entropy floor
    # instead, which resists collapse toward flat and toward a held side alike,
    # rather than from a hand-picked asymmetry.
    initial_flat_persistence: float = 0.9965
    flat_start_fraction: float = 1/3

    # --- reward: differential Sharpe (Moody & Saffell 2001) ----------------
    # eta = 1/1440 is a five-day window: longer than the 23.8 h credit window so
    # the baseline is stable across it, shorter than the 14.2-day episode so it
    # converges in-episode, and wide enough to hold several trades.
    dsr_eta: float = 1/1440
    dsr_initial_second_moment: float = FIVE_MINUTE_RETURN_SD ** 2
    dsr_variance_floor_fraction: float = 1e-3
    dsr_clip: float = 10.0

    # --- counterfactual ----------------------------------------------------
    counterfactual_stride: int = 128
    counterfactual_horizon: int = 48
    counterfactual_actor_weight: float = 0.5
    counterfactual_critic_weight: float = 0.5
    counterfactual_warmup: int = 10
    counterfactual_enabled: bool = True

    # --- schedule and evaluation -------------------------------------------
    # CleanRL's ppo_atari_lstm - recurrent PPO, and it uses 8 environments as we
    # do - runs 10M timesteps at 1024 per batch with 4 epochs over 4 minibatches:
    # 156,240 actor gradient steps. At 32 steps per update here, 5,000 updates
    # gives 160,000. The inherited 400 gave 12,800, twelve times short, and had
    # been carried from v4 without ever being justified.
    updates: int = 5000
    validation_every: int = 500
    threads: int = 1
    # Actions are SAMPLED at evaluation, never taken by argmax: with a
    # persistence gate, leaving flat under argmax would require P(switch) > 0.5,
    # a ten-minute mean hold, so argmax evaluates a policy that by construction
    # can never trade. Several seeds because one sampled path is noisy.
    evaluation_seeds: int = 4
    # Blocks multiply the work by the number of windows, so they use fewer
    # sampling seeds; the block distribution itself already supplies the spread.
    block_evaluation_seeds: int = 2
    block_days: int = 90
    block_step_days: int = 15
    stress_slippage: tuple = (0.0001, 0.0003, 0.0005)

    def __post_init__(self):
        if self.decision_minutes != 5:
            raise ValueError('this study fixes a five-minute decision interval')
        if self.rollout % self.sequence or self.episode_steps % self.rollout:
            raise ValueError('sequence must divide rollout and rollout must divide episode length')
        if min(self.environments, self.burn_in, self.sequence, self.updates) < 1:
            raise ValueError('positive training sizes required')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid discount')
        if not 0 <= self.fee < 0.01 or not 0 <= self.slippage < 0.01:
            raise ValueError('invalid execution cost')
        if not 0 < self.entry_leverage <= 1:
            raise ValueError('this study permits at most 1x entry contract notional')
        if self.cell not in ('gru', 'lstm') or self.normalization not in ('none', 'layer', 'batch'):
            raise ValueError('unsupported recurrent cell or normalization')
        if not 0 <= self.dropout < 1 or not 0 < self.initial_persistence < 1:
            raise ValueError('invalid dropout or initial persistence')
        if not 0 < self.initial_flat_persistence < 1 or not 0 <= self.flat_start_fraction <= 1:
            raise ValueError('invalid flat-state exploration setting')
        if self.local_width % self.attention_heads:
            raise ValueError('attention width must divide into heads')
        if not 0 < self.entropy_target < 0.6931471805599453:
            raise ValueError('direction entropy target must lie inside (0, ln 2)')
        if self.kl_stop_scope not in ('pass', 'update', 'off'):
            raise ValueError('kl_stop_scope must be pass, update or off')
        if self.critic_width_multiple < 1:
            raise ValueError('the value network may not be narrower than the policy')
        if self.environments % self.minibatch_sequences:
            raise ValueError('minibatch_sequences must divide the environment count')
        # The truncated-BPTT window must reach across the horizon over which
        # credit is assigned, or the recurrent gradient cannot carry it.
        if self.sequence < self.credit_window / 2:
            raise ValueError('BPTT sequence is too short for the credit window it must carry')
        if not 0 < self.dsr_eta < 1 or self.dsr_initial_second_moment <= 0:
            raise ValueError('invalid differential Sharpe settings')
        if not 0 < self.advantage_scale_decay <= 1:
            raise ValueError('invalid advantage scale decay')
        if min(self.evaluation_seeds, self.block_evaluation_seeds) < 1:
            raise ValueError('evaluation requires at least one sampling seed')
        if not 0 < self.block_step_days <= self.block_days:
            raise ValueError('invalid block window schedule')
        if self.counterfactual_enabled and (self.rollout % self.counterfactual_stride or
                not 0 < self.counterfactual_horizon < self.counterfactual_stride // 2):
            raise ValueError('counterfactual paths must fit inside each rollout interval')
        # GAE cannot see past the end of a rollout, so a rollout shorter than
        # the window it carries silently truncates it back to the failure the
        # horizon change exists to remove.
        if self.rollout < 3 * self.credit_window:
            raise ValueError('rollout truncates the credit window it is supposed to carry')
        # The DSR baseline must be slower than the window the advantage sees,
        # or the reward's own drift is mistaken for signal inside it.
        if 1 / self.dsr_eta < 2 * self.credit_window:
            raise ValueError('differential Sharpe window is shorter than the credit window')

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

    @property
    def gate_entropy_target(self):
        """Entropy floor for the keep/switch decision, in nats.

        Derived, not tuned: the entropy of a Bernoulli whose mean dwell time is
        twice the measured cost break-even. Below that the agent is no longer
        sampling alternatives often enough to learn from them; above it the dual
        applies no pressure at all, so this is a floor rather than a forced
        trading rate. At 44 h it is 0.0138 nats, against 0.0249 at the 22 h
        break-even itself and 0.003 in the collapsed runs.
        """
        steps = 2 * COST_BREAK_EVEN_HOURS * 60 / self.decision_minutes
        keep = 1 - 1/steps
        return float(-(1-keep)*math.log(1-keep) - keep*math.log(keep))

    def branch_widths(self, actor):
        """Per-branch layer widths.

        Only the value-estimation path widens. Andrychowicz et al. measure the
        width of the value MLP, not of shared feature encoders, so the minute
        encoder and the position encoder stay identical between the branches and
        only the market memory and the head scale up. Widening everything reached
        989k parameters against 630k training decisions, which buys capacity the
        citation does not support.
        """
        scale = 1 if actor else self.critic_width_multiple
        return {'hidden': self.hidden*scale, 'local_width': self.local_width,
                'position_hidden': self.position_hidden,
                'head_width': self.head_width*scale}

    @property
    def reward_scale(self):
        """Chosen so the discounted return has roughly unit scale.

        Derived, not tuned. The differential Sharpe ratio is approximately a
        z-score, so its per-step standard deviation is about one, and the
        discounted sum of an uncorrelated sequence has standard deviation
        1/sqrt(1 - gamma^2), here about 17. The critic head starts near zero and
        Adam moves a weight by roughly its learning rate per step, so 19,200
        critic steps at 3e-4 cannot travel that far: explained variance would sit
        near zero for the whole run and the advantage would be noise. Scaling by
        sqrt(1 - gamma^2) puts the value target back at unit scale.

        The predecessor's factor of 100 on log equity happened to land in the
        same place: 0.148 per step against gamma 0.99760 predicts a target
        standard deviation of 2.1, and 1.86 was observed.
        """
        return math.sqrt(1 - self.gamma ** 2)

    def as_dict(self):
        # Normalised through JSON so that tuple fields survive the round trip.
        # Without this, every training job compares its own tuples against the
        # lists read back from protocol.json and aborts on a false mismatch.
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


def protocol(base=Config()):
    return {
        'version': 'agent',
        'source_sha256': source_hash(),
        'instrument': 'single BitMEX XBTUSD inverse perpetual; BTC collateral',
        'objective': 'differential Sharpe ratio of net BTC equity log growth, including '
                     'commission, slippage and funding',
        'algorithm': 'recurrent PPO with a clipped enumerated counterfactual policy gradient and '
                     'action-value regression; automatically tuned direction entropy',
        'seeds': list(SEEDS), 'validation_years': list(YEARS),
        'config': base.as_dict(),
        'credit_window_steps': base.credit_window,
        'credit_window_hours': base.credit_window_hours,
        'cost_break_even_hours': COST_BREAK_EVEN_HOURS,
        'round_trip_cost_bp': base.round_trip_cost * 1e4,
        'reward_scale': base.reward_scale,
        'gate_entropy_target': base.gate_entropy_target,
        'actor_widths': base.branch_widths(True),
        'critic_widths': base.branch_widths(False),
        'actor_gradient_steps': base.updates*base.epochs*(base.rollout//base.sequence)
                                *(base.environments//base.minibatch_sequences),
        'references': {
            'horizon and holding priors': 'derived from the measured 14.8 bp five-minute return '
                                          'standard deviation and the resulting 22 h cost break-even',
            'update budget': 'CleanRL ppo_atari_lstm: 10M steps at 1024 batch, 4 epochs over 4 '
                             'minibatches = 156,240 actor gradient steps',
            'clip, gradient clip, annealing, advantage recomputation, separate and wider value '
            'network, tanh, 100x smaller final policy layer':
                'Andrychowicz et al., What Matters In On-Policy RL, ICLR 2021 (250,000 agents)',
            'burn-in ratio': 'Kapturowski et al., R2D2, ICLR 2019: 80-step sequences behind a '
                             '40-step burn-in',
            'automatic entropy coefficient': 'Haarnoja et al. 2018, arXiv:1812.05905, section 5',
            'differential Sharpe reward': 'Moody & Saffell, IEEE TNN 2001',
            'network width evidence': 'Tummon, Raja & Ryan 2020 reached 256 hidden units by Optuna '
                                      'search on a BitMEX crypto task; CleanRL ppo_atari_lstm uses '
                                      '128 for its recurrent baseline',
            'entropy on a discrete action space': 'CleanRL Atari configurations carry a fixed '
                                                  'ent_coef of 0.01; Andrychowicz find regularisation '
                                                  'unnecessary only because Gaussian policies in '
                                                  'continuous control explore by construction',
        },
        'reward': 'Moody & Saffell (2001) differential Sharpe ratio on per-step net log BTC equity '
                  'change; the running mean and second moment are part of the observation, without '
                  'which the reward depends on unobserved history and the critic cannot fit it',
        'cost_policy': 'fixed at the realistic 5 bp commission plus 1 bp adverse slippage per leg '
                       'throughout training; no curriculum; robustness reported by re-scoring at '
                       'higher slippage',
        'exploration': 'trainable side-specific continuation bias initialised to 0.995 flat and '
                       '0.9965 holding; entropy target applied to P(side | switch) only, uniformly '
                       'across states and never weighted by P(switch), which would let the agent '
                       'escape the penalty by never switching',
        'sizing': 'entry notional is 1x account equity and is not rebalanced; the differential '
                  'Sharpe reward prices out the leverage-reset flicker on its own, because raising '
                  'exposure scales return and volatility together while the fees are pure loss',
        'evaluation': 'sampled actions over several seeds, reported as a distribution; overlapping '
                      f'{base.block_days}-day blocks stepped {base.block_step_days} days at '
                      'selection; argmax is never used',
        'initial_training_states': 'one-third untouched flat histories, otherwise a persistent '
                                   'random-action prefix builds valid flat/long/short histories '
                                   'including costs; evaluation always starts flat',
        'counterfactual': 'all three first actions on copied ledgers, then sampled current-policy '
                          'continuation and a value bootstrap; no fake on-policy samples',
        'training_start': str(TRAIN_START), 'test_start': str(TEST_START),
        'test_locked': True, 'test_performance_accessed': False,
        'data': str(DATA), 'cache': str(CACHE),
    }
