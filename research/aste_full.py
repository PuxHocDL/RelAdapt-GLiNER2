"""Full ASTE: extract (aspect, opinion, sentiment) triplets end to end.

The previous round fed the model gold aspect spans, which turned out to be a
saturated task (92.5% accuracy) where aspect-conditioned interaction had nothing
left to fix. Here nothing is given: the model enumerates spans, decides which are
aspects and which are opinions, pairs them, and labels the pair. Scoring is exact
triplet F1 - all three elements must match - which is the standard benchmark
metric and is far from saturated.

Design follows the span-enumeration family (Span-ASTE, Xu et al. 2021):

    encoder -> span representations -> aspect/opinion scores -> prune to top-k
            -> [optional ACSI over pruned aspects] -> pair classifier (4-way)

ACSI is optional so the same code answers "does sparse aspect-conditioned
interaction help when the pairing decision is actually hard".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NONE, POS, NEU, NEG = 0, 1, 2, 3
PAIR_LABELS = ("none", "POS", "NEU", "NEG")
POLARITY_TO_PAIR = {"POS": POS, "NEU": NEU, "NEG": NEG}


@dataclass
class AsteConfig:
    hidden_size: int = 768
    max_span_width: int = 8
    # 20x20 = 400 candidate pairs per sentence against ~2 positives; the pair
    # classifier collapses to "none" under that ratio. Fewer kept spans plus an
    # explicit down-weight on the negative class keeps the problem learnable.
    n_aspect_keep: int = 10
    n_opinion_keep: int = 10
    none_weight: float = 0.1
    span_dim: int = 256
    pair_dim: int = 256
    dropout: float = 0.2
    max_rel_distance: int = 32

    # Ablation switches.
    use_acsi: bool = False
    use_gated_pooling: bool = True
    use_distance_feature: bool = True
    acsi_top_k: int = 8
    acsi_heads: int = 4
    acsi_dim: int = 256

    # ABSA-adapter additions.
    # role_symmetric: one parameter set applied to both span roles, separated only
    #   by a role embedding. Motivated by the error split - 25% of false positives
    #   have the aspect right and the opinion wrong, 22% the reverse - so
    #   conditioning only the aspect leaves half the binding errors untouched.
    # assignment: Sinkhorn with dustbins over the aspect x opinion score matrix.
    #   85% of gold aspects bind exactly one opinion, yet the classifier scores
    #   every cell independently and emits 578 surplus triplets. This injects the
    #   combinatorial constraint instead of more capacity: it adds one scalar.
    use_role_symmetric: bool = False
    use_assignment: bool = False
    sinkhorn_iters: int = 20
    assignment_weight: float = 1.0
    # Auxiliary token-level supervision. With the assignment constraint in place,
    # 72.7% of the remaining false positives are span-selection errors and 10-15%
    # of gold spans never enter the top-k candidate set at all - a hard ceiling no
    # pairing layer can lift. This is one linear layer trained alongside, and it
    # costs nothing at inference: it only sharpens the span scores.
    use_token_tagger: bool = False
    w_tagger: float = 0.5

    # Schema-grounded span scoring, added to B2 rather than replacing it.
    # A zero-shot probe showed GLiNER2 already scores these labels well
    # (aspect F1 0.757 on 14res with no training), and a same-pipeline ablation
    # showed schema scoring beating a scratch scorer with 33% fewer parameters.
    # The gate starts at zero, so step 0 is bit-identical to B2 and the learned
    # value is itself the measurement of how much the pretrained signal is worth.
    use_schema_score: bool = False
    # The prior can act on the model in two independent ways, and the first
    # attempt conflated them. schema_bias adds `schema_gate * prior` to the span
    # logit - a soft nudge whose size the model learns. schema_reserve is
    # structural: it hands the prior a fixed number of the candidate slots
    # outright. Measured on 15res the learned gate stopped at 0.03-0.04 against a
    # prior in [0,1], i.e. a bias of four hundredths of a logit, so the soft route
    # never got to say anything. The reserved route cannot be throttled that way.
    schema_bias: bool = True
    schema_reserve: int = 0

    # Stability work. Diagnosis: with a 450M encoder on 600-1300 sentences, 2 of
    # 20 runs collapse. They do not diverge - loss plateaus flat at ~0.58 for the
    # rest of training - and the collapse lands right after the one-cycle LR peak,
    # not at initialisation. Two degenerate states were observed: the dustbin
    # swallowing everything (n_pred 71 vs 514 gold) and unconstrained
    # over-prediction (901 vs 485). Both are fixed points of the assignment layer.
    #
    # sinkhorn_temp_start > 1 softens the assignment early, so the sharp-but-wrong
    # fixed point does not exist while the encoder is still moving.
    sinkhorn_temp_start: float = 1.0
    sinkhorn_temp_end: float = 1.0

    # Precision work. Our recall matches or beats published systems while
    # precision trails them by 10-14 points, and every published system has
    # P > R while we are the only one with R >> P. Each sentence offers ~100
    # candidate pairs against ~2 gold ones, and none_weight currently damps all
    # ~98 negatives equally. Most of them are trivial - neither span is a real
    # aspect or opinion - so they dominate the gradient while the negatives that
    # actually teach pairing (a real aspect against the wrong real opinion) are
    # a fraction of it. hard_none_weight raises just that second group.
    # P1, listwise pairing. The pair loss scores 100 cells as 100 independent
    # yes/no questions, but the decision is not independent: given an aspect, the
    # question is which of the ten opinions binds to it, if any. Measured on
    # 15res, 16.9% of gold triplets have both spans already in the candidate set
    # and are still rejected as "none", against 4.8% lost in span selection, so
    # the ordering inside the grid - not the candidate set - is where the loss is.
    # B2h changed the *weights* over cells and won +1.90; this changes the *unit*
    # from a cell to a row and a column. The "no partner" slot sits at log-odds 0,
    # so the arm adds no parameters.
    use_listwise: bool = False
    w_listwise: float = 0.5
    # L-series. `use_listwise` above normalises every positive inside ONE softmax
    # over [opinions, none], so two gold opinions for the same aspect compete for
    # the same probability mass. Measured on train that mis-specifies 12.9% of
    # aspects and 8.2% of opinions. `use_multi_positive` gives each gold partner
    # its own denominator over the negatives and NONE, never over another gold
    # partner. See `aste_relation.py`.
    use_multi_positive: bool = False
    w_multi_positive: float = 0.5
    multi_positive_symmetric: bool = True
    # M-series: gold > NULL > negative. B2hl raised gold-pair emission (the
    # rejected bucket fell in all four domains) and still lost 1.92 dev F1,
    # because a shared softmax's optimum drives NONE's mass to zero. NULL is the
    # decoder's boundary, not a competitor, so this asks for a margin on both
    # sides of it instead of pushing it out of the way.
    use_null_rank: bool = False
    w_null_rank: float = 0.5
    null_m_pos: float = 1.0
    null_m_neg: float = 1.0
    null_alpha: float = 1.0
    null_hard: bool = False
    # Total relation mass per row/column against the gold degree. Baseline
    # under-emits, B2hl over-emits; this constrains the quantity both get wrong.
    use_degree_loss: bool = False
    w_degree: float = 0.2
    hard_none_weight: float = 0.0    # 0 disables: every negative keeps none_weight
    # Raising hard negatives to 1.0 while leaving the rest at 0.1 multiplied the
    # total weight on "none" by about four. The base encoder absorbed that, but
    # on the large encoder it re-triggered the very failure none_weight was added
    # to prevent: 3 of 15 runs went to n_pred = 0 with the loss flat from epoch 1.
    # With this flag the per-cell negative weights are rescaled to sum to what a
    # uniform none_weight would give, so emphasis is redistributed among the
    # negatives instead of added on top of them.
    neutral_negative_mass: bool = False

    # Dustbin abstention gate. log_sinkhorn already returns a dustbin row and
    # column - the probability that an aspect, or an opinion, takes no partner at
    # all - and the non-coupled decode path throws both away, keeping only the
    # matched cell. That discarded quantity is a calibrated "this span should not
    # be paired" signal, which is exactly what precision needs. The gate scores a
    # cell by how far its match beats the better of the two abstentions instead of
    # by its matched probability alone. One scalar, zero-initialised, so the model
    # starts exactly at the arm this is built on and learns how much to lean on it.
    use_dustbin_gate: bool = False

    # E1 + E2, the architectural change.
    # E1 (coupled selection): a span's role score is derived FROM the match matrix
    #   instead of from an independent head. A span is an aspect because some
    #   opinion matches it, not on its own merits. Targets the dominant remaining
    #   error - 30.5% of false positives keep the right aspect but attach the
    #   wrong opinion, which is what independent selection produces.
    # E2 (single relation decision): the 4-way classifier's "none" class and the
    #   Sinkhorn dustbin were two mechanisms answering the same question. The
    #   classifier drops to 3 polarity classes; the dustbin decides existence.
    use_coupled_selection: bool = False
    n_prune: int = 30
    w_match: float = 1.0
    w_role: float = 1.0
    w_prune: float = 1.0
    # logsumexp spreads gradient across a whole row, which flattens the match
    # matrix; max sends it to the single best partner only.
    role_aggregation: str = "logsumexp"
    deep_span_encoder: bool = False
    # Merging "none" into the dustbin re-created the imbalance that made the
    # original pair classifier collapse: ~2 matched cells against ~18 dustbin
    # terms per sentence. This is the same remedy as none_weight, carried over.
    dustbin_weight: float = 0.1

    # D1: domain-fusion adapters. The span pathway is the bottleneck (pairing is
    # at 94.8% while aspect/opinion sit at 0.863/0.852) and five attempts to widen
    # it with randomly initialised capacity all lost ground. These adapters carry
    # capacity that is trained on the leakage-filtered union of the other three
    # domains first - 2.3x to 4.6x the target's own training set - so the extra
    # parameters have data behind them rather than only a wider layer.
    # `shared` is stacked before `domain`, MAD-X order; both are zero-initialised
    # at the up-projection, so an injected encoder starts bit-identical to B2h.
    use_adapter: bool = False
    adapter_shared_dim: int = 256
    adapter_domain_dim: int = 64
    adapter_dropout: float = 0.1
    # Frozen backbone is the point of the arm, not an economy: the large-encoder
    # collapses land right after the one-cycle LR peak, i.e. they are encoder
    # drift, and bounding that drift is half of what this design is testing.
    freeze_backbone: bool = False

    # H-series: the polarity decision. See `aste_polarity.py` for the full error
    # decomposition this comes from. Short version, measured on the stored B2h
    # predictions: a pair that is found and paired correctly but labelled with
    # the wrong polarity is 23.1-31.6% of all missed triplets and 17.0-26.7% of
    # all false positives, because that error costs one of each at once.
    # Relabelling those and emitting nothing extra is worth +5.34 to +10.19 F1
    # (avg +7.31) - the largest remaining bucket and the only one free of a
    # precision trade. 49-77% of it is NEU, which the model predicts 0 times on
    # 15res and 5 times on 16res against 145 gold.
    #
    # factor_polarity replaces the 4-way softmax with P(exists) x P(pol|exists).
    # The function class is identical - softmax is shift-invariant, so pinning
    # the none logit to 0 costs nothing - and `pair_dim -> 1` plus
    # `pair_dim -> 3` is exactly as many parameters as `pair_dim -> 4`. What
    # changes is that NEU competes against POS and NEG over ~2 real cells rather
    # than against `none` over 100.
    factor_polarity: bool = False
    w_polarity: float = 1.0
    # "none" | "sqrt" (1/sqrt f) | "balanced" (1/f), from the training split,
    # applied inside the 3-way head only. Weighting NEU there cannot re-open the
    # existence imbalance `none_weight` exists to control, which is what goes
    # wrong when the same fix is attempted on the 4-way loss.
    polarity_class_weight: str = "none"
    # Prototype contrastive term on the pair representation: the mechanism
    # MiniConGTS (-2.12 to -7.04 F1 without it), MPGM (D2SCL) and GAC all share.
    # EMA centroids rather than in-batch positives because at batch size 8 with
    # NEU at 2-7% a batch usually holds zero or one NEU pair. Buffers, so zero
    # trainable parameters, and they travel in the state dict - which is how the
    # fusion arm carries auxiliary-domain sentiment geometry into the target.
    use_polarity_contrast: bool = False
    w_pol_contrast: float = 0.3
    pol_contrast_temp: float = 0.1
    pol_contrast_momentum: float = 0.95
    # Boundary errors are 18-32% of misses and 27-32% of false positives, and
    # ~1000 independent per-span sigmoids cannot express the competition between
    # `battery`, `battery life` and `life`. One softmax per start position over
    # the end positions, plus a "no span starts here" slot. No parameters.
    use_boundary_listwise: bool = False
    w_boundary: float = 0.5

    # J-series: explicit type-conditioned pair modelling.
    #
    # Chosen over another polarity or graph variant because of what the H runs
    # measured: polarity accuracy *given both spans are correct* is already
    # 86-93% and H0b moved it by at most 0.4 points, so P(sentiment | a, o) is
    # not the bottleneck. EPMEI's own ablation says the same from the other side
    # - removing its GCN costs 0.02 F1 on 14res while removing typed markers
    # costs 8.8 - and the snap-boundary oracle here says 6.3-7.7 F1 sits in span
    # localisation and pair formation.
    #
    # Markers are not the route: two implementations of them were tried and
    # closed, and both failed for the same reason,
    # that a marker is a token slot with no pretraining trained on 605-1265
    # sentences. Type *embeddings* added to the already-pretrained span
    # representations avoid that: nothing new has to be learned in the encoder's
    # input space.
    #
    # Worth being precise about what this can and cannot do. The pair MLP
    # consumes [z_a ; z_o ; ...] with separate weight blocks per slot, so it
    # already knows which argument is the aspect; a constant added to z_a is
    # absorbed by the first layer's bias and changes nothing in those two blocks.
    # The effect is confined to the interaction terms, where it is real:
    #   (z_a + e_A) * (z_o + e_O) = z_a*z_o + z_a*e_O + e_A*z_o + e_A*e_O
    # and |z_a - z_o + (e_A - e_O)| shifts the elbow of the absolute value. So
    # this arm tests interaction conditioning, not "telling the model the type".
    use_pair_type_embed: bool = False
    # |z_a - z_o| as a fourth feature block. Widens the pair MLP input from 3x to
    # 4x span_dim, which is the only capacity either flag adds.
    use_pair_diff: bool = False

    # R-series: is the *relation scorer* the bottleneck, or the representation?
    # Every objective reweighting tried so far (H, K, L, M) trades precision for
    # a small recall gain and lands below B2h, while the existence oracle says
    # +6.04 F1 is sitting there. 13.1% of gold relations score below -2 on the
    # existence log-odds, i.e. the model is confidently wrong, not miscalibrated.
    # That is either a weak scorer over adequate features, or an inadequate pair
    # representation. R1 changes only the scorer, so it separates the two.
    #
    # pair_width appends both span widths; combined with use_distance_feature and
    # use_pair_diff this gives [h_a; h_o; h_a*h_o; |h_a-h_o|; d(a,o); w_a; w_o].
    use_pair_width: bool = False
    # Two hidden layers instead of one in the pair head.
    pair_deep: bool = False
    # s(a,o) = h_a^T W h_o + U[h_a; h_o] + b, the standard biaffine relation
    # scorer, replacing the concatenation MLP entirely.
    use_biaffine_pair: bool = False

    # S1: a keep-or-reject verifier over the pairs the model already wants to
    # emit, trained on shared-span hard negatives. Targets fp_one_span_real,
    # the largest false-positive category and the one R1bm churns rather than
    # resolves (512 removed, 459 added). See `aste_verifier.py`.
    # R2q: a pair query attends back over the sentence before the relation is
    # scored. Distinct hypothesis from S1 - not "reject better" but "represent
    # the pair better before deciding". Appended to R1bm's feature set, output
    # projection zero-initialised so step 0 is R1bm exactly.
    use_pair_context: bool = False
    pair_context_dim: int = 128

    use_verifier: bool = False
    verifier_dim: int = 64
    verifier_context: bool = True
    verifier_easy_weight: float = 0.1
    verifier_threshold: float = 0.5

    # K-series: overlap-aware exact-span ranking. See `aste_ranking.py`.
    #
    # The nested error budget on B2h says where the recoverable F1 is:
    # polarity +7.31, linking +1.67, boundary +6.92, unreachable 14.86. Polarity
    # was tried and does not convert (NEU precision 0.12-0.51 against a 0.50
    # break-even); linking is capped at +1.67 and link accuracy given the spans
    # is already 97.6-99.1%. Boundary is what is left, and it is worth +6.3 to
    # +7.7 on every domain.
    #
    # H0b already failed at it, which is informative rather than discouraging:
    # it asked whether the gold boundary looks plausible in absolute terms. The
    # oracle says the model usually already lands on an *overlapping* span and
    # picks the wrong member of the local set, and 96.9% of those errors move
    # exactly one edge. So the objective is a ranking one - the gold span must
    # outscore its overlapping rivals by a margin - not another plausibility one.
    use_overlap_ranking: bool = False
    w_overlap_rank: float = 0.5
    overlap_margin: float = 1.0
    # Restrict the rival set to spans sharing one edge with the gold span. Those
    # carry 96.9% of the observed errors and the set is an order of magnitude
    # smaller, so this is a hypothesis about where the signal is, not a speedup.
    overlap_single_edge: bool = False

    w_aspect: float = 1.0
    w_opinion: float = 1.0
    w_pair: float = 1.0

    @classmethod
    def preset(cls, name: str, **overrides) -> "AsteConfig":
        presets = {
            # Span-enumeration baseline without any aspect-conditioned interaction.
            "A0": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False),
            "A1": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=True),
            "A2": dict(use_acsi=False, use_gated_pooling=True, use_distance_feature=True),
            "A3": dict(use_acsi=True, use_gated_pooling=False, use_distance_feature=False),
            "A4": dict(use_acsi=True, use_gated_pooling=True, use_distance_feature=True),

            # ABSA-adapter ablation. A3 (ACSI, aspect-only) is the reference the
            # new components must beat, since it is the best configuration so far.
            "B1": dict(use_acsi=True, use_gated_pooling=False, use_distance_feature=False,
                       use_role_symmetric=True),
            "B2": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                       use_assignment=True),
            "B3": dict(use_acsi=True, use_gated_pooling=False, use_distance_feature=False,
                       use_role_symmetric=True, use_assignment=True),

            # Follow-ups to B2, the configuration that generalised across all four
            # domains. Kept deliberately small: one hyperparameter change, one
            # linear layer.
            "B2t": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                        use_assignment=True, n_aspect_keep=20, n_opinion_keep=20),
            "C1": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                       use_assignment=True, use_token_tagger=True),
            "C2": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                       use_assignment=True, use_token_tagger=True,
                       n_aspect_keep=20, n_opinion_keep=20),

            # Architectural change on top of B2: selection derived from matching,
            # and a single existence decision. Fewer mechanisms, not more.
            "E1": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                       use_assignment=True, use_coupled_selection=True),
            # E1b: pruning supervised, role scores left to emerge from matching.
            "E1b": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                        use_assignment=True, use_coupled_selection=True, w_role=0.0),
            # E1c: role scores still supervised, but aggregated with max so the
            # gradient reaches one partner instead of the whole row.
            "E1c": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                        use_assignment=True, use_coupled_selection=True,
                        role_aggregation="max"),

            # Span extraction is the bottleneck: pairing is already at 94.8% while
            # aspect/opinion F1 sit at 0.863/0.852, and triplet F1 is roughly their
            # product. B2w widens the span representation, which currently squeezes
            # 2x768 hidden dims into 256.
            "B2w": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                        use_assignment=True, span_dim=512, deep_span_encoder=True),
            # B2 plus the schema-grounded span signal.
            "B2s": dict(use_acsi=False, use_gated_pooling=False, use_distance_feature=False,
                        use_assignment=True, use_schema_score=True),
            # B2h isolates the structured negatives: B2 with nothing else changed,
            # so it is a single-variable comparison against the existing B2 runs.
            "B2h": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0),
            # P2: the same emphasis, but redistributed rather than added, which is
            # what the large encoder needs.
            # D1: dustbin abstention gate on top of the T1 stability fix.
            "B2d": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        sinkhorn_temp_start=4.0, sinkhorn_temp_end=1.0,
                        use_dustbin_gate=True),
            "B2p2": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         sinkhorn_temp_start=4.0, sinkhorn_temp_end=1.0,
                         hard_none_weight=1.0, neutral_negative_mass=True),
            # P1: structured negative weighting on top of the T1 stability fix.
            "B2p1": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         sinkhorn_temp_start=4.0, sinkhorn_temp_end=1.0,
                         hard_none_weight=1.0),
            # T1: annealed assignment temperature, aimed at the collapse modes.
            # D1 domain-fusion arms. The stage schedule and which corpus each
            # stage sees live in aste_full_train.FUSION_ARMS; these entries only
            # describe the model, so F0/F1/F3 share one config.
            "F": dict(use_acsi=False, use_gated_pooling=False,
                      use_distance_feature=False, use_assignment=True,
                      hard_none_weight=1.0, use_adapter=True, freeze_backbone=True),
            # F2 is the control that decides whether the adapter is doing the
            # work: same auxiliary data, same two stages, plain full fine-tuning.
            "F2": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, use_adapter=False, freeze_backbone=False),
            # B2hs: B2h plus the frozen GLiNER2 span prior, nothing else. The
            # gate starts at zero, so step 0 is bit-identical to B2h and any gain
            # has to be earned by the prior during training rather than handed to
            # it by initialisation. No extra training data of any kind, so the
            # result stays comparable with the published per-domain systems.
            "B2hs": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         hard_none_weight=1.0, use_schema_score=True),
            # B2hr: the prior decides 3 of the 10 candidate slots per role and
            # touches nothing else - no bias, no gate, zero added parameters.
            "B2hr": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         hard_none_weight=1.0, use_schema_score=True,
                         schema_bias=False, schema_reserve=3),
            # B2hb: the soft route again, but the gate gets its own learning rate
            # (`gate_lr` in run_one) so that a null result means the model chose
            # not to use the prior rather than that it could not reach a value
            # where the prior does anything.
            "B2hb": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         hard_none_weight=1.0, use_schema_score=True),
            # B2hl: B2h plus the listwise row/column objective, one variable.
            "B2hl": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         hard_none_weight=1.0, use_listwise=True, w_listwise=0.5),
            "B2hl1": dict(use_acsi=False, use_gated_pooling=False,
                          use_distance_feature=False, use_assignment=True,
                          hard_none_weight=1.0, use_listwise=True, w_listwise=1.0),
            "B2t1": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         sinkhorn_temp_start=4.0, sinkhorn_temp_end=1.0),

            # H-series. Each arm is B2h plus exactly one more thing, and none of
            # them adds a trainable parameter, so the ladder is measurable all
            # the way up and nothing in it can win by capacity.
            # H1: the factorisation alone. Same function class, same parameter
            #     count, same existence objective - only the polarity decision
            #     moves out of the 4-way loss.
            "H1": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, factor_polarity=True),
            # H2: + class weights inside the 3-way head. 15res has 25 NEU
            #     triplets in training against 783 POS.
            "H2": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, factor_polarity=True,
                       polarity_class_weight="sqrt"),
            # H2b: the same, at full inverse frequency. Kept separate because
            #     "sqrt" and "balanced" differ by a factor of 5 on 15res and
            #     guessing between them from one run is how the earlier rounds
            #     produced rankings that inverted at convergence.
            "H2b": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, factor_polarity=True,
                        polarity_class_weight="balanced"),
            # H3: + sentiment prototypes. The contrastive lever, zero parameters.
            "H3": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, factor_polarity=True,
                       polarity_class_weight="sqrt", use_polarity_contrast=True),
            # H4: + the boundary-listwise span objective, the second bucket.
            "H4": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, factor_polarity=True,
                       polarity_class_weight="sqrt", use_polarity_contrast=True,
                       use_boundary_listwise=True),
            # H0b: the boundary objective on plain B2h, with polarity untouched.
            #     Without it a gain in H4 cannot be attributed to either half.
            "H0b": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, use_boundary_listwise=True),
            # R2q: R1bm plus pair-query context. One extra attention over the
            # already-encoded token states, so ~1x the encoder cost rather than
            # the ~6x a marker re-encoding would need.
            "R2q": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=True, use_assignment=True,
                        hard_none_weight=1.0, use_biaffine_pair=True,
                        use_pair_diff=True, use_pair_width=True,
                        use_pair_context=True),
            # S1: R1bm plus the shared-span pair verifier. Identical generator,
            # one extra keep-or-reject stage trained on a frozen base.
            "S1": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=True, use_assignment=True,
                       hard_none_weight=1.0, use_biaffine_pair=True,
                       use_pair_diff=True, use_pair_width=True, use_verifier=True),
            # R-series: is the relation scorer the bottleneck? Encoder, data,
            # pruning, threshold and the main loss are all identical to B2h, so
            # only the pair scorer differs.
            "R1m": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=True, use_assignment=True,
                        hard_none_weight=1.0, use_pair_diff=True,
                        use_pair_width=True, pair_deep=True),
            # Follow-ups to R1b, the first arm to beat B2h on dev (+0.57 via
            # recall with precision flat). R1bm adds R1m's features to the
            # biaffine; R1bf gives it separate existence and polarity heads,
            # which is where the oracle says the remaining +6.04 and +3.35 are.
            "R1bm": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=True, use_assignment=True,
                         hard_none_weight=1.0, use_biaffine_pair=True,
                         use_pair_diff=True, use_pair_width=True),
            "R1bf": dict(use_acsi=False, use_gated_pooling=False,
                         use_distance_feature=False, use_assignment=True,
                         hard_none_weight=1.0, use_biaffine_pair=True,
                         factor_polarity=True),
            "R1b": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, use_biaffine_pair=True),
            # M-series: gold > NULL > negative. M1 all cells, M2 hardest rival
            # per row/column, M3 the same on the factorised existence/polarity
            # head (H1), M4 adds the degree regulariser.
            "M1": dict(**{k: v for k, v in dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0).items()}, use_null_rank=True),
            "M2": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, use_null_rank=True, null_hard=True),
            "M3": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, factor_polarity=True,
                       use_null_rank=True, null_hard=True),
            "M4": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, factor_polarity=True,
                       use_null_rank=True, null_hard=True, use_degree_loss=True),
            # L-series: relation existence as a candidate-set decision. L1 is the
            # multi-positive listwise objective on top of B2h, targeting the
            # 16.94% of gold triplets whose spans both survive pruning and whose
            # pair is still rejected as NONE. L1a drops the column term to test
            # whether the aspect->opinion direction carries it on its own.
            "L1": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, use_multi_positive=True),
            "L1a": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, use_multi_positive=True,
                        multi_positive_symmetric=False),
            "L1w": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, use_multi_positive=True,
                        w_multi_positive=1.0),
            # K-series: overlap-aware exact-span ranking. K1 competes the gold
            # span against every overlapping rival, K1e only against rivals that
            # share one edge - which is where 96.9% of the measured errors are.
            "K1": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, use_overlap_ranking=True),
            "K1e": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, use_overlap_ranking=True,
                        overlap_single_edge=True),
            # J-series: type-conditioned pair modelling, the direction the H runs
            # pointed at. J1 is B2h plus the two pair-side changes; J2 adds the
            # boundary objective on top, and is the candidate main model. Kept to
            # exactly two arms so a gain is attributable to one of them.
            "J1": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, use_pair_type_embed=True,
                       use_pair_diff=True),
            "J2": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, use_pair_type_embed=True,
                       use_pair_diff=True, use_boundary_listwise=True),
            # J1d / J1t split J1 so that, if it moves, the two halves separate.
            "J1d": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, use_pair_diff=True),
            "J1t": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, use_pair_type_embed=True),
            # FH / FH2: the H3 model inside the D1 two-stage fusion schedule.
            #     This is the arm the NEU counts argue for - 15res has 25 NEU
            #     triplets of its own and 343 in the clean auxiliary pool - and
            #     the prototypes are buffers, so stage 1's sentiment geometry is
            #     loaded into stage 2 with the rest of the state dict.
            "FH": dict(use_acsi=False, use_gated_pooling=False,
                       use_distance_feature=False, use_assignment=True,
                       hard_none_weight=1.0, factor_polarity=True,
                       polarity_class_weight="sqrt", use_polarity_contrast=True,
                       use_adapter=True, freeze_backbone=True),
            # FH2: the control that separates the adapter from the extra data,
            #     same as F2 does for B2h.
            "FH2": dict(use_acsi=False, use_gated_pooling=False,
                        use_distance_feature=False, use_assignment=True,
                        hard_none_weight=1.0, factor_polarity=True,
                        polarity_class_weight="sqrt", use_polarity_contrast=True,
                        use_adapter=False, freeze_backbone=False),
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; have {sorted(presets)}")
        return cls(**{**presets[name], **overrides})


def enumerate_spans(seq_len: int, max_width: int, device) -> torch.Tensor:
    """All [start, end) token spans up to ``max_width``, as (N, 2)."""
    starts, ends = [], []
    for start in range(seq_len):
        for width in range(1, min(max_width, seq_len - start) + 1):
            starts.append(start)
            ends.append(start + width)
    return torch.tensor([starts, ends], device=device).t()


class SpanEncoder(nn.Module):
    """Span representation: boundaries, optional gated pooling, and width."""

    def __init__(self, cfg: AsteConfig):
        super().__init__()
        d = cfg.hidden_size
        self.cfg = cfg
        self.width_embed = nn.Embedding(cfg.max_span_width + 1, 32)
        self.attn = nn.Linear(d, 1)
        self.gate = nn.Parameter(torch.zeros(3))
        if cfg.deep_span_encoder:
            self.project = nn.Sequential(
                nn.Linear(2 * d + 32, cfg.span_dim), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.span_dim, cfg.span_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            )
        else:
            self.project = nn.Sequential(
                nn.Linear(2 * d + 32, cfg.span_dim), nn.GELU(), nn.Dropout(cfg.dropout)
            )

    def forward(self, hidden: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        """hidden (B,T,H), spans (N,2) shared across the batch -> (B,N,span_dim)"""
        b, t, h = hidden.shape
        starts, ends = spans[:, 0], spans[:, 1] - 1
        h_start = hidden[:, starts]                       # (B,N,H)
        h_end = hidden[:, ends]

        if self.cfg.use_gated_pooling:
            positions = torch.arange(t, device=hidden.device).view(1, t)
            inside = (positions >= spans[:, 0:1]) & (positions < spans[:, 1:2])  # (N,T)
            scores = self.attn(hidden).squeeze(-1)                                # (B,T)
            masked = scores.unsqueeze(1).masked_fill(
                ~inside.unsqueeze(0), torch.finfo(scores.dtype).min
            )
            weights = masked.softmax(-1)
            h_attn = torch.einsum("bnt,bth->bnh", weights, hidden)
            g = self.gate.softmax(0)
            pooled = g[0] * h_start + g[1] * h_end + g[2] * h_attn
        else:
            pooled = h_start
        widths = (spans[:, 1] - spans[:, 0]).clamp(max=self.cfg.max_span_width)
        w = self.width_embed(widths).unsqueeze(0).expand(b, -1, -1)
        return self.project(torch.cat([pooled, h_end, w], dim=-1))


class AcsiInteraction(nn.Module):
    """Sparse top-k attention from each kept aspect span to token context."""

    def __init__(self, cfg: AsteConfig):
        super().__init__()
        self.cfg = cfg
        g, heads = cfg.acsi_dim, cfg.acsi_heads
        self.heads, self.head_dim = heads, g // heads
        self.q = nn.Linear(cfg.span_dim, g)
        self.k = nn.Linear(cfg.hidden_size, g)
        self.v = nn.Linear(cfg.hidden_size, g)
        self.o = nn.Linear(g, cfg.span_dim)
        self.norm = nn.LayerNorm(cfg.span_dim)
        self.rel = nn.Embedding(2 * cfg.max_rel_distance + 3, heads)
        nn.init.zeros_(self.rel.weight)
        # Two roles share every projection; only this embedding separates them,
        # so covering opinions as well costs 2 x span_dim parameters, not a copy.
        self.role = nn.Embedding(2, cfg.span_dim)
        nn.init.zeros_(self.role.weight)

    def forward(self, anchors, anchor_pos, hidden, mask, role: int = 0):
        b, n, _ = anchors.shape
        t = hidden.shape[1]
        anchors = anchors + self.role.weight[role].view(1, 1, -1)
        q = self.q(anchors).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(hidden).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(hidden).view(b, t, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhnd,bhtd->bhnt", q, k) / (self.head_dim ** 0.5)
        positions = torch.arange(t, device=hidden.device).view(1, 1, t)
        delta = (positions - anchor_pos.unsqueeze(-1)).clamp(
            -self.cfg.max_rel_distance, self.cfg.max_rel_distance
        ) + self.cfg.max_rel_distance + 1
        scores = scores + self.rel(delta).permute(0, 3, 1, 2)

        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.view(b, 1, 1, t), neg)
        keep = min(self.cfg.acsi_top_k, t)
        if keep < t:
            top = scores.topk(keep, dim=-1).indices
            gate = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top, True)
            scores = scores.masked_fill(~gate, neg)
        attn = torch.nan_to_num(scores.softmax(-1), nan=0.0)
        ctx = torch.einsum("bhnt,bhtd->bhnd", attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.norm(anchors + self.o(ctx))


def log_sinkhorn(scores: torch.Tensor, dustbin: torch.Tensor, n_iter: int) -> torch.Tensor:
    """Differentiable near-bijective matching with slack (SuperGlue formulation).

    A plain Sinkhorn would force every aspect to take an opinion and vice versa,
    which is wrong here: candidate sets are pruned to 10x10 while a sentence holds
    only one or two real pairs, and 15% of aspects legitimately bind more than one
    opinion. The dustbin row and column absorb everything unmatched, so the
    constraint is "at most one, softly" rather than "exactly one".
    """
    b, na, no = scores.shape
    bins_a = dustbin.expand(b, na, 1)
    bins_o = dustbin.expand(b, 1, no)
    corner = dustbin.expand(b, 1, 1)
    top = torch.cat([scores, bins_a], dim=-1)
    bottom = torch.cat([bins_o, corner], dim=-1)
    couplings = torch.cat([top, bottom], dim=1)                     # (B, na+1, no+1)

    norm = -torch.log(torch.tensor(na + no, dtype=scores.dtype, device=scores.device))
    log_mu = torch.cat([norm.expand(na),
                        (norm + torch.log(torch.tensor(float(no), device=scores.device)))
                        .reshape(1)])
    log_nu = torch.cat([norm.expand(no),
                        (norm + torch.log(torch.tensor(float(na), device=scores.device)))
                        .reshape(1)])
    log_mu = log_mu.expand(b, -1)
    log_nu = log_nu.expand(b, -1)

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(n_iter):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)
    return couplings + u.unsqueeze(2) + v.unsqueeze(1)


class AsteModel(nn.Module):
    def __init__(self, encoder, cfg: AsteConfig):
        super().__init__()
        # Every optional submodule built below draws from the global RNG, and the
        # DataLoader's shuffle permutation is drawn from that same stream - so an
        # arm with one extra module silently trains on a different data order
        # from the arm it is compared against. That is what made S1's "frozen
        # R1bm base" a different model (n_pred 255 vs 296 on the same seed).
        # Restoring the state at the end leaves parameter values untouched while
        # making the post-init stream identical across arms.
        self._rng_state = torch.get_rng_state()
        self.encoder = encoder
        self.cfg = cfg
        self.adapter_report = None
        if cfg.use_adapter:
            from aste_adapter import inject_adapters
            self.adapter_report = inject_adapters(
                encoder,
                [("shared", cfg.adapter_shared_dim), ("domain", cfg.adapter_domain_dim)],
                cfg.adapter_dropout,
            )
        s = cfg.span_dim
        self.span_encoder = SpanEncoder(cfg)
        self.aspect_scorer = nn.Sequential(
            nn.Linear(s, s), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(s, 1)
        )
        self.opinion_scorer = nn.Sequential(
            nn.Linear(s, s), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(s, 1)
        )
        self.acsi = AcsiInteraction(cfg) if cfg.use_acsi else None
        if cfg.use_assignment and cfg.use_dustbin_gate:
            self.dustbin_gate = nn.Parameter(torch.zeros(()))
        if cfg.use_assignment:
            self.match = nn.Bilinear(s, s, 1)
            # The only parameter the assignment constraint introduces.
            self.dustbin = nn.Parameter(torch.zeros(1))
        if cfg.use_coupled_selection:
            # One cheap head prunes the span set so the full match matrix stays
            # tractable; after that, matching is the only scoring mechanism.
            self.prune_scorer = nn.Sequential(
                nn.Linear(s, s), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(s, 1)
            )
            self.match_a = nn.Linear(s, s)
            self.match_o = nn.Linear(s, s)
            # Polarity only - existence is the dustbin's job (E2).
            self.polarity_head = nn.Sequential(
                nn.Linear(3 * s, cfg.pair_dim), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.pair_dim, 3),
            )
        if cfg.use_schema_score:
            # One gate per role, initialised to zero.
            self.schema_gate = nn.Parameter(torch.zeros(2))
        if cfg.use_token_tagger:
            # Two independent binary decisions per token rather than a joint BIO
            # scheme: an aspect and an opinion never share a token in ASTE, and
            # independence keeps the loss trivial.
            self.token_tagger = nn.Linear(cfg.hidden_size, 2)
        self.dist_embed = nn.Embedding(2 * cfg.max_rel_distance + 3, 32)
        # The verifier is inert during stage 1. Its dropout draws from the global
        # RNG, so computing it while the base trains desynchronises the base's own
        # dropout stream and the "frozen R1bm base" stops being the R1bm run it is
        # compared against - measured: pair_logits diverge from the second forward
        # pass onward. Gating it off during stage 1 keeps the base bit-identical.
        self.verifier_active = False
        if cfg.use_pair_type_embed:
            # Zero-initialised, so step 0 is bit-identical to the arm this is
            # built on and the learned magnitude is itself the measurement of
            # how much type conditioning is worth.
            self.pair_role = nn.Embedding(2, s)
            nn.init.zeros_(self.pair_role.weight)
        pair_in = (4 if cfg.use_pair_diff else 3) * s
        pair_in += 32 if cfg.use_distance_feature else 0
        if cfg.use_pair_width:
            self.pair_width_embed = nn.Embedding(cfg.max_span_width + 1, 32)
            pair_in += 64
        if cfg.use_pair_context:
            from aste_context import PairContext
            self.pair_ctx = PairContext(pair_in, cfg.hidden_size,
                                        cfg.pair_context_dim, cfg.dropout)
            pair_in = pair_in + cfg.pair_context_dim
        if cfg.use_biaffine_pair:
            n_out = 1 + 3 if cfg.factor_polarity else 4
            self.biaffine = nn.Bilinear(s, s, n_out)
            self.biaffine_linear = nn.Linear(2 * s, n_out)
            nn.init.zeros_(self.biaffine.weight)
            nn.init.zeros_(self.biaffine.bias)
        if cfg.factor_polarity:
            # Same layers as `pair_classifier`, cut in two after the trunk:
            # pair_dim -> 1 and pair_dim -> 3 is exactly as many parameters as
            # pair_dim -> 4, so H1 is parameter-identical to B2h by construction
            # and any difference it makes is the objective, not the capacity.
            trunk = [nn.Linear(pair_in, cfg.pair_dim), nn.GELU(), nn.Dropout(cfg.dropout)]
            if cfg.pair_deep:
                trunk += [nn.Linear(cfg.pair_dim, cfg.pair_dim), nn.GELU(),
                          nn.Dropout(cfg.dropout)]
            self.pair_trunk = nn.Sequential(*trunk)
            self.exist_scorer = nn.Linear(cfg.pair_dim, 1)
            self.polarity_scorer = nn.Linear(cfg.pair_dim, 3)
            self.register_buffer("polarity_weight", torch.ones(3))
            if cfg.use_polarity_contrast:
                from aste_polarity import SentimentPrototypes
                self.sentiment_protos = SentimentPrototypes(
                    cfg.pair_dim, 3, cfg.pol_contrast_momentum, cfg.pol_contrast_temp
                )
        else:
            layers = [nn.Linear(pair_in, cfg.pair_dim), nn.GELU(), nn.Dropout(cfg.dropout)]
            if cfg.pair_deep:
                layers += [nn.Linear(cfg.pair_dim, cfg.pair_dim), nn.GELU(),
                           nn.Dropout(cfg.dropout)]
            layers.append(nn.Linear(cfg.pair_dim, 4))
            self.pair_classifier = nn.Sequential(*layers)
        if cfg.use_verifier:
            self.build_verifier(pair_in)
        torch.set_rng_state(self._rng_state)
        del self._rng_state

    def build_verifier(self, pair_in: int) -> None:
        from aste_verifier import PairVerifier
        self.verifier = PairVerifier(pair_in, self.cfg.hidden_size,
                                     self.cfg.verifier_dim, self.cfg.verifier_context,
                                     self.cfg.dropout)

    def set_polarity_weights(self, counts) -> None:
        """Fill the 3-way class weights from the training split's own counts.

        Called by the trainer once the data is loaded, rather than stored in the
        config, so a run records the frequencies it actually saw. A no-op unless
        ``factor_polarity`` and ``polarity_class_weight`` are both set.
        """
        if not self.cfg.factor_polarity:
            return
        from aste_polarity import class_weights
        w = class_weights(counts, self.cfg.polarity_class_weight,
                          device=self.polarity_weight.device)
        if w is not None:
            self.polarity_weight.copy_(w)

    def set_sinkhorn_temp(self, progress: float) -> float:
        """Cosine anneal from temp_start to temp_end; progress in [0,1]."""
        a, b = self.cfg.sinkhorn_temp_start, self.cfg.sinkhorn_temp_end
        import math
        t = b + 0.5 * (a - b) * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        self._sinkhorn_temp = t
        return t

    def added_parameter_count(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        adapters = 0
        if self.cfg.use_adapter:
            from aste_adapter import backbone_parameters
            backbone = sum(p.numel() for p in backbone_parameters(self.encoder))
            adapters = enc - backbone
            enc = backbone
        head = total - enc - adapters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"encoder": enc, "adapters": adapters, "head": head,
                "added": total - enc, "total": total, "trainable": trainable,
                "added_pct_of_encoder": round(100 * (total - enc) / enc, 3)}

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def forward_coupled(self, input_ids, attention_mask, spans, span_valid,
                        gold_aspects=None, gold_opinions=None):
        """E1+E2: prune, match, derive role scores from the match, assign, label."""
        hidden = self.encode(input_ids, attention_mask)
        z = self.span_encoder(hidden, spans)
        neg = torch.finfo(z.dtype).min

        prune = self.prune_scorer(z).squeeze(-1).masked_fill(~span_valid, neg)
        if gold_aspects is not None:
            prune = prune + (gold_aspects | gold_opinions).float() * 1e4
        n_p = min(self.cfg.n_prune, z.shape[1])
        p_idx = prune.topk(n_p, dim=-1).indices                      # (B,P)
        zp = torch.gather(z, 1, p_idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]))

        # The single scoring mechanism: every pruned span against every other.
        match = torch.einsum("bad,bod->bao", self.match_a(zp), self.match_o(zp))
        match = match / (zp.shape[-1] ** 0.5)                        # (B,P,P)

        # E1: a span's role score is what the match matrix says about it.
        if self.cfg.role_aggregation == "max":
            a_score = match.max(dim=2).values
            o_score = match.max(dim=1).values
        else:
            a_score = torch.logsumexp(match, dim=2)                  # (B,P)
            o_score = torch.logsumexp(match, dim=1)

        gold_a_p = gold_o_p = None
        if gold_aspects is not None:
            gold_a_p = torch.gather(gold_aspects, 1, p_idx)
            gold_o_p = torch.gather(gold_opinions, 1, p_idx)
            a_rank = a_score + gold_a_p.float() * 1e4
            o_rank = o_score + gold_o_p.float() * 1e4
        else:
            a_rank, o_rank = a_score, o_score

        na = min(self.cfg.n_aspect_keep, n_p)
        no = min(self.cfg.n_opinion_keep, n_p)
        a_sel = a_rank.topk(na, dim=-1).indices                      # indices into P
        o_sel = o_rank.topk(no, dim=-1).indices

        sub = torch.gather(match, 1, a_sel.unsqueeze(-1).expand(-1, -1, n_p))
        sub = torch.gather(sub, 2, o_sel.unsqueeze(1).expand(-1, na, -1))   # (B,na,no)
        log_p = log_sinkhorn(sub, self.dustbin, self.cfg.sinkhorn_iters)

        z_a = torch.gather(zp, 1, a_sel.unsqueeze(-1).expand(-1, -1, zp.shape[-1]))
        z_o = torch.gather(zp, 1, o_sel.unsqueeze(-1).expand(-1, -1, zp.shape[-1]))
        ea = z_a.unsqueeze(2).expand(-1, -1, no, -1)
        eo = z_o.unsqueeze(1).expand(-1, na, -1, -1)
        polarity = self.polarity_head(torch.cat([ea, eo, ea * eo], dim=-1))  # (B,na,no,3)

        # Map back to original span indices so decoding is unchanged.
        a_idx = torch.gather(p_idx, 1, a_sel)
        o_idx = torch.gather(p_idx, 1, o_sel)

        return {
            "coupled": True,
            "prune_logits": prune, "p_idx": p_idx,
            "gold_full_a": gold_aspects, "gold_full_b": gold_opinions,
            "a_score": a_score, "o_score": o_score,
            "gold_a_pruned": gold_a_p, "gold_o_pruned": gold_o_p,
            "a_sel": a_sel, "o_sel": o_sel,
            "aspect_idx": a_idx, "opinion_idx": o_idx,
            "log_assign": log_p,                 # (B,na+1,no+1) with dustbins
            "polarity_logits": polarity,
        }

    def forward(self, input_ids, attention_mask, spans, span_valid,
                gold_aspects=None, gold_opinions=None, schema_scores=None):
        if self.cfg.use_coupled_selection:
            return self.forward_coupled(input_ids, attention_mask, spans, span_valid,
                                        gold_aspects, gold_opinions)
        """
        spans:      (N,2) enumerated once for the padded length
        span_valid: (B,N) spans lying inside the real (unpadded) sequence
        gold_*:     (B,N) bool, injected into the kept set during training so the
                    pair classifier always sees the positives it must learn.
        """
        hidden = self.encode(input_ids, attention_mask)
        z = self.span_encoder(hidden, spans)                       # (B,N,S)

        neg = torch.finfo(z.dtype).min
        a_logit = self.aspect_scorer(z).squeeze(-1)
        o_logit = self.opinion_scorer(z).squeeze(-1)
        if self.cfg.use_schema_score and self.cfg.schema_bias and schema_scores is not None:
            # schema_scores: (B, N, 2) aligned with the enumerated spans.
            a_logit = a_logit + self.schema_gate[0] * schema_scores[..., 0]
            o_logit = o_logit + self.schema_gate[1] * schema_scores[..., 1]
        a_logit = a_logit.masked_fill(~span_valid, neg)
        o_logit = o_logit.masked_fill(~span_valid, neg)

        a_rank = a_logit + (gold_aspects.float() * 1e4 if gold_aspects is not None else 0)
        o_rank = o_logit + (gold_opinions.float() * 1e4 if gold_opinions is not None else 0)
        n_a = min(self.cfg.n_aspect_keep, z.shape[1])
        n_o = min(self.cfg.n_opinion_keep, z.shape[1])
        reserve = self.cfg.schema_reserve if schema_scores is not None else 0
        if reserve > 0:
            # The last `reserve` candidate slots are filled by the frozen prior
            # rather than by the learned scorer, from the spans the learned
            # scorer did not already take. 10-15% of gold spans never reach the
            # candidate set, which is a ceiling no downstream layer can lift, and
            # the prior's own recall@10 on spans whose surface string is absent
            # from training runs 84.3-96.5% against the trained model's 58-73%.
            # Gold spans keep their +1e4 in `a_rank`, so during training they are
            # inside the learned part and the reserve never displaces a positive.
            n_learn_a, n_learn_o = max(1, n_a - reserve), max(1, n_o - reserve)
            a_idx = a_rank.topk(n_learn_a, dim=-1).indices
            o_idx = o_rank.topk(n_learn_o, dim=-1).indices
            a_prior = schema_scores[..., 0].masked_fill(~span_valid, -1.0)
            o_prior = schema_scores[..., 1].masked_fill(~span_valid, -1.0)
            a_prior = a_prior.scatter(1, a_idx, -1.0)     # do not re-pick
            o_prior = o_prior.scatter(1, o_idx, -1.0)
            a_idx = torch.cat([a_idx, a_prior.topk(n_a - n_learn_a, dim=-1).indices], 1)
            o_idx = torch.cat([o_idx, o_prior.topk(n_o - n_learn_o, dim=-1).indices], 1)
        else:
            a_idx = a_rank.topk(n_a, dim=-1).indices                   # (B,Na)
            o_idx = o_rank.topk(n_o, dim=-1).indices

        z_a = torch.gather(z, 1, a_idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]))
        z_o = torch.gather(z, 1, o_idx.unsqueeze(-1).expand(-1, -1, z.shape[-1]))

        pos_all = spans[:, 0].unsqueeze(0).expand(z.shape[0], -1)
        if self.acsi is not None:
            pos_a = torch.gather(pos_all, 1, a_idx)
            z_a = self.acsi(z_a, pos_a, hidden, attention_mask.bool(), role=0)
            if self.cfg.use_role_symmetric:
                pos_o = torch.gather(pos_all, 1, o_idx)
                z_o = self.acsi(z_o, pos_o, hidden, attention_mask.bool(), role=1)

        na, no = z_a.shape[1], z_o.shape[1]
        if self.cfg.use_pair_type_embed:
            z_a = z_a + self.pair_role.weight[0].view(1, 1, -1)
            z_o = z_o + self.pair_role.weight[1].view(1, 1, -1)
        ea = z_a.unsqueeze(2).expand(-1, -1, no, -1)
        eo = z_o.unsqueeze(1).expand(-1, na, -1, -1)
        feats = [ea, eo, ea * eo]
        if self.cfg.use_pair_diff:
            feats.append((ea - eo).abs())
        if self.cfg.use_pair_width:
            widths = (spans[:, 1] - spans[:, 0]).clamp(max=self.cfg.max_span_width)
            wa = self.pair_width_embed(torch.gather(
                widths.unsqueeze(0).expand(z.shape[0], -1), 1, a_idx))
            wo = self.pair_width_embed(torch.gather(
                widths.unsqueeze(0).expand(z.shape[0], -1), 1, o_idx))
            feats.append(wa.unsqueeze(2).expand(-1, -1, no, -1))
            feats.append(wo.unsqueeze(1).expand(-1, na, -1, -1))
        if self.cfg.use_distance_feature:
            pos = spans[:, 0].unsqueeze(0).expand(z.shape[0], -1)
            pa = torch.gather(pos, 1, a_idx).unsqueeze(2)
            po = torch.gather(pos, 1, o_idx).unsqueeze(1)
            delta = (po - pa).clamp(
                -self.cfg.max_rel_distance, self.cfg.max_rel_distance
            ) + self.cfg.max_rel_distance + 1
            feats.append(self.dist_embed(delta))
        pair_feats_cat = torch.cat(feats, dim=-1)
        if self.cfg.use_pair_context:
            # q_ao -> attend over H -> c_ao, appended to the pair features.
            ctx = self.pair_ctx(pair_feats_cat, hidden, attention_mask.bool())
            pair_feats_cat = torch.cat([pair_feats_cat, ctx], dim=-1)
        pair_hidden = exist = polarity = None
        if self.cfg.factor_polarity:
            pair_hidden = self.pair_trunk(pair_feats_cat)               # (B,Na,No,P)
            exist = self.exist_scorer(pair_hidden).squeeze(-1)          # (B,Na,No)
            polarity = self.polarity_scorer(pair_hidden)                # (B,Na,No,3)
            pair_logits = None
        else:
            pair_logits = self.pair_classifier(pair_feats_cat)            # (B,Na,No,4)
            if self.cfg.use_biaffine_pair:
                # h_a^T W h_o + U[h_a; h_o] + b. Zero-initialised bilinear weight,
                # so the arm starts exactly at the concat head it is added to and
                # the biaffine term has to earn whatever it contributes.
                pair_logits = pair_logits + self.biaffine(
                    ea.contiguous(), eo.contiguous()
                ) + self.biaffine_linear(torch.cat([ea, eo], dim=-1))

        assignment = None
        if self.cfg.use_assignment:
            match = self.match(ea.contiguous(), eo.contiguous()).squeeze(-1)  # (B,Na,No)
            # Temperature is set externally each epoch; dividing the scores before
            # Sinkhorn flattens the coupling early and sharpens it later.
            temp = getattr(self, "_sinkhorn_temp", self.cfg.sinkhorn_temp_end)
            log_p = log_sinkhorn(match / temp, self.dustbin / temp,
                                 self.cfg.sinkhorn_iters)
            log_match = log_p[:, :na, :no]
            assignment = log_match
            # Shift the decision: a cell that the assignment leaves unmatched is
            # pushed toward "none", one it claims is pushed away from it. The
            # classifier still decides which polarity.
            w = self.cfg.assignment_weight
            evidence = w * log_match
            if self.cfg.use_dustbin_gate:
                # log_p is (B, na+1, no+1): the last column says aspect i stays
                # unmatched, the last row says opinion j stays unmatched.
                a_bin = log_p[:, :na, no].unsqueeze(2)        # (B,na,1)
                o_bin = log_p[:, na, :no].unsqueeze(1)        # (B,1,no)
                abstain = torch.maximum(a_bin, o_bin).expand_as(log_match)
                evidence = evidence + self.dustbin_gate * (log_match - abstain)
            if self.cfg.factor_polarity:
                # In the 4-way form the assignment moves the none logit down by
                # `evidence` and every polarity logit up by the same amount, so
                # the gap it opens is 2 x evidence. With the none logit pinned at
                # 0 the whole gap has to live in `exist`, and the factor of two
                # is what keeps the arm a single-variable change from B2h.
                exist = exist + 2.0 * evidence
            else:
                bias = torch.zeros_like(pair_logits)
                bias[..., 0] = -evidence
                bias[..., 1:] = evidence.unsqueeze(-1)
                pair_logits = pair_logits + bias

        if self.cfg.factor_polarity:
            from aste_polarity import factored_pair_logits
            pair_logits = factored_pair_logits(exist, polarity)

        verifier_logits = None
        if self.cfg.use_verifier and self.verifier_active:
            # The verifier sees the same features the scorer saw plus the scorer's
            # own existence opinion, so it can only re-weigh an emitted decision -
            # it has no path to choose a different partner or a different label.
            score = torch.logsumexp(pair_logits[..., 1:], dim=-1) - pair_logits[..., 0]
            verifier_logits = self.verifier(
                pair_feats_cat.detach(), score.detach(), hidden.detach(),
                attention_mask.bool())

        return {
            "aspect_logits": a_logit, "opinion_logits": o_logit,
            "aspect_idx": a_idx, "opinion_idx": o_idx,
            "pair_logits": pair_logits, "assignment": assignment,
            "verifier_logits": verifier_logits,
            "exist_logits": exist, "polarity_logits_f": polarity,
            "pair_hidden": pair_hidden, "spans": spans, "span_valid": span_valid,
            "token_logits": self.token_tagger(hidden) if self.cfg.use_token_tagger else None,
        }

    def compute_loss_coupled(self, out, gold_pairs, span_valid):
        """Losses for E1+E2.

        ``gold_pairs`` maps batch index -> list of (a_slot, o_slot, polarity) in
        the selected aspect/opinion coordinates. Existence is supervised through
        the assignment matrix (matched cells versus their dustbins), so there is
        no separate "none" objective.
        """
        losses = {}
        # Stage-1 pruning must retain every gold span. This head sits on the
        # inference path but was previously unsupervised: during training gold
        # spans are force-injected into the kept set, which masked the omission
        # completely, and at test time the untrained gate discarded them. That is
        # what produced high precision with near-zero recall.
        prune_logits = out["prune_logits"]
        prune_gold = (out["gold_full_a"] | out["gold_full_b"]).to(prune_logits.dtype)
        valid = span_valid.to(prune_logits.dtype)
        pr_loss = F.binary_cross_entropy_with_logits(
            prune_logits.masked_fill(~span_valid, 0.0), prune_gold, reduction="none")
        pr_loss = (pr_loss * valid).sum() / valid.sum().clamp(min=1)
        losses["prune"] = pr_loss
        # Role scores derived from the match matrix are supervised directly, which
        # is what forces the match matrix to explain both roles at once.
        r_loss = torch.tensor(0.0, device=out["a_score"].device)
        if out["gold_a_pruned"] is not None:
            r_loss = (
                F.binary_cross_entropy_with_logits(
                    out["a_score"], out["gold_a_pruned"].to(out["a_score"].dtype))
                + F.binary_cross_entropy_with_logits(
                    out["o_score"], out["gold_o_pruned"].to(out["o_score"].dtype))
            ) / 2
        losses["role"] = r_loss

        log_p = out["log_assign"]
        b, na1, no1 = log_p.shape
        na, no = na1 - 1, no1 - 1
        pol_logits = out["polarity_logits"]

        match_terms, pol_terms = [], []
        n_pos = 0
        matched_a = torch.zeros((b, na), dtype=torch.bool, device=log_p.device)
        matched_o = torch.zeros((b, no), dtype=torch.bool, device=log_p.device)
        for i, pairs in enumerate(gold_pairs):
            for ai, oi, lab in pairs:
                match_terms.append(log_p[i, ai, oi])
                pol_terms.append(F.cross_entropy(
                    pol_logits[i, ai, oi].unsqueeze(0),
                    torch.tensor([lab - 1], device=log_p.device)))
                matched_a[i, ai] = True
                matched_o[i, oi] = True
                n_pos += 1
        # Everything unmatched should be routed to its dustbin.
        for i in range(b):
            for a in range(na):
                if not matched_a[i, a]:
                    match_terms.append(log_p[i, a, no])
            for o in range(no):
                if not matched_o[i, o]:
                    match_terms.append(log_p[i, na, o])

        pos = torch.stack(match_terms[:n_pos]) if n_pos else None
        negs = torch.stack(match_terms[n_pos:]) if len(match_terms) > n_pos else None
        m_loss = torch.tensor(0.0, device=log_p.device)
        if pos is not None:
            m_loss = m_loss - pos.mean()
        if negs is not None:
            m_loss = m_loss - self.cfg.dustbin_weight * negs.mean()
        p_loss = torch.stack(pol_terms).mean() if pol_terms else             torch.tensor(0.0, device=log_p.device)
        losses["match"] = m_loss
        losses["polarity"] = p_loss
        losses["total"] = (self.cfg.w_prune * pr_loss + self.cfg.w_role * r_loss
                           + self.cfg.w_match * m_loss + self.cfg.w_pair * p_loss)
        return losses

    def compute_loss(self, out, gold_aspects, gold_opinions, pair_target, span_valid,
                     token_target=None, token_mask=None):
        a_loss = F.binary_cross_entropy_with_logits(
            out["aspect_logits"].masked_fill(~span_valid, 0.0),
            gold_aspects.float(), reduction="none",
        )
        a_loss = (a_loss * span_valid).sum() / span_valid.sum().clamp(min=1)
        o_loss = F.binary_cross_entropy_with_logits(
            out["opinion_logits"].masked_fill(~span_valid, 0.0),
            gold_opinions.float(), reduction="none",
        )
        o_loss = (o_loss * span_valid).sum() / span_valid.sum().clamp(min=1)

        logits = out["pair_logits"]
        if self.cfg.factor_polarity:
            from aste_polarity import existence_loss, polarity_loss
            # The existence half is B2h's objective unchanged: same cells, same
            # hard/trivial split, same weights. Only the polarity decision has
            # left it.
            ga = gold_aspects.gather(1, out["aspect_idx"])
            go = gold_opinions.gather(1, out["opinion_idx"])
            hard = ga.unsqueeze(2) | go.unsqueeze(1)
            if self.cfg.hard_none_weight > 0.0:
                base = torch.where(hard, self.cfg.hard_none_weight, self.cfg.none_weight)
                if self.cfg.neutral_negative_mass:
                    neg = pair_target == 0
                    budget = self.cfg.none_weight * neg.sum()
                    base = base * (budget / (base * neg).sum().clamp(min=1e-6))
            else:
                base = torch.full_like(hard, self.cfg.none_weight, dtype=logits.dtype)
            cell_w = torch.where(pair_target > 0, torch.ones_like(base), base)
            p_loss = existence_loss(out["exist_logits"], pair_target, cell_w)
            pol_w = (self.polarity_weight
                     if self.cfg.polarity_class_weight != "none" else None)
            pol_loss = polarity_loss(out["polarity_logits_f"], pair_target, pol_w)
        elif self.cfg.hard_none_weight > 0.0:
            # Per-cell weights: a negative is "hard" when one of its two spans is
            # a gold aspect or a gold opinion, because then the model has to get
            # the pairing right rather than merely reject junk spans.
            ga = gold_aspects.gather(1, out["aspect_idx"])          # (B,Na)
            go = gold_opinions.gather(1, out["opinion_idx"])        # (B,No)
            hard = ga.unsqueeze(2) | go.unsqueeze(1)                # (B,Na,No)
            w = torch.where(hard, self.cfg.hard_none_weight, self.cfg.none_weight)
            neg = pair_target == 0
            if self.cfg.neutral_negative_mass:
                budget = self.cfg.none_weight * neg.sum()
                w = w * (budget / (w * neg).sum().clamp(min=1e-6))
            w = torch.where(pair_target > 0, torch.ones_like(w), w)
            p_loss = (F.cross_entropy(logits.reshape(-1, 4), pair_target.reshape(-1),
                                      reduction="none") * w.reshape(-1)).sum()
            p_loss = p_loss / w.sum().clamp(min=1e-6)
        else:
            weight = torch.tensor(
                [self.cfg.none_weight, 1.0, 1.0, 1.0],
                device=logits.device, dtype=logits.dtype,
            )
            p_loss = F.cross_entropy(logits.reshape(-1, 4), pair_target.reshape(-1),
                                     weight=weight)
        total = (self.cfg.w_aspect * a_loss + self.cfg.w_opinion * o_loss
                 + self.cfg.w_pair * p_loss)
        losses = {"aspect": a_loss, "opinion": o_loss, "pair": p_loss}

        if self.cfg.factor_polarity:
            losses["polarity"] = pol_loss
            total = total + self.cfg.w_polarity * pol_loss
            if self.cfg.use_polarity_contrast:
                pos = pair_target > 0
                c_loss = self.sentiment_protos(
                    out["pair_hidden"][pos], pair_target[pos] - 1, pol_w)
                losses["pol_contrast"] = c_loss
                total = total + self.cfg.w_pol_contrast * c_loss

        if self.cfg.use_overlap_ranking:
            from aste_ranking import overlap_ranking_loss
            spans_k = out["spans"]
            r_loss = 0.5 * (
                overlap_ranking_loss(out["aspect_logits"], gold_aspects, spans_k,
                                     span_valid, self.cfg.overlap_margin,
                                     self.cfg.overlap_single_edge)
                + overlap_ranking_loss(out["opinion_logits"], gold_opinions, spans_k,
                                       span_valid, self.cfg.overlap_margin,
                                       self.cfg.overlap_single_edge)
            )
            losses["overlap_rank"] = r_loss
            total = total + self.cfg.w_overlap_rank * r_loss

        if self.cfg.use_boundary_listwise:
            from aste_polarity import boundary_listwise_loss
            spans = out["spans"]
            # spans[:, 1] is the exclusive end, and span_valid is exactly
            # `end <= n_tokens`, so the largest valid end is the unpadded length.
            n_tokens = torch.where(
                span_valid, spans[:, 1].unsqueeze(0), torch.zeros_like(span_valid, dtype=spans.dtype)
            ).max(dim=1).values
            b_loss = 0.5 * (
                boundary_listwise_loss(out["aspect_logits"], gold_aspects, spans,
                                       span_valid, self.cfg.max_span_width, n_tokens)
                + boundary_listwise_loss(out["opinion_logits"], gold_opinions, spans,
                                         span_valid, self.cfg.max_span_width, n_tokens)
            )
            losses["boundary"] = b_loss
            total = total + self.cfg.w_boundary * b_loss

        if self.cfg.use_null_rank:
            from aste_relation import null_rank_loss
            n_loss = null_rank_loss(logits, pair_target, self.cfg.null_m_pos,
                                    self.cfg.null_m_neg, self.cfg.null_alpha,
                                    self.cfg.null_hard)
            losses["null_rank"] = n_loss
            total = total + self.cfg.w_null_rank * n_loss

        if self.cfg.use_degree_loss:
            from aste_relation import degree_loss
            d_loss = degree_loss(logits, pair_target)
            losses["degree"] = d_loss
            total = total + self.cfg.w_degree * d_loss

        if self.cfg.use_multi_positive:
            from aste_relation import multi_positive_listwise
            r_loss = multi_positive_listwise(logits, pair_target,
                                             self.cfg.multi_positive_symmetric)
            losses["multi_positive"] = r_loss
            total = total + self.cfg.w_multi_positive * r_loss

        if self.cfg.use_listwise:
            # Existence log-odds per cell, polarity marginalised out: the row
            # question is which opinion binds, not with what sentiment.
            e = torch.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]
            pos = (pair_target > 0)
            zeros = torch.zeros_like(e[..., :1])
            n_a, n_o = e.shape[1], e.shape[2]

            def listwise(scores, positive):
                """-log p of the gold partners, or of "no partner" for an empty row."""
                lp = torch.cat([scores, torch.zeros_like(scores[..., :1])],
                               dim=-1).log_softmax(-1)
                k = scores.shape[-1]
                hit = -(lp[..., :k] * positive.to(lp.dtype)).sum(-1)
                empty = -lp[..., k] * (~positive.any(-1)).to(lp.dtype)
                return (hit + empty).mean()

            row = listwise(e, pos)                                  # aspect -> opinion
            col = listwise(e.transpose(1, 2), pos.transpose(1, 2))  # opinion -> aspect
            l_loss = 0.5 * (row + col)
            losses["listwise"] = l_loss
            total = total + self.cfg.w_listwise * l_loss
            del zeros, n_a, n_o

        if out.get("token_logits") is not None and token_target is not None:
            t_loss = F.binary_cross_entropy_with_logits(
                out["token_logits"], token_target.to(out["token_logits"].dtype),
                reduction="none",
            )
            m = token_mask.unsqueeze(-1).to(t_loss.dtype)
            t_loss = (t_loss * m).sum() / m.sum().clamp(min=1) / 2
            losses["tagger"] = t_loss
            total = total + self.cfg.w_tagger * t_loss

        losses["total"] = total
        return losses
