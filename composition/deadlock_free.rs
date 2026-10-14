// verus/deadlock_free.rs
//
// Deadlock freedom for a finite, blocking collective schedule.
//
// This file proves a source-level scheduling property.  It does not define
// `deadlock_free` as an uninterpreted predicate and then assume that predicate.
// Instead it models:
//
//   * a concrete per-rank trace of collective events;
//   * a program counter for every rank;
//   * when a collective is enabled (all members are at the matching event);
//   * the reachable-frontier invariant of atomic collective completion; and
//   * absence of a non-final state in which no collective is enabled.
//
// Different ranks may name different groups at the same phase.  This is
// essential for TP x DP x EP schedules: ranks 0..3 may use one TP group while
// ranks 4..7 simultaneously use another.  What is required is agreement among
// the members of each named group, not equality of every rank's group handle.
//
// TRUST BOUNDARY: the theorem below is about the abstract schedule.  Turning an
// enabled transition into real progress assumes that NCCL completes a matching
// collective in the absence of process/device/network failure and under fair
// scheduling.  The correspondence between the Python call sites and a
// `Schedule` value must also be established by extraction, generation, or
// audit.  Neither assumption is disguised as a Verus proof in this file.
//
// Run with:
//   verus --crate-type=lib verus/deadlock_free.rs


use vstd::prelude::*;

verus! {

pub type Rank = nat;

#[derive(PartialEq, Eq)]
pub enum CollOp {
    AllReduce,
    AllToAll,
    AllGather,
}

/// A collective event.  `site` distinguishes different call sites that use
/// the same primitive on the same communicator (for example count exchange,
/// payload dispatch, and reverse dispatch).
pub struct Step {
    pub site: nat,
    pub op: CollOp,
    pub group: Set<Rank>,
}

/// `trace[r][p]` is rank `r`'s collective at phase `p`.
/// Every trace has exactly `phases` entries in a well-formed schedule.
pub struct Schedule {
    pub world_size: nat,
    pub phases: nat,
    pub trace: Seq<Seq<Step>>,
}

/// Per-rank program counters.  `pc[r] == p` means rank `r` has completed
/// phases `[0, p)` and is about to post phase `p`.
pub struct State {
    pub pc: Seq<int>,
}

pub open spec fn valid_rank(schedule: Schedule, r: Rank) -> bool {
    r < schedule.world_size
}

pub open spec fn step_at(schedule: Schedule, r: Rank, phase: int) -> Step
    recommends
        valid_rank(schedule, r),
        0 <= phase < schedule.phases,
{
    schedule.trace[r as int][phase]
}

/// Static schedule obligations.
///
/// `peer agreement` says that if q is named by r's phase-p group, q has the
/// same complete event at phase p.  Together with self-membership this makes
/// the groups at each phase a partition of the world, while still allowing
/// different partitions at different phases.
pub open spec fn well_formed_schedule(schedule: Schedule) -> bool {
    &&& schedule.world_size > 0
    &&& schedule.phases > 0
    &&& schedule.trace.len() == schedule.world_size
    &&& forall|r: Rank| #![auto] valid_rank(schedule, r)
        ==> schedule.trace[r as int].len() == schedule.phases
    &&& forall|r: Rank, phase: int| #![auto]
        valid_rank(schedule, r) && 0 <= phase < schedule.phases
        ==> step_at(schedule, r, phase).group.contains(r)
    &&& forall|r: Rank, q: Rank, phase: int|
        valid_rank(schedule, r)
        && 0 <= phase < schedule.phases
        && step_at(schedule, r, phase).group.contains(q)
        ==> valid_rank(schedule, q)
            && step_at(schedule, q, phase) == step_at(schedule, r, phase)
}

pub open spec fn valid_state(schedule: Schedule, state: State) -> bool {
    &&& state.pc.len() == schedule.world_size
    &&& forall|r: Rank| valid_rank(schedule, r)
        ==> 0 <= state.pc[r as int] <= schedule.phases
}

pub open spec fn finished(schedule: Schedule, state: State) -> bool {
    forall|r: Rank| #![auto] valid_rank(schedule, r)
        ==> state.pc[r as int] == schedule.phases
}

/// No reachable frontier may cut through one completed collective.  For every
/// phase and every two peers in that phase's group, either both have completed
/// the phase or neither has.  Atomic completion of an enabled collective
/// preserves this invariant.
pub open spec fn frontier_closed(schedule: Schedule, state: State) -> bool {
    forall|r: Rank, q: Rank, phase: int|
        valid_rank(schedule, r)
        && 0 <= phase < schedule.phases
        && step_at(schedule, r, phase).group.contains(q)
        ==> (state.pc[r as int] > phase) == (state.pc[q as int] > phase)
}

/// Rank `r` can lead the next collective transition exactly when every member
/// of its current group is at the same phase and sees the same event.
pub open spec fn enabled(schedule: Schedule, state: State, r: Rank) -> bool {
    &&& valid_rank(schedule, r)
    &&& 0 <= state.pc[r as int] < schedule.phases
    &&& forall|q: Rank| #![auto]
        step_at(schedule, r, state.pc[r as int]).group.contains(q)
        ==> valid_rank(schedule, q)
            && state.pc[q as int] == state.pc[r as int]
            && step_at(schedule, q, state.pc[q as int])
                == step_at(schedule, r, state.pc[r as int])
}

pub open spec fn deadlocked(schedule: Schedule, state: State) -> bool {
    &&& valid_state(schedule, state)
    &&& !finished(schedule, state)
    &&& forall|r: Rank| #![auto]
        valid_rank(schedule, r) ==> !enabled(schedule, state, r)
}

/// Initial state: no rank has posted a collective.
pub open spec fn initial_state(schedule: Schedule) -> State {
    State {
        pc: Seq::new(schedule.world_size, |i: int| 0int),
    }
}

/// Completing the collective led by `r` atomically advances exactly the
/// members of that collective's group.
pub open spec fn advance(schedule: Schedule, state: State, r: Rank) -> State
    recommends enabled(schedule, state, r),
{
    let event = step_at(schedule, r, state.pc[r as int]);
    State {
        pc: Seq::new(schedule.world_size, |i: int|
            if event.group.contains(i as nat) {
                state.pc[i] + 1
            } else {
                state.pc[i]
            }
        ),
    }
}

/// One transition in the abstract blocking-collective machine.
pub open spec fn collective_step(
    schedule: Schedule,
    before: State,
    after: State,
) -> bool {
    exists|r: Rank| #![auto]
        valid_rank(schedule, r)
        && enabled(schedule, before, r)
        && after == advance(schedule, before, r)
}

/// A finite execution starts at `initial_state` and contains only enabled
/// collective transitions.
pub open spec fn execution(schedule: Schedule, states: Seq<State>) -> bool {
    &&& states.len() > 0
    &&& states[0] == initial_state(schedule)
    &&& forall|i: int|
        #![trigger states[i]]
        0 <= i < states.len() - 1
        ==> collective_step(schedule, states[i], states[i + 1])
}

pub proof fn lemma_initial_state_invariant(schedule: Schedule)
    requires well_formed_schedule(schedule),
    ensures
        valid_state(schedule, initial_state(schedule)),
        frontier_closed(schedule, initial_state(schedule)),
{
    assert forall|r: Rank| valid_rank(schedule, r)
        implies 0 <= initial_state(schedule).pc[r as int]
            <= schedule.phases by {
        assert(initial_state(schedule).pc[r as int] == 0);
    }

    assert forall|r: Rank, q: Rank, phase: int|
        valid_rank(schedule, r)
        && 0 <= phase < schedule.phases
        && step_at(schedule, r, phase).group.contains(q)
        implies
            (initial_state(schedule).pc[r as int] > phase)
                == (initial_state(schedule).pc[q as int] > phase) by {
        assert(initial_state(schedule).pc[r as int] == 0);
        assert(valid_rank(schedule, q));
        assert(initial_state(schedule).pc[q as int] == 0);
    }
}

/// Completing an enabled collective preserves both the PC bounds and the
/// closed-frontier invariant.  The phase case split is the key argument:
/// advancing phase `current` cannot change whether a rank has crossed an
/// earlier or later phase; at `current`, peer agreement makes intersecting
/// groups identical, so the transition advances either both peers or neither.
pub proof fn lemma_advance_preserves_invariant(
    schedule: Schedule,
    before: State,
    leader: Rank,
)
    requires
        well_formed_schedule(schedule),
        valid_state(schedule, before),
        frontier_closed(schedule, before),
        enabled(schedule, before, leader),
    ensures
        valid_state(schedule, advance(schedule, before, leader)),
        frontier_closed(schedule, advance(schedule, before, leader)),
{
    let current = before.pc[leader as int];
    let event = step_at(schedule, leader, current);
    let after = advance(schedule, before, leader);

    assert(after.pc.len() == schedule.world_size);
    assert forall|r: Rank| valid_rank(schedule, r)
        implies 0 <= after.pc[r as int] <= schedule.phases by {
        assert(0 <= before.pc[r as int] <= schedule.phases);
        if event.group.contains(r) {
            assert(before.pc[r as int] == current);
            assert(current < schedule.phases);
            assert(after.pc[r as int] == before.pc[r as int] + 1);
        } else {
            assert(after.pc[r as int] == before.pc[r as int]);
        }
    }

    assert forall|r: Rank, q: Rank, phase: int|
        valid_rank(schedule, r)
        && 0 <= phase < schedule.phases
        && step_at(schedule, r, phase).group.contains(q)
        implies (after.pc[r as int] > phase) == (after.pc[q as int] > phase) by {
        assert(valid_rank(schedule, q));

        if phase < current {
            if event.group.contains(r) {
                assert(before.pc[r as int] == current);
                assert(after.pc[r as int] == current + 1);
            } else {
                assert(after.pc[r as int] == before.pc[r as int]);
            }
            if event.group.contains(q) {
                assert(before.pc[q as int] == current);
                assert(after.pc[q as int] == current + 1);
            } else {
                assert(after.pc[q as int] == before.pc[q as int]);
            }
            assert((before.pc[r as int] > phase) == (before.pc[q as int] > phase));
        } else if current < phase {
            if event.group.contains(r) {
                assert(before.pc[r as int] == current);
                assert(after.pc[r as int] == current + 1);
            } else {
                assert(after.pc[r as int] == before.pc[r as int]);
            }
            if event.group.contains(q) {
                assert(before.pc[q as int] == current);
                assert(after.pc[q as int] == current + 1);
            } else {
                assert(after.pc[q as int] == before.pc[q as int]);
            }
            assert((before.pc[r as int] > phase) == (before.pc[q as int] > phase));
        } else {
            assert(phase == current);

            assert(event.group.contains(r) == event.group.contains(q)) by {
                if event.group.contains(r) {
                    assert(step_at(schedule, r, current) == event);
                    assert(step_at(schedule, r, current).group == event.group);
                    assert(event.group.contains(q));
                } else if event.group.contains(q) {
                    assert(step_at(schedule, q, current) == event);
                    assert(step_at(schedule, q, current)
                        == step_at(schedule, r, current));
                    assert(step_at(schedule, r, current).group == event.group);
                    assert(event.group.contains(r));
                }
            }

            if event.group.contains(r) {
                assert(event.group.contains(q));
                assert(before.pc[r as int] == current);
                assert(before.pc[q as int] == current);
                assert(after.pc[r as int] == current + 1);
                assert(after.pc[q as int] == current + 1);
            } else {
                assert(!event.group.contains(q));
                assert(after.pc[r as int] == before.pc[r as int]);
                assert(after.pc[q as int] == before.pc[q as int]);
                assert((before.pc[r as int] > phase)
                    == (before.pc[q as int] > phase));
            }
        }
    }
}

/// An enabled transition is not a stuttering step: its leader advances, and no
/// rank's program counter moves backwards.  Together with the finite PC bound,
/// this is the ranking argument used when the runtime repeatedly schedules
/// enabled collectives.
pub proof fn lemma_advance_makes_progress(
    schedule: Schedule,
    before: State,
    leader: Rank,
)
    requires
        well_formed_schedule(schedule),
        valid_state(schedule, before),
        enabled(schedule, before, leader),
    ensures
        advance(schedule, before, leader).pc[leader as int]
            == before.pc[leader as int] + 1,
        forall|r: Rank| #![auto] valid_rank(schedule, r)
            ==> advance(schedule, before, leader).pc[r as int]
                >= before.pc[r as int],
{
    let current = before.pc[leader as int];
    let event = step_at(schedule, leader, current);
    let after = advance(schedule, before, leader);

    assert(event.group.contains(leader));
    assert(after.pc[leader as int] == before.pc[leader as int] + 1);

    assert forall|r: Rank| #![auto] valid_rank(schedule, r)
        implies after.pc[r as int] >= before.pc[r as int] by {
        if event.group.contains(r) {
            assert(after.pc[r as int] == before.pc[r as int] + 1);
        } else {
            assert(after.pc[r as int] == before.pc[r as int]);
        }
    }
}

/// Every state in a finite execution satisfies the two inductive invariants.
pub proof fn lemma_execution_state_invariant(
    schedule: Schedule,
    states: Seq<State>,
    i: nat,
)
    requires
        well_formed_schedule(schedule),
        execution(schedule, states),
        i < states.len(),
    ensures
        valid_state(schedule, states[i as int]),
        frontier_closed(schedule, states[i as int]),
    decreases i,
{
    if i == 0 {
        lemma_initial_state_invariant(schedule);
    } else {
        lemma_execution_state_invariant(schedule, states, (i as int - 1) as nat);
        assert(collective_step(
            schedule,
            states[i as int - 1],
            states[i as int],
        ));
        let leader = choose|r: Rank|
            valid_rank(schedule, r)
            && enabled(schedule, states[i as int - 1], r)
            && states[i as int] == advance(schedule, states[i as int - 1], r);
        lemma_advance_preserves_invariant(
            schedule,
            states[i as int - 1],
            leader,
        );
    }
}

/// Main machine-checked progress lemma.
///
/// Pick a rank with minimum program counter.  Frontier closure prevents any
/// peer in its current group from having crossed that phase already; minimality
/// prevents a peer from being behind.  Thus all peers are at the same matching
/// event and that collective is enabled.
pub proof fn lemma_nonfinal_state_has_enabled_collective(
    schedule: Schedule,
    state: State,
)
    requires
        well_formed_schedule(schedule),
        valid_state(schedule, state),
        frontier_closed(schedule, state),
        !finished(schedule, state),
    ensures exists|r: Rank| #![auto]
        valid_rank(schedule, r) && enabled(schedule, state, r),
{
    let pcs: Seq<int> = state.pc;
    pcs.min_ensures();

    let unfinished = choose|u: Rank| #![auto]
        valid_rank(schedule, u)
        && state.pc[u as int] != schedule.phases;
    assert(valid_rank(schedule, unfinished));
    assert(state.pc[unfinished as int] < schedule.phases);

    assert(pcs.min() < schedule.phases) by {
        assert(pcs.min() <= pcs[unfinished as int]);
    }

    let ri = choose|i: int|
        0 <= i < pcs.len() && pcs[i] == pcs.min();
    let r: Rank = ri as nat;
    assert(valid_rank(schedule, r));
    assert(state.pc[r as int] == pcs.min());
    assert(0 <= state.pc[r as int] < schedule.phases);

    assert forall|q: Rank| #![auto]
        step_at(schedule, r, state.pc[r as int]).group.contains(q)
        implies valid_rank(schedule, q)
            && state.pc[q as int] == state.pc[r as int]
            && step_at(schedule, q, state.pc[q as int])
                == step_at(schedule, r, state.pc[r as int]) by {
        let phase = state.pc[r as int];
        assert(valid_rank(schedule, q));
        assert(0 <= state.pc[q as int] <= schedule.phases);
        assert(pcs.min() <= pcs[q as int]);
        assert((state.pc[r as int] > phase) == (state.pc[q as int] > phase));
        assert(state.pc[q as int] == phase);
        assert(step_at(schedule, q, phase) == step_at(schedule, r, phase));
    }

    assert(enabled(schedule, state, r));
}

/// Deadlock freedom in the abstract blocking-collective semantics: every
/// invariant state is either finished or admits a collective transition.
pub proof fn theorem_deadlock_free(schedule: Schedule, state: State)
    requires
        well_formed_schedule(schedule),
        valid_state(schedule, state),
        frontier_closed(schedule, state),
    ensures !deadlocked(schedule, state),
{
    if !finished(schedule, state) {
        lemma_nonfinal_state_has_enabled_collective(schedule, state);
    }
}

/// End-to-end form for modeled executions: every reachable state is
/// non-deadlocked.  Unlike the old proof, reachability is not an unproved
/// premise; it is derived from the initial state and `collective_step`.
pub proof fn theorem_execution_deadlock_free(
    schedule: Schedule,
    states: Seq<State>,
    i: nat,
)
    requires
        well_formed_schedule(schedule),
        execution(schedule, states),
        i < states.len(),
    ensures !deadlocked(schedule, states[i as int]),
{
    lemma_execution_state_invariant(schedule, states, i);
    theorem_deadlock_free(schedule, states[i as int]);
}

// =====================================================================
// Concrete schedules compared by the DP=1 composition theorem.
//
//   phase 0: TP attention all-reduce (the TP group is the world at DP=1)
//   phase 1: EP count exchange (all-to-all path only)
//   phase 2: EP payload dispatch (all-to-all path only)
//   phase 3: EP reverse dispatch / combine (all-to-all path only)
//
// At DP=1, the TP and EP groups are both the whole world.  The all-reduce
// path replaces phases 1--3 with one world-group all-reduce.
// =====================================================================

pub open spec fn world_group(world_size: nat) -> Set<Rank> {
    Set::new(|r: Rank| r < world_size)
}

pub open spec fn tp_group_of(
    world_size: nat,
    tp_size: nat,
    r: Rank,
) -> Set<Rank>
    recommends tp_size > 0,
{
    Set::new(|q: Rank|
        q < world_size && q / tp_size == r / tp_size)
}

/// DP=1 all-reduce block schedule:
///   phase 0: TP attention all-reduce
///   phase 1: EP MoE all-reduce over the world group
pub open spec fn all_reduce_dp1_step(
    world_size: nat,
    tp_size: nat,
    r: Rank,
    phase: int,
) -> Step
    recommends
        tp_size > 0,
        r < world_size,
        0 <= phase < 2,
{
    if phase == 0 {
        Step {
            site: 0,
            op: CollOp::AllReduce,
            group: tp_group_of(world_size, tp_size, r),
        }
    } else {
        Step {
            site: 1,
            op: CollOp::AllReduce,
            group: world_group(world_size),
        }
    }
}

pub open spec fn all_reduce_dp1_schedule(world_size: nat, tp_size: nat) -> Schedule {
    Schedule {
        world_size,
        phases: 2,
        trace: Seq::new(world_size, |r: int|
            Seq::new(2, |phase: int|
                all_reduce_dp1_step(world_size, tp_size, r as nat, phase)
            )
        ),
    }
}

pub open spec fn all_to_all_dp1_step(
    world_size: nat,
    tp_size: nat,
    r: Rank,
    phase: int,
) -> Step
    recommends
        tp_size > 0,
        r < world_size,
        0 <= phase < 4,
{
    if phase == 0 {
        Step {
            site: 0,
            op: CollOp::AllReduce,
            group: tp_group_of(world_size, tp_size, r),
        }
    } else {
        Step {
            site: phase as nat,
            op: CollOp::AllToAll,
            group: world_group(world_size),
        }
    }
}

pub open spec fn all_to_all_dp1_schedule(world_size: nat, tp_size: nat) -> Schedule {
    Schedule {
        world_size,
        phases: 4,
        trace: Seq::new(world_size, |r: int|
            Seq::new(4, |phase: int|
                all_to_all_dp1_step(world_size, tp_size, r as nat, phase)
            )
        ),
    }
}

proof fn lemma_tp_peers_name_same_group(
    world_size: nat,
    tp_size: nat,
    r: Rank,
    q: Rank,
)
    requires
        tp_size > 0,
        r < world_size,
        tp_group_of(world_size, tp_size, r).contains(q),
    ensures
        q < world_size,
        tp_group_of(world_size, tp_size, q)
            == tp_group_of(world_size, tp_size, r),
{
    assert(q < world_size);
    assert(q / tp_size == r / tp_size);
    assert(tp_group_of(world_size, tp_size, q)
        =~= tp_group_of(world_size, tp_size, r)) by {
        assert forall|x: Rank|
            tp_group_of(world_size, tp_size, q).contains(x)
                == tp_group_of(world_size, tp_size, r).contains(x) by {
            assert((x / tp_size == q / tp_size)
                == (x / tp_size == r / tp_size));
        }
    }
}

/// The concrete two-phase DP=1 all-reduce schedule satisfies the generic static
/// collective-matching obligations.
pub proof fn lemma_all_reduce_dp1_schedule_well_formed(
    world_size: nat,
    tp_size: nat,
)
    requires
        world_size > 0,
        tp_size == world_size,
    ensures well_formed_schedule(all_reduce_dp1_schedule(world_size, tp_size)),
{
    let schedule = all_reduce_dp1_schedule(world_size, tp_size);

    assert(schedule.trace.len() == world_size);
    assert forall|r: Rank| #![auto] valid_rank(schedule, r)
        implies schedule.trace[r as int].len() == schedule.phases by {
        assert(schedule.trace[r as int].len() == 2);
    }

    assert forall|r: Rank, phase: int| #![auto]
        valid_rank(schedule, r) && 0 <= phase < schedule.phases
        implies step_at(schedule, r, phase).group.contains(r) by {
        if phase == 0 {
            assert(tp_group_of(world_size, tp_size, r).contains(r));
        } else {
            assert(world_group(world_size).contains(r));
        }
    }

    assert forall|r: Rank, q: Rank, phase: int|
        valid_rank(schedule, r)
        && 0 <= phase < schedule.phases
        && step_at(schedule, r, phase).group.contains(q)
        implies valid_rank(schedule, q)
            && step_at(schedule, q, phase) == step_at(schedule, r, phase) by {
        if phase == 0 {
            lemma_tp_peers_name_same_group(world_size, tp_size, r, q);
            assert(valid_rank(schedule, q));
            assert(tp_group_of(world_size, tp_size, q)
                == tp_group_of(world_size, tp_size, r));
        } else {
            assert(world_group(world_size).contains(q));
            assert(q < world_size);
            assert(valid_rank(schedule, q));
        }
    }
}

/// The concrete four-phase DP=1 all-to-all schedule satisfies the generic
/// collective-matching obligations.
pub proof fn lemma_all_to_all_dp1_schedule_well_formed(
    world_size: nat,
    tp_size: nat,
)
    requires
        world_size > 0,
        tp_size == world_size,
    ensures well_formed_schedule(all_to_all_dp1_schedule(world_size, tp_size)),
{
    let schedule = all_to_all_dp1_schedule(world_size, tp_size);

    assert(schedule.trace.len() == world_size);
    assert forall|r: Rank| #![auto] valid_rank(schedule, r)
        implies schedule.trace[r as int].len() == schedule.phases by {
        assert(schedule.trace[r as int].len() == 4);
    }

    assert forall|r: Rank, phase: int| #![auto]
        valid_rank(schedule, r) && 0 <= phase < schedule.phases
        implies step_at(schedule, r, phase).group.contains(r) by {
        if phase == 0 {
            assert(tp_group_of(world_size, tp_size, r).contains(r));
        } else {
            assert(world_group(world_size).contains(r));
        }
    }

    assert forall|r: Rank, q: Rank, phase: int|
        valid_rank(schedule, r)
        && 0 <= phase < schedule.phases
        && step_at(schedule, r, phase).group.contains(q)
        implies valid_rank(schedule, q)
            && step_at(schedule, q, phase) == step_at(schedule, r, phase) by {
        if phase == 0 {
            lemma_tp_peers_name_same_group(world_size, tp_size, r, q);
            assert(valid_rank(schedule, q));
            assert(tp_group_of(world_size, tp_size, q)
                == tp_group_of(world_size, tp_size, r));
        } else {
            assert(world_group(world_size).contains(q));
            assert(q < world_size);
            assert(valid_rank(schedule, q));
        }
    }
}

/// Paper-facing theorem for the concrete DP=1 all-to-all schedule: every state
/// in every modeled execution is either final or has an enabled collective.
pub proof fn theorem_all_to_all_dp1_execution_deadlock_free(
    world_size: nat,
    tp_size: nat,
    states: Seq<State>,
    i: nat,
)
    requires
        world_size > 0,
        tp_size == world_size,
        execution(all_to_all_dp1_schedule(world_size, tp_size), states),
        i < states.len(),
    ensures !deadlocked(
        all_to_all_dp1_schedule(world_size, tp_size),
        states[i as int],
    ),
{
    lemma_all_to_all_dp1_schedule_well_formed(world_size, tp_size);
    theorem_execution_deadlock_free(
        all_to_all_dp1_schedule(world_size, tp_size),
        states,
        i,
    );
}

/// Paper-facing theorem for the concrete DP=1 all-reduce schedule.
pub proof fn theorem_all_reduce_dp1_execution_deadlock_free(
    world_size: nat,
    tp_size: nat,
    states: Seq<State>,
    i: nat,
)
    requires
        world_size > 0,
        tp_size == world_size,
        execution(all_reduce_dp1_schedule(world_size, tp_size), states),
        i < states.len(),
    ensures !deadlocked(
        all_reduce_dp1_schedule(world_size, tp_size),
        states[i as int],
    ),
{
    lemma_all_reduce_dp1_schedule_well_formed(world_size, tp_size);
    theorem_execution_deadlock_free(
        all_reduce_dp1_schedule(world_size, tp_size),
        states,
        i,
    );
}

} // verus!
