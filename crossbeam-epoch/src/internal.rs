//! The global data and participant for garbage collection.
//!
//! # Registration
//!
//! In order to track all participants in one place, we need some form of participant
//! registration. When a participant is created, it is registered to a global lock-free
//! singly-linked list of registries; and when a participant is leaving, it is unregistered from the
//! list.
//!
//! # Pinning
//!
//! Every participant contains an integer that tells whether the participant is pinned and if so,
//! what was the global epoch at the time it was pinned. Participants also hold a pin counter that
//! aids in periodic global epoch advancement.
//!
//! When a participant is pinned, a `Guard` is returned as a witness that the participant is pinned.
//! Guards are necessary for performing atomic operations, and for freeing/dropping locations.
//!
//! # Thread-local bag
//!
//! Objects that get unlinked from concurrent data structures must be stashed away until the global
//! epoch sufficiently advances so that they become safe for destruction. Pointers to such objects
//! are pushed into a thread-local bag, and when it becomes full, the bag is marked with the current
//! global epoch and pushed into the global queue of bags. We store objects in thread-local storages
//! for amortizing the synchronization cost of pushing the garbages to a global queue.
//!
//! # Global queue
//!
//! Whenever a bag is pushed into a queue, the objects in some bags in the queue are collected and
//! destroyed along the way. This design reduces contention on data structures. The global queue
//! cannot be explicitly accessed: the only way to interact with it is by calling functions
//! `defer()` that adds an object tothe thread-local bag, or `collect()` that manually triggers
//! garbage collection.
//!
//! Ideally each instance of concurrent data structure may have its own queue that gets fully
//! destroyed as soon as the data structure gets dropped.

use core::cell::{Cell, UnsafeCell};
use core::cmp;
use core::convert::TryInto;
use core::mem::{self, ManuallyDrop};
use core::num::Wrapping;
use core::ptr;
use core::sync::atomic::Ordering;

use crossbeam_utils::CachePadded;
use membarrier;

use atomic::{Atomic, Owned, Shared};
use bloom_filter::BloomFilter;
use collector::{Collector, LocalHandle};
use garbage::{Bag, Garbage};
use guard::{unprotected, Guard};
use hazard::{HazardSet, Shield, ShieldError};
use sync::list::{repeat_iter, Entry, IsElement, List};
use sync::stack::Stack;
use tag::*;

/// The width of epoch's representation. In other words, there can be `1 << EPOCH_WIDTH` epochs that
/// are wrapping around.
const EPOCH_WIDTH: u32 = 5;

/// The width of the number of bags.
const BAGS_WIDTH: u32 = 3;

const_assert!(bags_epoch_width; BAGS_WIDTH <= EPOCH_WIDTH);

/// Compares two epochs.
fn epoch_cmp(a: usize, b: usize) -> cmp::Ordering {
    let diff = b.wrapping_sub(a) % (1 << EPOCH_WIDTH);
    if diff == 0 {
        cmp::Ordering::Equal
    } else if diff < (1 << (EPOCH_WIDTH - 1)) {
        cmp::Ordering::Less
    } else {
        cmp::Ordering::Greater
    }
}

bitflags! {
    /// Status flags tagged in a pointer to hazard pointer summary.
    struct StatusFlags: usize {
        const EJECTING = 1 << (EPOCH_WIDTH + 1);
        const PINNED   = 1 << EPOCH_WIDTH;
        const EPOCH    = (1 << EPOCH_WIDTH) - 1;
    }
}

impl StatusFlags {
    #[inline(always)]
    pub fn new(is_ejecting: bool, is_pinned: bool, epoch: usize) -> Self {
        debug_assert!(
            StatusFlags::all().bits() <= low_bits::<CachePadded<BloomFilter>>(),
            "StatusFlags should be tagged in a pointer to hazard pointer summary.",
        );

        let is_ejecting = if is_ejecting {
            Self::EJECTING
        } else {
            Self::empty()
        };
        let is_pinned = if is_pinned {
            Self::PINNED
        } else {
            Self::empty()
        };
        let epoch = Self::from_bits_truncate(epoch) & Self::EPOCH;
        is_ejecting | is_pinned | epoch
    }

    #[inline(always)]
    pub fn is_pinned(self) -> bool {
        !(self & Self::PINNED).is_empty()
    }

    #[inline(always)]
    pub fn epoch(self) -> usize {
        (self & Self::EPOCH).bits()
    }
}

/// The global data for a garbage collector.
pub struct Global {
    /// The intrusive linked list of `Local`s.
    locals: List<Local>,

    /// The global pool of bags of deferred functions.
    bags: CachePadded<Stack<Bag>>,
}

impl Global {
    /// Number of bags to destroy.
    const COLLECT_STEPS: usize = 8;

    /// Creates a new global data for garbage collection.
    #[inline]
    pub fn new() -> Self {
        // TODO(@jeehoonkang): it has a very ugly invariant...
        membarrier::heavy();

        Self {
            locals: List::new(),
            bags: CachePadded::new(Stack::new()),
        }
    }

    /// Pushes the bag into the global queue and replaces the bag with a new empty bag.
    pub fn push_bag(&self, bag: &mut Bag) {
        let bag = mem::replace(bag, Bag::new());
        self.bags.push(bag);
    }

    pub fn collect_hazards(&self, guard: &Guard) -> Option<BloomFilter> {
        // Heavy fence to synchronize with `Shield::defend()`.
        unsafe {
            membarrier::heavy_membarrier();
        }

        todo!()
    }

    /// Collects several bags from the global queue and executes deferred functions in them.
    ///
    /// Note: This may itself produce garbage and in turn allocate new bags.
    ///
    /// `pin()` rarely calls `collect()`, so we want the compiler to place that call on a cold
    /// path. In other words, we want the compiler to optimize branching for the case when
    /// `collect()` is not called.
    #[cold]
    #[must_use]
    pub fn collect<'g>(&'g self, guard: &'g Guard) -> Result<bool, ShieldError> {
        let summary = self.collect_hazards(guard);

        let steps = if cfg!(feature = "sanitize") {
            usize::max_value()
        } else {
            Self::COLLECT_STEPS
        };

        for _ in 0..steps {
            if let Some(mut bag) = self.bags.try_pop(guard)? {
                // Disposes the garbages (except for hazard pointers) in the bag popped from the
                // global queue.
                let disposed = bag.dispose(summary.as_ref());

                if let Some(local) = unsafe { guard.local.as_ref() } {
                    local.inc_reclaimed(disposed);
                }

                // TODO(@jeehoonkang): it must be a queue...  otherwise a garbage will be inspected
                // on and on and on.
                //
                // If the bag is not empty (due to hazard pointers), push it back to the global
                // queue.
                if !bag.is_empty() {
                    self.push_bag(&mut bag);
                }
            } else {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// Participant for garbage collection.
#[derive(Debug)]
pub struct Local {
    /// A node in the intrusive linked list of `Local`s.
    entry: Entry,

    /// The local status consisting of (1) the (approximate) summary of hazard pointers, and (2)
    /// `StatusFlags`.
    status: CachePadded<Atomic<CachePadded<BloomFilter>>>,

    /// A reference to the global data.
    ///
    /// When all guards and handles get dropped, this reference is destroyed.
    collector: UnsafeCell<ManuallyDrop<Collector>>,

    /// The local garbages of deferred functions.
    garbages: UnsafeCell<Vec<Garbage>>,

    /// The number of guards keeping this participant pinned.
    guard_count: Cell<usize>,

    /// The number of active handles.
    handle_count: Cell<usize>,

    /// Total number of pinnings performed.
    ///
    /// This is just an auxiliary counter that sometimes kicks off collection.
    pin_count: Cell<Wrapping<usize>>,

    /// The set of hazard pointers.
    pub(crate) hazards: HazardSet,

    /// The number of blocks retired by this thread.
    retired: Cell<usize>,
    /// The number of blocks reclaimed by this thread. Note that a block retired from a thread can
    /// be reclaimed by another thread. That is, `retired - reclaimed` can be negative and doesn't
    /// mean "locally retired but not yet reclaimed". However, the average (w.r.t num of threads)
    /// of thread-local average of `retired - reclaimed` at each op does make sense to be used as a
    /// space overhead criterion.
    reclaimed: Cell<usize>,
}

impl Local {
    /// Number of pinnings after which a participant will execute some deferred functions from the
    /// global queue.
    const PINNINGS_BETWEEN_COLLECT: usize = 8;

    /// Maximum number of unreclaimed garbages.
    const MAX_GARBAGES: usize = 256;

    /// Registers a new `Local` in the provided `Global`.
    pub fn register(collector: &Collector) -> LocalHandle {
        unsafe {
            // Since we dereference no pointers in this block, it is safe to use `unprotected`.

            let local = Owned::new(Local {
                entry: Entry::default(),
                status: CachePadded::new(Atomic::null()),
                collector: UnsafeCell::new(ManuallyDrop::new(collector.clone())),
                hazards: HazardSet::new(),
                garbages: UnsafeCell::new(Vec::new()),
                guard_count: Cell::new(0),
                handle_count: Cell::new(1),
                pin_count: Cell::new(Wrapping(0)),
                retired: Cell::new(0),
                reclaimed: Cell::new(0),
            })
            .into_shared(unprotected());
            collector.global.locals.insert(local);
            LocalHandle {
                local: local.as_raw(),
            }
        }
    }

    pub fn inc_retired(&self, ammount: usize) {
        let retired = self.retired.get();
        self.retired.set(retired.wrapping_add(ammount));
    }

    pub fn inc_reclaimed(&self, ammount: usize) {
        let reclaimed = self.reclaimed.get();
        self.reclaimed.set(reclaimed.wrapping_add(ammount));
    }

    pub fn retired_unreclaimed(&self) -> i64 {
        let retired: i64 = self.retired.get().try_into().unwrap();
        let reclaimed: i64 = self.reclaimed.get().try_into().unwrap();
        retired - reclaimed
    }

    /// Returns a reference to the `Global` in which this `Local` resides.
    #[inline]
    pub fn global(&self) -> &Global {
        &self.collector().global
    }

    /// Returns a reference to the `Collector` in which this `Local` resides.
    #[inline]
    pub fn collector(&self) -> &Collector {
        unsafe { &**self.collector.get() }
    }

    /// Returns `true` if the current participant is pinned.
    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.guard_count.get() > 0
    }

    /// Adds `deferred` to the thread-local bag.
    ///
    /// # Safety
    ///
    /// It should be safe for another thread to execute the given function.
    pub unsafe fn defer(&self, garbage: Garbage, guard: &Guard) {
        let garbages = &mut *self.garbages.get();
        garbages.push(garbage);

        if garbages.len() > Self::MAX_GARBAGES {
            self.collect(guard);
        }
    }

    pub fn collect(&self, guard: &Guard) {
        let hazards = self.global().collect_hazards(guard);

        let garbages = unsafe { &mut *self.garbages.get() };
        *garbages = garbages
            .drain(..)
            .filter_map(|g| {
                if g.is_hazardous(hazards.as_ref()) {
                    Some(g)
                } else {
                    g.dispose();
                    None
                }
            })
            .collect();
    }

    pub fn flush(&self, guard: &Guard) {
        let garbages = unsafe { &mut *self.garbages.get() };
        let mut bag = Bag::new();

        for mut garbage in garbages.drain(..) {
            while let Err(g) = unsafe { bag.try_push(garbage) } {
                self.global().push_bag(&mut bag);
                garbage = g;
            }

            if !bag.is_empty() {
                self.global().push_bag(&mut bag);
            }
        }

        let _ = self.global().collect(guard);
    }

    /// Pins the `Local`.
    #[inline]
    pub fn pin(&self) -> Guard {
        let guard = Guard { local: self };

        let guard_count = self.guard_count.get();
        self.guard_count.set(guard_count.checked_add(1).unwrap());

        if guard_count == 0 {
            // Loads the current local status. It's safe not to protect the access because no other
            // threads are modifying it.
            let local_status = unsafe { self.status.load(Ordering::Relaxed, unprotected()) };
            let local_flags = StatusFlags::from_bits_truncate(local_status.tag());
            debug_assert!(
                !local_flags.is_pinned(),
                "[Local::pin()] `self` should be unpinned"
            );

            // Increment the pin counter.
            let pin_count = self.pin_count.get();
            self.pin_count.set(pin_count + Wrapping(1));

            // After every `PINNINGS_BETWEEN_COLLECT` try collecting some old garbage bags.
            if pin_count.0 % Self::PINNINGS_BETWEEN_COLLECT == 0 {
                let _ = self.global().collect(&guard);
            }
        }

        guard
    }

    /// Unpins the `Local`.
    #[inline]
    pub fn unpin(&self) {
        let guard_count = self.guard_count.get();
        debug_assert_ne!(guard_count, 0, "[Local::unpin()] guard count should be > 0");

        if guard_count == 1 {
            unsafe {
                // We don't need to be protected because we're not accessing the shared memory.
                let guard = unprotected();

                // Loads the current status.
                let status = self.status.load(Ordering::Acquire, guard);
                let flags = StatusFlags::from_bits_truncate(status.tag());

                // Unpins `self` if it's not already unpinned.
                if flags.is_pinned() {
                    // Creates a summary of the set of hazard pointers.
                    let new_status = repeat_iter(|| self.hazards.make_summary(true, guard))
                        // `ShieldError` is impossible with the `unprotected()` guard.
                        .unwrap()
                        .map(|summary| Owned::new(CachePadded::new(summary)).into_shared(guard))
                        .unwrap_or_else(|| Shared::null())
                        .with_tag(StatusFlags::new(false, false, flags.epoch()).bits());

                    // Replaces `self.status` with the new status.
                    let old_status = self.status.swap(new_status, Ordering::AcqRel, guard);

                    // Defers to destroy the old summary with a "fake" guard, and returns the new
                    // status.
                    if !old_status.is_null() {
                        let guard = Guard { local: self };
                        guard.defer_destroy(old_status);
                        mem::forget(guard);
                    }
                }
            }
        }

        self.guard_count.set(guard_count - 1);

        if self.handle_count.get() == 0 {
            self.finalize();
        }
    }

    /// Increments the handle count.
    #[inline]
    pub fn acquire_handle(&self) {
        let handle_count = self.handle_count.get();
        self.handle_count.set(handle_count + 1);
    }

    /// Decrements the handle count.
    #[inline]
    pub fn release_handle(&self) {
        let guard_count = self.guard_count.get();
        let handle_count = self.handle_count.get();
        debug_assert!(handle_count >= 1);
        self.handle_count.set(handle_count - 1);

        if guard_count == 0 && handle_count == 1 {
            self.finalize();
        }
    }

    /// Removes the `Local` from the global linked list.
    #[cold]
    fn finalize(&self) {
        debug_assert_eq!(self.guard_count.get(), 0);
        debug_assert_eq!(self.handle_count.get(), 0);

        // Temporarily increment handle count. This is required so that the following call to `pin`
        // doesn't call `finalize` again.
        self.handle_count.set(1);
        let guard = Guard { local: self };
        {
            // Flushes the local garbages.
            self.flush(&guard);

            // Defers to destroy the local summary.
            let local_status = self.status.load(Ordering::Relaxed, &guard);
            if !local_status.is_null() {
                unsafe {
                    guard.defer_destroy(local_status);
                }
            }
        }
        // Revert the handle count back to zero.
        mem::forget(guard);
        self.handle_count.set(0);

        unsafe {
            // Take the reference to the `Global` out of this `Local`. Since we're not protected
            // by a guard at this time, it's crucial that the reference is read before marking the
            // `Local` as deleted.
            let collector: Collector = ptr::read(&*(*self.collector.get()));

            // Mark this node in the linked list as deleted.
            self.entry.delete();

            // Finally, drop the reference to the global. Note that this might be the last reference
            // to the `Global`. If so, the global data will be destroyed and all deferred functions
            // in its queue will be executed.
            drop(collector);
        }
    }
}

impl IsElement<Local> for Local {
    fn entry_of(local: &Local) -> &Entry {
        let entry_ptr = (local as *const Local as usize + offset_of!(Local, entry)) as *const Entry;
        unsafe { &*entry_ptr }
    }

    unsafe fn element_of(entry: &Entry) -> &Local {
        // offset_of! macro uses unsafe, but it's unnecessary in this context.
        #[allow(unused_unsafe)]
        let local_ptr = (entry as *const Entry as usize - offset_of!(Local, entry)) as *const Local;
        &*local_ptr
    }

    unsafe fn finalize(entry: &Entry, guard: &Guard) {
        guard.defer_destroy(Shared::from(Self::element_of(entry) as *const _));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use deferred::Deferred;

    #[test]
    fn check_defer() {
        static FLAG: AtomicUsize = AtomicUsize::new(0);
        fn set() {
            FLAG.store(42, Ordering::Relaxed);
        }

        let d = Deferred::new(set);
        assert_eq!(FLAG.load(Ordering::Relaxed), 0);
        d.call();
        assert_eq!(FLAG.load(Ordering::Relaxed), 42);
    }
}
