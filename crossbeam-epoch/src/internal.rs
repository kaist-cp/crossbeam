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
use core::convert::TryInto;
use core::mem::{self, ManuallyDrop};
use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};

use crossbeam_utils::CachePadded;
use membarrier;

use atomic::{Owned, Shared};
use bloom_filter::BloomFilter;
use collector::{Collector, LocalHandle};
use garbage::{Bag, Garbage};
use guard::{unprotected, Guard};
use hazard::{HazardSet, Shield, ShieldError};
use sync::list::{repeat_iter, Entry, IsElement, IterError, List};
use sync::stack::Stack;

/// The width of epoch's representation. In other words, there can be `1 << EPOCH_WIDTH` epochs that
/// are wrapping around.
const EPOCH_WIDTH: u32 = 5;

/// The width of the number of bags.
const BAGS_WIDTH: u32 = 3;

const_assert!(bags_epoch_width; BAGS_WIDTH <= EPOCH_WIDTH);

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

    #[inline]
    fn collect_hazards_inner(&self, guard: &Guard) -> Result<Option<BloomFilter>, IterError> {
        // Heavy fence to synchronize with `Shield::defend()`.
        unsafe {
            membarrier::heavy_membarrier();
        }

        // Creates a summary of the set of hazard pointers.
        let mut hazards: Option<BloomFilter> = None;
        let mut pred = Shield::null(guard);
        let mut curr = Shield::null(guard);

        // TODO(stjepang): `Local`s are stored in a linked list because linked lists are fairly easy
        // to implement in a lock-free manner. However, traversal can be slow due to cache misses
        // and data dependencies. We should experiment with other data structures as well.
        for local in self
            .locals
            .iter(&mut pred, &mut curr, true, guard)
            .map_err(IterError::ShieldError)?
        {
            let local = local?;
            let local = unsafe { &*local };

            // If the corresponding thread is not pinned, ignore it.
            if !local.status.load(Ordering::Acquire) {
                continue;
            }

            // Creates a summary of the set of hazard pointers.
            let local_hazards = repeat_iter(|| local.hazards.make_summary(false, &guard))
                .map_err(IterError::ShieldError)?;
            if let Some(local_hazards) = local_hazards {
                if let Some(hazards) = hazards.as_mut() {
                    hazards.union(&local_hazards);
                }
            }
        }

        Ok(hazards)
    }

    fn collect_hazards(&self, guard: &Guard) -> Result<Option<BloomFilter>, ShieldError> {
        repeat_iter(|| self.collect_hazards_inner(guard))
    }

    #[inline]
    #[must_use]
    pub fn collect<'g>(&'g self, guard: &'g Guard) -> Result<bool, ShieldError> {
        let hazards = self.collect_hazards(guard)?;
        self.collect_inner(hazards.as_ref(), guard)
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
    pub fn collect_inner<'g>(
        &'g self,
        hazards: Option<&BloomFilter>,
        guard: &'g Guard,
    ) -> Result<bool, ShieldError> {
        let steps = if cfg!(feature = "sanitize") {
            usize::max_value()
        } else {
            Self::COLLECT_STEPS
        };

        for _ in 0..steps {
            if let Some(mut bag) = self.bags.try_pop(guard)? {
                // Disposes the garbages (except for hazard pointers) in the bag popped from the
                // global queue.
                let disposed = bag.dispose(hazards);

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

    /// Whether the corresponding thread is accessing the shared memory.
    status: AtomicBool,

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
    collect_count: Cell<usize>,

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
    /// Maximum number of unreclaimed garbages.
    const MAX_GARBAGES: usize = 256;

    /// How often should we call global collection?
    const LOCAL_COLLECTS_BETWEEN_GLOBAL_COLLECT: usize = 8;

    /// Registers a new `Local` in the provided `Global`.
    pub fn register(collector: &Collector) -> LocalHandle {
        unsafe {
            // Since we dereference no pointers in this block, it is safe to use `unprotected`.

            let local = Owned::new(Local {
                entry: Entry::default(),
                status: AtomicBool::new(false),
                collector: UnsafeCell::new(ManuallyDrop::new(collector.clone())),
                hazards: HazardSet::new(),
                garbages: UnsafeCell::new(Vec::new()),
                guard_count: Cell::new(0),
                handle_count: Cell::new(1),
                collect_count: Cell::new(0),
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

        if garbages.len() >= Self::MAX_GARBAGES {
            // We ignore possibly happening shield errors.
            let _ = self.collect(guard);
        }
    }

    #[cold]
    pub fn collect(&self, guard: &Guard) -> Result<(), ShieldError> {
        let hazards = self.global().collect_hazards(guard)?;

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

        // Increment the pin counter.
        let collect_count = self.collect_count.get().wrapping_add(1);
        self.collect_count.set(collect_count);

        if collect_count % Self::LOCAL_COLLECTS_BETWEEN_GLOBAL_COLLECT == 0 {
            let _ = self.global().collect_inner(hazards.as_ref(), guard)?;
        }

        Ok(())
    }

    pub fn flush(&self) {
        let garbages = unsafe { &mut *self.garbages.get() };
        let mut bag = Bag::new();

        for mut garbage in garbages.drain(..) {
            while let Err(g) = unsafe { bag.try_push(garbage) } {
                self.global().push_bag(&mut bag);
                garbage = g;
            }
        }

        if !bag.is_empty() {
            self.global().push_bag(&mut bag);
        }
    }

    /// Pins the `Local`.
    #[inline]
    pub fn pin(&self) -> Guard {
        let guard = Guard { local: self };

        let guard_count = self.guard_count.get();
        self.guard_count.set(guard_count.checked_add(1).unwrap());

        if guard_count == 0 {
            // Sets the corresponding thread as pinned.
            self.status.store(true, Ordering::Relaxed);
        }

        guard
    }

    /// Unpins the `Local`.
    #[inline]
    pub fn unpin(&self) {
        let guard_count = self.guard_count.get();
        debug_assert_ne!(guard_count, 0, "[Local::unpin()] guard count should be > 0");

        if guard_count == 1 {
            // Sets the corresponding thread as unpinned.
            self.status.store(false, Ordering::Release);
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

        // Flushes the local garbages.
        self.flush();

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
