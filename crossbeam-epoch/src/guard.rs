use core::fmt;

use atomic::Shared;
use collector::Collector;
use deferred::Deferred;
use garbage::Garbage;
use internal::Local;

/// A guard that keeps the current thread pinned.
///
/// # Pinning
///
/// The current thread is pinned by calling [`pin`], which returns a new guard:
///
/// ```
/// use crossbeam_epoch as epoch;
///
/// // It is often convenient to prefix a call to `pin` with a `&` in order to create a reference.
/// // This is not really necessary, but makes passing references to the guard a bit easier.
/// let guard = &epoch::pin();
/// ```
///
/// When a guard gets dropped, the current thread is automatically unpinned.
///
/// # Pointers on the stack
///
/// Having a guard allows us to create pointers on the stack to heap-allocated objects.
/// For example:
///
/// ```
/// use crossbeam_epoch::{self as epoch, Atomic, Owned};
/// use std::sync::atomic::Ordering::SeqCst;
///
/// // Create a heap-allocated number.
/// let a = Atomic::new(777);
///
/// // Pin the current thread.
/// let guard = &epoch::pin();
///
/// // Load the heap-allocated object and create pointer `p` on the stack.
/// let p = a.load(SeqCst, guard);
///
/// // Dereference the pointer and print the value:
/// if let Some(num) = unsafe { p.as_ref() } {
///     println!("The number is {}.", num);
/// }
/// ```
///
/// # Multiple guards
///
/// Pinning is reentrant and it is perfectly legal to create multiple guards. In that case, the
/// thread will actually be pinned only when the first guard is created and unpinned when the last
/// one is dropped:
///
/// ```
/// use crossbeam_epoch as epoch;
///
/// let guard1 = epoch::pin();
/// let guard2 = epoch::pin();
/// assert!(epoch::is_pinned());
/// drop(guard1);
/// assert!(epoch::is_pinned());
/// drop(guard2);
/// assert!(!epoch::is_pinned());
/// ```
///
/// [`pin`]: fn.pin.html
pub struct Guard {
    pub(crate) local: *const Local,
}

impl Guard {
    #[inline]
    unsafe fn defer_garbage(&self, garbage: Garbage) {
        if let Some(local) = self.local.as_ref() {
            local.inc_retired(1);
            local.defer(garbage, self);
        } else {
            garbage.dispose();
        }
    }

    /// Stores a function so that it can be executed at some point after all currently pinned
    /// threads get unpinned.
    ///
    /// This method first stores `f` into the thread-local (or handle-local) cache. If this cache
    /// becomes full, some functions are moved into the global cache. At the same time, some
    /// functions from both local and global caches may get executed in order to incrementally
    /// clean up the caches as they fill up.
    ///
    /// There is no guarantee when exactly `f` will be executed. The only guarantee is that it
    /// won't be executed until all currently pinned threads get unpinned. In theory, `f` might
    /// never run, but the epoch-based garbage collection will make an effort to execute it
    /// reasonably soon.
    ///
    /// If this method is called from an [`unprotected`] guard, the function will simply be
    /// executed immediately.
    ///
    /// [`unprotected`]: fn.unprotected.html
    #[inline]
    pub fn defer<F, R>(&self, f: F)
    where
        F: FnOnce() -> R,
        F: Send + 'static,
    {
        unsafe {
            self.defer_unchecked(f);
        }
    }

    /// Stores a function so that it can be executed at some point after all currently pinned
    /// threads get unpinned.
    ///
    /// This method first stores `f` into the thread-local (or handle-local) cache. If this cache
    /// becomes full, some functions are moved into the global cache. At the same time, some
    /// functions from both local and global caches may get executed in order to incrementally
    /// clean up the caches as they fill up.
    ///
    /// There is no guarantee when exactly `f` will be executed. The only guarantee is that it
    /// won't be executed until all currently pinned threads get unpinned. In theory, `f` might
    /// never run, but the epoch-based garbage collection will make an effort to execute it
    /// reasonably soon.
    ///
    /// If this method is called from an [`unprotected`] guard, the function will simply be
    /// executed immediately.
    ///
    /// # Safety
    ///
    /// The given function must not hold reference onto the stack. It is highly recommended that
    /// the passed function is **always** marked with `move` in order to prevent accidental
    /// borrows.
    ///
    /// ```
    /// use crossbeam_epoch as epoch;
    ///
    /// let guard = &epoch::pin();
    /// let message = "Hello!";
    /// unsafe {
    ///     // ALWAYS use `move` when sending a closure into `defer_unchecked`.
    ///     guard.defer_unchecked(move || {
    ///         println!("{}", message);
    ///     });
    /// }
    /// ```
    ///
    /// Apart from that, keep in mind that another thread may execute `f`, so anything accessed by
    /// the closure must be `Send`.
    ///
    /// We intentionally didn't require `F: Send`, because Rust's type systems usually cannot prove
    /// `F: Send` for typical use cases. For example, consider the following code snippet, which
    /// exemplifies the typical use case of deferring the deallocation of a shared reference:
    ///
    /// ```ignore
    /// let shared = Owned::new(7i32).into_shared(guard);
    /// guard.defer_unchecked(move || shared.into_owned()); // `Shared` is not `Send`!
    /// ```
    ///
    /// While `Shared` is not `Send`, it's safe for another thread to call the deferred function,
    /// because it's called only after the grace period and `shared` is no longer shared with other
    /// threads. But we don't expect type systems to prove this.
    ///
    /// # Examples
    ///
    /// When a heap-allocated object in a data structure becomes unreachable, it has to be
    /// deallocated. However, the current thread and other threads may be still holding references
    /// on the stack to that same object. Therefore it cannot be deallocated before those references
    /// get dropped. This method can defer deallocation until all those threads get unpinned and
    /// consequently drop all their references on the stack.
    ///
    /// ```
    /// use crossbeam_epoch::{self as epoch, Atomic, Owned};
    /// use std::sync::atomic::Ordering::SeqCst;
    ///
    /// let a = Atomic::new("foo");
    ///
    /// // Now suppose that `a` is shared among multiple threads and concurrently
    /// // accessed and modified...
    ///
    /// // Pin the current thread.
    /// let guard = &epoch::pin();
    ///
    /// // Steal the object currently stored in `a` and swap it with another one.
    /// let p = a.swap(Owned::new("bar").into_shared(guard), SeqCst, guard);
    ///
    /// if !p.is_null() {
    ///     // The object `p` is pointing to is now unreachable.
    ///     // Defer its deallocation until all currently pinned threads get unpinned.
    ///     unsafe {
    ///         // ALWAYS use `move` when sending a closure into `defer_unchecked`.
    ///         guard.defer_unchecked(move || {
    ///             println!("{} is now being deallocated.", p.deref());
    ///             // Now we have unique access to the object pointed to by `p` and can turn it
    ///             // into an `Owned`. Dropping the `Owned` will deallocate the object.
    ///             drop(p.into_owned());
    ///         });
    ///     }
    /// }
    /// ```
    ///
    /// [`unprotected`]: fn.unprotected.html
    pub unsafe fn defer_unchecked<F, R>(&self, f: F)
    where
        F: FnOnce() -> R,
    {
        self.defer_garbage(Garbage::Deferred {
            inner: Deferred::new(move || drop(f())),
        });
    }

    /// Stores a destructor for an object so that it can be deallocated and dropped at some point
    /// after all currently pinned threads get unpinned.
    ///
    /// This method first stores the destructor into the thread-local (or handle-local) cache. If
    /// this cache becomes full, some destructors are moved into the global cache. At the same
    /// time, some destructors from both local and global caches may get executed in order to
    /// incrementally clean up the caches as they fill up.
    ///
    /// There is no guarantee when exactly the destructor will be executed. The only guarantee is
    /// that it won't be executed until all currently pinned threads get unpinned. In theory, the
    /// destructor might never run, but the epoch-based garbage collection will make an effort to
    /// execute it reasonably soon.
    ///
    /// If this method is called from an [`unprotected`] guard, the destructor will simply be
    /// executed immediately.
    ///
    /// # Safety
    ///
    /// The object must not be reachable by other threads anymore, otherwise it might be still in
    /// use when the destructor runs.
    ///
    /// Apart from that, keep in mind that another thread may execute the destructor, so the object
    /// must be sendable to other threads.
    ///
    /// We intentionally didn't require `T: Send`, because Rust's type systems usually cannot prove
    /// `T: Send` for typical use cases. For example, consider the following code snippet, which
    /// exemplifies the typical use case of deferring the deallocation of a shared reference:
    ///
    /// ```ignore
    /// let shared = Owned::new(7i32).into_shared(guard);
    /// guard.defer_destroy(shared); // `Shared` is not `Send`!
    /// ```
    ///
    /// While `Shared` is not `Send`, it's safe for another thread to call the destructor, because
    /// it's called only after the grace period and `shared` is no longer shared with other
    /// threads. But we don't expect type systems to prove this.
    ///
    /// # Examples
    ///
    /// When a heap-allocated object in a data structure becomes unreachable, it has to be
    /// deallocated. However, the current thread and other threads may be still holding references
    /// on the stack to that same object. Therefore it cannot be deallocated before those references
    /// get dropped. This method can defer deallocation until all those threads get unpinned and
    /// consequently drop all their references on the stack.
    ///
    /// ```
    /// use crossbeam_epoch::{self as epoch, Atomic, Owned};
    /// use std::sync::atomic::Ordering::SeqCst;
    ///
    /// let a = Atomic::new("foo");
    ///
    /// // Now suppose that `a` is shared among multiple threads and concurrently
    /// // accessed and modified...
    ///
    /// // Pin the current thread.
    /// let guard = &epoch::pin();
    ///
    /// // Steal the object currently stored in `a` and swap it with another one.
    /// let p = a.swap(Owned::new("bar").into_shared(guard), SeqCst, guard);
    ///
    /// if !p.is_null() {
    ///     // The object `p` is pointing to is now unreachable.
    ///     // Defer its deallocation until all currently pinned threads get unpinned.
    ///     unsafe {
    ///         guard.defer_destroy(p);
    ///     }
    /// }
    /// ```
    ///
    /// [`unprotected`]: fn.unprotected.html
    pub unsafe fn defer_destroy<T>(&self, ptr: Shared<T>) {
        unsafe fn dtor<T>(data: usize) {
            drop(Shared::from(data as *const T).into_owned());
        }

        self.defer_garbage(Garbage::Destroy {
            data: ptr.as_raw() as usize,
            dtor: dtor::<T>,
        });
    }

    pub fn collect(&self) {
        if let Some(local) = unsafe { self.local.as_ref() } {
            local.collect(self);
        }
    }

    /// Clears up the thread-local cache of deferred functions by executing them or moving into the
    /// global cache.
    ///
    /// Call this method after deferring execution of a function if you want to get it executed as
    /// soon as possible. Flushing will make sure it is residing in in the global cache, so that
    /// any thread has a chance of taking the function and executing it.
    ///
    /// If this method is called from an [`unprotected`] guard, it is a no-op (nothing happens).
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_epoch as epoch;
    ///
    /// let guard = &epoch::pin();
    /// unsafe {
    ///     guard.defer(move || {
    ///         println!("This better be printed as soon as possible!");
    ///     });
    /// }
    /// guard.flush();
    /// ```
    ///
    /// [`unprotected`]: fn.unprotected.html
    pub fn flush(&self) {
        if let Some(local) = unsafe { self.local.as_ref() } {
            local.flush(self);
        }
    }

    /// Returns the `Collector` associated with this guard.
    ///
    /// This method is useful when you need to ensure that all guards used with
    /// a data structure come from the same collector.
    ///
    /// If this method is called from an [`unprotected`] guard, then `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_epoch as epoch;
    ///
    /// let mut guard1 = epoch::pin();
    /// let mut guard2 = epoch::pin();
    /// assert!(guard1.collector() == guard2.collector());
    /// ```
    ///
    /// [`unprotected`]: fn.unprotected.html
    pub fn collector(&self) -> Option<&Collector> {
        unsafe { self.local.as_ref().map(|local| local.collector()) }
    }
}

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        if let Some(local) = unsafe { self.local.as_ref() } {
            local.unpin();
        }
    }
}

impl fmt::Debug for Guard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Guard { .. }")
    }
}

/// Returns a reference to a dummy guard that allows unprotected access to [`Atomic`]s.
///
/// This guard should be used in special occasions only. Note that it doesn't actually keep any
/// thread pinned - it's just a fake guard that allows loading from [`Atomic`]s unsafely.
///
/// Note that calling [`defer`] with a dummy guard will not defer the function - it will just
/// execute the function immediately.
///
/// # Safety
///
/// Loading and dereferencing data from an [`Atomic`] using this guard is safe only if the
/// [`Atomic`] is not being concurrently modified by other threads.
///
/// # Examples
///
/// ```
/// use crossbeam_epoch::{self as epoch, Atomic};
/// use std::sync::atomic::Ordering::Relaxed;
///
/// let a = Atomic::new(7);
///
/// unsafe {
///     // Load `a` without pinning the current thread.
///     a.load(Relaxed, epoch::unprotected());
///
///     epoch::unprotected().defer(move || {
///         println!("This gets executed immediately.");
///     });
///
///     // Dropping `dummy` doesn't affect the current thread - it's just a noop.
/// }
/// ```
///
/// The most common use of this function is when constructing or destructing a data structure.
///
/// For example, we can use a dummy guard in the destructor of a Treiber stack because at that
/// point no other thread could concurrently modify the [`Atomic`]s we are accessing.
///
/// If we were to actually pin the current thread during destruction, that would just unnecessarily
/// delay garbage collection and incur some performance cost, so in cases like these `unprotected`
/// is very helpful.
///
/// ```
/// use crossbeam_epoch::{self as epoch, Atomic};
/// use std::mem::ManuallyDrop;
/// use std::sync::atomic::Ordering::Relaxed;
///
/// struct Stack<T> {
///     head: Atomic<Node<T>>,
/// }
///
/// struct Node<T> {
///     data: ManuallyDrop<T>,
///     next: Atomic<Node<T>>,
/// }
///
/// impl<T> Drop for Stack<T> {
///     fn drop(&mut self) {
///         unsafe {
///             // Unprotected load.
///             let mut node = self.head.load(Relaxed, epoch::unprotected());
///
///             while let Some(n) = node.as_ref() {
///                 // Unprotected load.
///                 let next = n.next.load(Relaxed, epoch::unprotected());
///
///                 // Take ownership of the node, then drop its data and deallocate it.
///                 let mut o = node.into_owned();
///                 ManuallyDrop::drop(&mut o.data);
///                 drop(o);
///
///                 node = next;
///             }
///         }
///     }
/// }
/// ```
///
/// [`Atomic`]: struct.Atomic.html
/// [`defer`]: struct.Guard.html#method.defer
#[inline]
pub unsafe fn unprotected() -> &'static mut Guard {
    // HACK(stjepang): An unprotected guard is just a `Guard` with its field `local` set to null.
    // Since this function returns a `'static` reference to a `Guard`, we must return a reference
    // to a global guard. However, it's not possible to create a `static` `Guard` because it does
    // not implement `Sync`. To get around the problem, we create a static `usize` initialized to
    // zero and then transmute it into a `Guard`. This is safe because `usize` and `Guard`
    // (consisting of a single pointer) have the same representation in memory.
    static UNPROTECTED: usize = 0;
    &mut *(&UNPROTECTED as *const _ as *const _ as *mut Guard)
}
