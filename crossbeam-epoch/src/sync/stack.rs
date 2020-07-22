use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use atomic::{Atomic, Owned};
use guard::{unprotected, Guard};
use hazard::{Shield, ShieldError};

/// 's lock-free stack.
///
/// Usable with any number of producers and consumers.
#[derive(Debug)]
pub struct Stack<T> {
    head: Atomic<Node<T>>,
}

#[derive(Debug)]
struct Node<T> {
    data: ManuallyDrop<T>,
    next: Atomic<Node<T>>,
}

impl<T> Stack<T> {
    /// Creates a new, empty stack.
    pub fn new() -> Stack<T> {
        Stack {
            head: Atomic::null(),
        }
    }

    /// Pushes a value on top of the stack.
    pub fn push(&self, t: T) {
        let mut n = Owned::new(Node {
            data: ManuallyDrop::new(t),
            next: Atomic::null(),
        });

        loop {
            let head = self.head.load(Relaxed, unsafe { unprotected() });
            n.next.store(head, Relaxed);

            match self
                .head
                .compare_and_set(head, n, Release, unsafe { unprotected() })
            {
                Ok(_) => break,
                Err(e) => n = e.new,
            }
        }
    }

    /// Attempts to pop the top element from the stack.
    ///
    /// Returns `None` if the stack is empty.
    #[must_use]
    pub fn try_pop(&self, guard: &Guard) -> Result<Option<T>, ShieldError> {
        let mut head_shield = Shield::null(guard);

        loop {
            let head = self.head.load(Acquire, guard);
            head_shield.defend(head, Some(&self.head), guard)?;

            match unsafe { head_shield.as_ref() } {
                Some(h) => {
                    let next = h.next.load(Relaxed, guard);

                    let data = unsafe { ptr::read(&(*h).data) };
                    if self
                        .head
                        .compare_and_set(head, next, Release, guard)
                        .is_ok()
                    {
                        unsafe {
                            guard.defer_destroy(head);
                        }
                        return Ok(Some(ManuallyDrop::into_inner(data)));
                    }
                }
                None => return Ok(None),
            }
        }
    }
}

impl<T> Drop for Stack<T> {
    fn drop(&mut self) {
        let guard = unsafe { unprotected() };
        while self.try_pop(guard).unwrap().is_some() {}
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crossbeam_utils::thread;
    use pin;

    #[test]
    fn smoke() {
        let stack = Stack::new();

        thread::scope(|scope| {
            for _ in 0..10 {
                scope.spawn(|_| {
                    for i in 0..10_000 {
                        stack.push(i);
                        let mut guard = pin();
                        loop {
                            match stack.try_pop(&guard) {
                                Ok(r) => {
                                    assert!(r.is_some());
                                    break;
                                }
                                Err(_) => continue,
                            }
                        }
                    }
                });
            }
        })
        .unwrap();

        let guard = pin();
        assert!(stack.try_pop(&guard).unwrap().is_none());
    }
}
