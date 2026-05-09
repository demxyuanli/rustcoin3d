//! Bounded ring buffer built on `VecDeque`.
//!
//! Pushing beyond capacity automatically pops the oldest item.

use std::collections::VecDeque;

/// Fixed-capacity ring buffer. When full, new pushes evict the oldest entry.
pub struct RingBuffer<T> {
    inner: VecDeque<T>,
    cap: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given maximum capacity.
    pub fn new(cap: usize) -> Self {
        Self {
            inner: VecDeque::with_capacity(cap),
            cap,
        }
    }

    /// Push an item, evicting the oldest if at capacity.
    pub fn push(&mut self, item: T) {
        if self.inner.len() >= self.cap {
            self.inner.pop_front();
        }
        self.inner.push_back(item);
    }

    /// Number of items currently stored.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// View all items in insertion order (oldest first).
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.inner.as_slices()
    }

    /// Iterate over items in insertion order.
    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, T> {
        self.inner.iter()
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_within_capacity() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        assert_eq!(rb.len(), 2);
        let items: Vec<_> = rb.iter().copied().collect();
        assert_eq!(items, vec![1, 2]);
    }

    #[test]
    fn push_beyond_capacity_evicts() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.push(4); // evicts 1
        let items: Vec<_> = rb.iter().copied().collect();
        assert_eq!(items, vec![2, 3, 4]);
    }
}
