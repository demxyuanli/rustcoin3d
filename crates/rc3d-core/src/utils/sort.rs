//! Sort helpers for common patterns found across the codebase.

/// Sort a slice of `(key, count)` pairs by count in **descending** order.
///
/// The count type can be any `Ord` integer (`usize`, `u32`, etc.).
///
/// # Examples
///
/// ```
/// use rc3d_core::utils::sort::sort_by_count_desc;
/// let mut items = vec![("A", 3u32), ("B", 7), ("C", 2)];
/// sort_by_count_desc(&mut items);
/// assert_eq!(items[0], ("B", 7));
/// ```
pub fn sort_by_count_desc<K, C: Ord>(items: &mut [(K, C)]) {
    items.sort_by(|a, b| b.1.cmp(&a.1));
}

/// Sort a slice by a derived integer count in **descending** order.
///
/// The `key` closure extracts the count from each item.
///
/// # Examples
///
/// ```
/// use rc3d_core::utils::sort::sort_by_key_count_desc;
/// struct Item { name: &'static str, count: usize }
/// let mut items = vec![Item { name: "A", count: 3 }, Item { name: "B", count: 7 }];
/// sort_by_key_count_desc(&mut items, |it| it.count);
/// assert_eq!(items[0].name, "B");
/// ```
pub fn sort_by_key_count_desc<T, C: Ord, F: FnMut(&T) -> C>(items: &mut [T], mut key: F) {
    items.sort_by(|a, b| key(b).cmp(&key(a)));
}
