use crate::utils::UnsafeSlice;
use num::Num;

///////////////////
// Parallel for
///////////////////

/// Параллельно обработать каждый элемент слайса переданной функцией. O(log n) span
pub fn par_for<T: Send>(arr: &mut [T], action: impl Fn(usize, &mut T) + Copy + Sync) {
    par_for_helper(arr, 0, action);
}

fn par_for_helper<T: Send>(arr: &mut [T], l: usize, action: impl Fn(usize, &mut T) + Copy + Sync) {
    const SEQUENTIAL_BLOCK: usize = 4096;
    if arr.len() <= SEQUENTIAL_BLOCK {
        arr.iter_mut()
            .enumerate()
            .for_each(|(i, el)| action(i + l, el));
        return;
    }
    let m = arr.len() / 2;
    let (left, right) = arr.split_at_mut(m);
    let right_left = l + left.len();
    rayon::join(
        || par_for_helper(left, l, action),
        || par_for_helper(right, right_left, action),
    );
}

pub fn blocked_for<T: Send, const BLOCK_SIZE: usize>(
    arr: &mut [T],
    action: impl Fn(usize, &mut [T]) + Copy + Sync,
) {
    blocked_for_helper::<T, BLOCK_SIZE>(arr, 0, arr.len().div_ceil(SCAN_BLOCK_SIZE), action);
}

fn blocked_for_helper<T: Send, const BLOCK_SIZE: usize>(
    arr: &mut [T],
    block_left: usize,
    block_right: usize,
    action: impl Fn(usize, &mut [T]) + Copy + Sync,
) {
    if arr.len() <= BLOCK_SIZE {
        action(block_left, arr);
        return;
    }
    let m = (block_left + block_right) / 2;
    let split_point = (m - block_left) * BLOCK_SIZE;
    let (arr_left, arr_right) = arr.split_at_mut(split_point);
    rayon::join(
        || blocked_for_helper::<T, BLOCK_SIZE>(arr_left, block_left, m, action),
        || blocked_for_helper::<T, BLOCK_SIZE>(arr_right, m, block_right, action),
    );
}

///////////////////
// Map
///////////////////

/// Параллельно преобразовать слайс, используя переданную функцию. O(log n) span
pub fn par_map<T: Send + Sync, R: Send + Default + Clone>(
    arr: &[T],
    mapper: impl Fn(&T) -> R + Copy + Sync,
) -> Vec<R> {
    let mut res: Vec<R> = vec![Default::default(); arr.len()];
    par_map_helper(arr, &mut res, mapper);
    res
}

fn par_map_helper<T: Send + Sync, R: Send>(
    src_arr: &[T],
    result_arr: &mut [R],
    mapper: impl Fn(&T) -> R + Copy + Sync,
) {
    if src_arr.len() <= 4096 {
        result_arr
            .iter_mut()
            .zip(src_arr.iter())
            .for_each(|(res, src)| *res = mapper(src));
        return;
    }
    let m = src_arr.len() / 2;
    let (src_left, src_right) = src_arr.split_at(m);
    let (result_left, result_right) = result_arr.split_at_mut(m);
    rayon::join(
        || par_map_helper(src_left, result_left, mapper),
        || par_map_helper(src_right, result_right, mapper),
    );
}

///////////////////
// Scan
///////////////////

const SCAN_BLOCK_SIZE: usize = 1024 * 4;

/// Параллельно вычислить невключительные префиксные суммы.
/// Написанная реализация имеет O(log^2 n) span.
/// В то же время можно раскомментировать вызов [par_inline_prefix_sums_helper] (убрав рекурсию),
/// тогда будет O(log n) span.
/// Разницы по времени практически нет, зато рекурсивное сведение использует
/// меньше дополнительной памяти и проще для восприятия.
pub fn par_inline_prefix_sums<T: Num + Copy + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= SCAN_BLOCK_SIZE {
        inline_pref_sums(arr);
        return;
    }

    let block_count = arr.len().div_ceil(SCAN_BLOCK_SIZE);
    let mut block_sums: Vec<T> = vec![T::zero(); block_count];

    // Считаем суммы внутри блоков
    let block_sums_unsafe_slice = UnsafeSlice::new(&mut block_sums);
    blocked_for::<_, SCAN_BLOCK_SIZE>(arr, |block_num, block| unsafe {
        block_sums_unsafe_slice.write(block_num, inline_pref_sums(block));
    });

    // Теперь считаем префиксные суммы по блокам.
    // Можно рекурсивно свестись, что даст O(log^2 n) span.
    par_inline_prefix_sums(&mut block_sums);
    // Но можно посчитать суммы блоков за O(log n).
    // par_inline_prefix_sums_helper(&mut block_sums);

    // Наконец, окончательно вычисляем префиксные суммы,
    // добавляя к суммам внутри блоков префиксные суммы по блокам
    let block_sums_ref: &[T] = &block_sums;
    blocked_for::<_, SCAN_BLOCK_SIZE>(arr, |block_num, block| {
        let prev_sum = block_sums_ref[block_num];
        block.iter_mut().for_each(|el| *el = *el + prev_sum);
    });
}

/// Последовательно посчитать невключительные префиксные суммы.
/// Возвращает сумму всех чисел.
fn inline_pref_sums<T: Num + Copy>(arr: &mut [T]) -> T {
    let mut sum = T::zero();
    for el in arr.iter_mut() {
        let el_copy = *el;
        *el = sum;
        sum = sum + el_copy;
    }
    sum
}

/// Вычислить префиксные суммы за O(log n) span.
#[allow(dead_code)]
fn par_inline_prefix_sums_helper<T: Num + Copy + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= SCAN_BLOCK_SIZE {
        inline_pref_sums(arr);
        return;
    }
    let mut partial_sums = vec![T::zero(); 4 * arr.len()];
    let partial_sums_unsafe_slice = UnsafeSlice::new(&mut partial_sums);
    prefix_sums_up(arr, partial_sums_unsafe_slice, 0);
    prefix_sums_down(arr, &partial_sums, T::zero(), 0);
}

fn prefix_sums_up<T: Num + Copy + Send + Sync>(
    arr: &[T],
    partial_sums: UnsafeSlice<T>,
    id: usize,
) -> T {
    if arr.len() == 1 {
        unsafe {
            partial_sums.write(id, arr[0]);
        }
        return arr[0];
    }
    let (left, right) = arr.split_at(arr.len() / 2);
    let (left_sum, right_sum) = rayon::join(
        || prefix_sums_up(left, partial_sums, 2 * id + 1),
        || prefix_sums_up(right, partial_sums, 2 * id + 2),
    );
    let sum = left_sum + right_sum;
    unsafe {
        partial_sums.write(id, sum);
    }
    sum
}

fn prefix_sums_down<T: Num + Copy + Send + Sync>(
    arr: &mut [T],
    partial_sums: &[T],
    left_sum: T,
    id: usize,
) {
    if arr.len() == 1 {
        arr[0] = left_sum;
        return;
    }
    let (left, right) = arr.split_at_mut(arr.len() / 2);
    let right_left_sum = left_sum + partial_sums[2 * id + 1];
    rayon::join(
        || prefix_sums_down(left, partial_sums, left_sum, 2 * id + 1),
        || prefix_sums_down(right, partial_sums, right_left_sum, 2 * id + 2),
    );
}

///////////////////
// Filter
///////////////////

/// Параллельно отфильтровать массив по условию. Возвращает вектор с подходящими элементами.
pub fn par_filter<T: Send + Default + Sync + Copy>(
    arr: &[T],
    condition: impl Fn(&T) -> bool + Copy + Sync,
) -> Vec<T> {
    if arr.is_empty() {
        return vec![];
    }

    let mut mask: Vec<i32> = par_map(arr, |x| if condition(x) { 1 } else { 0 });
    par_inline_prefix_sums(&mut mask);

    let filtered_count =
        *mask.last().unwrap() as usize + if condition(arr.last().unwrap()) { 1 } else { 0 };
    let mut res_arr = vec![T::default(); filtered_count];

    let res_arr_ref = UnsafeSlice::new(&mut res_arr);
    par_for(&mut mask, |i, res_pos| unsafe {
        if condition(&arr[i]) {
            res_arr_ref.write(*res_pos as usize, arr[i]);
        }
    });

    res_arr
}

///////////////////
// Tests
///////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Random;

    #[test]
    fn par_for_test() {
        let mut arr = vec![1, 2, 3, 4, 5];
        par_for(&mut arr, |i, x| *x += 2 * i);
        assert_eq!(vec![1, 4, 7, 10, 13], arr);
    }

    #[test]
    fn par_map_test() {
        let arr = vec![1, 2, 3, 4, 5];
        let new_arr = par_map(&arr, |x| 2 * x);
        assert_eq!(vec![2, 4, 6, 8, 10], new_arr);
    }

    #[test]
    fn inline_pref_sums_test() {
        let mut arr = vec![1, 2, 3, 4, 5];
        let sum = inline_pref_sums(&mut arr);
        assert_eq!(15, sum);
        assert_eq!(vec![0, 1, 3, 6, 10], arr);
    }

    #[test]
    fn par_inline_prefix_sums_test() {
        let mut random = Random::new(3);
        for arr_len in [
            0,
            10,
            SCAN_BLOCK_SIZE,
            12 * SCAN_BLOCK_SIZE,
            SCAN_BLOCK_SIZE * SCAN_BLOCK_SIZE * 3 + 5,
        ] {
            let mut arr = random.next_vec_in_range(arr_len, -100, 100);

            let mut expected = arr.clone();
            inline_pref_sums(&mut expected);

            par_inline_prefix_sums(&mut arr);

            assert_eq!(expected, arr);
        }
    }

    #[test]
    fn par_filter_test() {
        let mut random = Random::new(3);
        for arr_len in [0, 10, SCAN_BLOCK_SIZE * SCAN_BLOCK_SIZE * 3 + 5] {
            let arr = random.next_vec_in_range(arr_len, -100, 100);

            let actual_filtered: Vec<i32> = par_filter(&arr, |&x| x > 0);
            let expected_filtered: Vec<i32> = arr.into_iter().filter(|&x| x > 0).collect();
            assert_eq!(expected_filtered, actual_filtered);
        }
    }
}
