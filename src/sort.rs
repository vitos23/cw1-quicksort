use crate::parallel_primitives::{blocked_for, par_filter};
use rayon::prelude::*;

pub fn sequential_quicksort<T: Ord>(arr: &mut [T]) {
    if arr.is_empty() {
        return;
    }
    let middle = partition(arr);
    let (left, right) = arr.split_at_mut(middle);
    sequential_quicksort(left);
    sequential_quicksort(&mut right[1..]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let mut m = 0;
    for i in 0..arr.len() {
        if arr[i] < arr[arr.len() - 1] {
            arr.swap(i, m);
            m += 1;
        }
    }
    arr.swap(m, arr.len() - 1);
    m
}

/// Максимально простая параллельная реализация быстрой сортировки,
/// обладающая work-ом последовательной реализации
/// (с точностью до константы, если исключить накладные расходы fork-join)
/// и O(n log n) span-ом.
pub fn simple_parallel_quicksort<T: Ord + Send>(arr: &mut [T]) {
    if arr.len() <= 1024 {
        sequential_quicksort(arr);
        return;
    }

    let middle = partition(arr);
    let (left, right) = arr.split_at_mut(middle);
    rayon::join(
        || simple_parallel_quicksort(left),
        || simple_parallel_quicksort(&mut right[1..]),
    );
}

/// Параллельная быстрая сортировка с O(polylog n) span (за исключением копирования).
/// В текущей реализации span = O(log^3 n),
/// но можно получить и O(log^2 n), если изменить [par_inline_prefix_sums]
/// в соответствии с описанным там способом.
///
/// Используются самописные параллельные примитивы.
///
/// Для конкатенации массивов используется последовательный memcpy
/// (при расчете span-а он считается за O(1))
pub fn parallel_quicksort_seq_memcpy<T: Ord + Default + Copy + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= 4096 {
        sequential_quicksort(arr);
        return;
    }

    let (less, eq, greater) = parallel_quicksort_helper(arr);

    arr[0..less.len()].copy_from_slice(&less);
    arr[less.len()..less.len() + eq.len()].copy_from_slice(&eq);
    arr[less.len() + eq.len()..].copy_from_slice(&greater);
}

/// Параллельная быстрая сортировка с O(polylog n) span (за исключением копирования).
/// В текущей реализации span = O(log^3 n),
/// но можно получить и O(log^2 n), если изменить [par_inline_prefix_sums]
/// в соответствии с описанным там способом.
///
/// Используются самописные параллельные примитивы.
///
/// Для конкатенации массивов используется последовательный memcpy
/// (при расчете span-а он считается за O(1)),
/// запущенный параллельно в трех копиях.
pub fn parallel_quicksort_3par_memcpy<T: Ord + Default + Copy + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= 4096 {
        sequential_quicksort(arr);
        return;
    }

    let (less, eq, greater) = parallel_quicksort_helper(arr);

    let (src_less, src_ge) = arr.split_at_mut(less.len());
    let (src_eq, src_greater) = src_ge.split_at_mut(eq.len());

    rayon::join(
        || {
            rayon::join(
                || src_less.copy_from_slice(&less),
                || src_eq.copy_from_slice(&eq),
            )
        },
        || src_greater.copy_from_slice(&greater),
    );
}

/// Параллельная быстрая сортировка с O(polylog n) span.
/// В текущей реализации span = O(log^3 n),
/// но можно получить и O(log^2 n), если изменить [par_inline_prefix_sums]
/// в соответствии с описанным там способом.
///
/// Используются самописные параллельные примитивы.
///
/// Для конкатенации массивов используется memcpy, запущенный параллельно через blocked_for.
/// Поэтому данная реализация имеет поистине полилогарифмический span.
pub fn parallel_quicksort_par_memcpy<T: Ord + Default + Copy + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= 4096 {
        sequential_quicksort(arr);
        return;
    }

    let (less, eq, greater) = parallel_quicksort_helper(arr);

    let (src_less, src_ge) = arr.split_at_mut(less.len());
    let (src_eq, src_greater) = src_ge.split_at_mut(eq.len());

    rayon::join(
        || rayon::join(|| par_copy(src_less, &less), || par_copy(src_eq, &eq)),
        || par_copy(src_greater, &greater),
    );
}

fn par_copy<T: Copy + Send + Sync>(dst: &mut [T], src: &[T]) {
    assert_eq!(dst.len(), src.len());
    const COPY_BLOCK: usize = 4096;
    blocked_for::<_, COPY_BLOCK>(dst, |block_index, dst_block| {
        let from = COPY_BLOCK * block_index;
        dst_block.copy_from_slice(&src[from..from + dst_block.len()]);
    });
}

fn parallel_quicksort_helper<T: Ord + Default + Copy + Send + Sync>(
    arr: &[T],
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let pivot = arr.last().unwrap();

    let mut less = par_filter(arr, |x| x < pivot);
    let eq = par_filter(arr, |x| x == pivot);
    let mut greater = par_filter(arr, |x| x > pivot);

    rayon::join(
        || parallel_quicksort_seq_memcpy(&mut less),
        || parallel_quicksort_seq_memcpy(&mut greater),
    );

    (less, eq, greater)
}

/// Параллельная реализация быстрой сортировки, аналогичная [parallel_quicksort_seq_memcpy],
/// но использующая параллельные примитивы из библиотеки `rayon`.
pub fn rayon_parallel_quicksort<T: Ord + Default + Copy + Send + Sync>(arr: &mut [T]) {
    if arr.len() <= 4096 {
        sequential_quicksort(arr);
        return;
    }

    let pivot = *arr.last().unwrap();

    // Можно написать даже так, однако это выходит за рамки стандартных примитивов:
    // let ((mut less, eq), (mut greater, _)): ((Vec<T>, Vec<T>), (Vec<T>, Vec<T>)) =
    //     arr.par_iter().partition_map(|&x| match x {
    //         x if x < pivot => Left(Left(x)),
    //         x if x > pivot => Right(Left::<T, T>(x)),
    //         _ => Left(Right(x)),
    //     });

    let mut less: Vec<T> = arr
        .par_iter()
        .filter_map(|&x| if x < pivot { Some(x) } else { None })
        .collect();
    let eq: Vec<T> = arr
        .par_iter()
        .filter_map(|&x| if x == pivot { Some(x) } else { None })
        .collect();
    let mut greater: Vec<T> = arr
        .par_iter()
        .filter_map(|&x| if x > pivot { Some(x) } else { None })
        .collect();

    rayon::join(
        || rayon_parallel_quicksort(&mut less),
        || rayon_parallel_quicksort(&mut greater),
    );

    arr[0..less.len()].copy_from_slice(&less);
    arr[less.len()..less.len() + eq.len()].copy_from_slice(&eq);
    arr[less.len() + eq.len()..].copy_from_slice(&greater);
}

///////////////////
// Tests
///////////////////

#[cfg(test)]
mod tests {
    use crate::sort::{
        parallel_quicksort_3par_memcpy, parallel_quicksort_par_memcpy,
        parallel_quicksort_seq_memcpy, rayon_parallel_quicksort, sequential_quicksort,
        simple_parallel_quicksort,
    };
    use crate::utils::Random;

    #[test]
    fn sort_test() {
        let sorters: &[fn(&mut [i32])] = &[
            sequential_quicksort,
            simple_parallel_quicksort,
            parallel_quicksort_seq_memcpy,
            parallel_quicksort_3par_memcpy,
            parallel_quicksort_par_memcpy,
            rayon_parallel_quicksort,
        ];
        for sorter in sorters {
            let mut random = Random::new(3);

            for arr_len in [0, 10, 5000, 300_000] {
                let mut arr = random.next_vec(arr_len);
                let mut expected_arr = arr.clone();
                expected_arr.sort();

                sorter(&mut arr);

                assert_eq!(expected_arr, arr);
            }
        }
    }
}
