use std::cell::UnsafeCell;

///////////////////
// UnsafeSlice
///////////////////

/// Невладеющая ссылка на слайс с возможностью записи
#[derive(Copy, Clone)]
pub struct UnsafeSlice<'a, T> {
    slice: &'a [UnsafeCell<T>],
}

unsafe impl<'a, T: Send + Sync> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }

    /// SAFETY: нельзя параллельно писать по одному и тому же индексу.
    pub unsafe fn write(&self, i: usize, value: T) {
        let ptr = self.slice[i].get();
        *ptr = value;
    }
}

///////////////////
// Random
///////////////////

pub struct Random {
    state: u32,
}

impl Random {
    pub fn new(initial_state: u32) -> Self {
        assert!(initial_state > 0);
        Self {
            state: initial_state,
        }
    }

    /// Xorshift by George Marsaglia
    pub fn next(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }

    pub fn next_in_range(&mut self, from: i32, to: i32) -> i32 {
        (self.next() as i64 % (to as i64 - from as i64) + from as i64) as i32
    }

    #[allow(dead_code)]
    pub fn next_vec_in_range(&mut self, len: usize, from: i32, to: i32) -> Vec<i32> {
        (0..len).map(|_| self.next_in_range(from, to)).collect()
    }

    pub fn next_vec(&mut self, len: usize) -> Vec<i32> {
        (0..len).map(|_| self.next() as i32).collect()
    }
}
