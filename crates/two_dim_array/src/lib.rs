use std::slice::SliceIndex;

/// A two-dimensional view of an underlying one-dimensional
/// buffer. Rows are considered contiguous.
///
/// Non-contiguous slicing is not supported as that requires
/// copying memory and returning a new object whereas this is
/// intended purely as a view of the data which supports mutation.
///
/// The caller must ensure no other mutable references to the
/// same buffer exist while this view is active.
///
/// # Slicing and indexing
///
/// Slicing and indexing through [] are not supported because
/// it cannot be done such that non-contiguous slices are
/// not possible. Instead use `get`, `get_unchecked` and
/// `get_panic`, the latter is equivalent to [] indexing.
/// Each has an `_mut` equivalent. All accept a column
/// index or slice and a row index which guarantees contiguous
/// slicing.
///
/// # Layout
///
/// The array is layed out in row-major order for the sake of
/// indexing and slicing. This results in row slices being
/// contiguous.
///
/// # Example
/// ```
/// use two_dim_array::TwoDimensionalArray;
///
/// let mut buffer = [1, 2, 3, 4];
/// let mut view = TwoDimensionalArray::new(&mut buffer, 2, 2).unwrap();
///
/// assert_eq!(view.shape(), (2, 2));
/// assert_eq!(view.get(0, 1), Some(&2));
/// ```
pub struct TwoDimensionalArray<'a, T> {
    buffer: &'a mut [T],
    num_rows: usize,
    num_cols: usize,
}

impl<'a, T> TwoDimensionalArray<'a, T> {
    /// Construct a `TwoDimensionalArray` from the buffer.
    ///
    /// See `from_mut_slice` for a const initialiser.
    ///
    /// # Errors
    ///
    /// Returns `ShapeError::InvalidShape` when the buffer cannot be arranged
    /// with `num_rows * num_cols`.
    pub fn new(buffer: &'a mut [T], num_rows: usize, num_cols: usize) -> Result<Self, ShapeError> {
        if buffer.len() != num_cols * num_rows {
            Err(ShapeError::InvalidShape {
                buffer_len: buffer.len(),
                num_rows,
                num_cols,
            })
        } else {
            Ok(Self {
                buffer,
                num_rows,
                num_cols,
            })
        }
    }

    /// Update the shape of the TwoDimensionalArray to have `num_rows`, `num_cols`.
    ///
    /// # Errors
    ///
    /// Returns `ShapeError::InvalidShape` when the buffer cannot be reshaped
    /// to the requested shape.
    pub fn reshape(&mut self, num_rows: usize, num_cols: usize) -> Result<(), ShapeError> {
        if self.buffer.len() != num_cols * num_rows {
            Err(ShapeError::InvalidShape {
                buffer_len: self.buffer.len(),
                num_rows,
                num_cols,
            })
        } else {
            self.num_rows = num_rows;
            self.num_cols = num_cols;
            Ok(())
        }
    }

    /// Returns the current shape that the buffer is being viewed as.
    /// Can be updated with `reshape`.
    ///
    /// Return order: `(num_rows, num_cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows, self.num_cols)
    }

    /// The number of rows in the current view of the buffer.
    /// Can be updated with reshape.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The number of columns in the current view of the buffer.
    /// Can be updated with reshape.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Returns the total number of elements in the underlying
    /// slice (`num_rows * num_cols`).
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns whether the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns a reference to an element or row subslice, without doing bounds
    /// checking.
    ///
    /// For a safe alternative see `get`. For a bounds checked alternative see
    /// `get_panic`.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// You can think of this like `.get(index).unwrap_unchecked()`.  It's UB
    /// to call `.get_unchecked(len)`, even if you immediately convert to a
    /// pointer.  And it's UB to call `.get_unchecked(..len + 1)`,
    /// `.get_unchecked(..=len)`, or similar.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    /// unsafe {
    ///     assert_eq!(*x.get_unchecked(0,1), 2);
    ///     assert_eq!(*x.get_unchecked(1,0..2), [3,4])
    /// }
    /// ```
    pub unsafe fn get_unchecked<I>(&self, row_idx: usize, col_idx: I) -> &I::Output
    where
        I: SliceIndex<[T]>,
    {
        unsafe {
            self.buffer
                .get_unchecked(row_idx * self.num_cols..row_idx * self.num_cols + self.num_cols)
                .get_unchecked(col_idx)
        }
    }

    /// Returns a mutable reference to an element or row subslice, without doing bounds
    /// checking.
    ///
    /// For a safe alternative see `get_mut`. For a bounds checked alternative see
    /// `get_mut_panic`.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// You can think of this like `.get_mut(index).unwrap_unchecked()`.  It's UB
    /// to call `.get_unchecked_mut(len)`, even if you immediately convert to a
    /// pointer.  And it's UB to call `.get_unchecked_mut(..len + 1)`,
    /// `.get_unchecked_mut(..=len)`, or similar.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    /// unsafe {
    ///     *x.get_unchecked_mut(0,1) = 42;
    /// }
    /// assert_eq!(a, [1, 42, 3, 4])
    /// ```
    pub unsafe fn get_unchecked_mut<I>(&mut self, row_idx: usize, col_idx: I) -> &mut I::Output
    where
        I: SliceIndex<[T]>,
    {
        unsafe {
            self.buffer
                .get_unchecked_mut(row_idx * self.num_cols..row_idx * self.num_cols + self.num_cols)
                .get_unchecked_mut(col_idx)
        }
    }

    /// Returns a reference to an element or row subslice depending on the type
    /// of index.
    ///
    /// - If given a position, returns a reference to the element at that
    ///   position or `None` if out of bounds.
    /// - If given a column range, returns the row subslice corresponding to
    ///   that range, or `None` if out of bounds.
    ///
    /// See `get_panic` for an equivalent to [] access, which does not return
    /// an option, but does bounds checking and `get_unchecked` which skips
    /// bounds checking.
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    ///
    /// assert_eq!(*x.get(0,1).unwrap(), 2);
    /// assert_eq!(*x.get(1,0..2).unwrap(), [3,4]);
    /// assert!(x.get(1,0..3).is_none());
    ///
    /// ```
    pub fn get<I>(&self, row_idx: usize, col_idx: I) -> Option<&I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.buffer
            .get(row_idx * self.num_cols..row_idx * self.num_cols + self.num_cols)?
            .get(col_idx)
    }

    /// Returns a mutable reference to an element or row subslice depending on the
    /// type of index (see `get`) or `None` if the index is out of bounds.
    ///
    /// See `get_mut_panic` for an equivalent to [] access, which does not return
    /// an option, but does bounds checking and `get_unchecked_mut` which skips
    /// bounds checking.
    ///
    /// # Examples
    ///
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    /// if let Some(elem) = x.get_mut(0,1) {
    ///     *elem = 42;
    /// }
    /// assert_eq!(a, [1, 42, 3, 4]);
    /// ```
    pub fn get_mut<I>(&mut self, row_idx: usize, col_idx: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.buffer
            .get_mut(row_idx * self.num_cols..row_idx * self.num_cols + self.num_cols)?
            .get_mut(col_idx)
    }

    /// Returns a bounds checked, reference to an element or row subslice
    /// depending on the type of col_idx (see `get`). Panics on out of bounds access.
    ///
    /// See `get_unchecked` which skips bounds checking. See `get` which
    /// bounds checks, returning an `Option`.
    ///
    /// # Examples
    ///
    /// In bounds access:
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    /// // Fine
    /// assert_eq!(*x.get_panic(0,1), 2);
    /// assert_eq!(*x.get_panic(1,0..2), [3,4])
    /// ```
    /// Out of bounds access:
    /// ```should_panic
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    /// // Panics
    /// x.get_panic(0,3);
    /// ```
    pub fn get_panic<I>(&self, row_idx: usize, col_idx: I) -> &I::Output
    where
        I: SliceIndex<[T]>,
    {
        &self.buffer[row_idx * self.num_cols..row_idx * self.num_cols + self.num_cols][col_idx]
    }

    /// Returns a bounds checked, mutable reference to an element or row subslice
    /// depending on the type of col_idx (see `get`). Panics on out of bounds access.
    ///
    /// See `get_unchecked_mut` which skips bounds checking. See `get_mut` which
    /// bounds checks, returning an `Option`.
    ///
    /// # Examples
    ///
    /// In bounds access:
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    /// // Fine
    /// *x.get_mut_panic(0,1) = 42;
    /// assert_eq!(a, [1, 42, 3, 4]);
    /// ```
    /// Out of bounds access:
    /// ```should_panic
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    ///
    /// // Panics
    /// *x.get_mut_panic(0,3) = 42;
    /// ```
    pub fn get_mut_panic<I>(&mut self, row_idx: usize, col_idx: I) -> &mut I::Output
    where
        I: SliceIndex<[T]>,
    {
        &mut self.buffer[row_idx * self.num_cols..row_idx * self.num_cols + self.num_cols][col_idx]
    }

    /// Returns an iterator yielding the array slices of the contiguous
    /// rows of the buffer.
    ///
    /// For mutable references see `rows_mut`.
    ///
    /// # Example
    ///
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    /// for row in x.rows() {
    ///   println!("{:?}", row);
    /// }
    /// ```
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.buffer.chunks(self.num_cols)
    }

    /// Returns an iterator yielding mutable references to the array
    /// slices of the contiguous rows of the buffer.
    ///
    /// See also `rows`.
    ///
    /// # Example
    ///
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    /// for row in x.rows_mut() {
    ///   row[0] = 42;
    /// }
    /// ```
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.buffer.chunks_mut(self.num_cols)
    }

    /// Returns a reference to the entire underlying one-dimensional
    /// buffer.
    ///
    /// See also `as_mut_slice`.
    ///
    /// # Example
    ///
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    /// assert!(std::ptr::eq(x.as_slice(), &a))
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.buffer
    }

    /// Returns a mutable reference to the entire underlying one-dimensional
    /// buffer.
    ///
    /// See also `as_slice`.
    ///
    /// # Example
    ///
    /// ```
    /// use two_dim_array::TwoDimensionalArray;
    /// let mut a = [1,2,3,4];
    /// let mut x = TwoDimensionalArray::new(&mut a, 2, 2).unwrap();
    /// x.as_mut_slice()[3] = 42;
    /// assert_eq!(a[3], 42)
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer
    }
}

/// Generic error for trying to assign an impossible shape
/// to `TwoDimensionalArray`.
#[derive(Debug)]
pub enum ShapeError {
    InvalidShape {
        buffer_len: usize,
        num_rows: usize,
        num_cols: usize,
    },
}
impl std::error::Error for ShapeError {}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape {
                buffer_len,
                num_rows,
                num_cols,
            } => f.write_fmt(format_args!(
                "Cannot reshape two dimensional array with number of elements {} into {}x{} array",
                buffer_len, num_rows, num_cols
            )),
        }
    }
}
