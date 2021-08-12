from random import shuffle

import numpy as np


class CoverLine:
    """
    Uses the minimum number of lines to cover all zeros in the cost matrix. Algorithm adapted from:
    https://econweb.ucsd.edu/~v2crawford/hungar.pdf
    """
    def __init__(self, cost_matrix, shape):
        """
        Initiates cover line class by converting cost matrix to boolean matrix, allocating all locations in matrix
        with 0's as being True
        Parameters
        ----------
        cost_matrix : passed from Hungarian class
        shape : shape of cost matrix
        """
        self.zero_locations = cost_matrix == 0
        self.shape = shape
        self.choices = np.zeros(self.shape, dtype=bool)
        self.marked_rows = []
        self.marked_columns = []
        self.calculate_choices()
        self.covered_columns = self.marked_columns

    def __mark_columns_with_zeros(self):
        """
        Marks all columns not marked with choices in marked rows
        Returns
        -------
        Number of newly marked columns
        """
        number_marked_columns = 0
        for index, column in enumerate(np.transpose(self.zero_locations)):
            if index not in self.marked_columns and column.any():
                row_indices = np.where(column)
                if (set(self.marked_rows) & set(row_indices)) != set([]):
                    self.marked_columns.append(index)
                    number_marked_columns += 1
        return number_marked_columns

    def __mark_rows_with_zeros(self):
        """
        Marks all rows not marked with choices in marked columns
        Returns
        -------
        Number of newly marked rows
        """
        number_marked_rows = 0
        for index, row in enumerate(self.choices):
            if index not in self.marked_rows and row.any():
                column_indices = np.where(row)
                if column_indices in self.marked_columns:
                    self.marked_rows.append(index)
                    number_marked_rows += 1
        return number_marked_rows

    def __choice_marked_columns(self):
        """
        Determines if there is a choice in all marked columns
        Returns
        -------
        True if there is a choice, otherwise, False
        """
        for marked_column in self.marked_columns:
            if not self.choices[:, marked_column].any():
                return False
        return True

    def __find_marked_column(self):
        """
        Finds marked column with no choice
        Returns
        -------
        Marked column
        """
        for marked_column in self.marked_columns:
            if not self.choices[:, marked_column].any():
                return marked_column

    def __find_row(self, choice_marked_column):
        """
        Finds a row without a choice for the marked column index
        Parameters
        ----------
        choice_marked_column : Marked column index

        Returns
        -------
        Row without a choice, otherwise, None if not found
        """
        row_indices, = np.where(self.zero_locations[:, choice_marked_column])
        for row_index in row_indices:
            if not self.choices[row_index].any():
                return row_index
        return None

    def __find_optimal(self, choice_marked_column):
        """
        Tries to find optimal row and column index so that column swap is ideal
        Parameters
        ----------
        choice_marked_column : Marked column

        Returns
        -------
        Optimal row and column index, otherwise, a random mutation if no optimal solution found
        """
        row_indices, _ = np.where(self.zero_locations[:, choice_marked_column])
        for row_index in row_indices:
            column_index = np.where(self.choices[row_index])[0]
            if self.__find_row(column_index):
                return row_index, column_index

        # Random mutation as no optimal row and column found
        shuffle(row_indices)
        column_indices, _ = np.where(self.choices[row_indices[0]])
        return row_indices[0], column_indices[0]

    def calculate_choices(self):
        """
        Main function of cover line algorithm. Calculates minimum number of lines necessary to cover all zeros in a
        matrix.
        Returns
        -------
        True if solution found
        """
        while True:
            # Reset marked rows and columns
            self.marked_rows = []
            self.marked_columns = []

            # Find rows with no choices made
            for index, row in enumerate(self.choices):
                if not row.any():
                    self.marked_rows.append(index)

            # Finish if no marked rows
            if not self.marked_rows:
                return True

            # Mark new columns with zeros in marked rows and finish if none
            if self.__mark_columns_with_zeros() == 0:
                return True

            # While still choices
            while self.__choice_marked_columns():
                # Finish if number of marked rows is 0
                if self.__mark_rows_with_zeros() == 0:
                    return True
                # Finish if number of marked columns is 0
                if self.__mark_columns_with_zeros() == 0:
                    return True

            # Find marked column with no choice
            choice_marked_column = self.__find_marked_column()

            # Loop while finding marked column with no choice
            while choice_marked_column is not None:
                # Find 0 in marked column without a row with a choice
                choice_marked_row = self.__find_row(choice_marked_column)

                update_choice_marked_column = None
                if choice_marked_row is None:
                    # Find row to swap with its accompanying column
                    choice_marked_row, update_choice_marked_column = self.__find_optimal(choice_marked_column)
                    # Delete old choices
                    self.choices[choice_marked_row, update_choice_marked_column] = False

                self.choices[choice_marked_row, choice_marked_column] = True
                choice_marked_column = update_choice_marked_column

    def get_rows(self):
        """
        Gets covered rows
        Returns
        -------
        Covered rows
        """
        return list(set(range(self.shape[0])) - set(self.marked_rows))

    def get_columns(self):
        """
        Get covered columns
        Returns
        -------
        Covered columns
        """
        return self.marked_columns
