from random import shuffle

import numpy as np


class CoverLine:
    def __init__(self, cost_matrix, shape):
        self.zero_locations = cost_matrix == 0
        self.shape = shape
        self.choices = np.zeros(self.shape, dtype=bool)
        self.marked_rows = []
        self.marked_columns = []
        self.calculate_choices()
        self.covered_columns = self.marked_columns

    def __mark_columns_with_zeros(self):
        number_marked_columns = 0
        for index, column in enumerate(np.transpose(self.zero_locations)):
            if index not in self.marked_columns and column.any():
                row_indices = np.where(column)
                if (set(self.marked_rows) & set(row_indices)) != set([]):
                    self.marked_columns.append(index)
                    number_marked_columns += 1
        return number_marked_columns

    def __mark_rows_with_zeros(self):
        number_marked_rows = 0
        for index, row in enumerate(self.choices):
            if index not in self.marked_rows and row.any():
                column_indices = np.where(row)
                if column_indices in self.marked_columns:
                    self.marked_rows.append(index)
                    number_marked_rows += 1
        return number_marked_rows

    def __choice_marked_columns(self):
        for marked_column in self.marked_columns:
            if not self.choices[:, marked_column].any():
                return False
        return True

    def __find_marked_column(self):
        for marked_column in self.marked_columns:
            if not self.choices[:, marked_column].any():
                return marked_column

    def __find_row(self, choice_marked_column):
        row_indices, = np.where(self.zero_locations[:, choice_marked_column])
        for row_index in row_indices:
            if not self.choices[row_index].any():
                return row_index
        return None

    def __find_optimal(self, choice_marked_column):
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
        while True:
            self.marked_rows = []
            self.marked_columns = []

            for index, row in enumerate(self.choices):
                if not row.any():
                    self.marked_rows.append(index)

            if not self.marked_rows:
                return True

            if self.__mark_columns_with_zeros() == 0:
                return True

            while self.__choice_marked_columns():
                if self.__mark_rows_with_zeros() == 0:
                    return True
                if self.__mark_columns_with_zeros() == 0:
                    return True

            choice_marked_column = self.__find_marked_column()

            while choice_marked_column is not None:
                choice_marked_row = self.__find_row(choice_marked_column)

                update_choice_marked_column = None
                if choice_marked_row is None:
                    choice_marked_row, update_choice_marked_column = self.__find_optimal(choice_marked_column)
                    self.choices[choice_marked_row, update_choice_marked_column] = False

                self.choices[choice_marked_row, choice_marked_column] = True
                choice_marked_column = update_choice_marked_column

    def get_rows(self):
        return list(set(range(self.shape[0])) - set(self.marked_rows))

    def get_columns(self):
        return self.marked_columns
