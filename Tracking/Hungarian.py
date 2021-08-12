import numpy as np

from CoverLine import CoverLine


def pad_cost_matrix(cost_matrix):
    """
    Pads cost matrix if not square
    Parameters
    ----------
    cost_matrix : Square or non square matrix

    Returns
    -------
    Padded cost matrix
    """
    if cost_matrix.shape[1] == cost_matrix.shape[0]:
        return cost_matrix
    cost_shape = np.shape(cost_matrix)
    padded_cost = np.zeros((max(cost_shape[0], cost_shape[1]), max(cost_shape[0], cost_shape[1])))
    padded_cost[:cost_shape[0], :cost_shape[1]] = cost_matrix
    return padded_cost


def __mark_rows_columns(marked_rows, marked_columns, row_index, column_index):
    """
    Checks if row or columns is marked, and marks them if not
    Parameters
    ----------
    marked_rows : All marked rows
    marked_columns : All marked columns
    row_index : Row index to check
    column_index : Column index to check

    Returns
    -------
    Updated rows and columns
    """
    new_marked_rows, new_marked_columns = marked_rows, marked_columns
    if not (marked_rows == row_index).any() and not (marked_columns == column_index).any():
        new_marked_rows = np.insert(marked_rows, len(marked_rows), row_index)
        new_marked_columns = np.insert(marked_columns, len(marked_columns), column_index)
    return new_marked_rows, new_marked_columns


def find_matches(zero_locations):
    """
    Marks all rows and columns with matches in them
    Parameters
    ----------
    zero_locations : All rows and columns with matches

    Returns
    -------
    Marked rows and columns
    """
    marked_rows, marked_columns = np.array([], dtype=int), np.array([], dtype=int)
    for index, row in enumerate(zero_locations):
        row_index = np.array([index])
        if np.sum(row) == 1:
            column_index = np.where(row)
            marked_rows, marked_columns = __mark_rows_columns(marked_rows, marked_columns, row_index, column_index)

    for index, column in enumerate(np.transpose(zero_locations)):
        column_index = np.array([index])
        if np.sum(column) == 1:
            row_index = np.where(column)
            marked_rows, marked_columns = __mark_rows_columns(marked_rows, marked_columns, row_index, column_index)

    return marked_rows, marked_columns


def get_match(zero_locations):
    """
    Finds row and column with minimum number of zeros in it
    Parameters
    ----------
    zero_locations : All rows and columns with matches

    Returns
    -------
    Row and column with min number of zeros
    """
    rows, columns = np.where(zero_locations)
    zero_count = []
    for index, row in enumerate(rows):
        zero_count.append(np.sum(zero_locations[row]) + np.sum(zero_locations[:, columns[index]]))

    indices = zero_count.index(min(zero_count))
    row = np.array([rows[indices]])
    column = np.array([columns[indices]])

    return row, column


class Hungarian:
    """
    Implementation of Hungarian Algorithm. Used to assign object detections to predicted paths.
    Pads input cost matrix if not square
    """
    def __init__(self, cost_matrix):
        """
        Initialises hungarian algorithm class.
        Cost matrix is an n x m size numpy matrix. Rows = Track location prediction. Columns = Object detections
        Parameters
        ----------
        cost_matrix : Prepared cost matrix from GlobalNearestNeighbour class
        """
        self.cost_matrix = pad_cost_matrix(cost_matrix)
        self.shape = self.cost_matrix.shape
        self.size = len(self.cost_matrix)
        self.results = []

    def __adjust_result_cost(self, result_cost, cover_rows, cover_columns):
        """
        Adjusts cost matrix to ensure it equals covered rows and columns
        Parameters
        ----------
        result_cost : unadjusted cost matrix
        cover_rows : Covered rows
        cover_columns : Covered columns

        Returns
        -------
        Adjusted matrix
        """
        elements = []
        for row_index, row in enumerate(result_cost):
            if row_index not in cover_rows:
                for index, element in enumerate(row):
                    if index not in cover_columns:
                        elements.append(element)

        min_uncovered = min(elements)
        for row in cover_rows:
            result_cost[row] += min_uncovered
        for column in cover_columns:
            result_cost[column] += min_uncovered
        result_cost -= np.ones(self.shape, dtype=int) * min_uncovered
        return result_cost

    def __save_result(self, matches):
        """
        Saves matches to resulting matrix. Ensures results are not out of bounds of input matrix
        Parameters
        ----------
        matches : matched rows and columns
        """
        for match in matches:
            row, column = match
            if row < self.shape[0] and column < self.shape[1]:
                self.results.append((int(row), int(column)))

    def calculate_assignment(self):
        """
        Calculates optimal assigment for object to track pairings
        Returns
        -------
        Optimized object to track assignments
        """
        result_cost = np.copy(self.cost_matrix)

        # Subtracts smallest entries from each row and column
        for index, row in enumerate(result_cost):
            result_cost[index] -= row.min()
        for index, column in enumerate(np.transpose(result_cost)):
            result_cost[:, index] -= column.min()

        # Minimum number of lines to cover all zeros in cost matrix. Ensures rows and columns is equal to size of
        # cost matrix
        total_covered = 0
        while total_covered < self.size:
            cover_line = CoverLine(result_cost, self.shape)
            cover_rows = cover_line.get_rows()
            cover_columns = cover_line.get_columns()
            total_covered = len(cover_rows) + len(cover_columns)
            if total_covered < self.size:
                result_cost = self.__adjust_result_cost(result_cost, cover_rows, cover_columns)

        # Create assignments. Find single zeros in rows and columns
        predicted_results = min(self.shape[1], self.shape[0])
        zero_locations = result_cost == 0
        while len(self.results) != predicted_results:
            if not zero_locations.any():
                raise TypeError

            matched_rows, matched_columns = find_matches(zero_locations)

            total_matched = len(matched_rows) + len(matched_columns)
            if total_matched == 0:
                matched_rows, matched_columns = get_match(zero_locations)

            for row in matched_rows:
                zero_locations[row] = False
            for column in matched_columns:
                zero_locations[:, column] = False

            self.__save_result(zip(matched_rows, matched_columns))
