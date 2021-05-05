import numpy as np
import copy


class Restaurant:
    r"""
    Restaurant class that models a restaurant given by a name, a location, a preparation time distribution and the
    corresponding mean preparation time.
    """

    def __init__(self, name, location, prep_time_generator, avg_prep_time):
        """
        Initialize the restaurant.

        Params
        =======
            name (string):  Name of the restaurant.
            location (2D array): Location of the restaurant.
                                   when orders are ready for pick-up.
            prep_time_generator (callable): Preparation time distribution.
            avg_prep_time (float): Mean preparation time.

        Attributes
        ===========
            queue (List): List of orders given by customer names.
            time_queue (List): List of exact times when orders are ready for pick-up.
            est_time_queue (List): List of estimated times (based on mean preparation times)
                                   when orders are ready for pick-up.
            storage (List): Name of customers of the finished orders.

        """
        self.name = name
        assert isinstance(location, np.ndarray)
        self.location = location
        self.queue = []
        self.time_queue = []
        self.est_time_queue = []
        self.storage = []
        assert callable(prep_time_generator)
        self.prep_time_generator = prep_time_generator
        self.avg_prep_time = avg_prep_time

    def full_deepcopy(self):
        r"""
        Returns a copy of the restaurant. Faster than copy.deepcopy.

        Returns
        ========
        Copy of self.
        """
        result = Restaurant(name=self.name,
                            location=self.location,
                            prep_time_generator=self.prep_time_generator,
                            avg_prep_time=self.avg_prep_time)
        result.queue = copy.deepcopy(self.queue)
        result.time_queue = copy.deepcopy(self.time_queue)
        result.est_time_queue = copy.deepcopy(self.est_time_queue)
        result.storage = copy.deepcopy(self.storage)
        return result

    def add_to_queue(self, order, current_time):
        r"""
        Adds an order (Customer) to the 'queue'. Adds the corresponding time it takes to prepare the
        order to the 'time_queue'.

        Arguments
        ==========
        order (str): Name of a customer.
        current_time (int): Current time.

        Returns
        ========
        None.

        """
        self.queue.append(order)
        processing_time = self.prep_time_generator()
        if len(self.time_queue) == 0:
            self.time_queue.append(current_time + processing_time)
            self.est_time_queue.append(current_time + self.avg_prep_time)
        else:
            self.time_queue.append(self.time_queue[-1] + processing_time)
            self.est_time_queue.append(self.est_time_queue[-1] + self.avg_prep_time)

    def check_queue(self, current_time):
        r"""
        Checks for orders in the queue that are ready. Prepared orders are added to the storage.

        Arguments
        ==========
        current_time (int): Current time.

        Returns
        ========
        None.

        """
        for time in self.time_queue:
            if time <= current_time:
                index = self.time_queue.index(time)
                self.storage.append(self.queue.pop(index))
                self.time_queue.pop(index)
                self.est_time_queue.pop(index)
            else:
                break

    def take_from_storage(self, order):
        r"""
        Checks if an order is in the storage and returns it fom the storage.

        Arguments
        ==========
        order (str): Name of a customer.

        Returns
        ========
        Customer name (str) if order in storage. None else.

        """
        if order in self.storage:
            self.storage.remove(order)
            return order
        else:
            return None

    def get_waiting_time(self, orders, current_time):
        r"""
        Returns the exact waiting time until a given list of orders is finished.

        Arguments
        ==========
        orders (List): List of customer names.
        current_time (int): Current time.

        Returns
        ========
        Estimated time (float) when the list of orders are all prepared.

        """
        orders = [order for order in orders if order not in self.storage]
        if len(orders) == 0:
            return 0
        max_index = max([self.queue.index(order) for order in orders])
        return self.time_queue[max_index] - current_time
