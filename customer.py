import copy


class Customer:
    r"""
    Customer class that models a customer given by a name, location, order time, list of favorite restaurants, and
    a maximal time that the customer is willing to wait.
    """

    def __init__(self, name, location, favorite_restaurants, time_constraint, dispatcher, order_time):
        r"""
        Initialize the customer.

        Params
        =======
            name (string):  Name of the customer.
            location (2D array): Location of the customer.
            favorite_restaurants (List): List of restaurant names that the customer considers ordering from.
            time_constraint (float): Maximal delivery time the customer is willing to accept. Influences the customers
                                     restaurant choice.
            dispatcher (Dispatcher): Dispatcher that informs the customer with arrival times and takes the order.
            order_time (int): Order time of customer.

        Attributes
        ===========
            day (int): Day at which the customer orders.
            status (int): Encodes whether the customer has not received their delivery yet (0), has received their
                          delivery (1), did not find a feasible restaurant to order from (-1).
            sum_shifted (float): Time that the customers delivery time was shifted to insert other customers in the
                                 tentative route.

        """
        # Customer information
        self.name = name
        self.location = location
        self.order_time = order_time
        self.day = None
        self.favorite_restaurants = favorite_restaurants
        self.time_constraint = time_constraint
        self.dispatcher = dispatcher
        self.status = 0  # 1 if served, -1 if rejected
        self.sum_shifted = 0

        # Possible customer information to write out
        self.etd = None
        self.atd = None
        #self.restaurant = None
        #self.restaurant_est_time_queue = None
        #self.other_restaurant_time_queues = None
        #self.vehicle = None
        #self.vehicle_location = None
        #self.vehicle_started_action_time = None
        #self.old_vehicle_route = None
        #self.new_vehicle_route = None
        #self.vehicle_route_to_customer = None
        #self.other_vehicle_routes = None
        #self.other_vehicle_started_action_times = None
        #self.max_pre_shift = None
        #self.max_post_shift = None
        #self.insertion_status = None  # 0 first action/inserted, 1 bundled orders
        #self.insertion_index = None
        #self.insertion_cost = None
        #self.myopic_etd = None
        #self.simulated_etd = None
        #self.one_step_simul = None
        #self.adv_one_step_simul = None
        #self.simulation_time = None

    def __deepcopy__(self, memo):
        r"""
        Custom deepcopy method to only copy relevant class information.

        Arguments
        ==========
            memo (Dict): Dicitionary of objects already copied during the current copying pass.

        Returns
        ========
            Deep copy of self.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__copydict__().items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __copydict__(self):
        r"""
        Relevant class information to deepcopy.

        Returns
        ========
        Dictionary of information to deepcopy.

        """
        copy_dict = {"name": self.name,
                     "location": self.location,
                     "sum_shifted": self.sum_shifted}
        return copy_dict

    def full_deepcopy(self):
        r"""
        Manual deepcopy that contains more information than the deepcopy method.

        Returns
        ========
        Deepcopy of self containing more information than the copy made by the custom deepcopy method.

        """
        result = Customer(name=self.name,
                          location=self.location,
                          favorite_restaurants=None,
                          time_constraint=self.time_constraint,
                          dispatcher=None,
                          order_time=self.order_time)
        result.status = self.status
        result.sum_shifted = self.sum_shifted
        result.etd = self.etd
        return result

    def order_restaurant(self, etd_dict, current_time, ignore_time_constraint=False):
        r"""
        Places a restaurant order via the dispatcher.
        The first feasible restaurant is chosen from the favorite list.

        Arguments
        ==========
            etd_dict (Dict): Dicitionary with restaurant names as keys and a tuple of estimated delivery time and a
                             a tentative route as value.
            current_time (int): Current time step in minutes.
            ignore_time_constraint (bool): Boolean indicating whether or not the customers delivery time preferences are
                                           ignored.

        Returns
        ========
            Returns a customer status. 0 is returned if the customer ordered and -1 else.

        """
        if ignore_time_constraint:
            self.restaurant = self.favorite_restaurants[0]
            self.dispatcher.take_order(customer=self, restaurant=self.restaurant,
                                       route=etd_dict[self.restaurant.name][1], current_time=current_time)
            self.etd = current_time + etd_dict[self.restaurant.name][0]
            return 0

        for restaurant in self.favorite_restaurants:
            try:
                if etd_dict[restaurant.name][0] < self.time_constraint:
                    self.restaurant = restaurant
                    self.dispatcher.take_order(customer=self, restaurant=restaurant,
                                               route=etd_dict[restaurant.name][1], current_time=current_time)
                    self.etd = current_time + etd_dict[restaurant.name][0]
                    return 0
            except KeyError:
                print([restaurant.name for restaurant in self.favorite_restaurants])
                print(restaurant.name)
                print([key for key, value in etd_dict.items()])
                print(etd_dict)
                raise Warning
        return -1

    def update_status(self):
        r"""
        Updates status to 1 to indicate that the customer has been served.
        """
        self.status = 1

    def to_dict(self):
        r"""
        Summarizes the customer data in a dict and returns it.

        Returns
        ========
        Dictionary with important customer information.
        """
        if self.status == 1:
            customer_dict = {'name': self.name,
                             'status': self.status,
                             'location': self.location.tolist(),
                             'order_time': self.order_time,
                             'favorite_restaurants': [restaurant.name for restaurant in self.favorite_restaurants],
                             'day': self.day,
                             'atd': self.atd,
                             'etd': self.etd,
                             'restaurant_name': self.restaurant.name,
                             #'myopic_etd': self.myopic_etd,
                             #'simulated_etd': self.simulated_etd,
                             #'one_step_simul': self.one_step_simul,
                             #'adv_one_step_simul': self.adv_one_step_simul,
                             #'restaurant_location': self.restaurant.location.tolist(),
                             #'restaurant_queue': self.restaurant_est_time_queue,
                             #'other_restaurant_queues': self.other_restaurant_time_queues,
                             #'vehicle_name': self.vehicle.name,
                             #'vehicle_started_action_time': self.vehicle_started_action_time,
                             #'old_vehicle_route': self.old_vehicle_route,
                             #'new_vehicle_route': self.new_vehicle_route,
                             #'vehicle_route_to_customer': self.vehicle_route_to_customer,
                             #'other_vehicle_routes': self.other_vehicle_routes,
                             #'other_vehicle_started_action_times': self.other_vehicle_started_action_times,
                             #'max_pre_shift': self.max_pre_shift,
                             #'max_post_shift': self.max_post_shift,
                             #'insertion_status': self.insertion_status,
                             #'insertion_index': self.insertion_index,
                             #'insertion_cost': self.insertion_cost,
                             #'simulation_time': self.simulation_time
                             }
        else:
            customer_dict = {'name': self.name,
                             'status': self.status,
                             'location': self.location.tolist(),
                             'order_time': self.order_time,
                             'favorite_restaurants': [restaurant.name for restaurant in self.favorite_restaurants],
                             'day': self.day,
                             'atd': None,
                             'etd': None,
                             'restaurant_name': None,
                             #'myopic_etd': None,
                             #'simulated_etd': None,
                             #'one_step_simul': None,
                             #'adv_one_step_simul': None,
                             #'restaurant_location': None,
                             #'restaurant_queue': None,
                             #'other_restaurant_queues': None,
                             #'vehicle_name': None,
                             #'vehicle_started_action_time': None,
                             #'old_vehicle_route': None,
                             #'new_vehicle_route': None,
                             #'vehicle_route_to_customer': None,
                             #'other_vehicle_routes': None,
                             #'other_vehicle_started_action_times': None,
                             #'max_pre_shift': None,
                             #'max_post_shift': None,
                             #'insertion_status': None,
                             #'insertion_index': None,
                             #'insertion_cost': None,
                             #'simulation_time': None
                             }
        return customer_dict

