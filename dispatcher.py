import numpy as np
from generator import raw_travel_time
from parameters import velocity, mean_wait_parking
from etd import get_all_customer_etd
import copy
import torch


class Action:
    r"""
    Route action class that models an action in a route. It is given by a status encoding the action type,
    an estimated time to perform the action, a destination, (sometimes) orders to pick up/drop off,
    (sometimes) the customer to deliver to, (sometimes) the restaurant to pick upa meal from.
    """

    def __init__(self, status, est_action_time, destination, orders=None, customer=None, restaurant=None):
        r"""
        Initialize the actiom.

        Params
        =======
            status (int):  Type of action to perform.
                            0 - waiting,
                            1 - driving to restaurant,
                            2 - driving to customer,
                            3 - pick up at restaurant,
                            4 - delivery to customer.
            est_action_time (float): Estimated time required to perform the action.
            destination (2D array): End location of the action.
            orders (List): List of orders to pickup (if status == 3). Else None.
            customer (Customer object): Customer (if status == 4). Else None.
            restaurant (Restaurant object): Restaurant (if status == 3). Else None.

        """
        self.destination = destination
        self.status = status
        self.orders = orders
        self.customer = customer
        self.restaurant = restaurant
        self.est_action_time = est_action_time

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
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __copydict__(self):
        r"""
        Relevant class information to deepcopy.

        Returns
        ========
        Dictionary of information to deepcopy.

        """
        copy_dict = {"destination": self.destination,
                     "status": self.status,
                     "customer": self.customer,
                     "restaurant": self.restaurant,
                     "est_action_time": self.est_action_time}
        return copy_dict

    def full_deepcopy(self, restaurant_dict, customer_dict):
        r"""
        Manual deepcopy that contains more information than the deepcopy method.

        Returns
        ========
        Deepcopy of self containing more information than the copy made by the custom deepcopy method.

        """
        if self.orders is not None:
            orders = copy.deepcopy(self.orders)
        else:
            orders = None
        if self.customer is not None:
            customer = customer_dict[self.customer.name]
        else:
            customer = None
        if self.restaurant is not None:
            restaurant = restaurant_dict[self.restaurant.name]
        else:
            restaurant = None

        result = Action(status=self.status,
                        est_action_time=self.est_action_time,
                        destination=self.destination,
                        orders=orders,
                        customer=customer,
                        restaurant=restaurant)
        return result

    def get_shift(self):
        r"""
        Returns the shift by which the action has been delayed due to insertion actions.

        Returns
        ========
        Time shift (int) in minutes.

        """
        if self.customer is None:
            return 0
        else:
            return self.customer.sum_shifted


class Dispatcher:
    r"""
    Dispatcher class that informs customer with estimated arrival times, forwards orders to restaurants,
    assigns orders to vehicles, and routes the vehicles.
    """

    def __init__(self, vehicles, restaurants):
        r"""
        Initialize the dispatcher.

        Params
        =======
            vehicles (List): List of all vehicles.
            restaurants (List): List of all restaurants.

        Attributes
        ===========
            bundling (bool): Boolean indicating whether or not orders may be consolidated.

        """
        self.vehicles = vehicles
        self.restaurants = restaurants
        self.bundling = True

    def take_order(self, customer, restaurant, route, current_time):
        r"""
        Forwards the order of a customer to the respective restaurant and updates the route of the responsible vehicle.

        Arguments
        ==========
            customer (Customer): Customer that is ordering.
            restaurant (Restaurant): Restaurant where the customer is ordering at.
            route (List): Tentative route containing customer and restaurant given by a list of actions (Action).
            current_time (int): Current time step in minutes.

        """
        self.insert(*route)
        restaurant.add_to_queue(order=customer.name, current_time=current_time)
        # Save various data to customer class to use as predictor later on
        customer.restaurant_est_time_queue = copy.deepcopy(restaurant.est_time_queue)
        restaurant_queues = []
        for restaurant in self.restaurants:
            restaurant_queues.append(copy.deepcopy(restaurant.est_time_queue))
        customer.other_restaurant_time_queues = restaurant_queues
        route_vehicles = []
        vehicle_started_action_times = []
        for vehicle in self.vehicles:
            if route[-3] != vehicle:
                vehicle_started_action_times.append(vehicle.started_action_time)
                if len(vehicle.route) > 0:
                    route_vehicles.append([(action.status, action.destination.tolist(), action.est_action_time)
                                           for action in vehicle.route])
                else:
                    route_vehicles.append([(0, vehicle.location.tolist(), 0)])
        customer.other_vehicle_routes = route_vehicles
        customer.other_vehicle_started_action_times = vehicle_started_action_times

    @staticmethod
    def insertion_cost(old_route, new_route, customer, penalty_factor=1.0, shift_constr=15):
        r"""
        Calculates the costs of an insertion. This is an auxiliary method for the 'insertion' method.

        Arguments
        ==========
            old_route (List): The vehicle's tentative route before restaurant and customer are inserted.
            new_route (List): The vehicle's tentative route after restaurant and customer are inserted.
            customer (Customer): Customer that is inserted in the route.
            penalty_factor (float): Factor that weights the cost of shifting previous customers due to the insertion.
            shift_constr (int): Number of minutes that previous customers are allowed to be shifted to a later time.

        Returns
        ========
            Returns the cost of the given insertion operation.

        """
        old_etd = get_all_customer_etd(old_route)
        new_etd = get_all_customer_etd(new_route)
        costs = new_etd[customer.name][0]
        for key in old_etd.keys():
            if shift_constr is not None:
                if new_etd[key][0] - old_etd[key][0] + old_etd[key][1] > shift_constr:
                    return np.inf
            costs += penalty_factor * (new_etd[key][0] - old_etd[key][0])
        return costs

    def insertion(self, customer, restaurant, current_time, vehicles=None):
        r"""
        Implementation of the insertion algorithm for vehicle routing. Appending a restaurant order to an
        existing restaurant order is allowed.

        Arguments
        ==========
            customer (Customer): Customer to be inserted.
            restaurant (Restaurant): Restaurant to be inserted.
            current_time (int): Current time step in minutes.
            vehicles (List): List of vehicles to consider in the assignment decision.

        Returns
        ========
            Returns a tentative route and additional information regarding the insertion.


        """
        p_star = np.inf
        i_star = None
        j_star = None
        drive_action_i_star = None
        wait_action_i_star = None
        drive_action_j_star = None
        wait_action_j_star = None
        append_switch_star = None
        vehicle_star = None

        if vehicles is None:
            vehicles = self.vehicles

        for vehicle in vehicles:  # loop over all routes
            route = copy.deepcopy(vehicle.route)
            if len(route) == 0:  # Case 1: Vehicle currently has empty route
                # drive to restaurant
                destination = restaurant.location
                travel_time = int(raw_travel_time(velocity, vehicle.location, destination)) + 1
                drive_action_i = Action(status=1,
                                        destination=destination,
                                        est_action_time=travel_time)
                route.append(drive_action_i)
                # wait at restaurant
                # calculate wait action
                if len(restaurant.est_time_queue) == 0:
                    wait_time = int(max(mean_wait_parking, restaurant.avg_prep_time - travel_time)) + 1
                else:
                    wait_time = int(max(mean_wait_parking, max(0, restaurant.est_time_queue[-1] - current_time)
                                        + restaurant.avg_prep_time - travel_time)) + 1
                wait_action_i = Action(status=3,
                                       destination=restaurant.location,
                                       orders=[customer.name],
                                       restaurant=restaurant,
                                       est_action_time=wait_time)
                route.append(wait_action_i)
                # drive to customer
                location = destination
                destination = customer.location
                travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                drive_action_j = Action(status=2,
                                        destination=destination,
                                        est_action_time=travel_time)
                route.append(drive_action_j)
                # parking and delivery
                wait_action_j = Action(status=4,
                                       destination=customer.location,
                                       customer=customer,
                                       est_action_time=int(mean_wait_parking) + 1)
                route.append(wait_action_j)
                # return values:
                p = self.insertion_cost(vehicle.route, route, customer)
                if p < p_star:
                    i_star = 0
                    j_star = 0
                    p_star = p
                    drive_action_i_star = drive_action_i
                    wait_action_i_star = wait_action_i
                    drive_action_j_star = drive_action_j
                    wait_action_j_star = wait_action_j
                    append_switch_star = 0
                    vehicle_star = vehicle
                continue

            elif len(route) > 0:  # Case 2: Vehicle currently has non-empty route.
                if not self.bundling:  # all actions must be appended to the end of the route
                    # drive to restaurant
                    location = route[-1].destination
                    destination = restaurant.location
                    travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                    drive_action_i = Action(status=1,
                                            destination=destination,
                                            est_action_time=travel_time)
                    route.append(drive_action_i)
                    # wait at restaurant
                    # calculate wait action
                    # calculate waiting time
                    if vehicle.started_action_time == 0:
                        vehicle.started_action_time = current_time
                    time_before_pickup = max(0, route[0].est_action_time - (current_time - vehicle.started_action_time))
                    for i in range(1, len(route)):
                        time_before_pickup += route[i].est_action_time
                    if len(restaurant.est_time_queue) == 0:
                        wait_time = int(max(mean_wait_parking, restaurant.avg_prep_time - time_before_pickup)) + 1
                    else:
                        wait_time = int(max(mean_wait_parking, max(0, restaurant.est_time_queue[-1] - current_time)
                                            + restaurant.avg_prep_time - time_before_pickup)) + 1
                    wait_action_i = Action(status=3,
                                           destination=destination,
                                           orders=[customer.name],
                                           restaurant=restaurant,
                                           est_action_time=wait_time)
                    route.append(wait_action_i)
                    # drive to customer
                    location = destination
                    destination = customer.location
                    travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                    drive_action_j = Action(status=2,
                                            destination=destination,
                                            est_action_time=travel_time)
                    route.append(drive_action_j)
                    # parking and delivery
                    wait_action_j = Action(status=4,
                                           destination=customer.location,
                                           customer=customer,
                                           est_action_time=int(mean_wait_parking) + 1)
                    route.append(wait_action_j)
                    # return values
                    p = self.insertion_cost(vehicle.route, route, customer)
                    if p < p_star:
                        i_star = 0
                        j_star = 0
                        p_star = p
                        drive_action_i_star = drive_action_i
                        wait_action_i_star = wait_action_i
                        drive_action_j_star = drive_action_j
                        wait_action_j_star = wait_action_j
                        append_switch_star = 0
                        vehicle_star = vehicle
                    continue

                # Test all possible points of insertion for pick up.
                # Insertion is only possible after waiting actions
                for i in range(2 - (len(route) % 2), len(route) + 2, 2):
                    route_i = copy.deepcopy(route)
                    append_switch = 0  # 1 if we just append an order instead of creating an action
                    # try appending to existing order
                    if route_i[i - 1].restaurant is not None and route_i[i - 1].restaurant.name == restaurant.name:
                        if i - 1 == 0:  # already waiting at restaurant
                            if len(restaurant.est_time_queue) > 0:
                                wait_time_append = int(max(0, restaurant.est_time_queue[-1] - current_time)
                                                       + restaurant.avg_prep_time) + 1
                            else:
                                wait_time_append = int(restaurant.avg_prep_time) + 1
                        else:
                            time_before_pickup = max(0, route_i[0].est_action_time -
                                                     (current_time - vehicle.started_action_time))
                            for ix in range(1, i - 1):
                                time_before_pickup += route_i[ix].est_action_time
                            if len(restaurant.est_time_queue) > 0:
                                wait_time_append = int(max(mean_wait_parking,
                                                           max(0, restaurant.est_time_queue[-1] - current_time)
                                                           + restaurant.avg_prep_time - time_before_pickup)) + 1
                            else:  # queue is empty because the order is in the storage already
                                wait_time_append = int(max(mean_wait_parking, restaurant.avg_prep_time
                                                           - time_before_pickup)) + 1
                        # update wait time
                        route_i[i - 1].est_action_time = wait_time_append
                        assert route_i[i - 1].status == 3
                        drive_action_i = None  # vehicle is already on its way to restaurant
                        wait_action_i = None  # Vehicle will wait at restaurant anyway
                        append_switch = 1
                    else:  # create a new action
                        # drive to restaurant
                        location = route_i[i - 1].destination
                        destination = restaurant.location
                        travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                        drive_action_i = Action(status=1,
                                                destination=destination,
                                                est_action_time=travel_time)
                        # wait at restaurant
                        # calculate waiting time
                        if vehicle.started_action_time == 0:
                            vehicle.started_action_time = current_time
                        time_before_pickup = max(0, route_i[0].est_action_time
                                                 - (current_time - vehicle.started_action_time)) + travel_time
                        for ix in range(1, i - 1):
                            time_before_pickup += route_i[ix].est_action_time
                        if len(restaurant.est_time_queue) == 0:
                            wait_time = int(max(mean_wait_parking, restaurant.avg_prep_time - time_before_pickup)) + 1
                        else:
                            wait_time = int(max(mean_wait_parking, max(0, restaurant.est_time_queue[-1] - current_time)
                                                + restaurant.avg_prep_time - time_before_pickup)) + 1
                        wait_action_i = Action(status=3,
                                               destination=destination,
                                               orders=[customer.name],
                                               restaurant=restaurant,
                                               est_action_time=wait_time)
                        route_i.insert(i, wait_action_i)
                        route_i.insert(i, drive_action_i)
                    for j in range(i + 2, len(route_i) + 2, 2):  # test all possible points insertion for drop off
                        route_j = copy.deepcopy(route_i)
                        # drive to customer
                        location = route_j[j - 1].destination
                        destination = customer.location
                        travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                        drive_action_j = Action(status=2,
                                                destination=destination,
                                                est_action_time=travel_time)
                        # wait at customer
                        wait_action_j = Action(status=4,
                                               destination=customer.location,
                                               customer=customer,
                                               est_action_time=int(mean_wait_parking) + 1)
                        route_j.insert(j, wait_action_j)
                        route_j.insert(j, drive_action_j)
                        # adapt est_action_time of altered drive actions
                        if i + 2 != j and append_switch == 0:
                            assert route_j[i + 2].status in [1, 2]
                            route_j[i + 2].est_action_time = int(raw_travel_time(
                                velocity, restaurant.location, route_j[i + 2].destination)) + 1
                        if j + 2 < len(route_j):
                            assert route_j[j + 2].status in [1, 2]
                            route_j[j + 2].est_action_time = int(raw_travel_time(
                                velocity, customer.location, route_j[j + 2].destination)) + 1
                        p = self.insertion_cost(vehicle.route, route_j, customer)
                        if p < p_star:
                            i_star = i
                            j_star = j
                            p_star = p
                            drive_action_i_star = drive_action_i
                            wait_action_i_star = wait_action_i
                            drive_action_j_star = drive_action_j
                            wait_action_j_star = wait_action_j
                            append_switch_star = append_switch
                            if append_switch_star == 1:
                                wait_action_i_star = wait_time_append
                            vehicle_star = vehicle
        if p_star == np.inf:
            print("Did not find a feasible route. This should not happen.")
            print("Customer was already shifted by {} minutes.".format(customer.sum_shifted))
            return [None, None, None, None, None, None, None, None, None, None]
        return [i_star, j_star, drive_action_i_star, wait_action_i_star, drive_action_j_star,
                wait_action_j_star, append_switch_star, vehicle_star, customer, p_star]

    @staticmethod
    def insert(i_star, j_star, drive_action_i_star, wait_action_i_star, drive_action_j_star,
               wait_action_j_star, append_switch_star, vehicle_star, customer, p_star, shift_customer=True):
        r"""
        This auxiliary method realizes a calculated insertion.

        Arguments
        ==========
            i_star (int): Action index defining the insertion point of restaurant in the route.
            j_star (int): Action index defining the insertion point of customer in the route.
            drive_action_i_star (Action): Drive action to the restaurant to be inserted.
            wait_action_i_star (Action): Wait action at the restaurant to be inserted.
            drive_action_j_star (Action): Drive action to the customer to be inserted.
            wait_action_j_star (Action): Wait action at the customer to be inserted.
            append_switch_star (int): Indicates whether the order was consolidated with previous orders.
            vehicle_star (int): Index of vehicle to which the order is assigned.
            customer (Customer): Customer that ordered.
            p_star (float): Costs of the insertion.
            shift_customer (bool): Boolean indicating whether the shift of customers due to the insertion is saved.

        """
        # could not find a feasible route
        if i_star is None:
            if shift_customer:
                customer.status = -1
            return None
        # save original vehicle route

        if len(vehicle_star.route) > 0:
            customer.old_vehicle_route = [(action.status, action.destination.tolist(),
                                           action.est_action_time, action.get_shift())
                                          for action in vehicle_star.route]
        else:
            customer.old_vehicle_route = [(0, vehicle_star.location.tolist(), 0, 0)]

        if i_star == 0 and j_star == 0:  # vehicle does not have a route yet
            vehicle_star.action_time = np.inf
            vehicle_star.route.append(drive_action_i_star)
            vehicle_star.route.append(wait_action_i_star)
            vehicle_star.route.append(drive_action_j_star)
            vehicle_star.route.append(wait_action_j_star)
            customer.insertion_status = 0

        elif i_star == -1 and j_star == -1:  # vehicle has a route but bundling is not allowed
            vehicle_star.route.append(drive_action_i_star)
            vehicle_star.route.append(wait_action_i_star)
            vehicle_star.route.append(drive_action_j_star)
            vehicle_star.route.append(wait_action_j_star)
            customer.insertion_status = 0

        elif append_switch_star == 0:  # insert all actions
            old_etd = get_all_customer_etd(vehicle_star.route)
            vehicle_star.route.insert(i_star, wait_action_i_star)
            vehicle_star.route.insert(i_star, drive_action_i_star)
            vehicle_star.route.insert(j_star, wait_action_j_star)
            vehicle_star.route.insert(j_star, drive_action_j_star)
            # adapt est_action_time of altered drive actions
            if i_star + 2 != j_star:
                assert vehicle_star.route[i_star + 2].status in [1, 2]
                vehicle_star.route[i_star + 2].est_action_time = int(raw_travel_time(
                    velocity, vehicle_star.route[i_star + 1].destination,
                    vehicle_star.route[i_star + 2].destination)) + 1
            if j_star + 2 < len(vehicle_star.route):
                assert vehicle_star.route[j_star + 2].status in [1, 2]
                vehicle_star.route[j_star + 2].est_action_time = int(raw_travel_time(
                    velocity, customer.location, vehicle_star.route[j_star + 2].destination)) + 1
            new_etd = get_all_customer_etd(vehicle_star.route)
            customer.insertion_status = 0
            # Update sum_shift for all customer that have been shifted due to the insertion
            if shift_customer:
                for action in vehicle_star.route:
                    if action.status == 4:
                        if action.customer != customer:
                            action.customer.sum_shifted += new_etd[action.customer.name][0] \
                                                           - old_etd[action.customer.name][0]
                            if action.customer.sum_shifted > 15:
                                raise Warning("Illegal time shift {} with append switch {}"
                                              .format(action.customer.sum_shifted, append_switch_star))

        elif append_switch_star == 1:  # append order to existing order
            old_etd = get_all_customer_etd(vehicle_star.route)
            vehicle_star.route[i_star - 1].orders.append(customer.name)
            vehicle_star.route[
                i_star - 1].est_action_time = wait_action_i_star  # special case: scalar instead of action
            vehicle_star.route.insert(j_star, wait_action_j_star)
            vehicle_star.route.insert(j_star, drive_action_j_star)
            # adapt est_action_time of altered drive actions
            if j_star + 2 < len(vehicle_star.route):
                assert vehicle_star.route[j_star + 2].status in [1, 2]
                vehicle_star.route[j_star + 2].est_action_time = int(raw_travel_time(
                    velocity, vehicle_star.route[j_star + 1].destination,
                    vehicle_star.route[j_star + 2].destination)) + 1
            new_etd = get_all_customer_etd(vehicle_star.route)
            customer.insertion_status = 1
            i_star = -1  # save i_star as -1 to indicate that the order was appended
            # Update sum_shift for all customer that have been shifted due to the insertion
            if shift_customer:
                for action in vehicle_star.route:
                    if action.status == 4:
                        if action.customer != customer:
                            action.customer.sum_shifted += new_etd[action.customer.name][0] \
                                                           - old_etd[action.customer.name][0]
                            if action.customer.sum_shifted > 15:
                                raise Warning("Illegal time shift {} with append switch {}"
                                              .format(action.customer.sum_shifted, append_switch_star))
        # customer data to save:
        #customer.vehicle = vehicle_star  # save vehicle that served customer
        #customer.vehicle_location = vehicle_star.location
        #customer.vehicle_started_action_time = vehicle_star.started_action_time
        #customer.new_vehicle_route = [(action.status, action.destination.tolist(),
        #                               action.est_action_time, action.get_shift())
        #                              for action in vehicle_star.route]
        #customer.insertion_cost = p_star
        #if i_star == 0 and j_star == 0:
        #    customer.vehicle_route_to_customer = customer.new_vehicle_route
        #else:
        #    customer.vehicle_route_to_customer = [(action.status, action.destination.tolist(),
        #                                           action.est_action_time, action.get_shift())
        #                                          for action in vehicle_star.route[0:j_star + 1]]
        #customer.insertion_index = [i_star, j_star]
        #max_pre_shift = 0
        #max_post_shift = 0
        #if i_star != -1 or j_star != -1:
        #    for index, action in enumerate(vehicle_star.route):
        #        if action.status == 4:
        #            if index < j_star or j_star == 0:
        #                max_pre_shift = max(action.customer.sum_shifted, max_pre_shift)
        #            if index > j_star:
        #                max_post_shift = max(action.customer.sum_shifted, max_post_shift)
        #customer.max_pre_shift = max_pre_shift
        #customer.max_post_shift = max_post_shift


class FakeDispatcher:
    r"""
    A clone of the actual dispatcher that uses a deep neural network to imitate a cheapest insertion routing heuristic.
    """

    def __init__(self, vehicles, restaurants, model):
        r"""
        Initialize the cloned dispatcher.

        Params
        =======
            vehicles (List): List of all vehicles.
            restaurants (List): List of all restaurants.
            model (torch.nn object): Neural network that predicts insertion decisions.

        """
        self.vehicles = vehicles  # list of all vehicles
        self.restaurants = restaurants  # list of all restaurants
        self.model = model
        self.model.eval()

    def take_order(self, customer, restaurant, route, current_time):
        r"""
        Forwards the order of a customer to the respective restaurant and updates the route of the responsible vehicle.

        Arguments
        ==========
            customer (Customer): Customer that is ordering.
            restaurant (Restaurant): Restaurant where the customer is ordering at.
            route (List): Tentative route containing customer and restaurant given by a list of actions (Action).
            current_time (int): Current time step in minutes.

        """
        if route is None:
            customer.status = -1
        else:
            self.insert(*route)
            restaurant.add_to_queue(order=customer.name, current_time=current_time)

    @staticmethod
    def scale(X, x_min=-1, x_max=1):
        r"""
        Scales the input of the DeepInsertion Model.

        Arguments
        ==========
            X (array): Unscaled input.
            x_min (float): Lower bound of scaled input.
            x_max (float): Upper bound of scaled input.

        Returns
        ========
        Array containing scaled input.

        """
        X_min = np.array([0.0, 0.0, 3.52070206, 3.71164996, 0.0, 3.0, 0.0, 0.0, 3.0, -20.0,
                          0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, -17.0, 0.0, 0.0, 0.0, 0.0,
                          -19.0, 0.0, 0.0, 0.0, 0.0, -15.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, -7.0,
                          0.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        X_max = np.array([15.35130731, 13.49346697, 12.46919309, 10.34123165, 51.0, 4.0, 15.35130731,
                          13.49346697, 35.0, 15.0, 4.0, 15.35130731, 13.49346697, 43.0, 15.0, 4.0, 15.35130731,
                          13.49346697, 32.0, 15.0, 4.0, 15.35130731, 13.49346697, 32.0, 15.0, 4.0, 15.35130731,
                          13.49346697, 34.0, 15.0, 4.0, 15.35130731, 13.49346697, 29.0, 15.0, 4.0, 15.35098815,
                          13.49346697, 28.0, 15.0, 4.0, 15.34983257, 13.41473213, 24.0, 15.0, 4.0, 15.33966073,
                          13.49346697, 21.0, 15.0, 4.0, 15.32294646, 12.45623837, 17.0, 15.0, 4.0, 15.15503898,
                          11.43302948, 14.0, 14.0, 4.0, 10.06925151, 10.76896486, 10.0, 12.0])

        nom = (X - X_min) * (x_max - x_min)
        denom = X_max - X_min
        denom[denom == 0] = 1
        return x_min + nom / denom

    @staticmethod
    def insertion_cost(old_route, new_route, customer, penalty_factor=1.0, shift_constr=15):
        r"""
        Calculates the costs of an insertion. This is an auxiliary method for the 'insertion' method.

        Arguments
        ==========
            old_route (List): The vehicle's tentative route before restaurant and customer are inserted.
            new_route (List): The vehicle's tentative route after restaurant and customer are inserted.
            customer (Customer): Customer that is inserted in the route.
            penalty_factor (float): Factor that weights the cost of shifting previous customers due to the insertion.
            shift_constr (int): Number of minutes that previous customers are allowed to be shifted to a later time.

        Returns
        ========
            Returns the cost of the given insertion operation.

        """
        old_etd = get_all_customer_etd(old_route)
        new_etd = get_all_customer_etd(new_route)
        costs = new_etd[customer.name][0]
        for key in old_etd.keys():
            if shift_constr is not None:
                if new_etd[key][0] - old_etd[key][0] + old_etd[key][1] > shift_constr:
                    return np.inf
            costs += penalty_factor * (new_etd[key][0] - old_etd[key][0])
        return costs

    def insertion(self, customer, restaurant, current_time):
        r"""
        DNN imitation of the insertion algorithm for vehicle routing. Appending a restaurant order to an
        existing restaurant order is allowed.

        Arguments
        ==========
            customer (Customer): Customer to be inserted.
            restaurant (Restaurant): Restaurant to be inserted.
            current_time (int): Current time step in minutes.

        Returns
        ========
            Returns a tentative route and additional information regarding the insertion.


        """
        # start_time = timeit.default_timer()
        route_list = []
        flag_count = 0
        # preprocess routes
        rest_dicts = []
        for vehicle in self.vehicles:
            rest_dict = {}
            action_time = 0
            remove_indices = []
            route = [[action.status, action.destination.tolist(), action.est_action_time, action.get_shift()]
                     for action in vehicle.route]
            started_action_time = vehicle.started_action_time
            for index, action in enumerate(route):
                if index == 0:
                    route[index] = [action[0], action[1][0], action[1][1], action[2] - started_action_time, action[3]]
                # shorten routes by deleting drive actions and adding the action times to the wait actions
                # additionally discretize locations and adapt first action time
                if action[0] in [1, 2]:
                    action_time = action[2]
                    remove_indices.append(index)
                    continue
                if action[0] in [0, 3, 4]:
                    route[index] = [action[0], action[1][0], action[1][1],
                                    action[2] + action_time, action[3]]
                    action_time = 0
                    if action[0] == 3:
                        rest_dict[str(action[1])] = index
            rest_dicts.append(rest_dict)
            for index in sorted(remove_indices, reverse=True):
                del route[index]
            route_list.append(route)
        # create batch
        input_list = []
        customer_location_x, customer_location_y = customer.location
        restaurant_location_x, restaurant_location_y = restaurant.location
        # convert restaurant queue to minutes instead of list of daytimes
        restaurant_queue = copy.deepcopy(restaurant.est_time_queue)
        if len(restaurant_queue) == 0:
            restaurant_queue = current_time
        else:
            restaurant_queue = restaurant_queue[-1]
        restaurant_queue = int(max(0, restaurant_queue - current_time))
        for route in route_list:
            inputs = [customer_location_x, customer_location_y, restaurant_location_x, restaurant_location_y,
                      restaurant_queue] + np.array(route).flatten().tolist()
            if len(inputs) < 65:
                inputs = self.scale(np.pad(inputs, (0, 65 - len(inputs))))
            if len(inputs) > 65:
                inputs = self.scale(inputs[0:65])
            input_list.append(inputs)

        # feed data to neural network and predict insertion indices
        inputs = np.array(input_list)
        inputs = torch.from_numpy(inputs).float()  # .unsqueeze(2).permute(1, 0, 2)
        with torch.no_grad():
            pred_i, pred_j = self.model.forward(inputs)
        pred_i = np.reshape(pred_i.numpy().argmax(axis=1), (len(self.vehicles), -1))
        pred_j = np.reshape(pred_j.numpy().argmax(axis=1), (len(self.vehicles), -1))

        # calculate insertion cost for every vehicle and every index tuple
        p_star = np.inf
        i_star = None
        j_star = None
        drive_action_i_star = None
        wait_action_i_star = None
        drive_action_j_star = None
        wait_action_j_star = None
        append_switch_star = None
        vehicle_star = None

        for vehicle_index, vehicle in enumerate(self.vehicles):
            i, j = pred_i[vehicle_index][0], pred_j[vehicle_index][0]
            route = copy.deepcopy(vehicle.route)
            if len(route) == 0:
                destination = restaurant.location
                travel_time = int(raw_travel_time(velocity, vehicle.location, destination)) + 1
                drive_action_i = Action(status=1,
                                        destination=destination,
                                        est_action_time=travel_time)
                route.append(drive_action_i)
                # wait at restaurant
                # calculate wait action
                if len(restaurant.est_time_queue) == 0:
                    wait_time = int(max(mean_wait_parking, restaurant.avg_prep_time - travel_time)) + 1
                else:
                    wait_time = int(max(mean_wait_parking, max(0, restaurant.est_time_queue[-1] - current_time)
                                        + restaurant.avg_prep_time - travel_time)) + 1
                wait_action_i = Action(status=3,
                                       destination=restaurant.location,
                                       orders=[customer.name],
                                       restaurant=restaurant,
                                       est_action_time=wait_time)
                route.append(wait_action_i)
                # drive to customer
                location = destination
                destination = customer.location
                travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                drive_action_j = Action(status=2,
                                        destination=destination,
                                        est_action_time=travel_time)
                route.append(drive_action_j)
                # parking and delivery
                wait_action_j = Action(status=4,
                                       destination=destination,
                                       customer=customer,
                                       est_action_time=int(mean_wait_parking) + 1)
                route.append(wait_action_j)
                # return values:
                p = self.insertion_cost(vehicle.route, route, customer)
                if p < p_star:
                    i_star = 0
                    j_star = 0
                    p_star = p
                    drive_action_i_star = drive_action_i
                    wait_action_i_star = wait_action_i
                    drive_action_j_star = drive_action_j
                    wait_action_j_star = wait_action_j
                    append_switch_star = 0
                    vehicle_star = vehicle
                continue

            else:
                # calculate converted insertion indices
                i -= 1
                j += 2
                modulo_index = len(route) % 2
                if i >= 0:
                    i = 2 * (i + 1) - modulo_index
                j = 2 * j - modulo_index
                # filter out impossible predictions
                flag = False
                if i == len(route) and j == len(route) + 1:
                    # standard append via insertion case
                    flag = False
                elif i != 0 and j != 0 and i >= j:
                    # customer is visited before restaurant
                    flag = True
                elif i == 0 and j == 0 and len(route) > 0:
                    # i=j=0 indicate empty route but route is actually non-empty
                    flag = True
                elif i == -1:
                    if j > len(route):
                        # restaurant is appended. So j has to be leq len(route)
                        flag = True
                    elif str(restaurant.location.tolist()) not in rest_dicts[vehicle_index].keys():
                        # order i appended when the corresponding restaurant is not visited
                        flag = True
                    elif route[j - 1].status not in [3, 4]:
                        flag = True
                    else:
                        append_index = rest_dicts[vehicle_index][str(restaurant.location.tolist())]
                        if append_index >= i:
                            flag = True
                else:
                    try:
                        if route[i - 1].status not in [3, 4] or route[j - 1].status not in [3, 4]:
                            flag = True
                    except IndexError:
                        if not (j == len(route) or j == len(route) + 2):
                            flag = True
                # if false prediction, just append
                if flag:
                    pred_i[vehicle_index] = len(route_list[vehicle_index])
                    pred_j[vehicle_index] = len(route_list[vehicle_index]) + 1

                    flag_count += 1
                    # drive to restaurant
                    location = route[-1].destination
                    destination = restaurant.location
                    travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                    drive_action_i = Action(status=1,
                                            destination=destination,
                                            est_action_time=travel_time)
                    route.append(drive_action_i)
                    # wait at restaurant
                    # calculate wait action
                    # calculate waiting time
                    if vehicle.started_action_time == 0:
                        vehicle.started_action_time = current_time
                    time_before_pickup = max(0, route[0].est_action_time - (
                            current_time - vehicle.started_action_time))
                    for ix in range(1, len(route)):
                        time_before_pickup += route[ix].est_action_time
                    if len(restaurant.est_time_queue) == 0:
                        wait_time = int(max(mean_wait_parking, restaurant.avg_prep_time - time_before_pickup)) + 1
                    else:
                        wait_time = int(max(mean_wait_parking, max(0, restaurant.est_time_queue[-1] - current_time)
                                            + restaurant.avg_prep_time - time_before_pickup)) + 1
                    wait_action_i = Action(status=3,
                                           destination=destination,
                                           orders=[customer.name],
                                           restaurant=restaurant,
                                           est_action_time=wait_time)
                    route.append(wait_action_i)
                    # drive to customer
                    location = destination
                    destination = customer.location
                    travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                    drive_action_j = Action(status=2,
                                            destination=destination,
                                            est_action_time=travel_time)
                    route.append(drive_action_j)
                    # parking and delivery
                    wait_action_j = Action(status=4,
                                           destination=customer.location,
                                           customer=customer,
                                           est_action_time=int(mean_wait_parking) + 1)
                    route.append(wait_action_j)
                    # return values:
                    p = self.insertion_cost(vehicle.route, route, customer)
                    if p < p_star:
                        i_star = len(vehicle.route)
                        j_star = len(vehicle.route) + 2
                        p_star = p
                        drive_action_i_star = drive_action_i
                        wait_action_i_star = wait_action_i
                        drive_action_j_star = drive_action_j
                        wait_action_j_star = wait_action_j
                        append_switch_star = 0
                        vehicle_star = vehicle
                    continue

                # create restaurant action
                append_switch = 0
                if i == -1:
                    if append_index - 1 == 0:  # already waiting at restaurant
                        if len(restaurant.est_time_queue) > 0:
                            wait_time_append = int(max(0, restaurant.est_time_queue[-1] - current_time)
                                                   + restaurant.avg_prep_time) + 1
                        else:
                            wait_time_append = int(restaurant.avg_prep_time) + 1
                    else:
                        time_before_pickup = max(0, route[0].est_action_time -
                                                 (current_time - vehicle.started_action_time))
                        for ix in range(1, append_index - 1):
                            time_before_pickup += route[ix].est_action_time
                        if len(restaurant.est_time_queue) > 0:
                            wait_time_append = int(max(mean_wait_parking,
                                                       max(0, restaurant.est_time_queue[-1] - current_time)
                                                       + restaurant.avg_prep_time - time_before_pickup)) + 1
                        else:  # queue is empty because the order is in the storage already
                            wait_time_append = int(max(mean_wait_parking, restaurant.avg_prep_time
                                                       - time_before_pickup)) + 1
                    # update wait time
                    route[append_index].est_action_time = wait_time_append
                    assert route[append_index].status == 3
                    drive_action_i = None  # vehicle is already on its way to restaurant
                    wait_action_i = None  # Vehicle will wait at restaurant anyway
                    append_switch = 1
                else:
                    location = route[i - 1].destination
                    destination = restaurant.location
                    travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                    drive_action_i = Action(status=1,
                                            destination=destination,
                                            est_action_time=travel_time)
                    # wait at restaurant
                    if vehicle.started_action_time == 0:
                        vehicle.started_action_time = current_time
                    time_before_pickup = max(0, route[0].est_action_time
                                             - (current_time - vehicle.started_action_time)) + travel_time
                    for ix in range(1, i - 1):
                        time_before_pickup += route[ix].est_action_time
                    if len(restaurant.est_time_queue) == 0:
                        wait_time = int(max(mean_wait_parking, restaurant.avg_prep_time - time_before_pickup)) + 1
                    else:
                        wait_time = int(max(mean_wait_parking, max(0, restaurant.est_time_queue[-1] - current_time)
                                            + restaurant.avg_prep_time - time_before_pickup)) + 1
                    wait_action_i = Action(status=3,
                                           destination=destination,
                                           orders=[customer.name],
                                           restaurant=restaurant,
                                           est_action_time=wait_time)
                    route.insert(i, wait_action_i)
                    route.insert(i, drive_action_i)
                # create customer action
                # drive to customer
                location = route[j - 1].destination
                destination = customer.location
                travel_time = int(raw_travel_time(velocity, location, destination)) + 1
                drive_action_j = Action(status=2,
                                        destination=destination,
                                        est_action_time=travel_time)
                # wait at customer
                wait_action_j = Action(status=4,
                                       destination=destination,
                                       customer=customer,
                                       est_action_time=int(mean_wait_parking) + 1)
                route.insert(j, wait_action_j)
                route.insert(j, drive_action_j)
                # adapt est_action_time of altered drive actions
                p_flag = False
                if i != -1:
                    if i + 2 != j:
                        if route[i + 2].status not in [1, 2]:
                            p_flag = True
                            flag_count += 1
                        route[i + 2].est_action_time = int(raw_travel_time(
                            velocity, restaurant.location, route[i + 2].destination)) + 1
                if j + 2 < len(route):
                    if route[j + 2].status not in [1, 2]:
                        p_flag = True
                        flag_count += 1
                    route[j + 2].est_action_time = int(raw_travel_time(
                        velocity, customer.location, route[j + 2].destination)) + 1
                if not p_flag:
                    p = self.insertion_cost(vehicle.route, route, customer)
                else:
                    p = np.inf
                if p < p_star:
                    i_star = i
                    j_star = j
                    p_star = p
                    drive_action_i_star = drive_action_i
                    wait_action_i_star = wait_action_i
                    drive_action_j_star = drive_action_j
                    wait_action_j_star = wait_action_j
                    append_switch_star = append_switch
                    if append_switch_star == 1:
                        i_star = append_index
                        wait_action_i_star = wait_time_append
                    vehicle_star = vehicle
        if p_star == np.inf:
            return [None, None, None, None, None, None, None, None, None, None]
        return [i_star, j_star, drive_action_i_star, wait_action_i_star, drive_action_j_star,
                wait_action_j_star, append_switch_star, vehicle_star, customer, p_star]

    @staticmethod
    def insert(i_star, j_star, drive_action_i_star, wait_action_i_star, drive_action_j_star,
               wait_action_j_star, append_switch_star, vehicle_star, customer, p_star, shift_customer=True):
        r"""
        This auxiliary method realizes a calculated insertion.

        Arguments
        ==========
            i_star (int): Action index defining the insertion point of restaurant in the route.
            j_star (int): Action index defining the insertion point of customer in the route.
            drive_action_i_star (Action): Drive action to the restaurant to be inserted.
            wait_action_i_star (Action): Wait action at the restaurant to be inserted.
            drive_action_j_star (Action): Drive action to the customer to be inserted.
            wait_action_j_star (Action): Wait action at the customer to be inserted.
            append_switch_star (int): Indicates whether the order was consolidated with previous orders.
            vehicle_star (int): Index of vehicle to which the order is assigned.
            customer (Customer): Customer that ordered.
            p_star (float): Costs of the insertion.
            shift_customer (bool): Boolean indicating whether the shift of customers due to the insertion is saved.

        """
        # could not find a feasible route
        if i_star is None:
            if shift_customer:
                customer.status = -1
            return None

        if i_star == 0 and j_star == 0:  # vehicle does not have a route yet
            vehicle_star.action_time = np.inf
            vehicle_star.route.append(drive_action_i_star)
            vehicle_star.route.append(wait_action_i_star)
            vehicle_star.route.append(drive_action_j_star)
            vehicle_star.route.append(wait_action_j_star)
            customer.insertion_status = 0

        elif i_star == -1 and j_star == -1:  # vehicle has a route but bundling is not allowed
            vehicle_star.route.append(drive_action_i_star)
            vehicle_star.route.append(wait_action_i_star)
            vehicle_star.route.append(drive_action_j_star)
            vehicle_star.route.append(wait_action_j_star)
            customer.insertion_status = 0

        elif append_switch_star == 0:  # insert all actions
            old_etd = get_all_customer_etd(vehicle_star.route)
            vehicle_star.route.insert(i_star, wait_action_i_star)
            vehicle_star.route.insert(i_star, drive_action_i_star)
            vehicle_star.route.insert(j_star, wait_action_j_star)
            vehicle_star.route.insert(j_star, drive_action_j_star)
            # adapt est_action_time of altered drive actions
            if i_star + 2 != j_star:
                assert vehicle_star.route[i_star + 2].status in [1, 2]
                vehicle_star.route[i_star + 2].est_action_time = int(raw_travel_time(
                    velocity, vehicle_star.route[i_star + 1].destination,
                    vehicle_star.route[i_star + 2].destination)) + 1
            if j_star + 2 < len(vehicle_star.route):
                assert vehicle_star.route[j_star + 2].status in [1, 2]
                vehicle_star.route[j_star + 2].est_action_time = int(raw_travel_time(
                    velocity, customer.location, vehicle_star.route[j_star + 2].destination)) + 1
            new_etd = get_all_customer_etd(vehicle_star.route)
            customer.insertion_status = 0
            # Update sum_shift for all customer that have been shifted due to the insertion
            if shift_customer:
                for action in vehicle_star.route:
                    if action.status == 4:
                        if action.customer != customer:
                            action.customer.sum_shifted += new_etd[action.customer.name][0] \
                                                           - old_etd[action.customer.name][0]
                            if action.customer.sum_shifted > 15:
                                action.customer.sum_shifted = 14

        elif append_switch_star == 1:  # append order to existing order
            old_etd = get_all_customer_etd(vehicle_star.route)
            vehicle_star.route[i_star].orders.append(customer.name)
            vehicle_star.route[
                i_star - 1].est_action_time = wait_action_i_star  # special case: scalar instead of action
            vehicle_star.route.insert(j_star, wait_action_j_star)
            vehicle_star.route.insert(j_star, drive_action_j_star)
            # adapt est_action_time of altered drive actions
            if j_star + 2 < len(vehicle_star.route):
                assert vehicle_star.route[j_star + 2].status in [1, 2]
                vehicle_star.route[j_star + 2].est_action_time = int(raw_travel_time(
                    velocity, vehicle_star.route[j_star + 1].destination,
                    vehicle_star.route[j_star + 2].destination)) + 1
            new_etd = get_all_customer_etd(vehicle_star.route)
            customer.insertion_status = 1
            i_star = -1  # save i_star as -1 to indicate that the order was appended
            # Update sum_shift for all customer that have been shifted due to the insertion
            if shift_customer:
                for action in vehicle_star.route:
                    if action.status == 4:
                        if action.customer != customer:
                            action.customer.sum_shifted += new_etd[action.customer.name][0] \
                                                           - old_etd[action.customer.name][0]
                            if action.customer.sum_shifted > 15:
                                action.customer.sum_shifted = 14
