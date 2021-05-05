import copy
import numpy as np
import parameters as param
from generator import raw_travel_time


def get_all_customer_etd(route):
    r"""
    Returns a dictionary with each customer's name served in the route as key. The value to each key is a
    tuple containing the sum of estimated action times until the customer is served and the time the customer has been
    shifted from its original etd.

    Arguments
    ==========
    route (List): List of actions (Action) to be performed by the vehicle.

    Returns
    ========
    Dictionary with customers in the route as key and a tuple of estimated delivery time for the customer and
    sum of shifts of the customer due to insertions.

    """
    etd_dict = {}
    etd = 0
    for action in route:
        assert action.est_action_time >= 0
        etd += action.est_action_time
        if action.status == 4:
            if action.customer != -1:
                etd_dict[action.customer.name] = [etd, action.customer.sum_shifted]
            else:
                etd_dict[action.customer] = [etd, 0]
    return etd_dict


def get_all_restaurants_etd(customer, dispatcher, restaurants, current_time):
    r"""
    Returns the etd for each restaurant and the corresponding route given a customer.
    The etd is calculated based on planning on means.
    The route is in the same format as the return of the dispatcher.insertion method.

    Arguments
    ==========
    customer (Customer): Ordering customer that requires an etd for each restaurant.
    dispatcher (Dispatcher): Dispatcher taking the order.
    restaurants (List): List of restaurants (Restaurant) for which the delivery times must be estimated.
    current_time (int): Current time.

    Returns
    ========
    Dictionary with restaurant names as key and a tuple of estimated delivery time and tentative route as value.

    """
    etd_dict = {}
    for restaurant in restaurants:
        # get best insertion
        route = dispatcher.insertion(customer=customer, restaurant=restaurant, current_time=current_time)
        # create a virtual copy of vehicle determined by insertion
        virtual_route = copy.deepcopy(route)
        virtual_route[8] = customer
        vehicle_star = virtual_route[7]
        # change the route of the virtual vehicle
        dispatcher.insert(*virtual_route, shift_customer=False)
        # calculate etd for customer
        if vehicle_star is None:
            etd_dict[restaurant.name] = [np.inf, None]
            continue
        if vehicle_star.started_action_time == 0:
            vehicle_star.started_action_time = current_time
        etd = max(0, vehicle_star.route[0].est_action_time - (current_time - vehicle_star.started_action_time))
        for action in vehicle_star.route[1:]:
            etd += action.est_action_time
            if action.status == 4:
                if action.customer.name == customer.name:
                    etd_dict[restaurant.name] = [etd, route]
                    break
        if etd < param.time_constraint and not advanced_eta:
            break
    return etd_dict


def advanced_eta(etd_dict, restaurant_dict, order_time, model):
    r"""
    Returns the etd for each restaurant and the corresponding route for a given customer.
    The etd is calculated based on the offline method using gradient boosted decision trees.

    Arguments
    ==========
    etd_dict (Dict): Dict returned by the 'get_all_restaurants_etd' function.
    restaurant_dict (Dict): Dict with restaurant names as key and restaurants (Restaurant) as value.
    order_time (int): Time at which the customer orders.
    model (LightGBM.Regressor): Gradient boosted decision tree model trained to predict delivery times.

    Returns
    ========
    Dictionary with restaurant names as key and a tuple of estimated delivery time and tentative route as value.

    """
    gbdt_input = []
    for restaurant_name, (etd, insertion_details) in etd_dict.items():
        # create the real route
        (i_star, j_star, drive_action_i_star, wait_action_i_star, drive_action_j_star,
         wait_action_j_star, append_switch_star, vehicle_star, customer, p_star) = insertion_details
        route = copy.deepcopy(vehicle_star.route)
        if i_star == 0 and j_star == 0:  # vehicle does not have a route yet
            route.append(drive_action_i_star)
            route.append(wait_action_i_star)
            route.append(drive_action_j_star)
            route.append(wait_action_j_star)

        elif i_star == -1 and j_star == -1:  # vehicle has a route but bundling is not allowed
            route.append(drive_action_i_star)
            route.append(wait_action_i_star)
            route.append(drive_action_j_star)
            route.append(wait_action_j_star)

        elif append_switch_star == 0:  # insert all actions
            route.insert(i_star, wait_action_i_star)
            route.insert(i_star, drive_action_i_star)
            route.insert(j_star, wait_action_j_star)
            route.insert(j_star, drive_action_j_star)
            # adapt est_action_time of altered drive actions
            if i_star + 2 != j_star:
                assert route[i_star + 2].status in [1, 2]
                route[i_star + 2].est_action_time = int(raw_travel_time(param.velocity, route[i_star + 1].destination,
                                                                        route[i_star + 2].destination)) + 1
            if j_star + 2 < len(route):
                assert route[j_star + 2].status in [1, 2]
                route[j_star + 2].est_action_time = int(raw_travel_time(
                    param.velocity, customer.location, route[j_star + 2].destination)) + 1

        elif append_switch_star == 1:  # append order to existing order
            route[i_star-1].orders.append(customer.name)
            route[i_star-1].est_action_time = wait_action_i_star  # special case: scalar instead of action
            route.insert(j_star, wait_action_j_star)
            route.insert(j_star, drive_action_j_star)
            # adapt est_action_time of altered drive actions
            if j_star + 2 < len(route):
                assert route[j_star + 2].status in [1, 2]
                route[j_star + 2].est_action_time = int(raw_travel_time(
                    param.velocity, route[j_star + 1].destination,
                    route[j_star + 2].destination)) + 1

        # extract gbdt features
        max_pre_shift = 0
        max_post_shift = 0
        restaurants_before_customer = 0
        customers_before_customer = 1  # for some reason its shifted in the training data of gdbt
        len_vehicle_route_to_customer = 1  # also shifted
        post_switch = False
        for index, action in enumerate(route):
            if action.status == 3 and not post_switch:
                restaurants_before_customer += 1
            if action.status == 4:
                if action.customer.name != customer.name and not post_switch:
                    max_pre_shift = max(action.customer.sum_shifted, max_pre_shift)
                    customers_before_customer += 1
                elif action.customer.name == customer.name:
                    post_switch = True
                    len_vehicle_route_to_customer = index + 1
                elif action.customer.name != customer.name and post_switch:
                    max_post_shift = max(action.customer.sum_shifted, max_post_shift)
        len_vehicle_route_total = len(route)
        customer.max_pre_shift = max_pre_shift
        customer.max_post_shift = max_post_shift
        restaurant = restaurant_dict[restaurant_name]
        if len(restaurant.est_time_queue) > 0:
            restaurant_queue = int(restaurant.est_time_queue[-1] + restaurant.avg_prep_time)
            if restaurant_queue > 0:
                restaurant_queue = int(restaurant_queue - order_time)
        else:
            restaurant_queue = int(restaurant.avg_prep_time)

        gbdt_input.append([etd + order_time, order_time, len_vehicle_route_to_customer, max_pre_shift, max_post_shift,
                           len_vehicle_route_total, customer.location[0], customer.location[1], restaurant.location[0],
                           restaurant.location[1], restaurant_queue,
                           restaurants_before_customer, customers_before_customer])

    etds = np.array([value[0] for value in etd_dict.values()])
    refined_etds = etds + model.predict(np.array(gbdt_input))
    for index, (key, value) in enumerate(etd_dict.items()):
        etd_dict[key] = [int(refined_etds[index]), value[1]]
    return etd_dict
