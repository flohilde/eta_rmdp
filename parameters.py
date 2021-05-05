# general parameters
area_size = [[2, 14], [4, 10]]
start_time_day = 0
end_time_day = 1440
n_days = 1
n_restaurants = 15
n_vehicles = 15

# dispatcher parameters
bundling = True
advanced_eta = False
n_simulations = None
simulation_nodes = None
n_approx_simulations = 4

# demand time parameters
n_lunch_mu = 200  # expected customers at lunch time
n_lunch_sigma = 10  # std of customers at lunch time
n_dinner_mu = 250  # expected customers at dinner time
n_dinner_sigma = 10  # std of customers at dinner time
t_lunch_mu = 12 * 60  # expected lunch time (in min)
t_lunch_sigma = 60  # std of lunch time (in min)
t_dinner_mu = 18 * 60  # expected dinner time (in min)
t_dinner_sigma = 60  # std of dinner time (in min)
time_error_std = 0

# customer parameters
n_fav_restaurants = 15
n_fav_restaurants_std = 0.0
time_constraint = 60
time_constraint_std = 0

# restaurant parameters
mu_prep_time = 8.3
sigma_prep_time = 1.3

# vehicle parameters
velocity = 0.5
mean_wait_parking = 2.5  # Average parking time of vehicles
std_wait_parking = 0.5  # std parking time of vehicles
mean_travel_time_perturbation = 0.0  # Average perturbation in travel times
std_travel_time_perturbation = 0.0  # std perturbation in travel times


visualize = False
write_data = True
write_to = "data/"


