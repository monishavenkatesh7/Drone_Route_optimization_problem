import pandas as pd
import numpy as np
import json
from itertools import permutations, combinations, product

# Load input data from JSON file
with open("input.json", "r") as file:
    data = json.load(file)

# Preprocess Orders DataFrame
# - Add unique ID column starting from 1
# - Rename columns for clarity
orders_df = pd.DataFrame(data["orders"])
orders_df.insert(0, "ID", range(1, len(orders_df) + 1))
orders_df.rename(columns={
    "delivery_x": "x", 
    "delivery_y": "y", 
    "deadline": "time", 
    "package_weight": "weight"
}, inplace=True)

# Preprocess Drones DataFrame
# - Add unique Drone ID column starting from 1
# - Rename columns
# - Filter only available drones
drones_df = pd.DataFrame(data["drones"]["fleet"])
drones_df.insert(0, "Drone_ID", range(1, len(drones_df) + 1))
drones_df.rename(columns={
    "max_payload": "max_payload", 
    "max_distance": "max_dist"
}, inplace=True)
drones_df = drones_df[drones_df['available']].reset_index(drop=True)

# Generate all possible order combinations and their permutations
order_ids = orders_df['ID'].tolist()
all_combos_perms = []
for r in range(1, len(order_ids) + 1):
    for combo in combinations(order_ids, r):
        all_combos_perms.extend(permutations(combo))

# Calculate distances, weights, and constraints for each order combination
dist_data = []
for combo in all_combos_perms:
    # Extract coordinates for each order in the combination
    dist_list = [(
        orders_df.loc[orders_df['ID'] == order, 'x'].values[0], 
        orders_df.loc[orders_df['ID'] == order, 'y'].values[0]
    ) for order in combo]
    
    # Calculate Manhattan distances
    # 1. Distance from origin to each point
    distance_list = [abs(x) + abs(y) for (x, y) in dist_list]
    
    # 2. Relative distances between consecutive points
    relative_distance = [abs(dist_list[0][0]) + abs(dist_list[0][1])]
    for i in range(len(dist_list) - 1):
        x1, y1 = dist_list[i]
        x2, y2 = dist_list[i + 1]
        relative_distance.append(abs(x2 - x1) + abs(y2 - y1))
    
    # 3. Cumulative distances (progressive sum)
    cumulative_distance = [relative_distance[0]]
    for i in range(1, len(relative_distance)):
        cumulative_distance.append(cumulative_distance[-1] + relative_distance[i])
    
    # Total round trip distance
    total_distance = cumulative_distance[-1] + abs(dist_list[-1][0]) + abs(dist_list[-1][1])

    # Calculate total package weight
    total_weight = sum(orders_df.loc[orders_df['ID'].isin(combo), 'weight'])

    # Extract order delivery times
    time_list = orders_df.loc[orders_df['ID'].isin(combo), 'time'].tolist()

    # Check drone-specific constraints
    drone_columns = {}
    for _, drone in drones_df.iterrows():
        drone_id = drone['Drone_ID']
        max_payload = drone['max_payload']
        max_dist = drone['max_dist']
        speed = drone['speed']

        # Weight constraint check
        weight_check = 1 if total_weight <= max_payload else 0
        
        # Distance constraint check
        distance_check = 1 if total_distance <= max_dist else 0
        
        # Delivery time constraint check
        delivery_time_check = 1 if all(
            time >= cum_dist / speed 
            for time, cum_dist in zip(time_list, cumulative_distance)
        ) else 0

        # Overall feasibility for this drone
        overall_check = 1 if (weight_check and distance_check and delivery_time_check) else 0

        # Store constraint results
        drone_columns.update({
            f'drone_{drone_id}_weight': weight_check,
            f'drone_{drone_id}_distance': distance_check,
            f'drone_{drone_id}_delivery_time_constraints': delivery_time_check,
            f'drone_{drone_id}_overall': overall_check
        })

    # Append combination details
    dist_data.append({
        'Order_Combos': combo, 
        'dist_list': dist_list, 
        'distance_list': distance_list,
        'relative_distance': relative_distance,
        'cumulative_distance': cumulative_distance,
        'total_distance': total_distance,
        'weight': total_weight,
        'time': time_list,
        **drone_columns
    })

# Create DataFrame from calculated distances and constraints
order_data_df = pd.DataFrame(dist_data)

# Generate all possible drone assignments
# - Create lists of valid order combinations for each drone
# - Add a 0 placeholder for "no assignment"
combos_lists = []
for drone in drones_df['Drone_ID']:
    valid_combos = list(order_data_df[order_data_df[f'drone_{drone}_overall'] == 1]['Order_Combos'])
    combos_lists.append(valid_combos + [0])

# Generate all possible drone assignment combinations
all_combinations = list(product(*combos_lists))

# Validate drone assignments to ensure no order is assigned multiple times
def is_valid_combination(combination):
    orders = set()
    for combo in combination:
        if combo == 0:  # Ignore placeholder
            continue
        for order in combo:
            if order in orders:
                return False  # Order is repeated
            orders.add(order)
    return True

# Filter out invalid combinations
valid_combinations = [combo for combo in all_combinations if is_valid_combination(combo)]

# Create DataFrame with valid combinations
valid_combos_df = pd.DataFrame(valid_combinations, columns=[f'drone_{drone}' for drone in drones_df['Drone_ID']])
valid_combos_df.insert(0, 'Valid_combinations', valid_combos_df.apply(tuple, axis=1))

# Function to retrieve metrics for each drone's order combination
def get_drone_metrics(combo, column_name):
    return 0 if combo == 0 else order_data_df.loc[order_data_df['Order_Combos'] == combo, column_name].values[0]

# Extract detailed metrics for each drone's order combination
for drone in drones_df['Drone_ID']:
    valid_combos_df[f'drone_{drone}_weight'] = valid_combos_df[f'drone_{drone}'].apply(lambda x: get_drone_metrics(x, 'weight'))
    valid_combos_df[f'drone_{drone}_order_coord'] = valid_combos_df[f'drone_{drone}'].apply(lambda x: get_drone_metrics(x, 'dist_list'))
    valid_combos_df[f'drone_{drone}_distance'] = valid_combos_df[f'drone_{drone}'].apply(lambda x: get_drone_metrics(x, 'distance_list'))
    valid_combos_df[f'drone_{drone}_relative_distance'] = valid_combos_df[f'drone_{drone}'].apply(lambda x: get_drone_metrics(x, 'relative_distance'))
    valid_combos_df[f'drone_{drone}_cumulative_distance'] = valid_combos_df[f'drone_{drone}'].apply(lambda x: get_drone_metrics(x, 'cumulative_distance'))
    valid_combos_df[f'drone_{drone}_total_distance'] = valid_combos_df[f'drone_{drone}'].apply(lambda x: get_drone_metrics(x, 'total_distance'))

# Function to extract the last value from a list
def get_last_value(value):
    return value[-1] if isinstance(value, list) and value else 0

# Calculate total time for each drone based on speed
for drone, speed in zip(drones_df['Drone_ID'], drones_df['speed']):
    valid_combos_df[f'drone_{drone}_total_time'] = (
        valid_combos_df[f'drone_{drone}_cumulative_distance'].apply(get_last_value) / speed
    )

# Calculate total time and total distance across all drones
valid_combos_df["Total_Time"] = valid_combos_df[[f'drone_{drone}_total_time' for drone in drones_df["Drone_ID"]]].sum(axis=1)
valid_combos_df["Total_Distance"] = valid_combos_df[[f'drone_{drone}_total_distance' for drone in drones_df["Drone_ID"]]].sum(axis=1)

# Get all unique order IDs
all_order_ids = set(orders_df["ID"])

# Function to check if all orders are assigned
def check_all_orders_present(combination):
    combo_orders = set()
    for drone_combo in combination:
        if drone_combo != 0:  # Ignore empty assignments
            combo_orders.update(drone_combo)
    return 1 if combo_orders == all_order_ids else 0

# Flag combinations that include all orders
valid_combos_df["contains_all_orders"] = valid_combos_df["Valid_combinations"].apply(check_all_orders_present)

# Function to count total unique orders in a combination
def count_orders(combination):
    combo_orders = set()
    for drone_combo in combination:
        if drone_combo != 0:  # Ignore empty assignments
            combo_orders.update(drone_combo)
    return len(combo_orders)

# Add order count column
valid_combos_df["count"] = valid_combos_df["Valid_combinations"].apply(count_orders)

# Sort combinations by total time
valid_combos_df = valid_combos_df.sort_values(by="Total_Time", ascending=True).reset_index(drop=True)

# Select optimal combination
# Priority 1: Combinations covering all orders with least total time
# Priority 2: Combinations with maximum order coverage and least total time
if (valid_combos_df["contains_all_orders"] == 1).any():
    final_output = valid_combos_df[valid_combos_df["contains_all_orders"] == 1].nsmallest(1, "Total_Time")
else:
    max_count = valid_combos_df["count"].max()
    final_output = valid_combos_df[valid_combos_df["count"] == max_count].nsmallest(1, "Total_Time")

drone_list = []
order_list = []
distance_list = []
for drone,id in zip(list(drones_df['Drone_ID']), list(drones_df['id'])):
    orders = list(final_output[f'drone_{drone}'])[0]
    orders_1 = []
    if orders == 0:
         orders_1 = []
    else:
        for order in orders:
            orders_1.append(list(orders_df[orders_df['ID'] == order]['id'])[0])
    order_list.append(orders_1)
    distance_list.append(list(final_output[f'drone_{drone}_total_distance'])[0])
    drone_list.append(id)

json_dict = {"assignments":
             [{"drone":d,
               "orders": o,
               "total_distance": l} for d,o,l in zip(drone_list, order_list, distance_list)]
             }

# Save to a JSON file
with open("output.json", "w") as file:
    json.dump(json_dict, file, indent=4)