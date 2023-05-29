import pandas as pd
import matplotlib.pyplot as plt

class ProcessGameState:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        data = pd.read_parquet(file_path)
        return data

    def process_data(self, side=None, team = None):
        filtered_data = self.data[self.data.apply(lambda row: row['side'] == side and row['team'] == team, axis=1)]
        return filtered_data


def is_within_boundary(x, y, z, boundary_vertices):
    x_bounds, y_bounds, z_bounds = boundary_vertices
    return (x >= min(x_bounds) and x <= max(x_bounds) and
            y >= min(y_bounds) and y <= max(y_bounds) and
            z >= min(z_bounds) and z <= max(z_bounds))

def convert_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + int(seconds)

# Instantiate the ProcessGameState class
game_state_processor = ProcessGameState("game_state_frame_data.parquet")

# Define the boundary vertices
boundary_vertices = ([-1735, -2024, -2806, -2472, -1565], [250, 398, 742, 1233, 580], [285, 421])



# !!! QUESTION 1: Check if entering via the light blue boundary is a common strategy used by Team2 on T side
# !! DIRECTION:
# I'll get the number of rounds there is at least one player of Team2 on T in the light blue boundary
# Dead or alive isn't important, they can get shot trying to enter the site this way.
team2_t = game_state_processor.process_data(side='T', team = 'Team2')
# Total of rounds they are in T-side (in case there is overtime)
print('Total number of rounds')
print(len(team2_t['round_num'].drop_duplicates()))

team2_t_enter_light_blue = team2_t[team2_t.apply(lambda row: is_within_boundary(row['x'], row['y'], row['z'], boundary_vertices), axis=1)]
distinct_rounds = team2_t_enter_light_blue['round_num'].drop_duplicates()
print('Number of rounds Team2 entered through the light blue area')
print(len(distinct_rounds))

# ANALYZE
# In 2/15 rounds, there is at least 1 player from Team2 trying to enter the site via the light blue boundary
# It's up to the coaches, but to me, this isn't a common strategy


# !!! Question 2: Calculate the average timer that Team2 on T side enters "BombsiteB" with at least 2 rifles or SMGs

# !! DIRECTION:
# I'll get average the clock_time when they FIRST enter BombsiteB (highest clock_time when area_name is "BombsiteB")
# where there are at least 2 players from Team2 in T side holding a Rifle or SMG

# Players in T side Team2 that's alive at BombsiteB
team2_t_bombsite_b = team2_t[team2_t.apply(lambda row: row['area_name'] == 'BombsiteB' and row['is_alive'] == True, axis=1)][['inventory', 'area_name', 'player', 'seconds', 'clock_time', 'round_num']]

# Convert 'clock_time' to numeric representation in seconds
team2_t_bombsite_b['clock_time'] = team2_t_bombsite_b['clock_time'].apply(convert_to_seconds)

# Select the rows with the highest clock_time for each player
distinct_players = team2_t_bombsite_b.loc[team2_t_bombsite_b.groupby(['round_num', 'player'])['clock_time'].idxmax()]

# Group by round_num and count the number of players
grouped = distinct_players.groupby('round_num').size()

# Filter players with rifles or SMGs
rifles_smgs = distinct_players[distinct_players['inventory'].apply(lambda inv: any(item['weapon_class'] in ['Rifle', 'SMG'] for item in inv))]

# Filter rifles_smgs based on at least two players in the same round_num
grouped_rifles_smgs = rifles_smgs.groupby('round_num').size()
filtered_rifles_smgs = rifles_smgs[rifles_smgs['round_num'].isin(grouped_rifles_smgs[grouped_rifles_smgs >= 2].index)]
# Show players when Team 2 in T enters site with rifles or smgs
print('----------------------------------------------------------------')
print('Final Data Frame for Question 2')
print(filtered_rifles_smgs)
print('----------------------------------------------------------------')
print('Average time they got into Bombsite B')
print(filtered_rifles_smgs['clock_time'].mean())

# ANALYZE:
# With the average time when Team 2 in T enters site with at least 2 riflers or SMGs of 83.7s (01:23.7)
# they tend to enter the site quite soon when they're well-equipped


# QUESTION 3: Analyze the CT positions within "BombsiteB"
# !! DIRECTION:
# I'll draw a heatmap of the coordinates of the players in Team2 when they're defending inside BombsiteB as CT side
# I'll use matplotlib.pyplot to draw a 3D heatmap
# The plot can be found at the file named "3d_heatmap.png"

team2_ct = game_state_processor.process_data(side='CT', team = 'Team2')

# Players in CT side Team2 that's alive at BombsiteB
team2_ct_bombsite_b = team2_ct[team2_ct.apply(lambda row: row['area_name'] == 'BombsiteB' and row['is_alive'] == True, axis=1)][['x', 'y', 'z', 'area_name', 'player', 'seconds', 'clock_time', 'round_num', 'player']]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the coordinates as x, y, and z
x = team2_ct_bombsite_b['x']
y = team2_ct_bombsite_b['y']
z = team2_ct_bombsite_b['z']

# Create the scatter plot
sc = ax.scatter(x, y, z, c=z, cmap='viridis')

# Set the title and axis labels
ax.set_title('3D Heat Map of Coordinates')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a colorbar to indicate the z coordinate
plt.colorbar(sc)

# Save the plot as a file
plt.savefig('3d_heatmap.png')
