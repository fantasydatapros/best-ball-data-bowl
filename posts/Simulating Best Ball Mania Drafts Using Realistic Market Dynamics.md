# Simulating Best Ball Mania Drafts Using Realistic Market Dynamics

## Tim Bryan

[@timbryan000](https://www.twitter.com/timbryan000)

>"I believe the market accurately reflects not the truth, which is what the efficient market hypothesis says, but it accurately and efficiently reflects everybody's opinion as to what's true." - Howard Marks

When it comes to modeling anything with an underlying market - stocks, commodities, sports - there are two different approaches. The first, fundamental, is where you build a model based on the underlying data of the system. In the case of fantasy football, a fundamental model might tackle player projections, strength of schedule, bye week analysis. The method I'll be using in this post is __technical modeling__. We'll only be using the movement of the market to find an optimal draft strategy. At the end you'll have a great understanding of how to pair this technical modeling strategy with your player projections and I'll give you the tools to simulate thousands of drafts quickly.


```python
# Basic Dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Data Loading and Preparation
Let's create a function for loading our data - the structure of 2021 and 2022 are a little different so we'll handle for that and then combine the DataFrames.


```python
def load_year_data(year):
    """
    Load the Best Ball data into a DataFrame, handles 2021 and 2022 differences in file structure
    """
    df = []
    # Determine the subdirectory based on the year
    subdir = '' if year == 2021 else '/fast'

    # Load regular season data
    for file in os.listdir(f'../data/{year}/regular_season{subdir}'):
        df.append(pd.read_csv(f'../data/{year}/regular_season{subdir}/{file}'))
    df = pd.concat(df, ignore_index=True)

    # Determine how to load post season data based on the year
    post_season_loaders = {'qf': [], 'sf': None, 'f': None}
    if year == 2021:
        post_season_loaders['qf'].append(pd.read_csv(
            f'../data/{year}/post_season/quarterfinals.csv'))
        post_season_loaders['sf'] = pd.read_csv(
            f'../data/{year}/post_season/semifinals.csv')
        post_season_loaders['f'] = pd.read_csv(
            f'../data/{year}/post_season/finals.csv')
    else:
        for file in os.listdir(f'../data/{year}/post_season/quarterfinals/'):
            post_season_loaders['qf'].append(pd.read_csv(
                f'../data/{year}/post_season/quarterfinals/{file}'))
        post_season_loaders['sf'] = pd.read_csv(
            f'../data/{year}/post_season/semifinals/part_00.csv')
        post_season_loaders['f'] = pd.read_csv(
            f'../data/{year}/post_season/finals/part_00.csv')

    post_season_loaders['qf'] = pd.concat(
        post_season_loaders['qf'], ignore_index=True)

    # Convert playoff dataframes to dictionaries and map onto df
    for k, playoffs_df in post_season_loaders.items():
        playoffs_dict = dict(
            zip(playoffs_df['tournament_entry_id'], [1]*len(playoffs_df)))
        df[k] = df['tournament_entry_id'].map(
            playoffs_dict).fillna(0).astype(int)

    # Add a column for the year
    df['year'] = year

    return df
```


```python
df21 = load_year_data(2021)
df22 = load_year_data(2022)
```


```python
# Combine the dataframes
df = pd.concat([df21, df22], ignore_index=True)
```


```python
# Drop columns that are not needed
df = df.drop(columns=['draft_entry_id', 'tournament_round_draft_entry_id'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>draft_id</th>
      <th>draft_time</th>
      <th>clock</th>
      <th>tournament_entry_id</th>
      <th>tournament_round_number</th>
      <th>player_name</th>
      <th>position_name</th>
      <th>bye_week</th>
      <th>projection_adp</th>
      <th>pick_order</th>
      <th>overall_pick_number</th>
      <th>team_pick_number</th>
      <th>pick_points</th>
      <th>roster_points</th>
      <th>playoff_team</th>
      <th>qf</th>
      <th>sf</th>
      <th>f</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64c5c57f-dfe6-49b7-a0c2-180a8e8e2ad9</td>
      <td>2021-07-20 05:54:15.422561+00:00</td>
      <td>30</td>
      <td>e762e1de-c639-431b-bbb6-1e30ee9e291f</td>
      <td>1</td>
      <td>Cam Akers</td>
      <td>RB</td>
      <td>11</td>
      <td>12.14</td>
      <td>10</td>
      <td>15</td>
      <td>2</td>
      <td>0.00</td>
      <td>1675.10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>64c5c57f-dfe6-49b7-a0c2-180a8e8e2ad9</td>
      <td>2021-07-20 05:54:15.422561+00:00</td>
      <td>30</td>
      <td>5e1f4e64-41fa-47ab-8c4b-5562fd1e9eb0</td>
      <td>1</td>
      <td>Brandin Cooks</td>
      <td>WR</td>
      <td>10</td>
      <td>91.48</td>
      <td>5</td>
      <td>92</td>
      <td>8</td>
      <td>103.60</td>
      <td>1554.92</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64c5c57f-dfe6-49b7-a0c2-180a8e8e2ad9</td>
      <td>2021-07-20 05:54:15.422561+00:00</td>
      <td>30</td>
      <td>f33b8689-4157-409d-950a-0b12244954e2</td>
      <td>1</td>
      <td>Noah Fant</td>
      <td>TE</td>
      <td>11</td>
      <td>100.64</td>
      <td>6</td>
      <td>102</td>
      <td>9</td>
      <td>72.30</td>
      <td>1656.58</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64c5c57f-dfe6-49b7-a0c2-180a8e8e2ad9</td>
      <td>2021-07-20 05:54:15.422561+00:00</td>
      <td>30</td>
      <td>aeba7ded-dfeb-4b96-bc59-501c4ca29202</td>
      <td>1</td>
      <td>Matt Ryan</td>
      <td>QB</td>
      <td>6</td>
      <td>138.23</td>
      <td>8</td>
      <td>152</td>
      <td>13</td>
      <td>97.56</td>
      <td>1732.12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>4</th>
      <td>64c5c57f-dfe6-49b7-a0c2-180a8e8e2ad9</td>
      <td>2021-07-20 05:54:15.422561+00:00</td>
      <td>30</td>
      <td>072a49a9-bc24-4176-b09b-e053fc9f05eb</td>
      <td>1</td>
      <td>A.J. Brown</td>
      <td>WR</td>
      <td>13</td>
      <td>23.25</td>
      <td>2</td>
      <td>23</td>
      <td>2</td>
      <td>103.50</td>
      <td>1614.82</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2021</td>
    </tr>
  </tbody>
</table>
</div>



## Define Player Rankings

In order to model the market, we first need to define the market. We will use __Average Draft Position (ADP)__ as our method for pricing players. I'll be using the earliest 2022 ADP rankings, but for the simulations coming up you really use any ranking system as long as it's in the format of the player_rankings DataFrame.


```python
# Filter down to just 2022
df = df[df['year'] == 2022]
```


```python
# Change draft_time to datetime
df['draft_time'] = pd.to_datetime(df['draft_time'], utc=True)

# Locate the first draft time for each player
mask = df.groupby(['player_name'])['draft_time'].transform('min') == df['draft_time']

# Filter down to this earliest draft time for each player
player_rankings = df.loc[mask, ['player_name','position_name', 'year', 'projection_adp', 'pick_points']].reset_index(drop=True)
```


```python
# Drop players where position is FB
player_rankings = player_rankings[player_rankings['position_name'] != 'FB']

# Drop players where adp_projection is 0
player_rankings = player_rankings[player_rankings['projection_adp'] != 0]
```


```python
player_rankings.sort_values('projection_adp').head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>position_name</th>
      <th>year</th>
      <th>projection_adp</th>
      <th>pick_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>142</th>
      <td>Jonathan Taylor</td>
      <td>RB</td>
      <td>2022</td>
      <td>1.0</td>
      <td>125.00</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Christian McCaffrey</td>
      <td>RB</td>
      <td>2022</td>
      <td>2.0</td>
      <td>227.66</td>
    </tr>
    <tr>
      <th>203</th>
      <td>Cooper Kupp</td>
      <td>WR</td>
      <td>2022</td>
      <td>3.0</td>
      <td>162.50</td>
    </tr>
    <tr>
      <th>170</th>
      <td>Derrick Henry</td>
      <td>RB</td>
      <td>2022</td>
      <td>4.0</td>
      <td>214.36</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Ja'Marr Chase</td>
      <td>WR</td>
      <td>2022</td>
      <td>5.0</td>
      <td>154.90</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Austin Ekeler</td>
      <td>RB</td>
      <td>2022</td>
      <td>6.0</td>
      <td>246.20</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Justin Jefferson</td>
      <td>WR</td>
      <td>2022</td>
      <td>7.0</td>
      <td>247.26</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Najee Harris</td>
      <td>RB</td>
      <td>2022</td>
      <td>8.0</td>
      <td>130.56</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Tyreek Hill</td>
      <td>WR</td>
      <td>2022</td>
      <td>9.0</td>
      <td>236.70</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Davante Adams</td>
      <td>WR</td>
      <td>2022</td>
      <td>10.0</td>
      <td>236.90</td>
    </tr>
  </tbody>
</table>
</div>



## Gathering Market Dynamics

Next, let's gather market dynamics from the Underdog Fantasy datasets. The market dyamics we'll be using are how people are drafting relative to ADP. We're calling this __market delta__ and it's simply the ADP minus the overall pick number. Some players employ a 0 RB strategy, others will "reach" relative to the market to secure a top QB - there's lots of different strategies and we can simulate them all.


```python
# Define the positions
positions = ['QB', 'RB', 'TE', 'WR']
```


```python
# Create a column for delta from ADP
df['market_delta'] = df['projection_adp'] - df['overall_pick_number']

# Create a column for delta from ADP by position
for pos in positions:
    df['market_delta_' +
        pos] = np.where(df['position_name'] == pos, df['market_delta'], 0)
```


```python
# Drop the columns we don't need
market_dynamics_df = df.drop(columns=['draft_time', 'tournament_round_number',
                                      'player_name', 'position_name', 'bye_week',
                                      'projection_adp', 'overall_pick_number',
                                      'qf', 'sf', 'f', 'year', 'clock', 'market_delta'])
```


```python
# Add columns for market delta and standard deviation thereof by position
market_dynamics_df = market_dynamics_df.groupby('tournament_entry_id').agg({
    'market_delta_QB': ['mean', 'std'],
    'market_delta_RB': ['mean', 'std'],
    'market_delta_TE': ['mean', 'std'],
    'market_delta_WR': ['mean', 'std']
})

# Flatten the column names
market_dynamics_df.columns = ['_'.join(col).strip() for col in market_dynamics_df.columns.values]
```


```python
market_dynamics_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>market_delta_QB_mean</th>
      <th>market_delta_QB_std</th>
      <th>market_delta_RB_mean</th>
      <th>market_delta_RB_std</th>
      <th>market_delta_TE_mean</th>
      <th>market_delta_TE_std</th>
      <th>market_delta_WR_mean</th>
      <th>market_delta_WR_std</th>
    </tr>
    <tr>
      <th>tournament_entry_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0000146a-e245-49c3-8a4e-8e739cfe1e46</th>
      <td>0.377778</td>
      <td>1.259302</td>
      <td>5.035000</td>
      <td>14.108540</td>
      <td>0.021111</td>
      <td>2.782994</td>
      <td>0.870000</td>
      <td>3.350938</td>
    </tr>
    <tr>
      <th>0000225d-54fa-4334-aa1e-41e7adaa2542</th>
      <td>0.006667</td>
      <td>1.790672</td>
      <td>-0.147222</td>
      <td>2.457519</td>
      <td>-0.446111</td>
      <td>1.304730</td>
      <td>-0.466111</td>
      <td>1.952305</td>
    </tr>
    <tr>
      <th>00004939-5e99-4dfb-b597-d6463eacf6aa</th>
      <td>4.343889</td>
      <td>12.045567</td>
      <td>0.462222</td>
      <td>2.107054</td>
      <td>-0.539444</td>
      <td>4.492765</td>
      <td>-2.013333</td>
      <td>4.747164</td>
    </tr>
    <tr>
      <th>0000766d-7308-495e-ad86-6ddb69dbc3ab</th>
      <td>0.023333</td>
      <td>0.925063</td>
      <td>-0.858333</td>
      <td>4.370150</td>
      <td>-0.015556</td>
      <td>1.465297</td>
      <td>-2.045000</td>
      <td>3.523895</td>
    </tr>
    <tr>
      <th>00007ef0-3100-479d-be3c-e7263edb4b5b</th>
      <td>1.474444</td>
      <td>5.236630</td>
      <td>-2.068333</td>
      <td>4.801088</td>
      <td>-0.075000</td>
      <td>0.536823</td>
      <td>-1.862778</td>
      <td>3.800692</td>
    </tr>
  </tbody>
</table>
</div>



Take the first row as an example; we can see this person on average drafted their RBs 5 picks behind ADP, plus or minus 14 picks. This is a high standard deviation, but this person drafted RBs whose ADP was lower than their current pick - they were consistently reaching for RBs relative to the market. Below you'll see that a majority of players draft close to ADP with some tails on either side. Something interesting to note is the larger than normal right tail for RBs. There's so much talk of the 0 RB strategy but it seems the market still reaches for RBs more than other positions.


```python
# Select only the mean columns
mean_columns = ['market_delta_QB_mean', 'market_delta_RB_mean',
                'market_delta_TE_mean', 'market_delta_WR_mean']
mean_df = market_dynamics_df[mean_columns]

# Plot histograms
mean_df.hist(bins=30, figsize=(15, 10), grid=False,
             color='#86bf91', zorder=2, rwidth=0.9)

plt.tight_layout()
plt.show()
```


    
![png](Simulating%20Best%20Ball%20Mania%20Drafts%20Using%20Realistic%20Market%20Dynamics_files/Simulating%20Best%20Ball%20Mania%20Drafts%20Using%20Realistic%20Market%20Dynamics_25_0.png)
    


## Draft Simulator
Below you'll see two different classes. The first is a strategy generator, which will allow us to load the market dynamics strategy of any of these Best Ball strategies OR even our own custom strategy. The second is a very complex draft simulator class. You can review the code yourself but I'll explain in simple terms what it does:

1. Initializes 12 teams with the standard rosters of Best Ball Mania (QB1, RB1, RB2, WR1, WR2, WR3, TE1, FLEX, BENCH x 10)
2. If no strategy is loaded, by default each team will select the player that has the lowest ADP and is available
3. If a strategy is loaded for the team, it will calculate __adjusted ADP__ based on a normal distribution of the mean market delta and standad deviation for each position. For example, if we set a team's market delta for QB to +5 plus or minus 1 then the adjusted ADP for Josh Allen may go from 19 to 12. The simulation then selects the player with the lowest available adjusted ADP.
4. For each pick, the simulation is checking if the number of required roster slots is equal to the number of picks remaining. If this is the case, then the sim will select the lowest available ADP or adjusted ADP to fill those slots.
5. Reorder the players so the highest scoring players are in the starting slots. I realize Best Ball rosters are reordered every week, but this is a close approximation.

It's not a perfect representation of how a market works, but it gives us an idea of how the flow of a draft can go depending on how you set the teams' valuation of different positions. As a reminder, a positive market delta means the team is willing to wait on that position, a negative market delta means the team will reach for that position.


```python
class StrategyGenerator:
    def __init__(self, num_teams):
        self.num_teams = num_teams

    def generate(self, strategy_dict=None):
        if strategy_dict is None:
            strategy_dict = {}

        strategies = {i: None for i in range(self.num_teams)}
        strategies.update(strategy_dict)

        return strategies
```


```python
class DraftSimulator:
    def __init__(self, available_players: pd.DataFrame, strategies: dict):
        self.available_players = available_players.copy()
        self.strategies = strategies
        self.team_rosters = {i: {'QB1': None, 'RB1': None, 'RB2': None, 'WR1': None, 'WR2': None, 'WR3': None,
                                 'TE1': None, 'FLEX': None, 'BENCH': []} for i in range(12)}

        # Create snake draft order
        self.draft_order = []
        for i in range(20):  # 20 rounds
            round_order = list(range(12)) if i % 2 == 0 else list(
                range(11, -1, -1))  # Reverse order every other round
            self.draft_order.extend(round_order)

    def _calculate_adjusted_projection_adp(self, player, strategy):
        if strategy and player['position_name'] in strategy:
            mean_market_delta = strategy[player['position_name']]['mean']
            stddev_market_delta = strategy[player['position_name']]['stddev']
            return np.random.normal(loc=player['projection_adp'] - mean_market_delta, scale=stddev_market_delta)
        return player['projection_adp']

    def _best_available_player(self, team_number, remaining_picks):
        # Calculate adjusted_projection_adp for all available players first
        self.available_players['adjusted_projection_adp'] = self.available_players.apply(
            self._calculate_adjusted_projection_adp, axis=1, args=(self.strategies.get(team_number),))

        # If remaining slots equal remaining picks, must draft needed positions
        needed_positions = [pos for pos, player in self.team_rosters[team_number].items(
        ) if player is None and pos != 'BENCH']
        if len(needed_positions) == remaining_picks:
            for position in needed_positions:
                position_players = self.available_players[self.available_players['position_name']
                                                          == position[:-1] if position not in ['FLEX', 'BENCH'] else position]
                if len(position_players) > 0:
                    return position_players.nsmallest(1, 'adjusted_projection_adp').iloc[0]

        # Otherwise, return player with lowest adjusted ADP
        if len(self.available_players) > 0:
            return self.available_players.nsmallest(1, 'adjusted_projection_adp').iloc[0]

        # No players left to draft, throw an exception
        if len(self.available_players) == 0:
            raise Exception('No players left to draft. Check your drafting logic.')

    def _add_to_team(self, team_number, player):
        team = self.team_rosters[team_number]
        position = player['position_name'] + '1'
        slot = None
        if team[position] is None:
            team[position] = player
            slot = position
        elif position[:-1] + '2' in team and team[position[:-1] + '2'] is None:
            team[position[:-1] + '2'] = player
            slot = position[:-1] + '2'
        elif position[:-1] == 'WR' and team['WR3'] is None:
            team['WR3'] = player
            slot = 'WR3'
        elif position[:-1] in ['RB', 'WR', 'TE'] and team['FLEX'] is None:
            team['FLEX'] = player
            slot = 'FLEX'
        else:
            team['BENCH'].append(player)
            slot = 'BENCH'

        self.available_players.drop(player.name, inplace=True)
        return slot

    def simulate_draft(self):
        results = []
        for i, team_number in enumerate(self.draft_order):
            remaining_picks = len(self.draft_order) - i
            player = self._best_available_player(team_number, remaining_picks)
            slot = self._add_to_team(team_number, player)
            result = {
                'player_name': player['player_name'],
                'position_name': player['position_name'],
                'slot': slot,
                'year': player['year'],
                'projection_adp': player['projection_adp'],
                'adjusted_projection_adp': player['adjusted_projection_adp'],
                'team_number': team_number,
                'round_number': i//12 + 1,
                'pick_number': i + 1,
                'pick_points': player['pick_points'],  # add this line
            }
            results.append(result)

        return pd.DataFrame(results)
```

As a test, I'll just set the strategy of the team picking first. We'll make them a QB happy team, so I'm going to set their QB market delta to 5 and add some variance.


```python
# Example strategy dictionary
example_strategy = {
    'market_delta_QB_mean': 5,
    'market_delta_QB_std': 3,
    'market_delta_RB_mean': 0,
    'market_delta_RB_std': 0,
    'market_delta_TE_mean': 0,
    'market_delta_TE_std': 0,
    'market_delta_WR_mean': 0,
    'market_delta_WR_std': 0
}

# Package the strategy dictionary into a single object
team_strategy = {
    'QB': {'mean': example_strategy['market_delta_QB_mean'], 'stddev': example_strategy['market_delta_QB_std']},
    'RB': {'mean': example_strategy['market_delta_RB_mean'], 'stddev': example_strategy['market_delta_RB_std']},
    'WR': {'mean': example_strategy['market_delta_WR_mean'], 'stddev': example_strategy['market_delta_WR_std']},
    'TE': {'mean': example_strategy['market_delta_TE_mean'], 'stddev': example_strategy['market_delta_TE_std']},
}
```


```python
# Create strategy dict for all teams (teams without a strategy have None)
strategy_dict = {0: team_strategy}  # Team numbers start at 0

# Generate strategies
generator = StrategyGenerator(num_teams=12)
strategies = generator.generate(strategy_dict)
```

Now let's simulate the draft and see how our adjustments effect the outcome!


```python
# Build the draft simulator
draft = DraftSimulator(available_players=player_rankings, strategies=strategies)
```


```python
# Simulate the draft
results = draft.simulate_draft()
```

As you can see below, our adjustments made the team #0 jump on Patrick Mahomes at the end of the second round. You can imagine how turning the dials on multiple different team strategies and running these sims hundreds or thousands of times would gives us a ton of different outcomes.


```python
results.sort_values(by='pick_number', ascending=True).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>position_name</th>
      <th>slot</th>
      <th>year</th>
      <th>projection_adp</th>
      <th>adjusted_projection_adp</th>
      <th>team_number</th>
      <th>round_number</th>
      <th>pick_number</th>
      <th>pick_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jonathan Taylor</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>125.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Christian McCaffrey</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>2.0</td>
      <td>2.000000</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>227.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cooper Kupp</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>3.0</td>
      <td>3.000000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>162.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Derrick Henry</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>4.0</td>
      <td>4.000000</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>214.36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ja'Marr Chase</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>5.0</td>
      <td>5.000000</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>154.90</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Austin Ekeler</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>6.0</td>
      <td>6.000000</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>246.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Justin Jefferson</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>7.0</td>
      <td>7.000000</td>
      <td>6</td>
      <td>1</td>
      <td>7</td>
      <td>247.26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Najee Harris</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>8.0</td>
      <td>8.000000</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>130.56</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tyreek Hill</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>9.0</td>
      <td>9.000000</td>
      <td>8</td>
      <td>1</td>
      <td>9</td>
      <td>236.70</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Davante Adams</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>10.0</td>
      <td>10.000000</td>
      <td>9</td>
      <td>1</td>
      <td>10</td>
      <td>236.90</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dalvin Cook</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>11.0</td>
      <td>11.000000</td>
      <td>10</td>
      <td>1</td>
      <td>11</td>
      <td>170.90</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Deebo Samuel</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>12.0</td>
      <td>12.000000</td>
      <td>11</td>
      <td>1</td>
      <td>12</td>
      <td>137.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Joe Mixon</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>13.0</td>
      <td>13.000000</td>
      <td>11</td>
      <td>2</td>
      <td>13</td>
      <td>136.20</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Travis Kelce</td>
      <td>TE</td>
      <td>TE1</td>
      <td>2022</td>
      <td>14.0</td>
      <td>14.000000</td>
      <td>10</td>
      <td>2</td>
      <td>14</td>
      <td>216.90</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Stefon Diggs</td>
      <td>WR</td>
      <td>WR2</td>
      <td>2022</td>
      <td>15.0</td>
      <td>15.000000</td>
      <td>9</td>
      <td>2</td>
      <td>15</td>
      <td>217.50</td>
    </tr>
    <tr>
      <th>15</th>
      <td>A.J. Brown</td>
      <td>WR</td>
      <td>WR2</td>
      <td>2022</td>
      <td>16.0</td>
      <td>16.000000</td>
      <td>8</td>
      <td>2</td>
      <td>16</td>
      <td>184.60</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Alvin Kamara</td>
      <td>RB</td>
      <td>RB2</td>
      <td>2022</td>
      <td>17.0</td>
      <td>17.000000</td>
      <td>7</td>
      <td>2</td>
      <td>17</td>
      <td>121.40</td>
    </tr>
    <tr>
      <th>17</th>
      <td>D'Andre Swift</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>18.0</td>
      <td>18.000000</td>
      <td>6</td>
      <td>2</td>
      <td>18</td>
      <td>79.30</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Josh Allen</td>
      <td>QB</td>
      <td>QB1</td>
      <td>2022</td>
      <td>19.0</td>
      <td>19.000000</td>
      <td>5</td>
      <td>2</td>
      <td>19</td>
      <td>193.96</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Nick Chubb</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>20.0</td>
      <td>20.000000</td>
      <td>4</td>
      <td>2</td>
      <td>20</td>
      <td>214.50</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Javonte Williams</td>
      <td>RB</td>
      <td>RB2</td>
      <td>2022</td>
      <td>21.0</td>
      <td>21.000000</td>
      <td>3</td>
      <td>2</td>
      <td>21</td>
      <td>30.80</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CeeDee Lamb</td>
      <td>WR</td>
      <td>WR2</td>
      <td>2022</td>
      <td>22.0</td>
      <td>22.000000</td>
      <td>2</td>
      <td>2</td>
      <td>22</td>
      <td>169.80</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Mark Andrews</td>
      <td>TE</td>
      <td>TE1</td>
      <td>2022</td>
      <td>23.0</td>
      <td>23.000000</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>128.90</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Patrick Mahomes</td>
      <td>QB</td>
      <td>QB1</td>
      <td>2022</td>
      <td>28.0</td>
      <td>21.451329</td>
      <td>0</td>
      <td>2</td>
      <td>24</td>
      <td>200.16</td>
    </tr>
    <tr>
      <th>24</th>
      <td>DK Metcalf</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>24.0</td>
      <td>24.000000</td>
      <td>0</td>
      <td>3</td>
      <td>25</td>
      <td>138.20</td>
    </tr>
  </tbody>
</table>
</div>



And here's team #0's full team for reference. You can see they have an absurd amount of QBs, but the logic in the class ensures they are able to fill out a full roster regardless.


```python
results[results['team_number']== 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>position_name</th>
      <th>slot</th>
      <th>year</th>
      <th>projection_adp</th>
      <th>adjusted_projection_adp</th>
      <th>team_number</th>
      <th>round_number</th>
      <th>pick_number</th>
      <th>pick_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jonathan Taylor</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>125.00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Patrick Mahomes</td>
      <td>QB</td>
      <td>QB1</td>
      <td>2022</td>
      <td>28.00</td>
      <td>21.451329</td>
      <td>0</td>
      <td>2</td>
      <td>24</td>
      <td>200.16</td>
    </tr>
    <tr>
      <th>24</th>
      <td>DK Metcalf</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>24.00</td>
      <td>24.000000</td>
      <td>0</td>
      <td>3</td>
      <td>25</td>
      <td>138.20</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Joe Burrow</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>50.00</td>
      <td>38.691575</td>
      <td>0</td>
      <td>4</td>
      <td>48</td>
      <td>266.18</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Kyler Murray</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>52.00</td>
      <td>45.564941</td>
      <td>0</td>
      <td>5</td>
      <td>49</td>
      <td>152.42</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Matthew Stafford</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>81.00</td>
      <td>71.962715</td>
      <td>0</td>
      <td>6</td>
      <td>72</td>
      <td>84.24</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Jalen Hurts</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>77.00</td>
      <td>68.502247</td>
      <td>0</td>
      <td>7</td>
      <td>73</td>
      <td>337.88</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Kirk Cousins</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>97.00</td>
      <td>90.779123</td>
      <td>0</td>
      <td>8</td>
      <td>96</td>
      <td>116.40</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Kenneth Walker</td>
      <td>RB</td>
      <td>RB2</td>
      <td>2022</td>
      <td>96.00</td>
      <td>96.000000</td>
      <td>0</td>
      <td>9</td>
      <td>97</td>
      <td>126.20</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Melvin Gordon</td>
      <td>RB</td>
      <td>FLEX</td>
      <td>2022</td>
      <td>120.00</td>
      <td>120.000000</td>
      <td>0</td>
      <td>10</td>
      <td>120</td>
      <td>36.20</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Ryan Tannehill</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>125.00</td>
      <td>118.553852</td>
      <td>0</td>
      <td>11</td>
      <td>121</td>
      <td>155.54</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Daniel Jones</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>147.00</td>
      <td>139.723449</td>
      <td>0</td>
      <td>12</td>
      <td>144</td>
      <td>131.80</td>
    </tr>
    <tr>
      <th>144</th>
      <td>Julio Jones</td>
      <td>WR</td>
      <td>WR2</td>
      <td>2022</td>
      <td>144.00</td>
      <td>144.000000</td>
      <td>0</td>
      <td>13</td>
      <td>145</td>
      <td>38.30</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Ronald Jones</td>
      <td>RB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>168.00</td>
      <td>168.000000</td>
      <td>0</td>
      <td>14</td>
      <td>168</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Nico Collins</td>
      <td>WR</td>
      <td>WR3</td>
      <td>2022</td>
      <td>169.00</td>
      <td>169.000000</td>
      <td>0</td>
      <td>15</td>
      <td>169</td>
      <td>55.00</td>
    </tr>
    <tr>
      <th>191</th>
      <td>Cole Beasley</td>
      <td>WR</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>192.00</td>
      <td>192.000000</td>
      <td>0</td>
      <td>16</td>
      <td>192</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Carson Wentz</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>201.00</td>
      <td>192.714013</td>
      <td>0</td>
      <td>17</td>
      <td>193</td>
      <td>28.78</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Feleipe Franks</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>215.94</td>
      <td>205.207254</td>
      <td>0</td>
      <td>18</td>
      <td>216</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>216</th>
      <td>Drew Brees</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>215.96</td>
      <td>206.668227</td>
      <td>0</td>
      <td>19</td>
      <td>217</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>239</th>
      <td>Chigoziem Okonkwo</td>
      <td>TE</td>
      <td>TE1</td>
      <td>2022</td>
      <td>215.98</td>
      <td>215.980000</td>
      <td>0</td>
      <td>20</td>
      <td>240</td>
      <td>43.80</td>
    </tr>
  </tbody>
</table>
</div>



## Large Scale Simulation
Now let's try out different market dynamic strategies and simulate them on a large scale. To do this we need to package our DraftSimulator() class into a DraftSimulatorManager() class. This will allow us to run N simulations of varying strategies.


```python
class DraftSimulationManager:
    def __init__(self, available_players: pd.DataFrame, strategies: dict):
        self.available_players = available_players
        self.strategies = strategies
        self.team_points = {i: [] for i in range(12)}

    def simulate_draft(self):
        draft_simulator = DraftSimulator(self.available_players.copy(), self.strategies)
        draft_results = draft_simulator.simulate_draft()
        team_points = draft_results.groupby('team_number').apply(lambda x: x[x['slot'] != 'BENCH']['pick_points'].sum())
        for team_number, points in team_points.iteritems():
            self.team_points[team_number].append(points)

    def run_simulations(self, n: int):
        for _ in range(n):
            self.simulate_draft()

    def points_stats_df(self):
        points_stats = {}
        for team_number, points in self.team_points.items():
            points_array = np.array(points)
            points_stats[team_number] = {
                'Median': np.median(points_array),
                'Mean': np.mean(points_array),
                'Q1': np.percentile(points_array, 25),
                'Q2': np.percentile(points_array, 50),
                'Q3': np.percentile(points_array, 75),
                'Max': np.max(points_array)
            }
        points_stats_df = pd.DataFrame(points_stats).T.rename_axis('team_number').reset_index()
        return points_stats_df
```

To test, we'll just leave our strategy alone and see how a QB happy team performs against ADP strict teams across 10 simulations


```python
simulation_manager = DraftSimulationManager(player_rankings, strategies)
simulation_manager.run_simulations(10)
points_stats_df = simulation_manager.points_stats_df()
```

As you can see below, team #0's strategy did not pay off. Next, we'll simulate different market dynamics strategies from Best Ball entries.


```python
points_stats_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team_number</th>
      <th>Median</th>
      <th>Mean</th>
      <th>Q1</th>
      <th>Q2</th>
      <th>Q3</th>
      <th>Max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>777.96</td>
      <td>780.040</td>
      <td>762.860</td>
      <td>777.96</td>
      <td>785.86</td>
      <td>873.76</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1183.32</td>
      <td>1135.940</td>
      <td>1063.320</td>
      <td>1183.32</td>
      <td>1183.32</td>
      <td>1189.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>864.90</td>
      <td>871.140</td>
      <td>839.100</td>
      <td>864.90</td>
      <td>906.30</td>
      <td>906.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>743.86</td>
      <td>746.728</td>
      <td>743.860</td>
      <td>743.86</td>
      <td>761.88</td>
      <td>807.38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1038.16</td>
      <td>1044.670</td>
      <td>1016.360</td>
      <td>1038.16</td>
      <td>1059.86</td>
      <td>1125.36</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>1091.76</td>
      <td>1063.260</td>
      <td>1045.935</td>
      <td>1091.76</td>
      <td>1091.76</td>
      <td>1091.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>881.86</td>
      <td>891.300</td>
      <td>881.860</td>
      <td>881.86</td>
      <td>881.86</td>
      <td>976.26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>710.60</td>
      <td>677.858</td>
      <td>628.745</td>
      <td>710.60</td>
      <td>710.60</td>
      <td>710.60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1036.44</td>
      <td>1025.098</td>
      <td>1036.440</td>
      <td>1036.44</td>
      <td>1036.44</td>
      <td>1036.44</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>872.74</td>
      <td>872.740</td>
      <td>872.740</td>
      <td>872.74</td>
      <td>872.74</td>
      <td>872.74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>811.50</td>
      <td>811.500</td>
      <td>811.500</td>
      <td>811.50</td>
      <td>811.50</td>
      <td>811.50</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>709.60</td>
      <td>709.600</td>
      <td>709.600</td>
      <td>709.60</td>
      <td>709.60</td>
      <td>709.60</td>
    </tr>
  </tbody>
</table>
</div>



## Simulating Actual Market Dynamics Strategies
This massive sample of different Best Ball entries gives us a quasi-realistic enviornment to simulate in. We can set one strategy as the __dependent variable__ and sample 11 random entries from real Best Ball entries as the __independent variable__ to emulate a real market enviornment.

Let's start by simulating the 0-RB strategy. In this strategy, the person drafting heavily underweights the importance of RBs on the roster.


```python
# Set the 0-RB strategy
zero_rb = {
    'market_delta_QB_mean': 0,
    'market_delta_QB_std': 0,
    'market_delta_RB_mean': -5,
    'market_delta_RB_std': 3,
    'market_delta_TE_mean': 0,
    'market_delta_TE_std': 0,
    'market_delta_WR_mean': 0,
    'market_delta_WR_std': 0
}

# Package the strategy dictionary into a single object
zero_rb_team_strategy = {
    'QB': {'mean': zero_rb['market_delta_QB_mean'], 'stddev': zero_rb['market_delta_QB_std']},
    'RB': {'mean': zero_rb['market_delta_RB_mean'], 'stddev': zero_rb['market_delta_RB_std']},
    'WR': {'mean': zero_rb['market_delta_WR_mean'], 'stddev': zero_rb['market_delta_WR_std']},
    'TE': {'mean': zero_rb['market_delta_TE_mean'], 'stddev': zero_rb['market_delta_TE_std']},
}

# Start the strategy dictionary with the 0-RB strategy for team 0
strategy_dict = {0: zero_rb_team_strategy}

# Sample 11 other strategies from the DataFrame
sampled_strategies = market_dynamics_df.sample(11)

# Add the sampled strategies to the strategy dictionary
for i, (index, row) in enumerate(sampled_strategies.iterrows()):
    strategy = {
        'QB': {'mean': row['market_delta_QB_mean'], 'stddev': row['market_delta_QB_std']},
        'RB': {'mean': row['market_delta_RB_mean'], 'stddev': row['market_delta_RB_std']},
        'WR': {'mean': row['market_delta_WR_mean'], 'stddev': row['market_delta_WR_std']},
        'TE': {'mean': row['market_delta_TE_mean'], 'stddev': row['market_delta_TE_std']}
    }
    strategy_dict[i+1] = strategy
```


```python
# Build the draft simulator
draft = DraftSimulator(available_players=player_rankings, strategies=strategy_dict)
```


```python
# Simulate the draft once just to test
results = draft.simulate_draft()
```


```python
# Look at the first two rounds
results.sort_values('pick_number', ascending=True).head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>position_name</th>
      <th>slot</th>
      <th>year</th>
      <th>projection_adp</th>
      <th>adjusted_projection_adp</th>
      <th>team_number</th>
      <th>round_number</th>
      <th>pick_number</th>
      <th>pick_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cooper Kupp</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>3.0</td>
      <td>3.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>162.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Christian McCaffrey</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>2.0</td>
      <td>-0.171291</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>227.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jonathan Taylor</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>1.0</td>
      <td>-1.431191</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>125.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ja'Marr Chase</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>5.0</td>
      <td>-6.779563</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>154.90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Davante Adams</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>10.0</td>
      <td>2.428839</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>236.90</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Austin Ekeler</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>6.0</td>
      <td>3.956579</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>246.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Derrick Henry</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>4.0</td>
      <td>0.830971</td>
      <td>6</td>
      <td>1</td>
      <td>7</td>
      <td>214.36</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Justin Jefferson</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>7.0</td>
      <td>7.125439</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>247.26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tyreek Hill</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>9.0</td>
      <td>1.165079</td>
      <td>8</td>
      <td>1</td>
      <td>9</td>
      <td>236.70</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Najee Harris</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>8.0</td>
      <td>4.977175</td>
      <td>9</td>
      <td>1</td>
      <td>10</td>
      <td>130.56</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Patrick Mahomes</td>
      <td>QB</td>
      <td>QB1</td>
      <td>2022</td>
      <td>28.0</td>
      <td>5.003400</td>
      <td>10</td>
      <td>1</td>
      <td>11</td>
      <td>200.16</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A.J. Brown</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>16.0</td>
      <td>5.101219</td>
      <td>11</td>
      <td>1</td>
      <td>12</td>
      <td>184.60</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Leonard Fournette</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>29.0</td>
      <td>7.972971</td>
      <td>11</td>
      <td>2</td>
      <td>13</td>
      <td>140.90</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Dalvin Cook</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>11.0</td>
      <td>7.574240</td>
      <td>10</td>
      <td>2</td>
      <td>14</td>
      <td>170.90</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Travis Kelce</td>
      <td>TE</td>
      <td>TE1</td>
      <td>2022</td>
      <td>14.0</td>
      <td>7.925679</td>
      <td>9</td>
      <td>2</td>
      <td>15</td>
      <td>216.90</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Stefon Diggs</td>
      <td>WR</td>
      <td>WR2</td>
      <td>2022</td>
      <td>15.0</td>
      <td>6.613392</td>
      <td>8</td>
      <td>2</td>
      <td>16</td>
      <td>217.50</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Joe Mixon</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>13.0</td>
      <td>10.355681</td>
      <td>7</td>
      <td>2</td>
      <td>17</td>
      <td>136.20</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Deebo Samuel</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>12.0</td>
      <td>10.943024</td>
      <td>6</td>
      <td>2</td>
      <td>18</td>
      <td>137.00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>D'Andre Swift</td>
      <td>RB</td>
      <td>RB2</td>
      <td>2022</td>
      <td>18.0</td>
      <td>12.123193</td>
      <td>5</td>
      <td>2</td>
      <td>19</td>
      <td>79.30</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Josh Allen</td>
      <td>QB</td>
      <td>QB1</td>
      <td>2022</td>
      <td>19.0</td>
      <td>15.260325</td>
      <td>4</td>
      <td>2</td>
      <td>20</td>
      <td>193.96</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Alvin Kamara</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>17.0</td>
      <td>8.418264</td>
      <td>3</td>
      <td>2</td>
      <td>21</td>
      <td>121.40</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Nick Chubb</td>
      <td>RB</td>
      <td>RB2</td>
      <td>2022</td>
      <td>20.0</td>
      <td>16.605769</td>
      <td>2</td>
      <td>2</td>
      <td>22</td>
      <td>214.50</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CeeDee Lamb</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>22.0</td>
      <td>18.790836</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>169.80</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Mark Andrews</td>
      <td>TE</td>
      <td>TE1</td>
      <td>2022</td>
      <td>23.0</td>
      <td>23.000000</td>
      <td>0</td>
      <td>2</td>
      <td>24</td>
      <td>128.90</td>
    </tr>
    <tr>
      <th>24</th>
      <td>DK Metcalf</td>
      <td>WR</td>
      <td>WR2</td>
      <td>2022</td>
      <td>24.0</td>
      <td>24.000000</td>
      <td>0</td>
      <td>3</td>
      <td>25</td>
      <td>138.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Look at team 0's draft
results[results['team_number'] == 0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>player_name</th>
      <th>position_name</th>
      <th>slot</th>
      <th>year</th>
      <th>projection_adp</th>
      <th>adjusted_projection_adp</th>
      <th>team_number</th>
      <th>round_number</th>
      <th>pick_number</th>
      <th>pick_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cooper Kupp</td>
      <td>WR</td>
      <td>WR1</td>
      <td>2022</td>
      <td>3.0</td>
      <td>3.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>162.50</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Mark Andrews</td>
      <td>TE</td>
      <td>TE1</td>
      <td>2022</td>
      <td>23.0</td>
      <td>23.000000</td>
      <td>0</td>
      <td>2</td>
      <td>24</td>
      <td>128.90</td>
    </tr>
    <tr>
      <th>24</th>
      <td>DK Metcalf</td>
      <td>WR</td>
      <td>WR2</td>
      <td>2022</td>
      <td>24.0</td>
      <td>24.000000</td>
      <td>0</td>
      <td>3</td>
      <td>25</td>
      <td>138.20</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Amari Cooper</td>
      <td>WR</td>
      <td>WR3</td>
      <td>2022</td>
      <td>48.0</td>
      <td>48.000000</td>
      <td>0</td>
      <td>4</td>
      <td>48</td>
      <td>135.20</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Joe Burrow</td>
      <td>QB</td>
      <td>QB1</td>
      <td>2022</td>
      <td>50.0</td>
      <td>50.000000</td>
      <td>0</td>
      <td>5</td>
      <td>49</td>
      <td>266.18</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Aaron Rodgers</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>71.0</td>
      <td>71.000000</td>
      <td>0</td>
      <td>6</td>
      <td>72</td>
      <td>191.56</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Dallas Goedert</td>
      <td>TE</td>
      <td>FLEX</td>
      <td>2022</td>
      <td>73.0</td>
      <td>73.000000</td>
      <td>0</td>
      <td>7</td>
      <td>73</td>
      <td>75.20</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Allen Robinson</td>
      <td>WR</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>94.0</td>
      <td>94.000000</td>
      <td>0</td>
      <td>8</td>
      <td>96</td>
      <td>52.60</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Kirk Cousins</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>97.0</td>
      <td>97.000000</td>
      <td>0</td>
      <td>9</td>
      <td>97</td>
      <td>116.40</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Noah Fant</td>
      <td>TE</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>118.0</td>
      <td>118.000000</td>
      <td>0</td>
      <td>10</td>
      <td>120</td>
      <td>68.70</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Jakobi Meyers</td>
      <td>WR</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>121.0</td>
      <td>121.000000</td>
      <td>0</td>
      <td>11</td>
      <td>121</td>
      <td>71.20</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Tyler Higbee</td>
      <td>TE</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>142.0</td>
      <td>142.000000</td>
      <td>0</td>
      <td>12</td>
      <td>144</td>
      <td>65.10</td>
    </tr>
    <tr>
      <th>144</th>
      <td>Julio Jones</td>
      <td>WR</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>144.0</td>
      <td>144.000000</td>
      <td>0</td>
      <td>13</td>
      <td>145</td>
      <td>38.30</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Malik Willis</td>
      <td>QB</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>164.0</td>
      <td>164.000000</td>
      <td>0</td>
      <td>14</td>
      <td>168</td>
      <td>2.40</td>
    </tr>
    <tr>
      <th>168</th>
      <td>Jahan Dotson</td>
      <td>WR</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>165.0</td>
      <td>165.000000</td>
      <td>0</td>
      <td>15</td>
      <td>169</td>
      <td>44.20</td>
    </tr>
    <tr>
      <th>191</th>
      <td>Allen Lazard</td>
      <td>WR</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>190.0</td>
      <td>190.000000</td>
      <td>0</td>
      <td>16</td>
      <td>192</td>
      <td>83.40</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Justin Jackson</td>
      <td>RB</td>
      <td>RB1</td>
      <td>2022</td>
      <td>191.0</td>
      <td>192.601169</td>
      <td>0</td>
      <td>17</td>
      <td>193</td>
      <td>20.50</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Jonnu Smith</td>
      <td>TE</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>213.0</td>
      <td>213.000000</td>
      <td>0</td>
      <td>18</td>
      <td>216</td>
      <td>13.60</td>
    </tr>
    <tr>
      <th>216</th>
      <td>Adam Trautman</td>
      <td>TE</td>
      <td>BENCH</td>
      <td>2022</td>
      <td>214.0</td>
      <td>214.000000</td>
      <td>0</td>
      <td>19</td>
      <td>217</td>
      <td>15.40</td>
    </tr>
    <tr>
      <th>239</th>
      <td>Rico Dowdle</td>
      <td>RB</td>
      <td>RB2</td>
      <td>2022</td>
      <td>216.0</td>
      <td>216.848107</td>
      <td>0</td>
      <td>20</td>
      <td>240</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Now lets run the RB zero strategy across 20 simulations in each draft position and see how we fare.


```python
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
fig.tight_layout(pad=5.0) 

for team_number in range(12):
    # Initialize the strategy dictionary with the 0-RB strategy for the current team
    strategy_dict = {team_number: zero_rb_team_strategy}

    # Sample 11 other strategies from the DataFrame, excluding the current team
    sampled_strategies = market_dynamics_df.sample(11)

    # Add the sampled strategies to the strategy dictionary
    for i, (index, row) in enumerate(sampled_strategies.iterrows()):
        strategy = {
            'QB': {'mean': row['market_delta_QB_mean'], 'stddev': row['market_delta_QB_std']},
            'RB': {'mean': row['market_delta_RB_mean'], 'stddev': row['market_delta_RB_std']},
            'WR': {'mean': row['market_delta_WR_mean'], 'stddev': row['market_delta_WR_std']},
            'TE': {'mean': row['market_delta_TE_mean'], 'stddev': row['market_delta_TE_std']}
        }
        strategy_dict[(i+1)%12] = strategy  # Modulo 12 to ensure team numbers are within 0-11

    # Run the simulations
    sim_manager = DraftSimulationManager(player_rankings.copy(), strategy_dict)
    sim_manager.run_simulations(20)

    # Compute the median points
    points_stats_df = sim_manager.points_stats_df()
    
    # Plot the median points for each team
    ax = axs[team_number // 3, team_number % 3]
    ax.bar(points_stats_df['team_number'], points_stats_df['Median'], color='b', alpha=0.7)
    ax.bar(team_number, points_stats_df.loc[points_stats_df['team_number'] == team_number, 'Median'], color='r', alpha=0.7)
    ax.set_xlabel('Team Number')
    ax.set_ylabel('Median Points')
    ax.set_title(f'Median Points for RB-0 Strategy at Position {team_number+1}')
    
plt.show()
```


    
![png](Simulating%20Best%20Ball%20Mania%20Drafts%20Using%20Realistic%20Market%20Dynamics_files/Simulating%20Best%20Ball%20Mania%20Drafts%20Using%20Realistic%20Market%20Dynamics_53_0.png)
    


## Conclusion

As you can see, the RB-0 strategy actually fares pretty well across many different draft positions. __This simulation tool is an excellent way to test different draft strategies in a realistic enviornment__. By sampling __actual Best Ball strategies__ and running them against the strategy you'd like to test through potentially thousands of simulations, you can get a great idea of how well your draft will go. Turning the dials on these market delta parameters even a little bit can changes the outcomes drastically. I hope someone finds this beneficial and uses this simulation tool to test their strategy for Best Ball Mania IV!
