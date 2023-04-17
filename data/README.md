# Best Ball Data Bowl Data Sets

## Data Source

BBM1: https://underblog.underdogfantasy.com/raw-data-from-best-ball-mania-i-dbb5f947311d

BBM2: https://underdognetwork.com/football/best-ball-research/best-ball-mania-ii-downloadable-pick-by-pick-data

BBM3: https://underdognetwork.com/football/best-ball-research/best-ball-mania-iii-downloadable-pick-by-pick-data

Datasets courtesy of Underdog Fantasy

Each of the datasets below contain BBM data for seperate BBM contests (so far there have been 3).

At the very least, your competition submission should use BBM III data (using I & II is not required). You're also welcome to use any outside dataset to merge with the BBM data below, although we prefer you use [nfl_data_py](https://github.com/cooperdff/nfl_data_py) if you're using Python, and [nflfastR](https://github.com/nflverse/nflfastR/) if you're using R. 

## Data Dictionary

### 2020

Data here is less rich than 2021 & 2022 and only available in a single CSV. We'd prefer you focus on 2021 & 2022, but in case you do want to bring in 2020:

| **Column**     | **Data Type** | **Description**                                  | **Example**                          |
|----------------|---------------|--------------------------------------------------|--------------------------------------|
| team           | string        | Team ID column                                   | 081ea432-dcf5-4370-932d-b53ac2296900 |
| player         | string        | NFL player selected                              | AJ Brown                             |
| drafted_round  | int           | Round player was drafted in                      | 4                                    |
| roster_points* | float         | How many points the team scored in that round    | 109.94                               |
| pick_points*   | float         | How many points that player scored in that round | 11.04                                |
| draft_time*    | utc timestamp | Time draft started                               | 2020-12-22 17:04:23 UTC              |
| playoff_round* | int           | What round the player made                       | Round 4                              |
| made_playoffs* | int (0 or 1)  | True/false whether entry made the playoffs       | 0                                    |

\* **Hayden Winks**: `playoff_round` is how far this team is in Best Ball Mania I. But pay attention to this because it’s important… The data in this column is wonky. A team that advances to the finals will have four different entries in this same data and they will have four different team IDs. Let me repeat. A team that advances to the finals will have a “Lost”, “Round 2”, “Round 3”, and “Round 4” playoff_round entry, and the team id doesn’t tell you it’s the same team. For this reason, we can’t directly use this to calculate “advance rates” or “best ball win rates”.
\* **Hayden Winks**: `made_playoffs` is whether or not this team is in the playoffs, but the weirdness of the data from playoff_round applies here. Some teams marked as “Lost” actually made the playoffs. I know, it’s confusing.
\* **Hayden Winks**: `roster_points` are how many points the team scored in that round, so the scores will be highest in “Lost” because that’s counting the entire regular season of best ball.
\* **Hayden Winks**: `pick_points` are how many points that player scored in that round, so the scores will be highest in “Lost” because that’s counting the entire regular season of best ball.
\* **Hayden Winks**: `draft_time` is when the draft took place, but like `roster_points` and `pick_points`, the time is dictated on the round. So the only draft times that are accurate in the data set are for teams labeled as “Lost”.

> As you can see, the data here is not the best and could lead to errors in analysis. For this reason, we prefer if you use BBMII (2021) and BBMIII (2022) data unless you have a good reason to use 2020 data.

### 2021

Data here is split up in regular season and postseason. Each CSV (whether regular season or postseason) has the same columns.

| **Column**              | **Data Type** | **Description**                                | **Example**                          |
|-------------------------|---------------|------------------------------------------------|--------------------------------------|
| draft_id                | string        | Entry id                                       | fde6d91d-04fb-4a16-b5d8-6e421361bb63 |
| draft_time              | nan           | Empty column                                   |                                      |
| clock                   | int           | Time per pick alloted                          | 30                                   |
| tournament_entry_id     | string        |                                                | 801f0c2a-3e43-49bb-aad5-0054cea26e23 |
| tournament_round_number | int           | Round player was selected                      | 4                                    |
| player_name             | string        | NFL player name                                | Jamaal Williams                      |
| position_name           | string        | Player Position                                | RB                                   |
| bye_week                | int           | Player bye week                                | 9                                    |
| projection_adp          | int           |                                                | 0                                    |
| pick_order              | int           |                                                | 0                                    |
| overall_pick_number     | int           |                                                | 1741                                 |
| team_pick_number        | int           |                                                | 13                                   |
| pick_points             | float         |                                                | 11.9                                 |
| roster_points           | float         |                                                | 138.52                               |
| playoff_team            | int           | True/False value. 1 if entry made the playoffs | 1                                    |

### 2022

Data here is again split up into regular season and postseason. There are multiple files that must be concatenated together for each stage of the postseason. Regular season is split up into mixed and fast time formats for drafts. 

| **Column**              | **Data Type** | **Description**                                | **Example**                          |
|-------------------------|---------------|------------------------------------------------|--------------------------------------|
| draft_id                | string        | Entry id                                       | fde6d91d-04fb-4a16-b5d8-6e421361bb63 |
| draft_time              | nan           | Empty column                                   |                                      |
| clock                   | int           | Time per pick alloted                          | 30                                   |
| tournament_entry_id     | string        |                                                | 801f0c2a-3e43-49bb-aad5-0054cea26e23 |
| tournament_round_number | int           | Round player was selected                      | 4                                    |
| player_name             | string        | NFL player name                                | Jamaal Williams                      |
| position_name           | string        | Player Position                                | RB                                   |
| bye_week                | int           | Player bye week                                | 9                                    |
| projection_adp          | int           |                                                | 0                                    |
| pick_order              | int           |                                                | 0                                    |
| overall_pick_number     | int           |                                                | 1741                                 |
| team_pick_number        | int           |                                                | 13                                   |
| pick_points             | float         |                                                | 11.9                                 |
| roster_points           | float         |                                                | 138.52                               |
| playoff_team            | int           | True/False value. 1 if entry made the playoffs | 1                                    |

## File Structure:

File structure for the datasets is below. You'll notice that for 2021 and 2022, the data is partioned in "parts". These are equal slices of the data which will need to be concatenated together. We did it this way because Github has a max file upload limit. 

```
/2020 (BBMI)
    part_00.csv


/2021 (BBMII)
    /post_season
        finals.csv
        quarterfinals.csv
        semifinals.csv
    /regular_season.csv
        part_00.csv
        part_01.csv
        ...
        part_05.csv


/2022 (BBMIII)
    /post_season
        /finals
            part_00.csv
        /quarterfinals
            part_00.csv
            part_01.csv
            part_02.csv
    /regular_season
        /fast
            part_00.csv
            part_01.csv
            ...
            part_26.csv
        /slow
            part_00.csv
            part_01.csv
            ...
            part_08.csv
```