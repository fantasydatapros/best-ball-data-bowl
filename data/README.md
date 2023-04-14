# Best Ball Data Bowl Data Sets

Datasets courtesy of Underdog Fantasy

Each of the datasets below contain BBM data for seperate BBM contests (so far there have been 3).

At the very least, your competition submission should use BBM III data (using I & II is not required). You're also welcome to use any outside dataset to merge with the BBM data below, although we prefer you use [nfl_data_py](https://github.com/cooperdff/nfl_data_py) if you're using Python, and [nflfastR](https://github.com/nflverse/nflfastR/) if you're using R. 

## Data Dictionary

### 2020

Data here is less rich than 2021 & 2022 and only available in a single CSV. We'd prefer you focus on 2021 & 2022, but in case you do want to bring in 2020:

| **Column**    | **Data Type** | **Description**                                   | **Example**                          |
|---------------|---------------|---------------------------------------------------|--------------------------------------|
| team          | string        | ID column for draft entry ID on Underdog          | 081ea432-dcf5-4370-932d-b53ac2296900 |
| player        | string        | NFL player selected                               | AJ Brown                             |
| drafted_round | int           | Round player was drafted in                       | 4                                    |
| roster_points | float         | Amount of points this player contributed to entry | 109.94                               |
| pick_points   | float         |                                                   | 11.04                                |
| draft_time    | utc timestamp | Time draft started                                | 2020-12-22 17:04:23 UTC              |
| playoff_round | int           | What round the player made                        | Round 4                              |
| make_playoffs | int (0 or 1)  | True/false whether entry made the playoffs        | 0                                    |

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