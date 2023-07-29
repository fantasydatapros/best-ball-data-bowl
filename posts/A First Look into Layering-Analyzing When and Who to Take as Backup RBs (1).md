# A First Look into Layering: Analyzing When and Who to Take as Backup RBs
## Participant: Dylan Sloan
Twitter: https://twitter.com/dsloan__ (two underscores after dsloan)

Linkedin: https://www.linkedin.com/in/dylan-s-681721127/
## Introduction
A common question amongst best ball drafters is do backup running backs matter. If I draft two running backs early, can I wait till very late to get my backup running backs? Are handcuff running backs a good choice for my backup running back slots? My project aims to answer these questions by creating archetypes for running backs and looking at where RBs were drafted for all teams, playoff teams, and top 1% teams.


## Utilities


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
```


```python
dir_fast = '/Users/dylansloan/Desktop/best-ball-data-bowl-master/data/2022/regular_season/fast/part_'
dir_mixed = '/Users/dylansloan/Desktop/best-ball-data-bowl-master/data/2022/regular_season/mixed/part_'
extension = '.csv'
filepaths = [dir_fast + str(i).zfill(2) + extension for i in range(27)]
filepaths += [dir_mixed + str(i).zfill(2) + extension for i in range(9)]
df_2022 = pd.concat([pd.read_csv(fp) for fp in filepaths])
```


```python
dir_path = '/Users/dylansloan/Desktop/best-ball-data-bowl-master/data/2021/regular_season/part_'
extension = '.csv'
filepaths = [dir_path + str(i).zfill(2) + extension for i in range(6)]
df_2021 = pd.concat([pd.read_csv(fp) for fp in filepaths])
```


```python
clay_proj2021=pd.read_csv('/Users/dylansloan/Downloads/Mike Clay 2021 Projection - Sheet1 (3).csv')
clay_proj2022=pd.read_csv('/Users/dylansloan/Downloads/2022 Clay Projections - Sheet1 (3).csv')
```

## Establishing Closing Line ADP
In order to create adp_difference which will be seen later, I needed to gather the ADPs of all the running backs. However, the ADPs of players change from the start of draft season to the end. In order to get more of a source of truth, I took the ADPs from within a week before the contest closed.


```python
df_2021['draft_time'] = pd.to_datetime(df_2021['draft_time'])
filtered_df2021 = df_2021[df_2021['draft_time'].dt.date == pd.to_datetime('2021-09-08').date()]
df_2022['draft_time'] = pd.to_datetime(df_2022['draft_time'])
filtered_df2022 = df_2022[df_2022['draft_time'].dt.date == pd.to_datetime('2022-09-06').date()]
```

## Creating A Playoff Team Subset in 2022
While the 2021 dataset had playoff_team properly assigned, the 2022 data did not. In the code below, I go about assigning this column as it will be used later in the project.


```python
df_2022['rank'] = df_2022.groupby('draft_id')['roster_points'].rank(method='dense', ascending=False)
df_2022['playoff_team'] = np.where(df_2022['rank'] <= 2, 1, 0)
df_2022 = df_2022.drop('rank', axis=1)
```

## ADP Difference
The ADP Difference idea comes from RotoViz’s Charles Kleinheksel who deciphered that a gap in adp of 97.5 from RB1 to RB2 on a team was enough to decipher whether a RB was a backup/starter situation or it was a committee situation. If the gap is larger than 97.5 it is declared a starter and backup and if less than it is a committee. I went about doing this in the code by getting a unique list of RBs and their closing line adp (projection_adp) and then creating adp_difference between the first two running backs selected.


```python
running_backs2021 = filtered_df2021[filtered_df2021['position_name'] == 'RB']
rbs2021 = pd.merge(running_backs2021, clay_proj2021, on='player_name')
unique_players2021 = rbs2021[['player_name', 'projection_adp','team']].drop_duplicates('player_name')
excluded_players=['Adrian Peterson','Kerryon Johnson']
unique_players2021=unique_players2021[~unique_players2021['player_name'].isin(excluded_players)]
unique_players2021 = unique_players2021[unique_players2021['projection_adp'] != 0]
lowest_adp2021 = unique_players2021.groupby(['team','player_name'])['projection_adp'].apply(lambda x: x.nsmallest(2))
lowest_adp2021 = lowest_adp2021.reset_index().sort_values(['team','projection_adp'])
unique_players2021.loc[unique_players2021['player_name'] == "Le'Veon Bell", 'team'] = 'BAL'
sorted_data2021 = unique_players2021.sort_values('team')
unique_players2021=unique_players2021[~unique_players2021['player_name'].isin(excluded_players)]
def adp_difference(adp):
    if len(adp)>1:
        sorted_adp2021=sorted(adp)
        return sorted_adp2021[1]-sorted_adp2021[0]
unique_players2021['adp_difference']=unique_players2021.groupby('team')['projection_adp'].transform(adp_difference)
unique_players2021=unique_players2021[['team','player_name','projection_adp','adp_difference']]
sorted_data2021=unique_players2021.sort_values('team')
print(sorted_data2021)
```

          team            player_name  projection_adp  adp_difference
    10205  ARI           James Conner          118.36           40.70
    15423  ARI           Eno Benjamin          215.79           40.70
    8706   ARI          Chase Edmonds           77.66           40.70
    9978   ATL          Wayne Gallman          198.16          136.20
    5254   ATL             Mike Davis           61.96          136.20
    8204   BAL        Ty'Son Williams          160.83           54.57
    8074   BAL           Le'Veon Bell          215.40           54.57
    4250   BUF              Zack Moss          115.17           16.90
    752    BUF       Devin Singletary          132.07           16.90
    15386  BUF            Matt Breida          215.81           16.90
    5565   CAR          Chuba Hubbard          170.55          169.51
    1254   CAR    Christian McCaffrey            1.04          169.51
    15276  CHI            Tarik Cohen          215.62          133.57
    11769  CHI        Damien Williams          166.49          133.57
    15382  CHI         Khalil Herbert          215.96          133.57
    501    CHI       David Montgomery           32.92          133.57
    13759  CIN              Joe Mixon           21.04          194.49
    15439  CIN            Chris Evans          215.66          194.49
    15425  CIN          Samaje Perine          215.53          194.49
    7070   CLE            Kareem Hunt           69.26           54.40
    5815   CLE             Nick Chubb           14.86           54.40
    13508  DAL           Tony Pollard          122.76          116.98
    2083   DAL        Ezekiel Elliott            5.78          116.98
    3999   DEN       Javonte Williams           60.17           41.00
    2999   DEN          Melvin Gordon          101.17           41.00
    4752   DET          D'Andre Swift           43.30           75.08
    6066   DET        Jamaal Williams          118.38           75.08
    15313  DET       Jermar Jefferson          215.66           75.08
    14512   GB            Aaron Jones            9.18           83.32
    15270   GB             Kylin Hill          215.92           83.32
    12271   GB              AJ Dillon           92.50           83.32
    15319  HOU            Mark Ingram          211.54           32.52
    13046  HOU          David Johnson          192.68           32.52
    251    HOU        Phillip Lindsay          160.16           32.52
    3250   IND           Nyheim Hines          136.00          120.97
    15403  IND            Marlon Mack          214.09          120.97
    9087   IND        Jonathan Taylor           15.03          120.97
    15014  JAX            Carlos Hyde          205.79          169.01
    2748   JAX         James Robinson           36.78          169.01
    9589    KC        Darrel Williams          208.59          183.66
    6819    KC  Clyde Edwards-Helaire           24.93          183.66
    15156   KC        Jerick McKinnon          210.57          183.66
    4501   LAC          Austin Ekeler           11.07          202.32
    15450  LAC         Larry Rountree          215.42          202.32
    11710  LAC         Justin Jackson          213.39          202.32
    15437  LAR              Jake Funk          215.97           41.98
    11208  LAR      Darrell Henderson           60.51           41.98
    1003   LAR            Sony Michel          102.49           41.98
    14261   LV            Josh Jacobs           57.59           68.81
    1505    LV           Kenyan Drake          126.40           68.81
    13019  MIA           Salvon Ahmed          214.95          134.00
    0      MIA           Myles Gaskin           70.24          134.00
    8957   MIA          Malcolm Brown          204.24          134.00
    6317   MIN     Alexander Mattison          150.21          147.95
    2497   MIN            Dalvin Cook            2.26          147.95
    7572    NE            James White          144.57           73.34
    15131   NE            J.J. Taylor          215.08           73.34
    9338    NE    Rhamondre Stevenson          138.42           73.34
    11459   NE          Damien Harris           65.08           73.34
    6568    NO           Alvin Kamara            3.00          195.34
    9730    NO             Tony Jones          198.34          195.34
    5505   NYG        Devontae Booker          212.82          200.35
    12768  NYG         Saquon Barkley           12.47          200.35
    3501   NYJ             Ty Johnson          174.31           56.82
    10707  NYJ          Tevin Coleman          171.31           56.82
    3748   NYJ         Michael Carter          114.49           56.82
    12522  PHI       Kenneth Gainwell          181.76          130.76
    2007   PHI           Boston Scott          212.62          130.76
    7321   PHI          Miles Sanders           51.00          130.76
    15466  PIT            Benny Snell          215.95          200.61
    14763  PIT           Najee Harris           15.34          200.61
    2334   SEA          Rashaad Penny          199.06          162.58
    10456  SEA           Chris Carson           36.48          162.58
    15393  SEA           Alex Collins          215.56          162.58
    15389   SF        Elijah Mitchell          215.83            4.26
    10957   SF            Trey Sermon           70.54            4.26
    12020   SF         Raheem Mostert           74.80            4.26
    5003    TB      Leonard Fournette          129.99           32.14
    7823    TB           Ronald Jones           97.85           32.14
    15465   TB        Ke'Shawn Vaughn          215.99           32.14
    13257   TB        Giovani Bernard          164.41           32.14
    15287  TEN        Darrynton Evans          213.77          209.00
    1756   TEN          Derrick Henry            4.77          209.00
    15383  TEN          Mekhi Sargent          215.84          209.00
    15286  TEN       Jeremy McNichols          215.96          209.00
    14010  WAS         Antonio Gibson           16.65          136.72
    15360  WAS        Jaret Patterson          214.66          136.72
    8455   WAS          J.D. McKissic          153.37          136.72



```python
running_backs2022 = filtered_df2022[filtered_df2022['position_name'] == 'RB']
rbs2022 = pd.merge(clay_proj2022, running_backs2022, on='player_name')
unique_players2022 = rbs2022[['player_name', 'projection_adp','team']].drop_duplicates('player_name')
unique_players2022 = unique_players2022[unique_players2022['projection_adp'] != 0]
lowest_adp2022 = unique_players2022.groupby(['team','player_name'])['projection_adp'].apply(lambda x: x.nsmallest(2))
lowest_adp2022 = lowest_adp2022.reset_index().sort_values(['team','projection_adp'])
sorted_data2022 = unique_players2022.sort_values('team')
unique_players2022=unique_players2022[~unique_players2022['player_name'].isin(excluded_players)]
def adp_difference(adp):
    if len(adp)>1:
        sorted_adp2022=sorted(adp)
        return sorted_adp2022[1]-sorted_adp2022[0]
unique_players2022['adp_difference']=unique_players2022.groupby('team')['projection_adp'].transform(adp_difference)
unique_players2022=unique_players2022[['team','player_name','projection_adp','adp_difference']]
sorted_data2022=unique_players2022.sort_values('team')
print(sorted_data2022)
```

          team            player_name  projection_adp  adp_difference
    0      ARI           James Conner           34.57          154.09
    763    ARI        Darrel Williams          212.78          154.09
    954    ARI           Eno Benjamin          188.66          154.09
    1690   ARI        Keaontay Ingram          215.94          154.09
    1694   ATL  Cordarrelle Patterson          103.42           44.79
    2457   ATL        Damien Williams          215.28           44.79
    2565   ATL         Tyler Allgeier          148.21           44.79
    3328   BAL           J.K. Dobbins           67.57          128.58
    4091   BAL            Gus Edwards          213.61          128.58
    4194   BAL             Mike Davis          196.15          128.58
    4831   BAL           Kenyan Drake          203.85          128.58
    6811   BUF              Zack Moss          208.06            0.14
    6048   BUF             James Cook          106.67            0.14
    5285   BUF       Devin Singletary          106.81            0.14
    7147   CAR    Christian McCaffrey            1.75          210.87
    7910   CAR          Chuba Hubbard          212.62          210.87
    8128   CAR         D'Onta Foreman          213.08          210.87
    8348   CHI       David Montgomery           65.46           87.11
    9111   CHI         Khalil Herbert          152.57           87.11
    9874   CHI          Trestan Ebner          215.94           87.11
    9876   CIN              Joe Mixon           13.00          201.61
    10639  CIN          Samaje Perine          214.61          201.61
    10758  CIN            Chris Evans          215.06          201.61
    12481  CLE            Jerome Ford          215.96           67.92
    12332  CLE       D'Ernest Johnson          213.93           67.92
    10806  CLE             Nick Chubb           27.15           67.92
    11569  CLE            Kareem Hunt           95.07           67.92
    12483  DAL        Ezekiel Elliott           41.75           39.51
    13246  DAL           Tony Pollard           81.26           39.51
    14009  DEN       Javonte Williams           19.86           93.17
    14772  DEN          Melvin Gordon          113.03           93.17
    15535  DET          D'Andre Swift           16.27          140.85
    16298  DET        Jamaal Williams          157.12          140.85
    17061  DET         Craig Reynolds          215.98          140.85
    17062   GB            Aaron Jones           18.47           35.59
    17825   GB              AJ Dillon           54.06           35.59
    19351  HOU           Rex Burkhead          210.33          149.07
    18588  HOU          Dameon Pierce           61.26          149.07
    19726  IND        Jonathan Taylor            1.54          121.80
    20489  IND           Nyheim Hines          123.34          121.80
    21252  JAX         Travis Etienne           36.60           99.21
    22015  JAX         James Robinson          135.81           99.21
    22778  JAX           Snoop Conner          215.97           99.21
    24300   KC          Isiah Pacheco          144.62           60.50
    25063   KC           Ronald Jones          214.72           60.50
    22779   KC  Clyde Edwards-Helaire           84.12           60.50
    23542   KC        Jerick McKinnon          179.20           60.50
    26435  LAC         Isaiah Spiller          181.55          176.17
    27172  LAC          Joshua Kelley          215.13          176.17
    25133  LAC          Austin Ekeler            5.38          176.17
    25896  LAC            Sony Michel          202.91          176.17
    27203  LAR              Cam Akers           59.27           54.90
    27966  LAR      Darrell Henderson          114.17           54.90
    28729  LAR         Kyren Williams          214.93           54.90
    28799   LV            Josh Jacobs           81.00           79.98
    29562   LV            Zamir White          160.98           79.98
    30325   LV         Ameer Abdullah          213.97           79.98
    30456   LV         Brandon Bolden          215.96           79.98
    30457  MIA          Chase Edmonds           71.28           80.64
    31220  MIA         Raheem Mostert          151.92           80.64
    31983  MIA           Myles Gaskin          215.85           80.64
    33527  MIN            Ty Chandler          215.66          126.29
    33525  MIN           Kene Nwangwu          215.98          126.29
    32762  MIN     Alexander Mattison          134.51          126.29
    31999  MIN            Dalvin Cook            8.22          126.29
    33553   NE    Rhamondre Stevenson           80.66           19.81
    34316   NE          Damien Harris          100.47           19.81
    35079   NE          Pierre Strong          215.93           19.81
    35087   NO           Alvin Kamara           14.80          173.90
    35850   NO            Mark Ingram          188.70          173.90
    36578   NO             Tony Jones          216.00          173.90
    37368  NYG       Antonio Williams          215.97          201.89
    36579  NYG         Saquon Barkley           13.81          201.89
    37342  NYG            Matt Breida          215.70          201.89
    38895  NYJ             Ty Johnson          215.99           78.81
    38132  NYJ         Michael Carter          129.85           78.81
    37369  NYJ            Breece Hall           51.04           78.81
    38896  PHI          Miles Sanders          100.54           30.26
    39659  PHI       Kenneth Gainwell          130.80           30.26
    40422  PHI           Boston Scott          210.32           30.26
    40689  PHI            Trey Sermon          215.37           30.26
    40721  PIT           Najee Harris           12.52          196.78
    41484  PIT          Jaylen Warren          209.30          196.78
    41988  PIT            Benny Snell          215.93          196.78
    43523  SEA           Travis Homer          215.92           30.66
    41994  SEA          Rashaad Penny           90.40           30.66
    42757  SEA         Kenneth Walker          121.06           30.66
    43520  SEA          DeeJay Dallas          215.89           30.66
    45032   SF     Tyrion Davis-Price          178.14          110.20
    43530   SF        Elijah Mitchell           67.94          110.20
    44293   SF            Jeff Wilson          188.25          110.20
    45790   TB      Leonard Fournette           23.94          111.52
    46553   TB          Rachaad White          135.46          111.52
    47316   TB        Ke'Shawn Vaughn          215.92          111.52
    48285  TEN        Julius Chestnut          215.97          207.22
    47323  TEN          Derrick Henry            7.35          207.22
    48086  TEN      Dontrell Hilliard          214.57          207.22
    48244  TEN         Hassan Haskins          215.40          207.22
    49050  WAS          J.D. McKissic          163.71           74.52
    49813  WAS         Brian Robinson          164.07           74.52
    48287  WAS         Antonio Gibson           89.19           74.52
    50571  WAS      Jonathan Williams          215.99           74.52


## Mike Clay Projections
I decided to use Mike Clay projections because of an article written by Jack Miller on how he created archetypes. By using the Mike Clay projections, it gives a sense of running backs who have a role based on how many points he projects them to score. It also helps decipher receiving running backs that if an injury happened would not have a lot of upside.


```python
clay_proj2021['clay_rushingpoints']=(clay_proj2021['clay_rushingyards']/10)+(clay_proj2021['clay_rushingtds']*6)
clay_proj2021['clay_receivingpoints']=(clay_proj2021['clay_receptions']*.5)+clay_proj2021['clay_receivingyards']/10+(clay_proj2021['clay_receivingtds']*6)
clay_proj2021['receivingoverrushingpoints']=clay_proj2021['clay_receivingpoints']/clay_proj2021['clay_rushingpoints']
clay_proj2021['clay_halfpprpoints']=(clay_proj2021['clay_receivingpoints']+clay_proj2021['clay_rushingpoints'])
clay_proj2021['clay_receivingoverrushing']=(clay_proj2021['clay_receivingpoints']/clay_proj2021['clay_rushingpoints'])
clay_proj2022['clay_rushingpoints'] = (clay_proj2022['clay_rushingyards'] / 10) + (clay_proj2022['clay_rushingtds'] * 6)
clay_proj2022['clay_receivingpoints'] = (clay_proj2022['clay_receptions'] * .5) + clay_proj2022['clay_receivingyards'] / 10 + (clay_proj2022['clay_receivingtds'] * 6)
clay_proj2022['receivingoverrushingpoints'] = clay_proj2022['clay_receivingpoints'] / clay_proj2022['clay_rushingpoints']
clay_proj2022['clay_halfpprpoints'] = (clay_proj2022['clay_receivingpoints'] + clay_proj2022['clay_rushingpoints'])
clay_proj2022['clay_receivingoverrushing'] = (clay_proj2022['clay_receivingpoints'] / clay_proj2022['clay_rushingpoints'])
```

## Archetypes
        Archetype 1 = Bell Cow Running Back in a Starter/Backup situation
        Archetype 2 = First Running Back Taken in a Committee situation
        Archetype 3 = Second Running Back Taken in a Committee situation
        Archetype 4 = Backup Running Back with a Role in a Starter/Backup situation
        Archetype 5 = Backup Running Back with no Role in a Starter/Backup situation
        Archetype 6 = Third or later Running Back taken in committee with role
        Archetype 7 = Third or later Running Back taken  in committee with no role
        Archetype 8 = Backup in backup/starter situation where backup is receiving back and doesn't have upside if injury of starter


```python
archetypes2021=pd.merge(unique_players2021,clay_proj2021,on='player_name')
archetypes2021 = archetypes2021.sort_values(['team_x', 'projection_adp'])
archetypes2021['depth_chart'] = archetypes2021.groupby('team_x')['projection_adp'].rank(method='min')
archetypes2021['archetype']=0 
archetypes2021.loc[(archetypes2021['adp_difference'] > 97.5) & (archetypes2021['projection_adp'] < 100), 'archetype'] = 1 
archetypes2021.loc[(archetypes2021['adp_difference'] > 97.5) & (archetypes2021['projection_adp'] > 100) & (archetypes2021['clay_halfpprpoints'] < 65), 'archetype'] = 5
archetypes2021.loc[(archetypes2021['adp_difference'] > 97.5) & (archetypes2021['clay_halfpprpoints'] >= 65) & (archetypes2021['projection_adp'] > 100), 'archetype'] = 4 
archetypes2021.loc[(archetypes2021['adp_difference'] < 97.5) & (archetypes2021['depth_chart'] == 1), 'archetype'] = 2
archetypes2021.loc[(archetypes2021['adp_difference'] < 97.5) & (archetypes2021['depth_chart'] == 2), 'archetype'] = 3
archetypes2021.loc[(archetypes2021['adp_difference'] < 97.5) & (archetypes2021['projection_adp'] > 100) & (archetypes2021['clay_halfpprpoints'] < 65) & (archetypes2021['depth_chart'] >= 3) , 'archetype'] = 7
archetypes2021.loc[(archetypes2021['adp_difference'] < 97.5) & (archetypes2021['projection_adp'] > 100) & (archetypes2021['clay_halfpprpoints'] > 65) & (archetypes2021['depth_chart'] >= 3), 'archetype'] = 6
archetypes2021.loc[(archetypes2021['adp_difference'] > 97.5) & (archetypes2021['clay_receivingoverrushing'] > 1), 'archetype'] = 8
archetypes2021 = archetypes2021.sort_values(['team_x', 'projection_adp'])
archetypes2021['depth_chart'] = archetypes2021.groupby('team_x')['projection_adp'].rank(method='min')
archetypes_sorted2021 = archetypes2021.sort_values(by='team_x') 
print(archetypes_sorted2021[['player_name','team_x','projection_adp','adp_difference','archetype']])
```

                  player_name team_x  projection_adp  adp_difference  archetype
    37          Chase Edmonds    ARI           77.66           40.70          2
    44           James Conner    ARI          118.36           40.70          3
    81           Eno Benjamin    ARI          215.79           40.70          7
    22             Mike Davis    ATL           61.96          136.20          1
    43          Wayne Gallman    ATL          198.16          136.20          4
    35        Ty'Son Williams    BAL          160.83           54.57          2
    34           Le'Veon Bell    BAL          215.40           54.57          3
    18              Zack Moss    BUF          115.17           16.90          2
    3        Devin Singletary    BUF          132.07           16.90          3
    77            Matt Breida    BUF          215.81           16.90          7
    24          Chuba Hubbard    CAR          170.55          169.51          5
    5     Christian McCaffrey    CAR            1.04          169.51          1
    2        David Montgomery    CHI           32.92          133.57          1
    51        Damien Williams    CHI          166.49          133.57          5
    69            Tarik Cohen    CHI          215.62          133.57          8
    75         Khalil Herbert    CHI          215.96          133.57          5
    60              Joe Mixon    CIN           21.04          194.49          1
    82          Samaje Perine    CIN          215.53          194.49          5
    84            Chris Evans    CIN          215.66          194.49          8
    25             Nick Chubb    CLE           14.86           54.40          2
    30            Kareem Hunt    CLE           69.26           54.40          3
    59           Tony Pollard    DAL          122.76          116.98          4
    9         Ezekiel Elliott    DAL            5.78          116.98          1
    17       Javonte Williams    DEN           60.17           41.00          2
    13          Melvin Gordon    DEN          101.17           41.00          3
    20          D'Andre Swift    DET           43.30           75.08          2
    26        Jamaal Williams    DET          118.38           75.08          3
    72       Jermar Jefferson    DET          215.66           75.08          7
    63            Aaron Jones     GB            9.18           83.32          2
    53              AJ Dillon     GB           92.50           83.32          3
    68             Kylin Hill     GB          215.92           83.32          7
    73            Mark Ingram    HOU          211.54           32.52          6
    57          David Johnson    HOU          192.68           32.52          3
    1         Phillip Lindsay    HOU          160.16           32.52          2
    39        Jonathan Taylor    IND           15.03          120.97          1
    14           Nyheim Hines    IND          136.00          120.97          8
    80            Marlon Mack    IND          214.09          120.97          5
    12         James Robinson    JAX           36.78          169.01          1
    65            Carlos Hyde    JAX          205.79          169.01          4
    29  Clyde Edwards-Helaire     KC           24.93          183.66          1
    41        Darrel Williams     KC          208.59          183.66          4
    67        Jerick McKinnon     KC          210.57          183.66          8
    85         Larry Rountree    LAC          215.42          202.32          5
    50         Justin Jackson    LAC          213.39          202.32          4
    19          Austin Ekeler    LAC           11.07          202.32          1
    48      Darrell Henderson    LAR           60.51           41.98          2
    4             Sony Michel    LAR          102.49           41.98          3
    83              Jake Funk    LAR          215.97           41.98          7
    62            Josh Jacobs     LV           57.59           68.81          2
    6            Kenyan Drake     LV          126.40           68.81          3
    0            Myles Gaskin    MIA           70.24          134.00          1
    38          Malcolm Brown    MIA          204.24          134.00          4
    56           Salvon Ahmed    MIA          214.95          134.00          4
    27     Alexander Mattison    MIN          150.21          147.95          4
    11            Dalvin Cook    MIN            2.26          147.95          1
    49          Damien Harris     NE           65.08           73.34          2
    40    Rhamondre Stevenson     NE          138.42           73.34          3
    32            James White     NE          144.57           73.34          6
    66            J.J. Taylor     NE          215.08           73.34          7
    28           Alvin Kamara     NO            3.00          195.34          1
    42             Tony Jones     NO          198.34          195.34          4
    55         Saquon Barkley    NYG           12.47          200.35          1
    23        Devontae Booker    NYG          212.82          200.35          4
    46          Tevin Coleman    NYJ          171.31           56.82          3
    15             Ty Johnson    NYJ          174.31           56.82          6
    16         Michael Carter    NYJ          114.49           56.82          2
    31          Miles Sanders    PHI           51.00          130.76          1
    54       Kenneth Gainwell    PHI          181.76          130.76          5
    8            Boston Scott    PHI          212.62          130.76          4
    64           Najee Harris    PIT           15.34          200.61          1
    87            Benny Snell    PIT          215.95          200.61          5
    45           Chris Carson    SEA           36.48          162.58          1
    10          Rashaad Penny    SEA          199.06          162.58          4
    79           Alex Collins    SEA          215.56          162.58          5
    78        Elijah Mitchell     SF          215.83            4.26          7
    47            Trey Sermon     SF           70.54            4.26          2
    52         Raheem Mostert     SF           74.80            4.26          3
    33           Ronald Jones     TB           97.85           32.14          2
    21      Leonard Fournette     TB          129.99           32.14          3
    58        Giovani Bernard     TB          164.41           32.14          6
    86        Ke'Shawn Vaughn     TB          215.99           32.14          7
    76          Mekhi Sargent    TEN          215.84          209.00          5
    7           Derrick Henry    TEN            4.77          209.00          1
    71        Darrynton Evans    TEN          213.77          209.00          4
    70       Jeremy McNichols    TEN          215.96          209.00          5
    36          J.D. McKissic    WAS          153.37          136.72          8
    61         Antonio Gibson    WAS           16.65          136.72          1
    74        Jaret Patterson    WAS          214.66          136.72          5



```python
archetypes2022=pd.merge(unique_players2022,clay_proj2022,on='player_name')
archetypes2022 = archetypes2022.sort_values(['team_x', 'projection_adp'])
archetypes2022['depth_chart'] = archetypes2022.groupby('team_x')['projection_adp'].rank(method='min')
archetypes2022['archetype']=0 
archetypes2022.loc[(archetypes2022['adp_difference'] > 97.5) & (archetypes2022['projection_adp'] < 100), 'archetype'] = 1 
archetypes2022.loc[(archetypes2022['adp_difference'] > 97.5) & (archetypes2022['projection_adp'] > 100) & (archetypes2022['clay_halfpprpoints'] < 65), 'archetype'] = 5
archetypes2022.loc[(archetypes2022['adp_difference'] > 97.5) & (archetypes2022['clay_halfpprpoints'] >= 65) & (archetypes2022['projection_adp'] > 100), 'archetype'] = 4 
archetypes2022.loc[(archetypes2022['adp_difference'] < 97.5) & (archetypes2022['depth_chart'] == 1), 'archetype'] = 2
archetypes2022.loc[(archetypes2022['adp_difference'] < 97.5) & (archetypes2022['depth_chart'] == 2), 'archetype'] = 3
archetypes2022.loc[(archetypes2022['adp_difference'] < 97.5) & (archetypes2022['projection_adp'] > 100) & (archetypes2022['clay_halfpprpoints'] < 65) & (archetypes2022['depth_chart'] >= 3) , 'archetype'] = 7
archetypes2022.loc[(archetypes2022['adp_difference'] < 97.5) & (archetypes2022['projection_adp'] > 100) & (archetypes2022['clay_halfpprpoints'] > 65) & (archetypes2022['depth_chart'] >= 3), 'archetype'] = 6
archetypes2022.loc[(archetypes2022['adp_difference'] > 97.5) & (archetypes2022['clay_receivingoverrushing'] > 1), 'archetype'] = 8
archetypes2022.loc[archetypes2022['player_name'] == 'J.D. McKissic', 'archetype'] = 6
archetypes2022.loc[archetypes2022['player_name'] == 'Brian Robinson', 'archetype'] = 3
archetypes2022.loc[archetypes2022['player_name'] == 'Rex Burkhead', 'archetype'] = 4
archetypes2022 = archetypes2022.sort_values(['team_x', 'projection_adp'])
archetypes2022['depth_chart'] = archetypes2022.groupby('team_x')['projection_adp'].rank(method='min')
archetypes_sorted2022 = archetypes2022.sort_values(by='team_x') 
print(archetypes_sorted2022[['player_name','team_x','projection_adp','adp_difference','archetype']])
```

                   player_name team_x  projection_adp  adp_difference  archetype
    0             James Conner    ARI           34.57          154.09          1
    2             Eno Benjamin    ARI          188.66          154.09          5
    1          Darrel Williams    ARI          212.78          154.09          5
    3          Keaontay Ingram    ARI          215.94          154.09          5
    4    Cordarrelle Patterson    ATL          103.42           44.79          2
    6           Tyler Allgeier    ATL          148.21           44.79          3
    5          Damien Williams    ATL          215.28           44.79          6
    7             J.K. Dobbins    BAL           67.57          128.58          1
    9               Mike Davis    BAL          196.15          128.58          5
    10            Kenyan Drake    BAL          203.85          128.58          5
    8              Gus Edwards    BAL          213.61          128.58          4
    13               Zack Moss    BUF          208.06            0.14          7
    11        Devin Singletary    BUF          106.81            0.14          3
    12              James Cook    BUF          106.67            0.14          2
    14     Christian McCaffrey    CAR            1.75          210.87          1
    15           Chuba Hubbard    CAR          212.62          210.87          5
    16          D'Onta Foreman    CAR          213.08          210.87          4
    17        David Montgomery    CHI           65.46           87.11          2
    18          Khalil Herbert    CHI          152.57           87.11          3
    19           Trestan Ebner    CHI          215.94           87.11          7
    20               Joe Mixon    CIN           13.00          201.61          1
    21           Samaje Perine    CIN          214.61          201.61          4
    22             Chris Evans    CIN          215.06          201.61          8
    26             Jerome Ford    CLE          215.96           67.92          7
    25        D'Ernest Johnson    CLE          213.93           67.92          7
    23              Nick Chubb    CLE           27.15           67.92          2
    24             Kareem Hunt    CLE           95.07           67.92          3
    27         Ezekiel Elliott    DAL           41.75           39.51          2
    28            Tony Pollard    DAL           81.26           39.51          3
    29        Javonte Williams    DEN           19.86           93.17          2
    30           Melvin Gordon    DEN          113.03           93.17          3
    31           D'Andre Swift    DET           16.27          140.85          1
    32         Jamaal Williams    DET          157.12          140.85          4
    33          Craig Reynolds    DET          215.98          140.85          5
    34             Aaron Jones     GB           18.47           35.59          2
    35               AJ Dillon     GB           54.06           35.59          3
    37            Rex Burkhead    HOU          210.33          149.07          4
    36           Dameon Pierce    HOU           61.26          149.07          1
    38         Jonathan Taylor    IND            1.54          121.80          1
    39            Nyheim Hines    IND          123.34          121.80          8
    40          Travis Etienne    JAX           36.60           99.21          1
    41          James Robinson    JAX          135.81           99.21          4
    42            Snoop Conner    JAX          215.97           99.21          5
    44         Jerick McKinnon     KC          179.20           60.50          6
    46            Ronald Jones     KC          214.72           60.50          7
    43   Clyde Edwards-Helaire     KC           84.12           60.50          2
    45           Isiah Pacheco     KC          144.62           60.50          3
    48             Sony Michel    LAC          202.91          176.17          5
    50           Joshua Kelley    LAC          215.13          176.17          5
    47           Austin Ekeler    LAC            5.38          176.17          1
    49          Isaiah Spiller    LAC          181.55          176.17          5
    51               Cam Akers    LAR           59.27           54.90          2
    52       Darrell Henderson    LAR          114.17           54.90          3
    53          Kyren Williams    LAR          214.93           54.90          7
    54             Josh Jacobs     LV           81.00           79.98          2
    55             Zamir White     LV          160.98           79.98          3
    56          Ameer Abdullah     LV          213.97           79.98          7
    57          Brandon Bolden     LV          215.96           79.98          7
    58           Chase Edmonds    MIA           71.28           80.64          2
    59          Raheem Mostert    MIA          151.92           80.64          3
    60            Myles Gaskin    MIA          215.85           80.64          7
    63            Kene Nwangwu    MIN          215.98          126.29          5
    64             Ty Chandler    MIN          215.66          126.29          5
    62      Alexander Mattison    MIN          134.51          126.29          4
    61             Dalvin Cook    MIN            8.22          126.29          1
    65     Rhamondre Stevenson     NE           80.66           19.81          2
    66           Damien Harris     NE          100.47           19.81          3
    67           Pierre Strong     NE          215.93           19.81          7
    68            Alvin Kamara     NO           14.80          173.90          1
    69             Mark Ingram     NO          188.70          173.90          4
    70              Tony Jones     NO          216.00          173.90          5
    73        Antonio Williams    NYG          215.97          201.89          8
    71          Saquon Barkley    NYG           13.81          201.89          1
    72             Matt Breida    NYG          215.70          201.89          4
    76              Ty Johnson    NYJ          215.99           78.81          7
    75          Michael Carter    NYJ          129.85           78.81          3
    74             Breece Hall    NYJ           51.04           78.81          2
    77           Miles Sanders    PHI          100.54           30.26          2
    78        Kenneth Gainwell    PHI          130.80           30.26          3
    79            Boston Scott    PHI          210.32           30.26          7
    80             Trey Sermon    PHI          215.37           30.26          7
    81            Najee Harris    PIT           12.52          196.78          1
    82           Jaylen Warren    PIT          209.30          196.78          5
    83             Benny Snell    PIT          215.93          196.78          5
    87            Travis Homer    SEA          215.92           30.66          7
    84           Rashaad Penny    SEA           90.40           30.66          2
    85          Kenneth Walker    SEA          121.06           30.66          3
    86           DeeJay Dallas    SEA          215.89           30.66          7
    89             Jeff Wilson     SF          188.25          110.20          5
    88         Elijah Mitchell     SF           67.94          110.20          1
    90      Tyrion Davis-Price     SF          178.14          110.20          5
    91       Leonard Fournette     TB           23.94          111.52          1
    92           Rachaad White     TB          135.46          111.52          4
    93         Ke'Shawn Vaughn     TB          215.92          111.52          5
    97         Julius Chestnut    TEN          215.97          207.22          8
    94           Derrick Henry    TEN            7.35          207.22          1
    95       Dontrell Hilliard    TEN          214.57          207.22          8
    96          Hassan Haskins    TEN          215.40          207.22          5
    99           J.D. McKissic    WAS          163.71           74.52          6
    100         Brian Robinson    WAS          164.07           74.52          3
    98          Antonio Gibson    WAS           89.19           74.52          2
    101      Jonathan Williams    WAS          215.99           74.52          7


## All Teams, Playoff Teams, and Top 1% Teams
In order to make proper comparisons we need to split all of the teams into teams that made the playoff round and top 1% teams. Comparing against top 1% teams is very important because in Best Ball Mania 4 there is a portion of the prize pool awarded to the highest regular season finishers and there is a likely correlation between having a top 1% team and making the Best Ball Mania finals.


```python
archetypedrafts2021=pd.merge(archetypes2021,df_2021,on='player_name')
archetypedrafts2021 = archetypedrafts2021.rename(columns={'projection_adp_x': 'projection_adp'})
archetypedrafts2022=pd.merge(archetypes2021,df_2022,on='player_name')
archetypedrafts2022=archetypedrafts2022.rename(columns={'projection_adp_x': 'projection_adp'})
desired_columns = ['player_name', 'tournament_entry_id', 'projection_adp', 'pick_points', 'roster_points', 'archetype', 'playoff_team', 'position_name','overall_pick_number']
everybody2021 = archetypedrafts2021.groupby('tournament_entry_id')
everybody2022 = archetypedrafts2022.groupby('tournament_entry_id')
all_teams2021 = everybody2021.apply(lambda x: x[desired_columns]).reset_index(drop=True)
all_teams2022 = everybody2022.apply(lambda x: x[desired_columns]).reset_index(drop=True)
playoff_teams2021 = all_teams2021[all_teams2021['playoff_team'] == 1]
playoff_teams2022 = all_teams2022[all_teams2022['playoff_team'] == 1]
```

## Splitting Teams by Structure
Another important concept when trying to figure out which backup running backs to select is which RBs you chose before the backups and the best way to do this was to split up by structure. Hyper Fragile was defined as taking 2 or more running backs before pick 36, Hero RB was defined as taking 1 running back before pick 36 and Zero RB was defined as taking 0 running backs before pick 36.


```python
all_teams2021 = all_teams2021.sort_values('overall_pick_number')
all_teams2021['order_taken'] = all_teams2021.groupby('tournament_entry_id').cumcount() + 1
all_teams2021['group_type'] = ''
for name, group in all_teams2021.groupby('tournament_entry_id'):
    count_under_36 = np.sum(group['overall_pick_number'] < 36)
    if count_under_36 >= 2:
        all_teams2021.loc[group.index, 'group_type'] = 'Hyper Fragile'
    elif count_under_36 == 1:
        all_teams2021.loc[group.index, 'group_type'] = 'Hero RB'
    else:
        all_teams2021.loc[group.index, 'group_type'] = 'Zero RB'
playoff_teams2021 = all_teams2021[all_teams2021['playoff_team'] == 1]
playoff_teams2021 = playoff_teams2021.sort_values('overall_pick_number')
playoff_teams2021['order_taken'] = playoff_teams2021.groupby('tournament_entry_id').cumcount() + 1
playoff_teams2021['group_type'] = ''

for name, group in playoff_teams2021.groupby('tournament_entry_id'):
    count_under_36 = np.sum(group['overall_pick_number'] < 36)
    if count_under_36 >= 2:
        playoff_teams2021.loc[group.index, 'group_type'] = 'Hyper Fragile'
    elif count_under_36 == 1:
        playoff_teams2021.loc[group.index, 'group_type'] = 'Hero RB'
    else:
        playoff_teams2021.loc[group.index, 'group_type'] = 'Zero RB'
top_1_percent2021_cutoffs = all_teams2021.groupby('group_type')['roster_points'].transform(lambda x: x.quantile(0.99))
top_1_percent2021_teams = all_teams2021[all_teams2021['roster_points'] >= top_1_percent2021_cutoffs].copy()

all_teams2022 = all_teams2022.sort_values('overall_pick_number')
all_teams2022['order_taken'] = all_teams2022.groupby('tournament_entry_id').cumcount() + 1
all_teams2022['group_type'] = ''
for name, group in all_teams2022.groupby('tournament_entry_id'):
    count_under_36 = np.sum(group['overall_pick_number'] < 36)
    if count_under_36 >= 2:
        all_teams2022.loc[group.index, 'group_type'] = 'Hyper Fragile'
    elif count_under_36 == 1:
        all_teams2022.loc[group.index, 'group_type'] = 'Hero RB'
    else:
        all_teams2022.loc[group.index, 'group_type'] = 'Zero RB'
        
playoff_teams2022 = all_teams2022[all_teams2022['playoff_team'] == 1]       
playoff_teams2022 = playoff_teams2022.sort_values('overall_pick_number')
playoff_teams2022['order_taken'] = playoff_teams2022.groupby('tournament_entry_id').cumcount() + 1
playoff_teams2022['group_type'] = ''

for name, group in playoff_teams2022.groupby('tournament_entry_id'):
    count_under_36 = np.sum(group['overall_pick_number'] < 36)
    if count_under_36 >= 2:
        playoff_teams2022.loc[group.index, 'group_type'] = 'Hyper Fragile'
    elif count_under_36 == 1:
        playoff_teams2022.loc[group.index, 'group_type'] = 'Hero RB'
    else:
        playoff_teams2022.loc[group.index, 'group_type'] = 'Zero RB'
top_1_percent2022_cutoffs = all_teams2022.groupby('group_type')['roster_points'].transform(lambda x: x.quantile(0.99))
top_1_percent2022_teams = all_teams2022[all_teams2022['roster_points'] >= top_1_percent2022_cutoffs].copy()
```

## 2021 Third RB Pick Number Data


```python
datasets = [all_teams2021, playoff_teams2021, top_1_percent2021_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
fig, axs = plt.subplots(3, 3, figsize=(18, 18), sharey='row')
for i, data in enumerate(datasets):
    for j, group_type in enumerate(group_types):
        group_data = data[data['group_type'] == group_type]
        third_rb = group_data[group_data['order_taken'] == 3]
        axs[i, j].hist(third_rb['overall_pick_number'], bins=10, edgecolor='black')
        axs[i, j].set_title(f'{dataset_names[i]}: {group_type} Group', fontsize=18)
        max_adp = third_rb['overall_pick_number'].max()
        axs[i, j].set_xticks(np.arange(0, max_adp + 10, 20))
        mean_adp = third_rb['overall_pick_number'].mean()
        std_adp = third_rb['overall_pick_number'].std()
        axs[i, j].text(0.7, 0.8, f'Mean: {mean_adp:.2f}\nStd: {std_adp:.2f}', transform=axs[i, j].transAxes, fontsize=15, color='red')
        axs[i, j].tick_params(axis='both', which='major', labelsize=16)
for ax in axs.flat:
    ax.set_xlabel('ADP', fontsize=16)
for ax in axs[:, 0]:
    ax.set_ylabel('Count', fontsize=16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_23_0.png)
    


The overall trend from the 2021 third RB pick number data is that it was better to draft your third RB later than the average drafter in 2021. I think an interesting result versus my expectation was how early the hyper fragile third RB was taken. One might think if you draft your first two RB within the first 36 picks that you can and should wait till a lot later but that wasn't really the case.

## 2022 Third RB Pick Number Data


```python
datasets = [all_teams2022, playoff_teams2022, top_1_percent2022_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
fig, axs = plt.subplots(3, 3, figsize=(18, 18), sharey='row')
for i, data in enumerate(datasets):
    for j, group_type in enumerate(group_types):
        group_data = data[data['group_type'] == group_type]
        third_rb = group_data[group_data['order_taken'] == 3]
        axs[i, j].hist(third_rb['overall_pick_number'], bins=10, edgecolor='black')
        axs[i, j].set_title(f'{dataset_names[i]}: {group_type} Group', fontsize=18)
        max_adp = third_rb['overall_pick_number'].max()
        axs[i, j].set_xticks(np.arange(0, max_adp + 10, 20))
        mean_adp = third_rb['overall_pick_number'].mean()
        std_adp = third_rb['overall_pick_number'].std()
        axs[i, j].text(0.7, 0.8, f'Mean: {mean_adp:.2f}\nStd: {std_adp:.2f}', transform=axs[i, j].transAxes, fontsize=15, color='red')
        axs[i, j].tick_params(axis='both', which='major', labelsize=16)
for ax in axs.flat:
    ax.set_xlabel('ADP', fontsize=16)
for ax in axs[:, 0]:
    ax.set_ylabel('Count', fontsize=16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_26_0.png)
    


It seemed that the exact opposite trend compared to 2021 occured. In 2022 picking your third RB earlier than the mean was the better strategy. The mean decreasing means that those teams that went later faded off and didn't make the cut of playoff teams or top 1 percent teams. However, the other trend of picking your third hyper fragile RB pretty early around pick 85 remained the same. 

## 2021 4th Running Back Pick Number 


```python
datasets = [all_teams2021, playoff_teams2021, top_1_percent2021_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
fig, axs = plt.subplots(3, 3, figsize=(18, 18), sharey='row')
for i, data in enumerate(datasets):
    for j, group_type in enumerate(group_types):
        group_data = data[data['group_type'] == group_type]
        third_rb = group_data[group_data['order_taken'] == 4]
        axs[i, j].hist(third_rb['overall_pick_number'], bins=10, edgecolor='black')
        axs[i, j].set_title(f'{dataset_names[i]}: {group_type} Group', fontsize=18)
        max_adp = third_rb['overall_pick_number'].max()
        axs[i, j].set_xticks(np.arange(0, max_adp + 10, 20))
        mean_adp = third_rb['overall_pick_number'].mean()
        std_adp = third_rb['overall_pick_number'].std()
        axs[i, j].text(0.7, 0.8, f'Mean: {mean_adp:.2f}\nStd: {std_adp:.2f}', transform=axs[i, j].transAxes, fontsize=15, color='red')
        axs[i, j].tick_params(axis='both', which='major', labelsize=16)
for ax in axs.flat:
    ax.set_xlabel('ADP', fontsize=16)
for ax in axs[:, 0]:
    ax.set_ylabel('Count', fontsize=16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_29_0.png)
    


Not much change between the playoff, top 1% and all teams in Hyper Fragile and Hero but with the Zero RB teams it was definitely better to draft your fourth RB earlier and I think this can be correlated to Hayden Wink's idea that gets highlighted by Pat Kerrane a lot of needing to draft 4 RBs through no later than round 13. This data definitely proves that because the thirteenth round ends at pick 156. The only subset that has a mean reasonably close to that number was all teams Zero RB which saw a decrease among the playoff and top 1% teams.

## 2022 4th Running Back Pick Number


```python
datasets = [all_teams2022, playoff_teams2022, top_1_percent2022_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
fig, axs = plt.subplots(3, 3, figsize=(18, 18), sharey='row')
for i, data in enumerate(datasets):
    for j, group_type in enumerate(group_types):
        group_data = data[data['group_type'] == group_type]
        third_rb = group_data[group_data['order_taken'] == 4]
        axs[i, j].hist(third_rb['overall_pick_number'], bins=10, edgecolor='black')
        axs[i, j].set_title(f'{dataset_names[i]}: {group_type} Group', fontsize=18)
        max_adp = third_rb['overall_pick_number'].max()
        axs[i, j].set_xticks(np.arange(0, max_adp + 10, 20))
        mean_adp = third_rb['overall_pick_number'].mean()
        std_adp = third_rb['overall_pick_number'].std()
        axs[i, j].text(0.1, 0.8, f'Mean: {mean_adp:.2f}\nStd: {std_adp:.2f}', transform=axs[i, j].transAxes, fontsize=15, color='red')
        axs[i, j].tick_params(axis='both', which='major', labelsize=16)
for ax in axs.flat:
    ax.set_xlabel('ADP', fontsize=16)
for ax in axs[:, 0]:
    ax.set_ylabel('Count', fontsize=16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_32_0.png)
    


In 2022 it was better to draft the fourth RB below the mean of all teams for every type of structure. This Hayden Wink's idea of drafting 4 running backs through 13 rounds can be seen here again as this time there are multiple subset at or near 156 but each one of them ended up decreasing in the top 1% of teams

## The Upper and Lower Bounds of Top 1% Teams by Structure


```python
bounds_df = pd.DataFrame(columns=["Group Type", "Order Taken", "Lower Bound", "Upper Bound"])
index = 0
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
datasets = [top_1_percent2021_teams, top_1_percent2022_teams]
for order_taken in [3, 4]: 
    for group_type in group_types:
        for i, data in enumerate(datasets):
            group_data = data[data['group_type'] == group_type]
            rb_data = group_data[group_data['order_taken'] == order_taken]
            Q1 = rb_data['overall_pick_number'].quantile(0.25)
            Q3 = rb_data['overall_pick_number'].quantile(0.75)
            low_guardrail = ((Q1))
            high_guardrail = ((Q3))
            bounds_df.loc[index] = [group_type, order_taken, low_guardrail, high_guardrail]
            index += 1           
guardrails = bounds_df.groupby(['Group Type', 'Order Taken']).mean().reset_index()
print(guardrails)
```

          Group Type  Order Taken  Lower Bound  Upper Bound
    0        Hero RB            3         89.5       130.75
    1        Hero RB            4        121.5       169.50
    2  Hyper Fragile            3         64.0       102.50
    3  Hyper Fragile            4        104.5       160.50
    4        Zero RB            3        102.5       145.50
    5        Zero RB            4        130.0       170.50


The code above creates 25% and 75% bounds for the top 1 percent of teams based on the order taken of the running back and which structure the team was in. It then averages the 2021 bounds with the 2022 bounds.

## 2021 3rd RB Archetype


```python
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
datasets = [all_teams2021, playoff_teams2021, top_1_percent2021_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']

for i, group_type in enumerate(group_types):
    for j, dataset in enumerate(datasets):
        data = dataset[(dataset['group_type'] == group_type) & (dataset['order_taken'] == 3) & (dataset['archetype'] >= 1)]
        weights = np.ones_like(data['archetype']) / len(data['archetype'])
        counts, bins, patches = axs[i, j].hist(data['archetype'], bins=7, edgecolor='black', alpha=0.6, weights=weights)
        axs[i, j].set_xticks(bins)
        title = axs[i, j].set_title(f'{group_type} - {dataset_names[j]}')
        title.set_fontsize(18)
        text = axs[i, j].text(0.6, 0.8, f'Mean: {data["archetype"].mean():.2f}\nStd: {data["archetype"].std():.2f}', transform=axs[i, j].transAxes, fontsize=16)
        for item in [axs[i, j].xaxis.label, axs[i, j].yaxis.label]:
            item.set_fontsize(16)
        for item in axs[i, j].get_xticklabels() + axs[i, j].get_yticklabels():
            item.set_fontsize(16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_38_0.png)
    


The one trend that pops out is that regardless of structure having a third RB with an archetype of three or less was a very common strategy. Also, as you go from all teams to playoff teams to top 1 percent teams, no matter the structure the archetype three was increasingly a higher percent of overall RB archetypes.

## 2022 3rd RB Archetype


```python
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
datasets = [all_teams2022, playoff_teams2022, top_1_percent2022_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']

for i, group_type in enumerate(group_types):
    for j, dataset in enumerate(datasets):
        data = dataset[(dataset['group_type'] == group_type) & (dataset['order_taken'] == 3) & (dataset['archetype'] >= 1)]
        weights = np.ones_like(data['archetype']) / len(data['archetype'])
        counts, bins, patches = axs[i, j].hist(data['archetype'], bins=7, edgecolor='black', alpha=0.6, weights=weights)
        axs[i, j].set_xticks(bins)
        title = axs[i, j].set_title(f'{group_type} - {dataset_names[j]}')
        title.set_fontsize(18)
        text = axs[i, j].text(0.6, 0.8, f'Mean: {data["archetype"].mean():.2f}\nStd: {data["archetype"].std():.2f}', transform=axs[i, j].transAxes, fontsize=16)
        for item in [axs[i, j].xaxis.label, axs[i, j].yaxis.label]:
            item.set_fontsize(16)
        for item in axs[i, j].get_xticklabels() + axs[i, j].get_yticklabels():
            item.set_fontsize(16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_41_0.png)
    


Two things stick out from the 2022 3rd RB Archetype Data: the increase of the archetype one in the Hero RB structure and the increase of the archetype two in the hyper fragile. The increase in the archtype one in the Hero RB structure makes a lot of sense, if you can get a RB that is a bellcow later than your competitors it is very easy to see how that can translate into a top 1% team. The increase of the archetype two in the hyper fragile is very interesting and I know a common strategy among drafters is take the cheaper guy in some of the ambigious backfields but I think this proves the point of really trying to mix both the archetype two and archetype three into your portfolio being a preferable strategy.

## 2021 4th RB Archetype


```python
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
datasets = [all_teams2021, playoff_teams2021, top_1_percent2021_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']

for i, group_type in enumerate(group_types):
    for j, dataset in enumerate(datasets):
        data = dataset[(dataset['group_type'] == group_type) & (dataset['order_taken'] == 4) & (dataset['archetype'] >= 1)]
        weights = np.ones_like(data['archetype']) / len(data['archetype'])
        counts, bins, patches = axs[i, j].hist(data['archetype'], bins=7, edgecolor='black', alpha=0.6, weights=weights)
        axs[i, j].set_xticks(bins)
        title = axs[i, j].set_title(f'{group_type} - {dataset_names[j]}')
        title.set_fontsize(18)
        text = axs[i, j].text(0.6, 0.8, f'Mean: {data["archetype"].mean():.2f}\nStd: {data["archetype"].std():.2f}', transform=axs[i, j].transAxes, fontsize=16)
        for item in [axs[i, j].xaxis.label, axs[i, j].yaxis.label]:
            item.set_fontsize(16)
        for item in axs[i, j].get_xticklabels() + axs[i, j].get_yticklabels():
            item.set_fontsize(16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_44_0.png)
    


The one trend that seems consistent throughout the 2021 data is the archetype three being dominant. We see again the increase of archetype three throughout all structures as you go from all team to playoff teams to top 1 percent teams. I think there is sometimes a belief among some drafters that your fourth RB will never really see the light of the day and doesn't matter so just punt the position off and take it way later. The 2021 data would disagree heavily with that statement.

## 2022 4th RB Archetype


```python
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
group_types = ['Hyper Fragile', 'Hero RB', 'Zero RB']
datasets = [all_teams2022, playoff_teams2022, top_1_percent2022_teams]
dataset_names = ['All Teams', 'Playoff Teams', 'Top 1% Teams']

for i, group_type in enumerate(group_types):
    for j, dataset in enumerate(datasets):
        data = dataset[(dataset['group_type'] == group_type) & (dataset['order_taken'] == 4) & (dataset['archetype'] >= 1)]
        weights = np.ones_like(data['archetype']) / len(data['archetype'])
        counts, bins, patches = axs[i, j].hist(data['archetype'], bins=7, edgecolor='black', alpha=0.6, weights=weights)
        axs[i, j].set_xticks(bins)
        title = axs[i, j].set_title(f'{group_type} - {dataset_names[j]}')
        title.set_fontsize(18)
        text = axs[i, j].text(0.6, 0.8, f'Mean: {data["archetype"].mean():.2f}\nStd: {data["archetype"].std():.2f}', transform=axs[i, j].transAxes, fontsize=16)
        for item in [axs[i, j].xaxis.label, axs[i, j].yaxis.label]:
            item.set_fontsize(16)
        for item in axs[i, j].get_xticklabels() + axs[i, j].get_yticklabels():
            item.set_fontsize(16)
plt.tight_layout()
plt.show()
```


    
![png](A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_files/A%20First%20Look%20into%20Layering-Analyzing%20When%20and%20Who%20to%20Take%20as%20Backup%20RBs%20%281%29_47_0.png)
    


The are two trends that are pretty interesting here: the archetype 3 dominance yet again and the decreasing mean from all teams to playoff to top 1 percent teams. The increasing archetype 3 among all structures as you go from all teams to playoff teams to top 1 percent teams just continues to be strong here and shows that the fourth RB points really do matter and you can't just have any backup. The decreasing mean tells a similar story of the necessity of points out of your fourth running back. The archetypes are assembled in a way that match what should be highest points to lowest points and therefore a decrease in mean correlates with an increase in projected points.

## Conclusion


```python
print(guardrails)
```

          Group Type  Order Taken  Lower Bound  Upper Bound
    0        Hero RB            3         89.5       130.75
    1        Hero RB            4        121.5       169.50
    2  Hyper Fragile            3         64.0       102.50
    3  Hyper Fragile            4        104.5       160.50
    4        Zero RB            3        102.5       145.50
    5        Zero RB            4        130.0       170.50


The goal of this project was to decipher who and when to pick your 3rd RB. For the when side of this question, the best answer I believe is a range under which it would be recommended to take your backup RBs based on top 1 percent scores from previous years. The table above creates 25% and 75% bounds for the top 1 percent of teams based on the order taken of the running back and which structure the team was in. It then averages the 2021 bounds with the 2022 bounds. This gives guardrails for when to take each RB depending on your structure. Going outside of these rails doesn't mean your team is destined for failure but I do think if I was to pick whether all my teams could fall outside or in between, I would chose the latter.


Now for the harder question of who you should take. The archetype data showed the prevalence of the 2/3 archetype and the non-existence of the 4/5 archetype across all structures amongst playoff and top 1 percent teams. It seems from the data that it is much better to pick RBs in the committee situation rather than going after RBs in a backup role whether they have a slight role or no role. This translates to this year's Best Ball Mania 4 by taking RBs like Khalil Herbert, Damien Harris, Rashaad Penny, and Devon Achane instead of RBs like Elijah Mitchell, Jaylen Warren, Tank Bigsby, and Jerome Ford.
