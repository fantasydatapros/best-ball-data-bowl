Surveying the Avalanche: What happens to the draft board when there’s a
run at a position?
================

**Participant:** Dan Falkenheim
([@thefalkon](https://twitter.com/thefalkon/))

**Introduction**

My project will focus on defining a positional run using pick-by-pick
data, show how a positional run affects the draft board and parse out
strategies for handling a positional run. To keep the project limited in
scope, I will use data from BBM2 and BBM3, focus on wide receiver runs
in the first ten rounds of a draft and analyze the success of different
strategies through advance rates and team regular season point totals.

**Import Libraries and Pick-by-Pick Data**

``` r
library(tidyverse)
library(gt)
library(gtExtras)
library(ggthemes)
library(slider)
set.seed(123)

setwd("C:/Users/danfa/Desktop/R/Best Ball Data Bowl Submission/best-ball-data-bowl-master/data/2022/regular_season/fast")

bbm3_fast <- list.files(pattern = "*.csv")[1:27] %>% map_df(~read_csv(.))

setwd("C:/Users/danfa/Desktop/R/Best Ball Data Bowl Submission/best-ball-data-bowl-master/data/2022/regular_season/mixed")

bbm3_slow <- list.files(pattern = "*.csv")[1:9] %>% map_df(~read_csv(.))

setwd("C:/Users/danfa/Desktop/R/Best Ball Data Bowl Submission/best-ball-data-bowl-master/data/2022/post_season/quarterfinals")

# We are going to use the quarterfinals data frame to create advance rate data.
bbm3_quarterfinals <- list.files(pattern = "*.csv")[1:10] %>% map_df(~read_csv(.)) %>%
  # We will grab the unique tournament entry ID's here.
  .$tournament_entry_id %>%
  unique()

bbm3 <- rbind(bbm3_fast, bbm3_slow) %>%
  # Next, we'll change the playoff_team column to reflect whether a team advanced.
  mutate(playoff_team = if_else(tournament_entry_id %in% bbm3_quarterfinals, 1, 0),
         # We are also going to give every player who has an ADP of 0 an ADP of 216.
         projection_adp = if_else(projection_adp == 0, 216, projection_adp))

setwd("C:/Users/danfa/Desktop/R/Best Ball Data Bowl Submission/best-ball-data-bowl-master/data/2021/BBM2")

bbm2 <- read.csv("BBM_II_Data_Dump_Regular_Season_01312022.csv") %>%
  # We need to filter out one draft since it doesn't contain every pick.
  filter(draft_id != "27bbd807-bca3-4486-b8aa-7101fda4c9eb") %>%
  # We are also going to give every player who has an ADP of 0 an ADP of 216.
  mutate(projection_adp = if_else(projection_adp == 0, 216, projection_adp))

# Data Cleanup
rm(bbm3_fast, bbm3_quarterfinals, bbm3_slow)
```

**Data Manipulation**

Once we have our data, we will create two new data frames.

1.  The first data frame will have four new columns which mark whether a
    drafter selected a specific position at each pick in the draft.
    Then, we will cumulatively sum each of these columns, providing a
    running count of how many players of each position were taken at
    each pick in the draft. (The resulting columns will be called
    WR/RB/TE/QB_taken)

2.  We will repeat this process for the second data frame, but we will
    sort each draft by projection_ADP (referred to as ADP for the rest
    of the article) rather than the actual draft selections. This will
    create a roadmap for how the draft was *expected* to go and how many
    of each position were expected to be taken at each pick as if the
    draft strictly followed ADP. (The resulting columns will be called
    WR/RB/TE/QB_expected)

We will join these data frames, keeping only the expected WR, RB, TE,
and QB columns from the second data frame. Next, we will create a column
for each position called “WR/RB/TE/QB_over_expected,” which is the
difference between how many of each position were taken and how many
were expected to have been taken. A positive number will indicate more
players of that position were taken in the draft than expected, and a
negative number will indicate fewer players were taken in the draft than
expected.

Finally, we will group the data frame by individual overall pick number,
and then scale the WR/RB/TE/QB_over_expected columns. This will provide
a standardized metric to analyze how many players of a position were
taken over or under expectation regarding each individual pick. The
scaled metric will be called WR/RB/TE/QB_oe.

``` r
bbm3_positions_taken <- bbm3 %>%
  arrange(draft_id, overall_pick_number) %>%
  # These create new columns that mark any time a specific position was taken with a 1
  mutate(picked_RB = apply(., 1, function(x) length(which(x=="RB"))),
         picked_WR = apply(., 1, function(x) length(which(x=="WR"))),
         picked_QB = apply(., 1, function(x) length(which(x=="QB"))),
         picked_TE = apply(., 1, function(x) length(which(x=="TE")))) %>%
  group_by(draft_id) %>%
  # Then we cumulatively sum those columns
  mutate(RB_taken = cumsum(picked_RB),
         WR_taken = cumsum(picked_WR), 
         QB_taken = cumsum(picked_QB), 
         TE_taken = cumsum(picked_TE))

bbm3_expected_positions <- bbm3 %>%
  arrange(draft_id, projection_adp) %>%
  mutate(xRB = apply(., 1, function(x) length(which(x=="RB"))),
         xWR = apply(., 1, function(x) length(which(x=="WR"))),
         xQB = apply(., 1, function(x) length(which(x=="QB"))),
         xTE = apply(., 1, function(x) length(which(x=="TE")))) %>%
  group_by(draft_id) %>%
  mutate(RB_expected = cumsum(xRB),
         WR_expected = cumsum(xWR), 
         QB_expected = cumsum(xQB), 
         TE_expected = cumsum(xTE))

bbm3_expected_positions <- bbm3_expected_positions %>% 
  #bbm3_expected_positions is sorted by projection adp. In order to join with the original data frame,
  #I have changed the overall_pick_number to be an ordered list of numbers from 1:216. That way, the expected 
  #data frame is sorted as if the projection adp was the actual pick order.
  mutate(overall_pick_number = 1:216) %>%
  select(draft_id, overall_pick_number, RB_expected, WR_expected, QB_expected, TE_expected) %>%
  ungroup()

bbm3 <- bbm3_positions_taken %>% 
  left_join(bbm3_expected_positions, by = c("draft_id", "overall_pick_number"))

#Create Position Over Expected Columns
bbm3 <- bbm3 %>%
  mutate(RB_over_expected = RB_taken - RB_expected,
         WR_over_expected = WR_taken - WR_expected,
         QB_over_expected = QB_taken - QB_expected,
         TE_over_expected = TE_taken - TE_expected)

# Scale results
bbm3 <- bbm3 %>%
  ungroup() %>%
  group_by(overall_pick_number) %>%
  mutate(RB_oe = scale(RB_over_expected),
         WR_oe = scale(WR_over_expected),
         TE_oe = scale(TE_over_expected),
         QB_oe = scale(QB_over_expected)) %>%
  ungroup()

# Now, onto BBM2
bbm2_positions_taken <- bbm2 %>%
  arrange(draft_id, overall_pick_number) %>%
  mutate(picked_RB = apply(., 1, function(x) length(which(x=="RB"))),
         picked_WR = apply(., 1, function(x) length(which(x=="WR"))),
         picked_QB = apply(., 1, function(x) length(which(x=="QB"))),
         picked_TE = apply(., 1, function(x) length(which(x=="TE")))) %>%
  group_by(draft_id) %>%
  mutate(RB_taken = cumsum(picked_RB),
         WR_taken = cumsum(picked_WR), 
         QB_taken = cumsum(picked_QB), 
         TE_taken = cumsum(picked_TE))

bbm2_expected_positions <- bbm2 %>%
  arrange(draft_id, projection_adp) %>%
  mutate(xRB = apply(., 1, function(x) length(which(x=="RB"))),
         xWR = apply(., 1, function(x) length(which(x=="WR"))),
         xQB = apply(., 1, function(x) length(which(x=="QB"))),
         xTE = apply(., 1, function(x) length(which(x=="TE")))) %>%
  group_by(draft_id) %>%
  mutate(RB_expected = cumsum(xRB),
         WR_expected = cumsum(xWR), 
         QB_expected = cumsum(xQB), 
         TE_expected = cumsum(xTE))

bbm2_expected_positions <- bbm2_expected_positions %>% 
  #bbm2_expected_positions is sorted by projection adp. In order to join with the original data frame,
  #I've changed the overall_pick_number to be an ordered list of numbers from 1:216. That way, the expected 
  #data frame is sorted as if the projection adp was the actual pick order.
  mutate(overall_pick_number = 1:216) %>%
  select(draft_id, overall_pick_number, RB_expected, WR_expected, QB_expected, TE_expected) %>%
  ungroup()

bbm2 <- bbm2_positions_taken %>% 
  left_join(bbm2_expected_positions, by = c("draft_id", "overall_pick_number"))

#Create Position Over Expected Columns
bbm2 <- bbm2 %>%
  mutate(RB_over_expected = RB_taken - RB_expected,
         WR_over_expected = WR_taken - WR_expected,
         QB_over_expected = QB_taken - QB_expected,
         TE_over_expected = TE_taken - TE_expected)

#Scale results
bbm2 <- bbm2 %>%
  ungroup() %>%
  group_by(overall_pick_number) %>%
  mutate(RB_oe = scale(RB_over_expected),
         WR_oe = scale(WR_over_expected),
         TE_oe = scale(TE_over_expected),
         QB_oe = scale(QB_over_expected)) %>%
  ungroup()

# Data cleanup
rm(bbm3_positions_taken, bbm3_expected_positions, bbm2_positions_taken, bbm2_expected_positions)
```

Next, we are going to create a few new columns to define ADP value:

- Raw ADP Value: The difference between where the player was drafted and
  his ADP.

- Percentage ADP Value: The difference between where the player was
  drafted and his ADP, divided by his ADP.

- Raw Positional ADP Value: The difference between where the player was
  drafted and his position’s ADP. (In other words: If James Cook was
  selected as the RB31, Raw Positional ADP Value grabs what the RB31’s
  ADP was rather than James Cook’s individual ADP.)

- Percentage Positional ADP Value: The difference between where the
  player was drafted and his position’s ADP, divided by his position’s
  ADP.

``` r
#Now, generate positional ADP's based on the actual draft (by overall pick number), and by how the draft was expected to go (by projection_adp)
bbm3_positions_taken <- bbm3 %>%
  arrange(draft_id, position_name, overall_pick_number) %>%
  group_by(draft_id, position_name) %>%
  mutate(rank = row_number(),
         # New position name identifies the position by number (the WR26, for example)
         new_position_name = str_c(position_name, rank)) %>%
  ungroup()

bbm3_positions_expected <- bbm3 %>%
  arrange(draft_id, position_name, projection_adp) %>%
  group_by(draft_id, position_name) %>%
  mutate(rank = row_number(),
         new_position_name = str_c(position_name, rank),
         position_adp = projection_adp) %>%
  ungroup() %>%
  select(draft_id, new_position_name, position_adp)

# Join the two data frames
bbm3 <- bbm3_positions_taken %>% 
  left_join(bbm3_positions_expected, by = c("draft_id", "new_position_name"))

# Create the new ADP value columns 
bbm3 <- bbm3 %>%
  arrange(draft_id, overall_pick_number) %>%
  mutate(raw_adp_value = overall_pick_number - projection_adp,
         pct_adp_value = (overall_pick_number - projection_adp)/(projection_adp),
         raw_positional_adp_value = overall_pick_number - position_adp,
         pct_positional_adp_value = (overall_pick_number - position_adp)/(position_adp))
  
#Now, BBM2
bbm2_positions_taken <- bbm2 %>%
  arrange(draft_id, position_name, overall_pick_number) %>%
  group_by(draft_id, position_name) %>%
  mutate(rank = row_number(),
         new_position_name = str_c(position_name, rank)) %>%
  ungroup()

bbm2_positions_expected <- bbm2 %>%
  arrange(draft_id, position_name, projection_adp) %>%
  group_by(draft_id, position_name) %>%
  mutate(rank = row_number(),
         new_position_name = str_c(position_name, rank),
         position_adp = projection_adp) %>%
  ungroup() %>%
  select(draft_id, new_position_name, position_adp)

bbm2 <- bbm2_positions_taken %>% 
  left_join(bbm2_positions_expected, by = c("draft_id", "new_position_name"))

bbm2 <- bbm2 %>%
  arrange(draft_id, overall_pick_number) %>%
  mutate(projection_adp = if_else(projection_adp == 0, 216, projection_adp),
         raw_adp_value = overall_pick_number - projection_adp,
         pct_adp_value = (overall_pick_number - projection_adp)/(projection_adp),
         raw_positional_adp_value = overall_pick_number - position_adp,
         pct_positional_adp_value = (overall_pick_number - position_adp)/(position_adp))  

# Data cleanup
rm(bbm3_positions_taken, bbm3_positions_expected, bbm2_positions_taken, bbm2_positions_expected)
```

**Defining a Positional Run**

The definition of a positional run at wide receiver should match what a
run “feels” like during a draft. A run should contain an area of the
draft where more WRs go than expected and a stretch of picks
encompassing at least half a round or more. At the same time, the
criteria need to be strong enough to capture runs accurately and provide
a robust-enough sample size to begin conducting analysis.

With that being the case, I have defined a wide receiver run as:

1.  An area of the draft of ***at least eight consecutive picks*** where
    the amount of wide receivers taken above expectation is ***at least
    one standard deviation above the mean***.

2.  Within that stretch of picks, the number of wide receivers taken is
    ***greater than half of the total players taken***. (For example*,*
    a run of eight picks would include at least five wide receivers
    taken in that range.)

Increasing the criteria–looking at more than eight consecutive picks,
raising the standard deviation cutoff line to two or requiring more wide
receivers selected in the run–will capture more pronounced runs.
Decreasing the thresholds will grab a larger sample at the risk of
snagging picks that are not really part of a run. The process is
somewhat arbitrary, but the above criteria produced the sample that
matched what a wide receiver run feels like in drafts.

``` r
# Create a new column ("wr_run") that's a indicator of whether the amount of WRs taken over expectation is at least 1 standard deviation from the mean.
bbm3_wr_runs <- bbm3 %>%
  arrange(draft_id, overall_pick_number) %>%
  group_by(draft_id) %>%
  mutate(wr_run_extended = WR_oe >= 1.0)

# Replace NA's
bbm3_wr_runs$wr_run_extended[is.na(bbm3_wr_runs$wr_run_extended)] <- FALSE

# Set a run ID for each draft. This will create a unique run identifier that we can later use for grouping.
bbm3_wr_runs <- bbm3_wr_runs %>%
  group_by(draft_id) %>%
  mutate(wr_run_id = cumsum(wr_run_extended & lag(wr_run_extended, default = FALSE) == FALSE)) %>%
  ungroup()

# Calculate the lengths of each run, and how many WRs were taken in each run
bbm3_wr_runs <- bbm3_wr_runs %>%
  filter(wr_run_extended) %>% #This will filter out all picks that do not have a WR_oe >= 1.0
  group_by(draft_id, wr_run_id) %>%
  mutate(wr_run_length = sum(wr_run_extended),
         wrs_taken = sum(position_name == "WR")) %>%
  ungroup()

# Mark the start and end of each run
bbm3_wr_runs <- bbm3_wr_runs %>%
  group_by(draft_id, wr_run_id) %>%
  mutate(wr_run_start = row_number() == 1,
         wr_run_end = row_number() == wr_run_length & wr_run_extended) %>%
  ungroup()

# Next, we are going to filter and pare down the wr_runs data frame to prepare it for rejoining with the bbm3 data frame
bbm3_wr_runs_filtered <- bbm3_wr_runs %>%
  select(draft_id, tournament_entry_id, overall_pick_number, wr_run_length, wrs_taken, wr_run_start, wr_run_end) %>%
  # This filter call sets our parameters for the run. These can be adjusted as we see fit.
  filter((wrs_taken > 0.5 * wr_run_length) & overall_pick_number <= 120 &
           wr_run_length >= 8)

# Now, we can rejoin with the original data frame
bbm3 <- bbm3 %>% 
  left_join(bbm3_wr_runs_filtered, by = c("draft_id", "tournament_entry_id", "overall_pick_number"))

# Add a TRUE/FALSE column to indicate if a run took place.
bbm3 <- bbm3 %>%
  mutate(wr_run_occurred = if_else(!is.na(wr_run_length), TRUE, FALSE))

# Repeat with BBM2
# Create a new column ("wr_run") that's a indicator of whether the amount of WRs taken over expectation is at least 1 standard deviation from the mean.
bbm2_wr_runs <- bbm2 %>%
  arrange(draft_id, overall_pick_number) %>%
  group_by(draft_id) %>%
  mutate(wr_run_extended = WR_oe >= 1.0)

# Replace NA's
bbm2_wr_runs$wr_run_extended[is.na(bbm2_wr_runs$wr_run_extended)] <- FALSE

# Set a run ID for each draft. This will create a unique run identifier that we can later use for grouping.
bbm2_wr_runs <- bbm2_wr_runs %>%
  group_by(draft_id) %>%
  mutate(wr_run_id = cumsum(wr_run_extended & lag(wr_run_extended, default = FALSE) == FALSE)) %>%
  ungroup()

# Calculate the lengths of each run, and how many WRs were taken in each run
bbm2_wr_runs <- bbm2_wr_runs %>%
  filter(wr_run_extended) %>% #This will filter out all picks that do not have a WR_oe >= 1.0
  group_by(draft_id, wr_run_id) %>%
  mutate(wr_run_length = sum(wr_run_extended),
         wrs_taken = sum(position_name == "WR")) %>%
  ungroup()

# Mark the start and end of each run
bbm2_wr_runs <- bbm2_wr_runs %>%
  group_by(draft_id, wr_run_id) %>%
  mutate(wr_run_start = row_number() == 1,
         wr_run_end = row_number() == wr_run_length & wr_run_extended) %>%
  ungroup()

# Next, we are going to filter and pare down the wr_runs data frame to prepare it for rejoining with the bbm2 data frame
bbm2_wr_runs_filtered <- bbm2_wr_runs %>%
  select(draft_id, tournament_entry_id, overall_pick_number, wr_run_length, wrs_taken, wr_run_start, wr_run_end) %>%
  # This filter call sets our parameters for the run. These can be adjusted as we see fit.
  filter((wrs_taken > 0.5 * wr_run_length) & overall_pick_number <= 120 &
           wr_run_length >= 8)

# Now, we can rejoin with the original data frame
bbm2 <- bbm2 %>% 
  left_join(bbm2_wr_runs_filtered, by = c("draft_id", "tournament_entry_id", "overall_pick_number"))

# Add a TRUE/FALSE column to indicate if a run took place.
bbm2 <- bbm2 %>%
  mutate(wr_run_occurred = if_else(!is.na(wr_run_length), TRUE, FALSE))

# Data cleanup
rm(bbm3_wr_runs, bbm3_wr_runs_filtered,bbm2_wr_runs, bbm2_wr_runs_filtered)
```

**Quick Exploratory Data Analysis**

Let’s dig into the data.

``` r
# Create a table to show the sample size
bbm3_sample_size <- tibble("Category" = c("Drafts", "Teams", "Picks"),
       "Sample" = c(bbm3 %>%
                      # This counts each unique draft where a WR run took place
                      filter(!is.na(wr_run_length)) %>%
                      .$draft_id %>%
                      unique() %>% 
                      length(),
                    bbm3 %>%
                      # This counts each unique team that selected a player while a WR run took place
                      filter(!is.na(wr_run_length)) %>%
                      .$tournament_entry_id %>%
                      unique() %>% 
                      length(),
                    bbm3 %>%
                      # This grabs each unique pick that occurred during a WR run
                      filter(!is.na(wr_run_length)) %>%
                      .$tournament_entry_id %>% 
                      length()),
       # Totals for unique drafts, teams and picks for BBM3
       "Total" = c(37600, 451200, 8121600)) %>%
  mutate("Percent" = Sample/Total,
         "Tournament" = "BBM3")

# BBM2
# Create a table to show the sample size
bbm2_sample_size <- tibble("Category" = c("Drafts", "Teams", "Picks"),
       "Sample" = c(bbm2 %>%
                      # This counts each unique draft where a WR run took place
                      filter(!is.na(wr_run_length)) %>%
                      .$draft_id %>%
                      unique() %>% 
                      length(),
                    bbm2 %>%
                      # This counts each unique team that selected a player while a WR run took place
                      filter(!is.na(wr_run_length)) %>%
                      .$tournament_entry_id %>%
                      unique() %>% 
                      length(),
                    bbm2 %>%
                      # This grabs each unique pick that occurred during a WR run
                      filter(!is.na(wr_run_length)) %>%
                      .$tournament_entry_id %>% 
                      length()),
       # Totals for unique drafts, teams and picks for BBM3
       "Total" = c(12947, 155364, 2796552)) %>%
  mutate("Percent" = Sample/Total, 
         "Tournament" = "BBM2")

rbind(bbm2_sample_size, bbm3_sample_size) %>%
  group_by(Tournament) %>%
  gt(rowname_col = "Category") %>%
  tab_header(
    title = "Wide Receiver Run Sample Size",
  ) %>%
  tab_footnote(
    footnote = "Sample includes all drafts where a WR took place, all teams that made a draft pick during a WR run and all draft picks that occurred during a WR run.",
    locations = cells_column_labels(columns = c("Sample"))
  ) %>%
  fmt_number(
    columns = c(Sample, Total),
    decimals = 0
  ) %>%
  fmt_percent(
    columns = c(Percent),
    decimals = 2
  ) %>%
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  tab_options(
    table.width = 700,
    data_row.padding = px(2.5)
  ) %>%
  gt_theme_538()  %>%
  # Saved locally, uploaded to personal Github for ease of use. Applies for all tables.
  gtsave_extra(filename = "table1.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table1.png)<!-- -->

Across BBM2 and BBM3, roughly one out of every six drafts experienced a
wide receiver run in the first ten rounds. Additionally, we can see the
distribution of wide receiver over expectation values that the criteria
captured:

``` r
# Create a new data frame that grabs the WR_oe values from BBM2 and BBM3
wr_oe_distribution <- rbind(
  bbm3 %>%
    filter(!is.na(wr_run_length)) %>%
    select(WR_oe) %>%
    # Add an identifier for grouping
    mutate(tournament = "BBM3"),
  bbm2 %>%
    filter(!is.na(wr_run_length)) %>%
    select(WR_oe) %>%
    mutate(tournament = "BBM2")
  )

ggplot(data = wr_oe_distribution, aes(x = WR_oe)) +
  geom_histogram(color = "black", binwidth = 0.25) +
  facet_wrap(~tournament) +
  scale_x_continuous(breaks = seq(1,4,0.5), limits = c(1,4)) +
  theme_minimal() +
  labs(x = "Wide Receivers Taken Over Expectation (Scaled)",
       y = "Count",
       title = "Distribution of Wide Receivers Taken Over Expectation (Scaled) in WR Runs") +
  theme(panel.background = element_rect(fill = "white", color = "black"),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        plot.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "none",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"))
```

![](Surveying_The_Avalanche_files/figure-gfm/WR_oe%20Values-1.png)<!-- -->

``` r
ggsave("figure1.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/figure1.png)<!-- -->

The graph’s cutoff is set to 4.0–less than 0.5% of picks during a run
were more than four standard deviations above the mean. At the same
time, 33% of picks during a WR run were at least two or more standard
deviations above the mean, showing that the criteria are robust enough
to capture more extreme WR runs.

Let’s also check the distribution of WR run lengths.

``` r
wr_run_length_dist <- rbind(
  bbm3 %>%
    # To get unique WR run's, grab the wr_run_length for the last pick in a run
    filter(!is.na(wr_run_length), wr_run_end) %>%
    select(wr_run_length) %>%
    # Add an identifier for grouping
    mutate(tournament = "BBM3"),
  bbm2 %>%
    filter(!is.na(wr_run_length), wr_run_end) %>%
    select(wr_run_length) %>%
    # Add an identifier for grouping
    mutate(tournament = "BBM2"))

ggplot(data = wr_run_length_dist, aes(x = wr_run_length)) +
  geom_histogram(color = "black", binwidth = 1) +
  facet_wrap(~tournament, scales = c("free_y")) +
  scale_x_continuous(breaks = seq(8,50,2), limits = c(7,50)) +
  theme_minimal() +
  labs(x = "Wide Receiver Run Lengths",
       y = "Count",
       title = "Distribution of Wide Receivers Taken Over Expectation (Scaled) in WR Runs") +
  theme(panel.background = element_rect(fill = "white", color = "black"),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        plot.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "none",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"))
```

![](Surveying_The_Avalanche_files/figure-gfm/WR%20Run%20Lengths-1.png)<!-- -->

``` r
ggsave("figure2.png")
```

    ## Saving 9 x 4 in image

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/figure2.png)<!-- -->

**How does a wide receiver run affect ADP?**

We can check how a WR run affects ADP by looking at ADP value during the
run itself.

``` r
run_adp_table <- rbind(bbm3 %>% 
                         filter(!is.na(wr_run_length)) %>%
                         group_by(position_name) %>%
                         reframe(Raw_ADP_Value = mean(raw_adp_value),
                                 Percentage_ADP_Value = mean(pct_adp_value),
                                 Positional_Raw_ADP_Value = mean(raw_positional_adp_value),
                                 Positional_Percentage_ADP_Value = mean(pct_positional_adp_value)) %>%
                         mutate(tournament = "BBM3"),
                       bbm2 %>% 
                         filter(!is.na(wr_run_length)) %>%
                         group_by(position_name) %>%
                         reframe(Raw_ADP_Value = mean(raw_adp_value),
                                 Percentage_ADP_Value = mean(pct_adp_value),
                                 Positional_Raw_ADP_Value = mean(raw_positional_adp_value),
                                 Positional_Percentage_ADP_Value = mean(pct_positional_adp_value)) %>%
                         mutate(tournament = "BBM2"))

run_adp_table %>%
  group_by(tournament) %>%
  gt(rowname_col = "position_name") %>%
  # Headers
  tab_header(
    title = "Average ADP Value During a WR Run",
    subtitle = "Data from BBM2 and BBM3"
  ) %>%
  # Column Labels
  cols_label(
    Raw_ADP_Value = "Raw ADP Value",
    Percentage_ADP_Value = "Percentage ADP Value",
    Positional_Raw_ADP_Value = "Positional Raw ADP Value",
    Positional_Percentage_ADP_Value = "Positional Percentage ADP Value",
    tournament = "Tournament"
  ) %>%
  # Formatting
  fmt_number(
    columns = c(Raw_ADP_Value, Positional_Raw_ADP_Value),
    decimals = 2
  ) %>%
  fmt_percent(
    columns = c(Percentage_ADP_Value, Positional_Percentage_ADP_Value),
    decimals = 2
  ) %>%
  # Styling
  gt_hulk_col_numeric(
    columns = c(Raw_ADP_Value, Positional_Raw_ADP_Value, Percentage_ADP_Value, Positional_Percentage_ADP_Value),
    trim = TRUE
    ) %>%
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  tab_options(
    table.width = 700, 
    data_row.padding = px(2.5),
  ) %>%
  gt_theme_538() %>%
  gtsave_extra(filename = "table2.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table2.png)<!-- -->

On average, wide receivers are about four picks more expensive and are
taken at around a 6-7% premium during a WR run. Conversely, running
backs become about four picks cheaper and are taken at around a 9-11%
discount. Quarterbacks and tight ends also become cheaper, but not to
the same extent as running backs.

To help visualize this further, let’s plot every pick that occurred
during a WR run in BBM2 and BBM3 in relation to the overall pick number
and the positional percentage ADP value of each pick.

``` r
# Combine BBM2 and BBM3 into one data frame and pare it down
run_adp_viz_data <- rbind(bbm3 %>%
                            # Add Underdog Position Colors
                mutate(position_color = case_when(position_name == "WR" ~ "#e57e23",
                                          position_name == "RB" ~ "#16997e",
                                          position_name == "TE" ~ "#2980b9",
                                          position_name == "QB" ~ "#9747b8")) %>% filter(!is.na(wr_run_length)) %>%
                select(overall_pick_number, pct_positional_adp_value, pct_adp_value, position_name, position_color),
              bbm2 %>% 
                mutate(position_color = case_when(position_name == "WR" ~ "#e57e23",
                                          position_name == "RB" ~ "#16997e",
                                          position_name == "TE" ~ "#2980b9",
                                          position_name == "QB" ~ "#9747b8")) %>% filter(!is.na(wr_run_length)) %>%
                select(overall_pick_number, pct_positional_adp_value, pct_adp_value, position_name, position_color))

# Visualize ADP values during a run
ggplot(data = run_adp_viz_data %>% 
         # Factor is used here to order the positions in the correct way with their corresponding colors.
         mutate(position_name = factor(run_adp_viz_data$position_name, levels = c("WR", "RB", "TE", "QB")),
                position_color = factor(run_adp_viz_data$position_color, 
                                        levels = c("#e57e23","#16997e","#2980b9","#9747b8"))), aes(x = overall_pick_number, y = pct_positional_adp_value, color = position_color)) +
  geom_point(alpha = 1/30) +
  geom_hline(yintercept = 0, color = "black") +
  geom_hline(yintercept = -.05, color = "black", linetype = "dotted") +
  geom_hline(yintercept = .05, color = "black", linetype = "dotted") +
  annotate(geom = "text", x = 9, y = 0.08, label = "5% discount", size = 3) +
  annotate(geom = "text", x = 9, y = -0.08, label = "5% premium", size = 3) +
  scale_x_continuous(breaks = seq(0, 120, 12), 
                     limits = c(0, 120)) +
  scale_y_continuous(breaks = seq(-0.25, 0.25, .05), 
                     limits = c(-.3, .3), 
                     labels = c("-25%", "-20%", "-15%", "-10%", "-5%", "0%", "5%", "10%", "15%", "20%", "25%")) +
  facet_wrap(~position_name) +
  labs(title = "ADP Value of Draft Picks During a WR Run",
       subtitle = "WRs tend to be selected at a premium, while RB's, QB's and TE's are taken at a discount.",
       x = "Overall Pick Number", 
       y = "Percentage Positional ADP Value",
       caption = "@thefalkon | Data: Underdog BBM2 and BBM3") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "white", color = "black"),
        #panel.grid = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        plot.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "none",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"))
```

![](Surveying_The_Avalanche_files/figure-gfm/WR%20Run%20ADP%20Viz-1.png)<!-- -->

``` r
ggsave("figure3.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/figure3.png)<!-- -->

While individual players may go higher or lower than their positional
ADP, the chart illustrates just how inflated WR prices are and the
massive discounts available at RB, TE and QB.

We can also check how a WR run affects ADP by looking at ADP value in
the 12 picks following a WR run.

``` r
# Function to get the next 12 observations after a run has ended
get_following_picks <- function(df) {
  run_indices <- which(df$wr_run_end)
  following_indices <- purrr::map(run_indices, ~.x + 1:min(12, nrow(df) - .x))
  following_indices <- unlist(following_indices)
  return(df[following_indices, ])
}

# Apply the function to each draft
bbm3_followed_run <- bbm3 %>%
  group_by(draft_id) %>%
  group_split() %>%
  purrr::map_df(get_following_picks)

# Prepare the data frame for joining
bbm3_followed_run <- bbm3_followed_run %>%
  unique() %>%
  # Add a TRUE column to indicate that these picks followed a run
  mutate(followed_run = TRUE) %>%
  select(draft_id, tournament_entry_id, overall_pick_number, followed_run)

# Add the followed_run data to the original data frame.
bbm3 <- bbm3 %>% 
  left_join(bbm3_followed_run, by = c("draft_id", "tournament_entry_id", "overall_pick_number"))

# Repeat the process for BBM2
# Apply the function to each draft
bbm2_followed_run <- bbm2 %>%
  group_by(draft_id) %>%
  group_split() %>%
  purrr::map_df(get_following_picks)

# Prepare the data frame for joining
bbm2_followed_run <- bbm2_followed_run %>%
  unique() %>%
  # Add a TRUE column to indicate that these picks followed a run
  mutate(followed_run = TRUE) %>%
  select(draft_id, tournament_entry_id, overall_pick_number, followed_run)

# Add the followed_run data to the original data frame.
bbm2 <- bbm2 %>% 
  left_join(bbm2_followed_run, by = c("draft_id", "tournament_entry_id", "overall_pick_number"))

followed_run_adp_table <- rbind(bbm3 %>% 
                         filter(followed_run == T) %>%
                         group_by(position_name) %>%
                         reframe(Raw_ADP_Value = mean(raw_adp_value),
                                 Percentage_ADP_Value = mean(pct_adp_value),
                                 Positional_Raw_ADP_Value = mean(raw_positional_adp_value),
                                 Positional_Percentage_ADP_Value = mean(pct_positional_adp_value)) %>%
                         mutate(tournament = "BBM3"),
                       bbm2 %>% 
                         filter(followed_run == T) %>%
                         group_by(position_name) %>%
                         reframe(Raw_ADP_Value = mean(raw_adp_value),
                                 Percentage_ADP_Value = mean(pct_adp_value),
                                 Positional_Raw_ADP_Value = mean(raw_positional_adp_value),
                                 Positional_Percentage_ADP_Value = mean(pct_positional_adp_value)) %>%
                         mutate(tournament = "BBM2"))


followed_run_adp_table %>%
  group_by(tournament) %>%
  gt(rowname_col = "position_name") %>%
  # Headers
  tab_header(
    title = "Average ADP Value One Round After a WR Run",
    subtitle = "Data from BBM2 and BBM3"
  ) %>%
  # Column Labels
  cols_label(
    Raw_ADP_Value = "Raw ADP Value",
    Percentage_ADP_Value = "Percentage ADP Value",
    Positional_Raw_ADP_Value = "Positional Raw ADP Value",
    Positional_Percentage_ADP_Value = "Positional Percentage ADP Value",
    tournament = "Tournament"
  ) %>%
  # Formatting
  fmt_number(
    columns = c(Raw_ADP_Value, Positional_Raw_ADP_Value),
    decimals = 2
  ) %>%
  fmt_percent(
    columns = c(Percentage_ADP_Value, Positional_Percentage_ADP_Value),
    decimals = 2
  ) %>%
  # Styling
  gt_hulk_col_numeric(
    columns = c(Raw_ADP_Value, Positional_Raw_ADP_Value, Percentage_ADP_Value, Positional_Percentage_ADP_Value),
    trim = TRUE
  ) %>%
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  tab_options(
    table.width = 700, 
    data_row.padding = px(2.5),
  ) %>%
  gt_theme_538()  %>%
  gtsave_extra(filename = "table3.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table3.png)<!-- -->

One round after a run ends, positional ADPs start stabilizing.

**Strategies To Consider**

Drafters face a dilemma when a positional run occurs: should they grab a
position other than wide receiver at a value, or should they continue to
take wide receivers at a premium to avoid getting buried at the
position? In terms of regular season scoring and advancing, the answer
is fickle.

To analyze the effectiveness of different strategies during a WR run, we
will create a new column called “wr_run_participation”. Every time a WR
run is occurring, *and* a drafter selects a WR, that pick will be marked
as “participated”. Every time a WR run is occurring, *and* a drafter
selects a RB, QB or TE, that pick will be marked as
“did_not_participate”. We’ll then filter BBM2 and BBM3 data down to
teams that drafted while a WR run was occurring.

``` r
# Pull drafts where a WR run occurred
bbm3_wr_run_draft_ids <- bbm3 %>%
  filter(!is.na(wr_run_length)) %>%
  pull(draft_id) %>%
  unique()

# Create a new data frame that ONLY contains drafts where a run occurred.
bbm3_wr_run_drafts <- bbm3 %>%
  filter(draft_id %in% bbm3_wr_run_draft_ids)

# Identify teams that participated in a positional run AND took a Wide Receiver
bbm3_wr_run_drafts <- bbm3_wr_run_drafts %>%
  mutate(wr_run_participation = case_when(
    wr_run_occurred == TRUE & position_name == "WR" ~ "participated",
    wr_run_occurred == TRUE & position_name != "WR" ~ "did_not_participate",
    TRUE ~ NA_character_ #If we get to the end of the code and nothing happened, mark as NA
  )) 

# Create a new data frame that contains advance rate and regular season scoring data.
bbm3_team_success <- bbm3_wr_run_drafts %>%
  group_by(tournament_entry_id) %>%
  reframe(roster_points, playoff_team, position_name, wr_run_participation, overall_pick_number) %>%
  unique()

# Grab the team id's of teams that drafted during in a WR run
bbm3_run_teams <- bbm3_wr_run_drafts %>%
  filter(wr_run_occurred == TRUE) %>%
  pull(tournament_entry_id) %>%
  unique()

# Filter the team_success by the teams who participated in a WR run.
bbm3_run_team_success <- bbm3_team_success %>%
  filter(tournament_entry_id %in% bbm3_run_teams)

### Repeat the Process for BBM2
# Pull drafts where a WR run occurred
bbm2_wr_run_draft_ids <- bbm2 %>%
  filter(!is.na(wr_run_length)) %>%
  pull(draft_id) %>%
  unique()

# Create a new data frame that ONLY contains drafts where a run occurred.
bbm2_wr_run_drafts <- bbm2 %>%
  filter(draft_id %in% bbm2_wr_run_draft_ids)

# Identify teams that participated in a positional run AND took a Wide Receiver
bbm2_wr_run_drafts <- bbm2_wr_run_drafts %>%
  mutate(wr_run_participation = case_when(
    wr_run_occurred == TRUE & position_name == "WR" ~ "participated",
    wr_run_occurred == TRUE & position_name != "WR" ~ "did_not_participate",
    TRUE ~ NA_character_ #If we get to the end of the code and nothing happened, mark as NA
  )) 

# Create a new data frame that contains advance rate and regular season scoring data.
bbm2_team_success <- bbm2_wr_run_drafts %>%
  group_by(tournament_entry_id) %>%
  reframe(roster_points, playoff_team, position_name, wr_run_participation, overall_pick_number) %>%
  unique()

# Grab the team id's of teams that drafted during in a WR run
bbm2_run_teams <- bbm2_wr_run_drafts %>%
  filter(wr_run_occurred == TRUE) %>%
  pull(tournament_entry_id) %>%
  unique()

# Filter the team_success by the teams who participated in a WR run.
bbm2_run_team_success <- bbm2_team_success %>%
  filter(tournament_entry_id %in% bbm2_run_teams)
```

We now have our data frames filtered down to teams that drafted during a
WR run and whether those draft picks went with the run (WR) or cut
against the grain (RB, QB, TE). We will now examine whether following
the run or taking a detour was more effective.

``` r
bbm3_run_performance <- bbm3_run_team_success %>%
  # Drop NA's to grab only the picks that happened during a run.
  drop_na() %>% 
    # Reshape the data frame so that each time a team participated or did not participate in a run is counted. Overall_Pick_Number is included to get each unique pick
  count(tournament_entry_id, wr_run_participation, roster_points, playoff_team, position_name, overall_pick_number) %>%
  pivot_wider(names_from = wr_run_participation, values_from = n, values_fill = 0) %>%
  select(-c(overall_pick_number)) %>%
  # Now, we are going to reshape the data frame again, this time separating each time a detour was taken by position. Then, we are going to sum the amount of times a team either participated and took a WR, or took a detour and drafted another position.
  group_by(tournament_entry_id, position_name) %>%
  reframe(roster_points, playoff_team, position_name,
          participated = sum(participated), did_not_participate = sum(did_not_participate)) %>%
  # Now, we are free to only keep the unique picks by each position. We are going to create new columns to mark how many times a drafter took a detour and drafted another position.
  distinct(tournament_entry_id, position_name, .keep_all = TRUE) %>%
  pivot_wider(names_from = position_name, names_prefix = "detour", values_from = did_not_participate, 
              values_fill = 0) %>%
  group_by(tournament_entry_id) %>%
  reframe(roster_points, playoff_team,
          participated = sum(participated), 
          detourRB = sum(detourRB),
          detourTE = sum(detourTE),
          detourQB = sum(detourQB)) %>%
  # Now, we can keep the unique teams
  distinct(tournament_entry_id, .keep_all = TRUE) 

bbm3_wrs_during_run_adv <- bbm3_run_performance %>%
  group_by(participated) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = participated)

bbm3_wrs_during_run_pts <- bbm3_run_performance %>%
  group_by(participated) %>%
  reframe(sample_size = n(),
         roster_points = mean(roster_points)) %>%
  rename("position_taken" = participated)

bbm3_participated_success <- left_join(bbm3_wrs_during_run_adv, bbm3_wrs_during_run_pts, 
          by = "position_taken") %>%
  select(-advancing_teams) %>%
  relocate(sample_size, .after = position_taken) %>%
  mutate(position = "WR", strategy = "Participated")

# Separate detours by position
bbm3_rbs_during_run_adv <- bbm3_run_performance %>%
  group_by(detourRB) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = detourRB) %>%
  mutate(position = "RB", strategy = "Did Not Participate")

bbm3_rbs_during_run_pts <- bbm3_run_performance %>%
  group_by(detourRB) %>%
  reframe(sample_size = n(),
         roster_points = mean(roster_points)) %>%
  rename("position_taken" = detourRB) %>%
  mutate(position = "RB", strategy = "Did Not Participate")

bbm3_qbs_during_run_adv <- bbm3_run_performance %>%
  group_by(detourQB) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = detourQB) %>%
  mutate(position = "QB", strategy = "Did Not Participate")

bbm3_qbs_during_run_pts <- bbm3_run_performance %>%
  group_by(detourQB) %>%
  reframe(sample_size = n(),
         roster_points = mean(roster_points)) %>%
  rename("position_taken" = detourQB) %>%
  mutate(position = "QB", strategy = "Did Not Participate")

bbm3_tes_during_run_adv <- bbm3_run_performance %>%
  group_by(detourTE) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = detourTE) %>%
  mutate(position = "TE", strategy = "Did Not Participate")

bbm3_tes_during_run_pts <- bbm3_run_performance %>%
  group_by(detourTE) %>%
  reframe(sample_size = n(),
         roster_points = mean(roster_points)) %>%
  rename("position_taken" = detourTE) %>%
  mutate(position = "TE", strategy = "Did Not Participate")

bbm3_rb_detour_success <- bbm3_rbs_during_run_adv %>% 
  left_join(bbm3_rbs_during_run_pts %>%
              select(position_taken, sample_size, roster_points), by = "position_taken")

bbm3_qb_detour_success <- bbm3_qbs_during_run_adv %>% 
  left_join(bbm3_qbs_during_run_pts %>%
              select(position_taken, sample_size, roster_points), by = "position_taken")

bbm3_te_detour_success <- bbm3_tes_during_run_adv %>% 
  left_join(bbm3_tes_during_run_pts %>%
              select(position_taken, sample_size, roster_points), by = "position_taken")

# Combine all of the detour data frames into one
bbm3_detour_success <- rbind(bbm3_rb_detour_success, bbm3_qb_detour_success, bbm3_te_detour_success) %>%
  select(-advancing_teams) %>%
  relocate(sample_size, .after = position_taken) %>%
  relocate(roster_points, .after = advance_rate)

bbm3_participated_success %>% 
  rbind(bbm3_detour_success) %>%
  ungroup() %>%
  select(-strategy) %>%
  mutate(adv_rate_oe = advance_rate - 2/12,
         roster_pts_oe = roster_points - 1520.915) %>%
  gt(groupname_col = "position") %>%
  cols_move(., columns = adv_rate_oe, after = advance_rate) %>%
  #Header
  tab_header(title = "Analyzing WR Run Strategy Effectiveness", 
             subtitle = "Data from BBM3") %>%
  # Column Labels
  cols_label(position_taken = "Position Picks",
             sample_size = "Sample Size",
             advance_rate = "Advance Rate",
             adv_rate_oe = "Adv. Rate Over Expected",
             roster_points = "Avg. Roster Points",
             roster_pts_oe = "Avg. Roster Points Over Expected") %>%
  tab_footnote(
    footnote = "Advance Rate over Expected = Advance Rate - Expected Advance Rate (16.667)",
    locations = cells_column_labels(columns = c("adv_rate_oe"))
  ) %>%
  tab_footnote(
    footnote = "Avg. Roster Points over Expected = Avg. Roster Points - Base Avg. Roster Points (1520.92)",
    locations = cells_column_labels(columns = c("roster_pts_oe"))
  ) %>%
  #Formatting
  fmt_number(columns = c(sample_size),
             decimals = 0) %>%
  fmt_number(columns = c(roster_points, roster_pts_oe),
             decimals = 2) %>%
  fmt_percent(columns = c(advance_rate, adv_rate_oe),
              decimals = 2) %>%
  # Styling
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  tab_options(
    table.width = 700, 
    data_row.padding = px(2.5),
  ) %>%
  gt_highlight_rows(
    rows = c(1, 8, 9, 12),
    fill = "#bbebb7"
  ) %>%
  gt_theme_538()  %>%
  gtsave_extra(filename = "table4.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table4.png)<!-- -->

At first glance, selecting a RB or QB during a WR run looks optimal in
BBM3, yielding a 1% or greater boost to advance rate and about a 3-11
point boost to how many points a team scored during the regular season.

Let’s see if that was the case in BBM2.

``` r
bbm2_run_performance <- bbm2_run_team_success %>%
  # Drop NA's to grab only the picks that happened during a run.
  drop_na() %>% 
  # Reshape the data frame so that each time a team participated or did not participate in a run is counted. Overall_Pick_Number is included to get each unique pick
  count(tournament_entry_id, wr_run_participation, roster_points, playoff_team, position_name, overall_pick_number) %>%
  pivot_wider(names_from = wr_run_participation, values_from = n, values_fill = 0) %>%
  select(-c(overall_pick_number)) %>%
  # Now, we are going to reshape the data frame again, this time separating each time a detour was taken by position. Then, we are going to sum the amount of times a team either participated and took a WR, or took a detour and drafted another position.
  group_by(tournament_entry_id, position_name) %>%
  reframe(roster_points, playoff_team, position_name,
          participated = sum(participated), did_not_participate = sum(did_not_participate)) %>%
  # Now, we are free to only keep the unique picks by each position. We are going to create new columns to mark how many times a drafter took a detour and drafted another position.
  distinct(tournament_entry_id, position_name, .keep_all = TRUE) %>%
  pivot_wider(names_from = position_name, names_prefix = "detour", values_from = did_not_participate, 
              values_fill = 0) %>%
  group_by(tournament_entry_id) %>%
  reframe(roster_points, playoff_team,
          participated = sum(participated), 
          detourRB = sum(detourRB),
          detourTE = sum(detourTE),
          detourQB = sum(detourQB)) %>%
  # Now, we can keep the unique teams
  distinct(tournament_entry_id, .keep_all = TRUE) 

bbm2_wrs_during_run_adv <- bbm2_run_performance %>%
  group_by(participated) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = participated)

bbm2_wrs_during_run_pts <- bbm2_run_performance %>%
  group_by(participated) %>%
  reframe(sample_size = n(),
          roster_points = mean(roster_points)) %>%
  rename("position_taken" = participated)

bbm2_participated_success <- left_join(bbm2_wrs_during_run_adv, bbm2_wrs_during_run_pts, 
                                       by = "position_taken") %>%
  select(-advancing_teams) %>%
  relocate(sample_size, .after = position_taken) %>%
  mutate(position = "WR", strategy = "Participated")

# Separate detours by position
bbm2_rbs_during_run_adv <- bbm2_run_performance %>%
  group_by(detourRB) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = detourRB) %>%
  mutate(position = "RB", strategy = "Did Not Participate")

bbm2_rbs_during_run_pts <- bbm2_run_performance %>%
  group_by(detourRB) %>%
  reframe(sample_size = n(),
          roster_points = mean(roster_points)) %>%
  rename("position_taken" = detourRB) %>%
  mutate(position = "RB", strategy = "Did Not Participate")

bbm2_qbs_during_run_adv <- bbm2_run_performance %>%
  group_by(detourQB) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = detourQB) %>%
  mutate(position = "QB", strategy = "Did Not Participate")

bbm2_qbs_during_run_pts <- bbm2_run_performance %>%
  group_by(detourQB) %>%
  reframe(sample_size = n(),
          roster_points = mean(roster_points)) %>%
  rename("position_taken" = detourQB) %>%
  mutate(position = "QB", strategy = "Did Not Participate")

bbm2_tes_during_run_adv <- bbm2_run_performance %>%
  group_by(detourTE) %>%
  count(playoff_team) %>%
  mutate(advance_rate = n / sum(n)) %>%
  filter(playoff_team == TRUE) %>%
  select(-playoff_team) %>%
  rename("advancing_teams" = n, "position_taken" = detourTE) %>%
  mutate(position = "TE", strategy = "Did Not Participate")

bbm2_tes_during_run_pts <- bbm2_run_performance %>%
  group_by(detourTE) %>%
  reframe(sample_size = n(),
          roster_points = mean(roster_points)) %>%
  rename("position_taken" = detourTE) %>%
  mutate(position = "TE", strategy = "Did Not Participate")

bbm2_rb_detour_success <- bbm2_rbs_during_run_adv %>% 
  left_join(bbm2_rbs_during_run_pts %>%
              select(position_taken, sample_size, roster_points), by = "position_taken")

bbm2_qb_detour_success <- bbm2_qbs_during_run_adv %>% 
  left_join(bbm2_qbs_during_run_pts %>%
              select(position_taken, sample_size, roster_points), by = "position_taken")

bbm2_te_detour_success <- bbm2_tes_during_run_adv %>% 
  left_join(bbm2_tes_during_run_pts %>%
              select(position_taken, sample_size, roster_points), by = "position_taken")

# Combine all of the detour data frames into one
bbm2_detour_success <- rbind(bbm2_rb_detour_success, bbm2_qb_detour_success, bbm2_te_detour_success) %>%
  select(-advancing_teams) %>%
  relocate(sample_size, .after = position_taken) %>%
  relocate(roster_points, .after = advance_rate)

bbm2_participated_success %>% 
  rbind(bbm2_detour_success) %>%
  ungroup() %>%
  select(-strategy) %>%
  mutate(adv_rate_oe = advance_rate - 2/12,
         roster_pts_oe = roster_points - 1534.633) %>%
  gt(groupname_col = "position") %>%
  cols_move(., columns = adv_rate_oe, after = advance_rate) %>%
  #Header
  tab_header(title = "Analyzing WR Run Strategy Effectiveness", 
             subtitle = "Data from BBM2") %>%
  # Column Labels
  cols_label(position_taken = "Position Picks",
             sample_size = "Sample Size",
             advance_rate = "Advance Rate",
             adv_rate_oe = "Adv. Rate Over Expected",
             roster_points = "Avg. Roster Points",
             roster_pts_oe = "Avg. Roster Points Over Expected") %>%
  tab_footnote(
    footnote = "Advance Rate over Expected = Advance Rate - Expected Advance Rate (16.667)",
    locations = cells_column_labels(columns = c("adv_rate_oe"))
  ) %>%
  tab_footnote(
    footnote = "Avg. Roster Points over Expected = Avg. Roster Points - Base Avg. Roster Points (1534.63)",
    locations = cells_column_labels(columns = c("roster_pts_oe"))
  ) %>%
  #Formatting
  fmt_number(columns = c(sample_size),
             decimals = 0) %>%
  fmt_number(columns = c(roster_points, roster_pts_oe),
             decimals = 2) %>%
  fmt_percent(columns = c(advance_rate, adv_rate_oe),
              decimals = 2) %>%
  # Styling
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  tab_options(
    table.width = 700, 
    data_row.padding = px(2.5),
  ) %>%
  gt_highlight_rows(
    rows = c(3, 7, 12, 15),
    fill = "#bbebb7"
  ) %>%
  gt_theme_538() %>%
  gtsave_extra(filename = "table5.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table5.png)<!-- -->

Nearly the opposite was true: Selecting WRs during a WR run was
necessary to get to base advance rates, and taking more than one yielded
a 1% boost to advance rates. At the same time, drafting a RB during a WR
run decreased a team’s advance rate by about 1.8% and reduced a team’s
end-of-year points by about 12 on average. Alternatively, drafters who
opted to take a QB or a TE received similar benefits to drafters who
took WRs.

The results make more sense when we dig into *which* players were
selected during a wide receiver run.

``` r
# Grab the top 20 most frequently drafted WRs during a WR run
bbm3_participated_players <- bbm3_wr_run_drafts %>%
  filter(wr_run_participation == "participated") %>%
  group_by(player_name) %>%
  reframe(position_name,
          count = n(),
          adv_rate = sum(playoff_team)/n()) %>%
  arrange(-count) %>%
  unique() %>% 
  head(20) %>%
  mutate(strategy = "Participated")

# Grab the top 20 most frequently drafted RB's, QB's and TE's during a WR run
bbm3_detour_players <- bbm3_wr_run_drafts %>%
  filter(wr_run_participation == "did_not_participate") %>%
  group_by(player_name) %>%
  reframe(position_name,
          count = n(),
          adv_rate = sum(playoff_team)/n()) %>%
  arrange(-count) %>%
  unique() %>%
  head(20) %>%
  mutate(strategy = "Did Not Participate")

# Combine the two and plot
rbind(bbm3_participated_players, bbm3_detour_players) %>%
  mutate(adv_rate = adv_rate) %>%
  ggplot(data = ., aes(x = player_name, y = adv_rate)) +
  geom_hline(yintercept = 2/12, linetype = "dotted") +
  geom_label(aes(label = paste0(round(adv_rate, 3) * 100, "%")),
             size = 2.5, nudge_y = 0.013, label.size = 0, label.padding = unit(0.01, "lines")) +
  geom_col(aes(fill = adv_rate > 2/12), color = "black", position= "dodge", width = 0.5) +
  facet_wrap(~strategy, scales = "free_x") +
  labs(x = "Player Name",
       y = "Advance Rate",
       fill = "Advance rate greater than 16.67%?",
       title = "Comparing Advance Rates for Players Taken During a WR Run",
       subtitle = "Plot of the top 20 players most frequently taken, split by strategy",
       caption = "@thefalkon | Data: BBM3") + 
  scale_y_continuous(breaks = seq(0,0.5,0.1),
                     labels = c("0%", "10%", "20%", "30%", "40%", "50%")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.ticks = element_line(color = "black"),
        panel.background = element_rect(fill = "white", color = "black"),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        plot.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "top",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"))
```

![](Surveying_The_Avalanche_files/figure-gfm/Specific%20Players%20BBM3-1.png)<!-- -->

``` r
ggsave("figure4.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/figure4.png)<!-- -->

The reason why taking a detour and selecting a RB or QB was optimal in
2022 is simple: player’s like Jalen Hurts, Josh Jacobs and Patrick
Mahomes provided meteoric advance rates, and other players were, by and
large, not backbreaking picks. On the other hand, many wide receivers
underperformed their ADP, and when WRs like Amari Cooper, Amon-Ra
St. Brown and DK Metcalf *did* hit, it wasn’t to the same extent as the
previous group.

BBM2 shows the opposite effect:

``` r
bbm2_participated_players <- bbm2_wr_run_drafts %>%
  filter(wr_run_participation == "participated") %>%
  group_by(player_name) %>%
  reframe(position_name,
          count = n(),
          adv_rate = sum(playoff_team)/n()) %>%
  arrange(-count) %>%
  unique() %>% 
  head(20) %>%
  mutate(strategy = "Participated")

bbm2_detour_players <- bbm2_wr_run_drafts %>%
  filter(wr_run_participation == "did_not_participate") %>%
  group_by(player_name) %>%
  reframe(position_name,
          count = n(),
          adv_rate = sum(playoff_team)/n()) %>%
  arrange(-count) %>%
  unique() %>%
  head(20) %>%
  mutate(strategy = "Did Not Participate")

rbind(bbm2_participated_players, bbm2_detour_players) %>%
  mutate(adv_rate = adv_rate) %>%
  ggplot(data = ., aes(x = player_name, y = adv_rate)) +
  geom_hline(yintercept = 2/12, linetype = "dotted") +
  geom_label(aes(label = paste0(round(adv_rate, 3) * 100, "%")),
             size = 2.5, nudge_y = 0.013, label.size = 0, label.padding = unit(0.01, "lines")) +
  geom_col(aes(fill = adv_rate > 2/12), color = "black", position = "dodge", width = 0.5) +
  facet_wrap(~strategy, scales = "free_x") +
  labs(x = "Player Name",
       y = "Advance Rate",
       fill = "Advance rate greater than 16.67%?",
       title = "Comparing Advance Rates for Players Taken During a WR Run",
       subtitle = "Plot of the top 20 players most frequently taken, split by strategy",
       caption = "@thefalkon | Data: BBM2") + 
  scale_y_continuous(breaks = seq(0,0.5,0.1),
                     labels = c("0%", "10%", "20%", "30%", "40%", "50%")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.ticks = element_line(color = "black"),
        panel.background = element_rect(fill = "white", color = "black"),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        plot.background = element_rect(fill = "white", color = NA),
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "top",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"))
```

![](Surveying_The_Avalanche_files/figure-gfm/Specific%20Players%20BBM2-1.png)<!-- -->

``` r
ggsave("figure5.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/figure5.png)<!-- -->

In 2021, WRs like Cooper Kupp, Deebo Samuel and Ja’Marr Chase were
incredible hits, while RBs like Raheem Mostert, Travis Etienne and Mike
Davis were landmines.

The effectiveness of each choice–whether to go with the run or step to
the side and take a detour–was *heavily* influenced by who the best
plays were that season.

The next set of strategies to consider is structural rather than
player-based. Instead of looking at drafts where WR runs occurred, we
can sort drafts by the highest average wide receiver over expectation
and pull the top 5% of all those drafts. Previously, Hayden Winks has
shown that drafting four WRs through seven rounds is the [golden rule of
best
ball](https://underblog.underdogfantasy.com/the-golden-rule-of-best-ball-8a7fc0df3983),
and [getting to around five WRs through 10 rounds performs well during
the regular
season](https://underdognetwork.com/football/best-ball-research/when-to-draft-wrs-in-best-ball-updated).
Pat Kerrane has also illustrated how [most WR spike weeks come within
the first nine
rounds](https://www.nbcsports.com/fantasy/football/news/article-best-ball-strategy-how-build-rb-wr-and-te-playoffs).

Does a WR-heavy room change the calculus of how many WRs to take and how
early to take them? Let’s look at BBM3 first.

``` r
# Grab the most WR heavy drafts through nine rounds
bbm3_wr_heavy_draft_ids <- bbm3 %>%
  filter(overall_pick_number <= 108) %>%
  group_by(draft_id) %>%
  reframe(avg_wr_pick = mean(WR_oe),
          avg_wr_adp_val = mean(pct_adp_value)) %>%
  arrange(-avg_wr_pick) %>%
  mutate(pct = rank(-avg_wr_pick)) %>%
  # 1880 = 37,600 * 0.05, to get the top 5% of drafts
  filter(pct <= 1880) %>%
  pull(draft_id)

# First, get the base rates of WR strategies through 7 and 9 rounds
bbm3_base_through7 <- bbm3 %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 7) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))

bbm3_base_through9 <- bbm3 %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 9) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))

# Then, grab the advance rates of WR strategies through 7 and 9 rounds in WR heavy Rooms
bbm3_wr_heavy_through7 <- bbm3 %>%
  filter(draft_id %in% bbm3_wr_heavy_draft_ids) %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 7) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))

bbm3_wr_heavy_through9 <- bbm3 %>%
  filter(draft_id %in% bbm3_wr_heavy_draft_ids) %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 9) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))

# Join the two data frams for plotting
bbm3_wr_heavy_through7 <- bbm3_wr_heavy_through7 %>% left_join(bbm3_base_through7, by = "total_wrs", suffix = c("", "_base")) %>%
  mutate(diff_adv_rate = adv_rate - adv_rate_base,
         diff_pts = pts - pts_base,
         rounds = "Through 7 Rounds") %>%
  select(-c(adv_rate_base, pts_base, n_base, n))

# Join the two data frams for plotting
bbm3_wr_heavy_through9 <- bbm3_wr_heavy_through9 %>% left_join(bbm3_base_through9, by = "total_wrs", suffix = c("", "_base")) %>%
  mutate(diff_adv_rate = adv_rate - adv_rate_base,
         diff_pts = pts - pts_base, 
         rounds = "Through 9 Rounds") %>%
  select(-c(adv_rate_base, pts_base, n_base, n))

rbind(bbm3_wr_heavy_through7, bbm3_wr_heavy_through9) %>%
  select(-c(utilization_base)) %>%
  group_by(rounds) %>%
  gt() %>%
  tab_header(title = "Analyzing WR Run Strategy Effectiveness", 
             subtitle = "Data from BBM3") %>%
  # Column Labels
  cols_label(
    total_wrs = "Total WRs",
    adv_rate = "Advance Rate",
    pts = "Regular Season Roster Points",
    utilization = "Utilization",
    diff_adv_rate = "Advance Rate Difference vs. Base",
    diff_pts = "Regular Season Roster Points Difference vs. Base"
  ) %>%
  #Formatting
  fmt_number(columns = c(pts, diff_pts),
             decimals = 2) %>%
  fmt_percent(columns = c(adv_rate, utilization, diff_adv_rate),
              decimals = 2) %>%
  # Footnotes
  tab_footnote(
    footnote = "Utilization is how often drafters had X wide receivers through seven or ten rounds.",
    locations = cells_column_labels(columns = c("utilization"))
  ) %>%
  tab_footnote(
    footnote = "Advance Rate Difference vs. Base = Advance Rate of each strategy in WR drafts - Advance Rate of each strategy in all drafts.",
    locations = cells_column_labels(columns = c("diff_adv_rate"))
  ) %>%
  # Styling
  gt_highlight_rows(
    rows = c(5, 6, 13, 14),
    fill = "#bbebb7"
  ) %>%
  gt_highlight_rows(
    rows = c(4, 12, 15),
    fill = "#e7ffe6", 
    font_weight = "normal"
  ) %>%
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  cols_move(columns = "diff_adv_rate", after = "adv_rate") %>%
  cols_move(columns = "diff_pts", after = "pts") %>%
  cols_move(columns = "utilization", after = "total_wrs") %>%
  tab_options(
    table.width = 700, 
    data_row.padding = px(2.5),
  ) %>%
  gt_theme_538() %>%
  gtsave_extra(filename = "table6.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table6.png)<!-- -->

Before drawing conclusions, let’s also look at BBM2.

``` r
bbm2_wr_heavy_draft_ids <- bbm2 %>%
  filter(overall_pick_number <= 108) %>%
  group_by(draft_id) %>%
  reframe(avg_wr_pick = mean(WR_oe),
          avg_wr_adp_val = mean(pct_adp_value)) %>%
  arrange(-avg_wr_pick) %>%
  mutate(pct = rank(-avg_wr_pick)) %>%
  filter(pct <= 648) %>%
  pull(draft_id)

bbm2_base_through7 <- bbm2 %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 7) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))

bbm2_base_through9 <- bbm2 %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 9) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))


bbm2_wr_heavy_through7 <- bbm2 %>%
  filter(draft_id %in% bbm2_wr_heavy_draft_ids) %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 7) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))

bbm2_wr_heavy_through9 <- bbm2 %>%
  filter(draft_id %in% bbm2_wr_heavy_draft_ids) %>%
  arrange(tournament_entry_id, overall_pick_number) %>%
  group_by(tournament_entry_id) %>%
  mutate(total_wrs = cumsum(picked_WR)) %>%
  filter(team_pick_number == 9) %>%
  group_by(total_wrs) %>%
  reframe(n = n(),
          adv_rate = sum(playoff_team)/n(),
          pts = mean(roster_points)) %>%
  mutate(utilization = n/sum(n))

bbm2_wr_heavy_through7 <- bbm2_wr_heavy_through7 %>% left_join(bbm2_base_through7, by = "total_wrs", suffix = c("", "_base")) %>%
  mutate(diff_adv_rate = adv_rate - adv_rate_base,
         diff_pts = pts - pts_base,
         rounds = "Through 7 Rounds") %>%
  select(-c(adv_rate_base, pts_base, n_base, n))

bbm2_wr_heavy_through9 <- bbm2_wr_heavy_through9 %>% left_join(bbm2_base_through9, by = "total_wrs", suffix = c("", "_base")) %>%
  mutate(diff_adv_rate = adv_rate - adv_rate_base,
         diff_pts = pts - pts_base, 
         rounds = "Through 9 Rounds") %>%
  select(-c(adv_rate_base, pts_base, n_base, n))

rbind(bbm2_wr_heavy_through7, bbm2_wr_heavy_through9) %>%
  select(-c(utilization_base)) %>%
  group_by(rounds) %>%
  gt() %>%
  tab_header(title = "Analyzing WR Run Strategy Effectiveness", 
             subtitle = "Data from BBM2") %>%
  # Column Labels
  cols_label(
    total_wrs = "Total WRs",
    adv_rate = "Advance Rate",
    pts = "Regular Season Roster Points",
    utilization = "Utilization",
    diff_adv_rate = "Advance Rate Difference vs. Base",
    diff_pts = "Regular Season Roster Points Difference vs. Base"
  ) %>%
  #Formatting
  fmt_number(columns = c(pts, diff_pts),
             decimals = 2) %>%
  fmt_percent(columns = c(adv_rate, utilization, diff_adv_rate),
              decimals = 2) %>%
  # Footnotes
  tab_footnote(
    footnote = "Utilization is how often drafters had X wide receivers through seven or ten rounds.",
    locations = cells_column_labels(columns = c("utilization"))
  ) %>%
  tab_footnote(
    footnote = "Advance Rate Difference vs. Base = Advance Rate of each strategy in WR drafts - Advance Rate of each strategy in all drafts.",
    locations = cells_column_labels(columns = c("diff_adv_rate"))
  ) %>%
  # Styling
  gt_highlight_rows(
    rows = c(5, 6, 13, 14),
    fill = "#bbebb7"
  ) %>%
  cols_move(columns = "diff_adv_rate", after = "adv_rate") %>%
  cols_move(columns = "diff_pts", after = "pts") %>%
  cols_move(columns = "utilization", after = "total_wrs") %>%
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  tab_options(
    table.width = 700, 
    data_row.padding = px(2.5),
  ) %>%
  gt_theme_538() %>%
  gtsave_extra(filename = "table7.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table7.png)<!-- -->

The following conclusions should be taken with a grain of salt, as the
sample size is limited to 5% of drafts from BBM2 and BBM3. While
drafting either four or five WRs by Round 7 and Round 9 underperformed
how a team scored during the regular season, both strategies appear to
remain optimal in WR-heavy rooms. Notably, drafting more WRs and
drafting them earlier [seemed optimal for regular season scoring in
BBM3](https://underdognetwork.com/_next/image?url=https%3A%2F%2Fimages.ctfassets.net%2Fl8eoog2wur2i%2F6UwDhRFtH21lpnaHKojFrV%2Fca6915069d920fc53bd1c444cd51466c%2FBBM3WROptimal.png&w=3840&q=75),
but that held less firmly in WR-heavy rooms.

**Conclusions**

Before diving into takeaways, it’s worth noting that regular season
scoring and advance rates shouldn’t be the only lens to quantify whether
a particular strategy was effective or ineffective. 67% of BBM4’s prize
pool is dedicated to playoff prizes, and 20% of the total prize pool is
awarded to finishing first. Constructing teams to have the requisite
weekly upside to take down an uncorrelated three-week tournament is as
important–if not more–than advancing out of a 12-person pod. Research on
how inflated WR ADPs impact a team’s weekly upside is an important next
step.

The present article’s data offer no definitive answer on the best
approach to handling WR runs. BBM2 and BBM3 illustrated that different
approaches can work. When there’s a WR run, and the WRs in that range
hit, it’s still important to draft a WR. Other times, when there’s a WR
run, and the WRs in that range are weak and different positions are
stronger, it’s important to draft those positions at a value.

Constructing teams well–drafting four or five WRs by Round 7 and 9–still
matters. It’s easy to say *play the best plays and move on*, but each
drafter should assess different positional tiers and evaluate whether
one position still provides an edge over another. There’s no sense in
grabbing a falling Mike Davis if the opportunity cost is a WR with more
upside.

The data may offer clues, and BBM3 provides a word of caution about
*only* grabbing WR’s when they are flying off the board. The ADP
landscape has changed in BBM4 with inflated WR prices. By just how much?
It’s an imperfect exercise, but we can take BBM4’s current ADP, treat it
as if a draft strictly followed ADP, and then join it with the closing
ADP from BBM3. We can apply the same methodology from before the gauge
how WR-heavy BBM4 is:

``` r
# Read in BBM4 ADP from July 30th
bbm4 <- read_csv("bbm4_073023.csv") %>%
  select(adp, slotName) %>%
  mutate(draft_id = "BBM4") %>%
  rename("position_name" = slotName, "overall_pick_number" = adp)
```

    ## Rows: 1571 Columns: 10
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (7): id, firstName, lastName, adp, positionRank, slotName, teamName
    ## dbl (1): projectedPoints
    ## lgl (2): lineupStatus, byeWeek
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# Treat the BBM4 ADP as if it was a draft that strictly followed ADP, and then repeat the methodology from earlier
bbm4 <- bbm4 %>%
  head(216) %>%
  mutate(overall_pick_number = 1:216) %>%
  mutate(picked_RB = apply(., 1, function(x) length(which(x=="RB"))),
         picked_WR = apply(., 1, function(x) length(which(x=="WR"))),
         picked_QB = apply(., 1, function(x) length(which(x=="QB"))),
         picked_TE = apply(., 1, function(x) length(which(x=="TE")))) %>%
  mutate(RB_taken = cumsum(picked_RB),
         WR_taken = cumsum(picked_WR), 
         QB_taken = cumsum(picked_QB), 
         TE_taken = cumsum(picked_TE))

# Grab the last draft from BBM3
last_draft_id <- bbm3 %>%
  arrange(draft_time) %>%
  tail(n = 1) %>%
  .$draft_id

last_draft <- bbm3 %>%
  filter(draft_id == last_draft_id)

# Join the BBM4 data frame to BBM3, as if it was a draft that actually took place with BBM3 closing line ADP as its ADP
bbm4 <- bbm4 %>%
  left_join(last_draft %>%
              select(overall_pick_number, RB_expected, WR_expected, QB_expected, TE_expected), 
            by = "overall_pick_number") %>%
  mutate(RB_over_expected = RB_taken - RB_expected,
         WR_over_expected = WR_taken - WR_expected,
         QB_over_expected = QB_taken - QB_expected,
         TE_over_expected = TE_taken - TE_expected)

# Add in the draft to BBM3
new_bbm <- bind_rows(bbm3, bbm4)

# Create new wr_oe values
new_bbm <- new_bbm %>%
  ungroup() %>%
  group_by(overall_pick_number) %>%
  mutate(RB_oe = scale(RB_over_expected),
         WR_oe = scale(WR_over_expected),
         TE_oe = scale(TE_over_expected),
         QB_oe = scale(QB_over_expected)) %>%
  ungroup()

new_bbm %>%
  filter(overall_pick_number <= 108) %>%
  group_by(draft_id) %>%
  summarize(mean_wr_oe = mean(WR_oe)) %>%
  arrange(-mean_wr_oe) %>%
  head(n = 10) %>%
  gt() %>%
  tab_header(title = "Most WR Heavy Rooms in BBM3") %>%
  # Column Labels
  cols_label(
    draft_id = "Draft ID",
    mean_wr_oe = "Average WR Over Expectation (Scaled)"
    ) %>%
  #Formatting
  fmt_number(columns = c(mean_wr_oe),
             decimals = 2)%>%
  gt_highlight_rows(
    rows = c(6),
    fill = "#bbebb7"
  ) %>%
  # Styling
  tab_style(
    style = cell_text(size = "medium"),
    locations = cells_body()
  ) %>%
  tab_options(
    table.width = 700, 
    data_row.padding = px(2.5),
  ) %>%
  gt_theme_538() %>%
  gtsave_extra(filename = "table8.png")
```

![](https://raw.githubusercontent.com/thefalkon-1/bbm-data-bowl-tables/main/table8.png)<!-- -->

BBM4’s current ADP through 10 rounds would be the *sixth heaviest WR
room in all of BBM3*. It’s more important than ever to secure WR
firepower early and also enticing to grab discounted RBs at an even
lower price. [Theory will only take you so
far](https://www.imdb.com/title/tt15398776/characters/nm0614165). Draft
wisely.
