# Data directory

The scripts download three unchanged source tables from the official dcaribou
data endpoint:

- `raw/player_valuations.csv.gz`: historical valuation records and the target.
- `raw/players.csv.gz`: profile attributes used for age, position, foot, and height.
- `raw/appearances.csv.gz`: match-level minutes, goals, assists, and cards used
  for cumulative and recent-five-match features.

Source: https://github.com/dcaribou/transfermarkt-datasets

Direct files:

- https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/player_valuations.csv.gz
- https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz
- https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/appearances.csv.gz

License: CC0 1.0, as stated by the source repository.
