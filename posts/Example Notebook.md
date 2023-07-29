# Submission Title Goes Here
## Participants: Fantasy Data Pros (Ben Dominguez, Peter Overzet, Loudog)

[@bendominguez011](https://www.twitter.com/bendominguez011)
[@peteroverzet](https://www.twitter.com/peteroverzet)
[@loudogvideo](https://www.twitter.com/loudogvideo)

This is an example notebook that should be as a guideline for your submission. Write a submission title with an H1 heading (#), and then either your name or your team name (and the names of your team members as well) with an H2 (##). If you'd like, you can also add your Twitter handles, any organizations you're associated with, or any other personal/professional social links.


Write a brief introduction explaining the topic you're exploring in your notebook, and then get started!


```python
import pandas as pd
```

Your notebook should contain both code and text, there's a 2000 word maximum to keep in mind when writing text, although it's okay if you go a bit over. After each code block or figure, you should include context and reasoning for each step in your analysis.

While this example is a Jupyter notebook, you're welcome to use R if you feel more comfortable doing so - you can either embed an R kernel within Jupyter or submit an R markdown notebook.


```python
df = pd.read_csv('../data/2020/part_00.csv')
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
      <th>team</th>
      <th>player</th>
      <th>drafted_round</th>
      <th>roster_points</th>
      <th>pick_points</th>
      <th>draft_time</th>
      <th>playoff_round</th>
      <th>made_playoffs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>081ea432-dcf5-4370-932d-b53ac2296900</td>
      <td>A.J. Brown</td>
      <td>4</td>
      <td>109.94</td>
      <td>0.0</td>
      <td>2020-12-22 17:04:23 UTC</td>
      <td>Round 4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>081ea432-dcf5-4370-932d-b53ac2296900</td>
      <td>Antonio Brown</td>
      <td>15</td>
      <td>109.94</td>
      <td>11.5</td>
      <td>2020-12-22 17:04:23 UTC</td>
      <td>Round 4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>081ea432-dcf5-4370-932d-b53ac2296900</td>
      <td>Baker Mayfield</td>
      <td>13</td>
      <td>109.94</td>
      <td>0.0</td>
      <td>2020-12-22 17:04:23 UTC</td>
      <td>Round 4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>081ea432-dcf5-4370-932d-b53ac2296900</td>
      <td>Chase Edmonds</td>
      <td>10</td>
      <td>109.94</td>
      <td>0.0</td>
      <td>2020-12-22 17:04:23 UTC</td>
      <td>Round 4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>081ea432-dcf5-4370-932d-b53ac2296900</td>
      <td>D.J. Chark</td>
      <td>5</td>
      <td>109.94</td>
      <td>14.2</td>
      <td>2020-12-22 17:04:23 UTC</td>
      <td>Round 4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.pick_points.plot.hist();
```


    
![png](Example%20Notebook_files/Example%20Notebook_4_0.png)
    


There's also a limit on how many tables / figures you can add. We've set that number to 15, although if you go 1 or 2 over it's okay.

At the end of the notebook, sum up your conclusions in a conclusion paragraph. This is an example so this notebook is much, much shorter than what the ideal submission should look like.

After you're done, make a PR to [the Github repository](https://github.com/fantasydatapros/best-ball-data-bowl) and add a comment that includes your submission title, team name / your name, and a brief description of your findings.

Good luck!
