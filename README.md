# RookLift

Analyze your Garmin fitness watch data against your Lichess rating.

Build a logistic regression model with your historical Garmin and Lichess data, then read it from your Garmin watch with the corresponding [RookLift-Frontend](https://github.com/dmvaldman/Rooklift-frontend)

# Deploy with Modal

Model creation are chess predictions are Cron jobs using Modal. Model creation happens once a week and predictions each morning.

```
modal deploy -m deploy
```