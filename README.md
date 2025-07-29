# RookLift

Personal backend for [RookLift](https://github.com/dmvaldman/Rooklift-frontend). Analyzes my Garmin fitness watch data against my Lichess rating to measure my intelligence any given day.

Builds a statistical model from time-series fitness/sleep metrics to predict whether I will win or lose at chess any given day.

Uses Modal for cron jobs (one to build the model, the other to update my intelligence score) and Supabase to store the data.

# Deploy with Modal

```
modal deploy -m deploy
```