-- Renames the column in the table
ALTER TABLE public.garmin_body_battery_sleep
RENAME COLUMN body_battery_during_sleep TO body_battery;

-- Drop the old function first because its return signature (the column names) is changing.
-- We specify the parameter types to uniquely identify the function to drop.
DROP FUNCTION IF EXISTS public.get_daily_signals(date, date);

-- Recreates the function to use the new column name
CREATE OR REPLACE FUNCTION get_daily_signals(
    end_date DATE,
    start_date DATE
)
RETURNS TABLE (
    -- lichess
    "date" DATE,
    "rating_morning" INTEGER,
    "rating_evening" INTEGER,
    -- garmin_stress
    "stress_avg" INTEGER,
    -- garmin_body_battery
    "battery_max" INTEGER,
    -- garmin_body_battery_sleep
    "body_battery" INTEGER,
    -- garmin_sleep
    "sleep_stress" REAL,
    "light_duration" INTEGER,
    "rem_duration" INTEGER,
    "deep_duration" INTEGER,
    "sleep_duration" INTEGER,
    "sleep_score" INTEGER,
    "awake_duration" INTEGER,
    -- garmin_activities
    "activity_calories" INTEGER,
    -- garmin_summary
    "steps" INTEGER,
    "sedentary_duration" INTEGER,
    "stress_duration" INTEGER,
    "low_stress_duration" INTEGER,
    "active_calories" INTEGER
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        -- lichess
        l.date,
        l.rating_morning,
        l.rating_evening,
        -- garmin_stress
        gs.stress_avg,
        -- garmin_body_battery
        gbb.battery_max,
        -- garmin_body_battery_sleep
        gbbs.body_battery,
        -- garmin_sleep
        gsl.sleep_stress,
        gsl.light_duration,
        gsl.rem_duration,
        gsl.deep_duration,
        gsl.sleep_duration,
        gsl.sleep_score,
        gsl.awake_duration,
        -- garmin_activities
        ga.activity_calories,
        -- garmin_summary
        gsm.steps,
        gsm.sedentary_duration,
        gsm.stress_duration,
        gsm.low_stress_duration,
        gsm.active_calories
    FROM
        public.lichess l
    LEFT JOIN public.garmin_stress gs ON l.date = gs.date
    LEFT JOIN public.garmin_body_battery gbb ON l.date = gbb.date
    LEFT JOIN public.garmin_body_battery_sleep gbbs ON l.date = gbbs.date
    LEFT JOIN public.garmin_sleep gsl ON l.date = gsl.date
    LEFT JOIN public.garmin_activities ga ON l.date = ga.date
    LEFT JOIN public.garmin_summary gsm ON l.date = gsm.date
    WHERE
        l.date >= start_date AND l.date <= end_date
        -- Rule: Lichess data must exist and have at least one rating.
        AND (l.rating_morning IS NOT NULL OR l.rating_evening IS NOT NULL);
END;
$$ LANGUAGE plpgsql;
