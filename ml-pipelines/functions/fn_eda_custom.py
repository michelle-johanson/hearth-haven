# fn_eda_custom.py
# Dataset-Specific Preparation Functions
# Covers: Chapter 7 (Reproducible Data Preparation Pipelines)
#
# These functions encode every cleaning decision made during EDA into a
# single callable function per domain. At the top of every pipeline notebook,
# call the relevant function and get back a clean, model-ready DataFrame.
#
# Design principle: no CSVs saved, no intermediate files.
# Every call reads from the original source CSVs and returns a fresh DataFrame.
# This means cleaning decisions stay in sync with the source data automatically.
#
# Functions:
#   prepare_social_media()  <- eda_social_media.ipynb
#   prepare_donors()        <- eda_donors.ipynb
#   prepare_residents()     <- eda_residents.ipynb (coming after that EDA)
#
# Standard usage pattern in every pipeline notebook:
#
#   from functions.fn_eda_custom import prepare_social_media
#   from functions.fn_prepare import define_features, split_data, build_preprocessor, build_pipelines
#
#   df, NUMERIC, CATEGORICAL, DROP = prepare_social_media()
#   X, y = define_features(df, target='engagement_rate',
#                          numeric=NUMERIC, categorical=CATEGORICAL,
#                          drop_cols=DROP['engagement_rate'])


def prepare_social_media(
    posts_path='data/social_media_posts.csv',
    donations_path='data/donations.csv',
):
    """
    Load, clean, and engineer features for the social media pipeline.
    Encodes all decisions from eda_social_media.ipynb.

    Steps:
        1.  Load raw CSVs
        2.  Drop structural columns (IDs, URLs, free text)
        3.  Fix column types
        4.  Engineer post_month and post_is_weekend from created_at
        5.  Fill intentional nulls with meaningful sentinels
        6.  Fill campaign_name nulls with 'None'
        7.  Bin rare campaign categories (GivingTuesday -> Other)
        8.  Cap outliers and transform skew on numeric features
        9.  Aggregate social-referred donations and left-join to posts
        10. Engineer led_to_donation classification target
        11. Return df + feature lists

    Key EDA Findings Encoded:
        - features_resident_story: #1 predictor of donations (F=212.6)
        - sentiment_tone + has_call_to_action: top engagement predictors
        - boost_budget_php: DROPPED -- 84% zeros collapsed IQR to [0,0]
          Signal preserved via is_boosted (boolean, in CATEGORICAL)
        - impressions vs reach: r=0.9965 -- near-identical, both leaky anyway
        - led_to_donation base rate: 7.6% -- use class_weight='balanced'
        - YouTube metrics: low variance after 0-fill, low expected signal

    Parameters:
        posts_path (str): path to social_media_posts.csv
        donations_path (str): path to donations.csv

    Returns:
        df (DataFrame): cleaned, merged, model-ready DataFrame (812 rows)
        NUMERIC_FEATURES (list): 9 numeric input feature names
        CATEGORICAL_FEATURES (list): 11 categorical input feature names
        DROP_ALWAYS (dict): per-target drop lists keyed by target name
            Keys: 'engagement_rate', 'donation_referrals', 'led_to_donation'

    Example:
        df, NUMERIC, CATEGORICAL, DROP = prepare_social_media()

        X, y = define_features(df, target='engagement_rate',
                               numeric=NUMERIC, categorical=CATEGORICAL,
                               drop_cols=DROP['engagement_rate'])
    """
    import pandas as pd
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from fn_clean import (
        drop_structural_columns, fix_column_types, handle_intentional_nulls,
        cap_outliers_iqr, transform_skewed_column, bin_rare_categories, merge_tables,
    )

    print("=" * 52)
    print("  prepare_social_media()")
    print("=" * 52)

    # 1. Load
    posts     = pd.read_csv(posts_path)
    donations = pd.read_csv(donations_path)
    print(f"\n[OK] Loaded: {posts.shape[0]} posts, {donations.shape[0]} donations")

    # 2. Drop structural columns
    # IDs, URLs, and free text -- no predictive value and cannot be modeled
    posts = drop_structural_columns(posts, [
        'platform_post_id',
        'post_url',
        'caption',
        'hashtags',
    ])

    # 3. Fix column types
    posts = fix_column_types(posts)

    # 4. Engineer temporal features
    # day_of_week and post_hour already exist in raw data
    # Add post_month (seasonality) and post_is_weekend
    posts['created_at']      = pd.to_datetime(posts['created_at'], errors='coerce')
    posts['post_month']      = posts['created_at'].dt.month
    posts['post_is_weekend'] = posts['created_at'].dt.dayofweek.ge(5).astype(int)
    posts = posts.drop(columns=['created_at'])
    print("[OK] Engineered: post_month, post_is_weekend")

    # 5. Handle intentional nulls -- nulls by design, not data quality problems
    posts = handle_intentional_nulls(posts, {
        'video_views':               0,
        'watch_time_seconds':        0,
        'avg_view_duration_seconds': 0,
        'subscriber_count_at_post':  0,
        'boost_budget_php':          0,
        'forwards':                  0,
        'call_to_action_type':       'None',
    })

    # 6. Fill campaign_name
    # Null = not linked to a campaign. 'None' lets model learn campaign effect.
    posts['campaign_name'] = posts['campaign_name'].fillna('None')

    # 7. Bin rare categories
    # GivingTuesday: 2.7% of posts -> 'Other'
    posts = bin_rare_categories(posts, 'campaign_name', threshold=0.05)

    # 8. Cap outliers and transform skew
    # boost_budget_php: NOT processed -- 84% zeros cause IQR fence collapse
    # impressions + reach: yeo-johnson (skew: ~1.16 -> ~0.04 each)
    posts = cap_outliers_iqr(posts, 'impressions')
    posts = cap_outliers_iqr(posts, 'reach')
    posts = cap_outliers_iqr(posts, 'follower_count_at_post')
    posts = transform_skewed_column(posts, 'impressions')
    posts = transform_skewed_column(posts, 'reach')

    # 9. Aggregate and join donations
    social_donations = (
        donations[donations['channel_source'] == 'SocialMedia']
        .groupby('referral_post_id')
        .agg(
            confirmed_donation_count=('donation_id', 'count'),
            confirmed_monetary_value=('amount',      'sum'),
        )
        .reset_index()
        .rename(columns={'referral_post_id': 'post_id'})
    )
    df = merge_tables(posts, social_donations, on='post_id', how='left')
    df['confirmed_donation_count'] = df['confirmed_donation_count'].fillna(0).astype(int)
    df['confirmed_monetary_value'] = df['confirmed_monetary_value'].fillna(0)
    df = df.drop(columns=['post_id'])

    # 10. Engineer classification target
    # Binary: did post lead to at least 1 confirmed social donation?
    # Base rate 7.6% -- use class_weight='balanced' in classifier
    df['led_to_donation'] = (df['confirmed_donation_count'] > 0).astype(int)
    print(f"[OK] Target 'led_to_donation': {df['led_to_donation'].mean():.1%} positive rate")

    # 11. Feature lists
    NUMERIC_FEATURES = [
        'num_hashtags',
        'mentions_count',
        'caption_length',
        'post_hour',
        'post_month',
        'post_is_weekend',
        'follower_count_at_post',
        'video_views',
        'forwards',
    ]

    CATEGORICAL_FEATURES = [
        'platform',
        'post_type',
        'media_type',
        'day_of_week',
        'call_to_action_type',
        'content_topic',
        'sentiment_tone',
        'campaign_name',
        'has_call_to_action',
        'is_boosted',
        'features_resident_story',
    ]

    DROP_ALWAYS = {
        'engagement_rate': [
            'likes', 'comments', 'shares', 'saves',
            'reach', 'impressions', 'click_throughs', 'profile_visits',
            'donation_referrals', 'estimated_donation_value_php',
            'confirmed_donation_count', 'confirmed_monetary_value', 'led_to_donation',
        ],
        'donation_referrals': [
            'estimated_donation_value_php',
            'confirmed_donation_count', 'confirmed_monetary_value', 'led_to_donation',
            'likes', 'comments', 'shares', 'saves',
            'reach', 'impressions', 'click_throughs', 'profile_visits', 'engagement_rate',
        ],
        'led_to_donation': [
            'confirmed_donation_count', 'confirmed_monetary_value',
            'donation_referrals', 'estimated_donation_value_php',
            'likes', 'comments', 'shares', 'saves',
            'reach', 'impressions', 'click_throughs', 'profile_visits', 'engagement_rate',
        ],
    }

    print(f"\n[OK] prepare_social_media() complete.")
    print(f"     Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"     Numeric ({len(NUMERIC_FEATURES)}), Categorical ({len(CATEGORICAL_FEATURES)})")
    print(f"     Targets: {list(DROP_ALWAYS.keys())}")

    return df, NUMERIC_FEATURES, CATEGORICAL_FEATURES, DROP_ALWAYS


def prepare_donors(
    supporters_path='data/supporters.csv',
    donations_path='data/donations.csv',
):
    """
    Load, clean, and engineer features for the donor pipeline.
    Encodes all decisions from eda_donors.ipynb.

    Steps:
        1.  Load raw CSVs
        2.  Drop PII and identifier columns
        3.  Fix column types
        4.  Engineer days_since_first_donation and days_since_created
        5.  Fill never-donated sentinel (-1)
        6.  Aggregate donation behavior per supporter
        7.  Left-join aggregates to supporters
        8.  Fill zero-donation supporter rows
        9.  Bin rare country categories
        10. Cap outliers and transform skewed features
        11. Engineer pipeline targets
        12. Return df + feature lists

    Key EDA Findings Encoded:
        - Dataset is small: only 60 supporters, 59 with any donations
        - is_recurring_donor: #1 predictor of total value (F=19.6)
        - days_since_last_donation: r=0.83 with is_lapsed -- LEAKY for lapse
        - total_donations vs monetary_donation_count: r=0.89 multicollinear
          total_donations DROPPED -- monetary_donation_count kept as proxy
        - is_lapsed base rate: 18.3% (11/60) -- use class_weight='balanced'
        - avg_monetary_gift: 4 outliers capped, skew resolved post-cap
        - monetary_donation_count: yeo-johnson applied (skew: ~1.49 -> ~0.02)
        - country: Singapore + Canada -> 'Other' (both below 5%)

    Parameters:
        supporters_path (str): path to supporters.csv
        donations_path (str): path to donations.csv

    Returns:
        df (DataFrame): cleaned, merged, model-ready DataFrame (60 rows)
        NUMERIC_FEATURES (list): 7 numeric input feature names
        CATEGORICAL_FEATURES (list): 7 categorical input feature names
        DROP_ALWAYS (dict): per-target drop lists keyed by target name
            Keys: 'is_lapsed', 'total_monetary_value'

    Note:
        For the total_monetary_value regression, filter to donors only:
            df_donors = df[df['monetary_donation_count'] > 0]
        before calling define_features(). The 1 non-donor row has
        total_monetary_value=0, which is not a meaningful regression target.

    Example:
        df, NUMERIC, CATEGORICAL, DROP = prepare_donors()

        # Lapse classification
        X, y = define_features(df, target='is_lapsed',
                               numeric=NUMERIC, categorical=CATEGORICAL,
                               drop_cols=DROP['is_lapsed'])

        # Lifetime value regression (donors only)
        df_donors = df[df['monetary_donation_count'] > 0]
        X, y = define_features(df_donors, target='total_monetary_value',
                               numeric=NUMERIC, categorical=CATEGORICAL,
                               drop_cols=DROP['total_monetary_value'])
    """
    import pandas as pd
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from fn_clean import (
        drop_structural_columns, fix_column_types, bin_rare_categories,
        cap_outliers_iqr, transform_skewed_column, merge_tables,
    )

    print("=" * 52)
    print("  prepare_donors()")
    print("=" * 52)

    # 1. Load
    supporters = pd.read_csv(supporters_path)
    donations  = pd.read_csv(donations_path)
    print(f"\n[OK] Loaded: {supporters.shape[0]} supporters, {donations.shape[0]} donations")

    # 2. Drop PII and identifier columns
    supporters = drop_structural_columns(supporters, [
        'display_name',
        'organization_name',
        'first_name',
        'last_name',
        'email',
        'phone',
    ])

    # 3. Fix column types
    supporters = fix_column_types(supporters)

    # 4. Engineer date-derived features
    # Convert raw dates to "days since" -- numeric, usable by any model
    supporters['first_donation_date'] = pd.to_datetime(
        supporters['first_donation_date'], errors='coerce'
    )
    supporters['created_at'] = pd.to_datetime(
        supporters['created_at'], errors='coerce'
    )
    ref_date = pd.Timestamp('today').normalize()

    supporters['days_since_first_donation'] = (
        ref_date - supporters['first_donation_date']
    ).dt.days
    supporters['days_since_created'] = (
        ref_date - supporters['created_at']
    ).dt.days

    supporters = supporters.drop(columns=['first_donation_date', 'created_at'])
    print("[OK] Engineered: days_since_first_donation, days_since_created")

    # 5. Fill never-donated sentinel
    # 1 supporter has no donation history -- null becomes -1
    # Sentinel -1 distinguishes "never donated" from "donated very recently"
    supporters['days_since_first_donation'] = (
        supporters['days_since_first_donation'].fillna(-1)
    )

    # 6. Aggregate donation behavior per supporter
    # total_donations intentionally excluded -- r=0.89 with monetary_donation_count
    donation_agg = donations.groupby('supporter_id').agg(
        monetary_donation_count=('donation_type',  lambda x: (x == 'Monetary').sum()),
        total_monetary_value   =('amount',          'sum'),
        avg_monetary_gift      =('amount',           'mean'),
        is_recurring_donor     =('is_recurring',     'max'),
        unique_campaigns       =('campaign_name',    'nunique'),
        last_donation_date     =('donation_date',    'max'),
        donation_types_count   =('donation_type',    'nunique'),
    ).reset_index()

    donation_agg['last_donation_date'] = pd.to_datetime(
        donation_agg['last_donation_date'], errors='coerce'
    )
    donation_agg['days_since_last_donation'] = (
        ref_date - donation_agg['last_donation_date']
    ).dt.days
    donation_agg = donation_agg.drop(columns=['last_donation_date'])

    # 7. Left-join -- keep all supporters
    df = merge_tables(supporters, donation_agg, on='supporter_id', how='left')

    # 8. Fill zero-donation rows
    df['monetary_donation_count']  = df['monetary_donation_count'].fillna(0).astype(int)
    df['total_monetary_value']     = df['total_monetary_value'].fillna(0)
    df['avg_monetary_gift']        = df['avg_monetary_gift'].fillna(0)
    df['is_recurring_donor']       = df['is_recurring_donor'].fillna(False)
    df['unique_campaigns']         = df['unique_campaigns'].fillna(0).astype(int)
    df['donation_types_count']     = df['donation_types_count'].fillna(0).astype(int)
    df['days_since_last_donation'] = df['days_since_last_donation'].fillna(-1)
    df = df.drop(columns=['supporter_id'])

    # 9. Bin rare categories
    # Singapore (1.7%) + Canada (1.7%) -> 'Other'
    # Region: all three values above 5%, no binning needed
    df = bin_rare_categories(df, 'country', threshold=0.05)

    # 10. Cap outliers and transform skew
    # avg_monetary_gift: 4 outliers capped; skew resolved post-cap (0.42 < 0.5)
    df = cap_outliers_iqr(df, 'avg_monetary_gift')
    df = transform_skewed_column(df, 'avg_monetary_gift')
    # monetary_donation_count: yeo-johnson applied
    df = cap_outliers_iqr(df, 'monetary_donation_count')
    df = transform_skewed_column(df, 'monetary_donation_count')
    # days_since_last_donation: skew=1.58 but contains -1 sentinel values
    # -- do NOT transform, sentinel meaning would be distorted

    # 11. Engineer targets
    # is_lapsed: has donated but nothing in last 365 days (18.3% base rate)
    df['is_lapsed'] = (
        (df['monetary_donation_count'] > 0) &
        (df['days_since_last_donation'] > 365)
    ).astype(int)

    print(f"[OK] Target 'is_lapsed': {df['is_lapsed'].mean():.1%} positive rate")
    print(f"[OK] Target 'total_monetary_value': mean={df['total_monetary_value'].mean():.0f} PHP, "
          f"median={df['total_monetary_value'].median():.0f} PHP")

    # 12. Feature lists
    NUMERIC_FEATURES = [
        'monetary_donation_count',    # how many monetary gifts (transformed)
        'avg_monetary_gift',          # average gift size PHP (capped)
        'unique_campaigns',           # breadth of campaign participation
        'donation_types_count',       # diversity of giving types
        'days_since_last_donation',   # recency (-1 if never donated)
        'days_since_first_donation',  # tenure (-1 if never donated)
        'days_since_created',         # time in system
    ]

    CATEGORICAL_FEATURES = [
        'supporter_type',       # MonetaryDonor, InKindDonor, Volunteer, etc.
        'relationship_type',    # Local, International, PartnerOrganization
        'country',              # Philippines, USA, Other
        'region',               # Luzon, Mindanao, Visayas
        'status',               # Active / Inactive
        'acquisition_channel',  # WordOfMouth, SocialMedia, Website, etc.
        'is_recurring_donor',   # boolean -> 0/1 -- #1 predictor of total value
    ]

    DROP_ALWAYS = {
        'is_lapsed': [
            # Leaky: lapse is literally defined by days_since_last_donation
            'days_since_last_donation',
            # Other target -- not predicting here
            'total_monetary_value',
        ],
        'total_monetary_value': [
            # Derived from donation behavior -- not a predictor
            'is_lapsed',
            # NOTE: also filter df to monetary_donation_count > 0 before
            # calling define_features() -- see docstring above
        ],
    }

    print(f"\n[OK] prepare_donors() complete.")
    print(f"     Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"     Numeric ({len(NUMERIC_FEATURES)}), Categorical ({len(CATEGORICAL_FEATURES)})")
    print(f"     Targets: {list(DROP_ALWAYS.keys())}")
    print(f"     WARNING: Small dataset (n=60). Treat outputs with caution.")

    return df, NUMERIC_FEATURES, CATEGORICAL_FEATURES, DROP_ALWAYS


def prepare_residents(
    residents_path='data/residents.csv',
    health_path='data/health_wellbeing_records.csv',
    education_path='data/education_records.csv',
    counseling_path='data/process_recordings.csv',
    incidents_path='data/incident_reports.csv',
    visitations_path='data/home_visitations.csv',
    interventions_path='data/intervention_plans.csv',
):
    """
    Load, clean, and engineer features for the resident pipeline.
    Encodes all decisions from eda_residents.ipynb.

    Steps:
        1.  Load all 7 raw CSVs
        2.  Drop structural/PII/zero-variance columns from residents
        3.  Fix column types
        4.  Engineer date-derived features (age, days_in_care, length_of_stay)
        5.  Handle intentional nulls (pwd_type, special_needs, reintegration)
        6.  Aggregate each supporting table to resident level
        7.  Left-join all aggregates onto residents
        8.  Fill zeros/medians for residents with no records in a table
        9.  Resolve perfect multicollinearity (r=1.0 pairs)
        10. Engineer targets
        11. Return df + feature lists

    Key EDA Findings Encoded:
        - risk_escalated: ZERO positive cases -- target replaced with
          current_risk_num (regression) which is more useful
        - has_special_needs: #1 predictor of education progress (F=26.4)
        - health_record_count == education_records exactly: r=1.0, drop health_record_count
        - achieved_plans == plan_achievement_rate * total_plans: r=1.0, drop achieved_plans
        - health_score_latest vs health_score_avg: r=0.92, drop health_score_avg
        - progress_avg: LEAKY for progress_percent_latest -- in DROP_ALWAYS
        - reintegration_status: LEAKY for reintegration_achieved -- in DROP_ALWAYS
        - religion: all 8 values above 5%, no binning needed
        - Counseling is richest table: 2819 sessions, avg ~47/resident

    Parameters:
        residents_path through interventions_path (str): CSV file paths

    Returns:
        df (DataFrame): cleaned, merged, model-ready DataFrame (60 rows x ~65 cols)
        NUMERIC_FEATURES (list): numeric input feature names
        CATEGORICAL_FEATURES (list): categorical input feature names
        DROP_ALWAYS (dict): per-target drop lists
            Keys: 'reintegration_achieved', 'progress_percent_latest',
                  'current_risk_num'

    Note:
        For current_risk_num regression, initial_risk_level is a valid
        feature (it's the intake snapshot, not the outcome). Include it.
        For reintegration_achieved, filter to residents with a reintegration
        pathway: df_reint = df[df['reintegration_type'] != 'None']

    Example:
        df, NUMERIC, CATEGORICAL, DROP = prepare_residents()

        # Reintegration classification (residents in pathway only)
        df_r = df[df['reintegration_type'] != 'None']
        X, y = define_features(df_r, target='reintegration_achieved',
                               numeric=NUMERIC, categorical=CATEGORICAL,
                               drop_cols=DROP['reintegration_achieved'])

        # Education progress regression (all residents)
        X, y = define_features(df, target='progress_percent_latest',
                               numeric=NUMERIC, categorical=CATEGORICAL,
                               drop_cols=DROP['progress_percent_latest'])

        # Current risk regression (all residents)
        X, y = define_features(df, target='current_risk_num',
                               numeric=NUMERIC, categorical=CATEGORICAL,
                               drop_cols=DROP['current_risk_num'])
    """
    import pandas as pd
    import numpy as np
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from fn_clean import (
        drop_structural_columns, fix_column_types, bin_rare_categories,
        cap_outliers_iqr, merge_tables,
    )

    print("=" * 52)
    print("  prepare_residents()")
    print("=" * 52)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    residents     = pd.read_csv(residents_path)
    health        = pd.read_csv(health_path)
    education     = pd.read_csv(education_path)
    counseling    = pd.read_csv(counseling_path)
    incidents     = pd.read_csv(incidents_path)
    visitations   = pd.read_csv(visitations_path)
    interventions = pd.read_csv(interventions_path)
    print(f"\n[OK] Loaded: {residents.shape[0]} residents, "
          f"{counseling.shape[0]} counseling sessions, "
          f"{health.shape[0]} health records")

    # ── 2. Drop structural columns ────────────────────────────────────────────
    residents = drop_structural_columns(residents, [
        'case_control_no',          # admin ID
        'internal_code',            # admin ID
        'sex',                      # all 'F' -- zero variance
        'place_of_birth',           # high-cardinality free text
        'referring_agency_person',  # free text name
        'assigned_social_worker',   # free text name
        'initial_case_assessment',  # free text narrative
        'notes_restricted',         # restricted free text (all null)
        'date_enrolled',            # exact duplicate of date_of_admission
        'date_colb_registered',     # admin paperwork date
        'date_colb_obtained',       # admin paperwork date
        'date_case_study_prepared', # admin paperwork date
    ])

    # ── 3. Fix column types ───────────────────────────────────────────────────
    residents = fix_column_types(residents)

    # ── 4. Engineer date features ─────────────────────────────────────────────
    for col in ['date_of_birth', 'date_of_admission', 'date_closed', 'created_at']:
        residents[col] = pd.to_datetime(residents[col], errors='coerce')

    ref_date = pd.Timestamp('today').normalize()

    residents['age_at_admission_days'] = (
        residents['date_of_admission'] - residents['date_of_birth']
    ).dt.days
    residents['days_in_care'] = (
        ref_date - residents['date_of_admission']
    ).dt.days
    residents['length_of_stay_days'] = (
        residents['date_closed'] - residents['date_of_admission']
    ).dt.days

    residents = residents.drop(columns=[
        'date_of_birth', 'date_of_admission', 'date_closed', 'created_at',
        'age_upon_admission', 'present_age', 'length_of_stay',
    ])
    print("[OK] Engineered: age_at_admission_days, days_in_care, length_of_stay_days")

    # ── 5. Handle intentional nulls ───────────────────────────────────────────
    residents['pwd_type']               = residents['pwd_type'].fillna('None')
    residents['special_needs_diagnosis']= residents['special_needs_diagnosis'].fillna('None')
    residents['reintegration_type']     = residents['reintegration_type'].fillna('None')
    # length_of_stay_days: null for active cases -> use days_in_care
    residents['length_of_stay_days']    = residents['length_of_stay_days'].fillna(
        residents['days_in_care']
    )

    # ── 6a. Aggregate health ──────────────────────────────────────────────────
    health['record_date'] = pd.to_datetime(health['record_date'], errors='coerce')
    health_latest = (
        health.sort_values('record_date').groupby('resident_id').last().reset_index()
        [['resident_id', 'general_health_score', 'nutrition_score',
          'sleep_quality_score', 'energy_level_score', 'bmi']]
        .rename(columns={
            'general_health_score': 'health_score_latest',
            'nutrition_score':      'nutrition_score_latest',
            'sleep_quality_score':  'sleep_score_latest',
            'energy_level_score':   'energy_score_latest',
            'bmi':                  'bmi_latest',
        })
    )
    health_avg = health.groupby('resident_id').agg(
        # health_score_avg DROPPED: r=0.92 with health_score_latest
        checkup_completion =('medical_checkup_done', 'mean'),
        # health_record_count DROPPED: r=1.0 with education_records
    ).reset_index()
    health_agg = health_latest.merge(health_avg, on='resident_id', how='left')

    # ── 6b. Aggregate education ───────────────────────────────────────────────
    education['record_date'] = pd.to_datetime(education['record_date'], errors='coerce')
    edu_latest = (
        education.sort_values('record_date').groupby('resident_id').last().reset_index()
        [['resident_id', 'attendance_rate', 'progress_percent',
          'education_level', 'enrollment_status', 'completion_status']]
        .rename(columns={
            'attendance_rate':   'attendance_rate_latest',
            'progress_percent':  'progress_percent_latest',
            'education_level':   'education_level_latest',
            'enrollment_status': 'enrollment_status_latest',
            'completion_status': 'completion_status_latest',
        })
    )
    edu_avg = education.groupby('resident_id').agg(
        progress_avg     =('progress_percent', 'mean'),
        attendance_avg   =('attendance_rate',  'mean'),
        education_records=('education_record_id', 'count'),
    ).reset_index()
    edu_agg = edu_latest.merge(edu_avg, on='resident_id', how='left')

    # ── 6c. Aggregate counseling ──────────────────────────────────────────────
    counseling_agg = counseling.groupby('resident_id').agg(
        session_count        =('recording_id',            'count'),
        avg_session_duration =('session_duration_minutes', 'mean'),
        progress_rate        =('progress_noted',           'mean'),
        concern_rate         =('concerns_flagged',         'mean'),
        referral_rate        =('referral_made',            'mean'),
    ).reset_index()

    # ── 6d. Aggregate incidents ───────────────────────────────────────────────
    severity_map = {'Low': 1, 'Medium': 2, 'High': 3}
    incidents['severity_num'] = incidents['severity'].map(severity_map)
    incidents_agg = incidents.groupby('resident_id').agg(
        incident_count      =('incident_id',   'count'),
        avg_severity        =('severity_num',  'mean'),
        high_severity_count =('severity_num',  lambda x: (x == 3).sum()),
        runaway_attempts    =('incident_type', lambda x: (x == 'RunawayAttempt').sum()),
        self_harm_incidents =('incident_type', lambda x: (x == 'SelfHarm').sum()),
    ).reset_index()

    # ── 6e. Aggregate visitations ─────────────────────────────────────────────
    coop_map = {'Highly Cooperative': 4, 'Cooperative': 3,
                'Neutral': 2, 'Uncooperative': 1}
    visitations['coop_num'] = visitations['family_cooperation_level'].map(coop_map)
    visitations_agg = visitations.groupby('resident_id').agg(
        visitation_count       =('visitation_id',        'count'),
        safety_concern_rate    =('safety_concerns_noted', 'mean'),
        favorable_outcome_rate =('visit_outcome',
                                 lambda x: (x == 'Favorable').sum() / len(x)),
        family_cooperation_score=('coop_num',            'mean'),
    ).reset_index()

    # ── 6f. Aggregate interventions ───────────────────────────────────────────
    interventions_agg = interventions.groupby('resident_id').agg(
        total_plans          =('plan_id',       'count'),
        open_plans           =('status',        lambda x: (x == 'Open').sum()),
        plan_categories      =('plan_category', 'nunique'),
        plan_achievement_rate=('status',        lambda x: (x == 'Achieved').sum() / len(x)),
    ).reset_index()

    # ── 7. Build master DataFrame ─────────────────────────────────────────────
    df = residents.copy()
    for agg_df in [health_agg, edu_agg, counseling_agg,
                   incidents_agg, visitations_agg, interventions_agg]:
        df = merge_tables(df, agg_df, on='resident_id', how='left')
    df = df.drop(columns=['resident_id'])

    # ── 8. Fill missing aggregates ────────────────────────────────────────────
    count_cols = ['session_count', 'incident_count', 'visitation_count',
                  'total_plans', 'open_plans', 'plan_categories',
                  'education_records', 'high_severity_count',
                  'runaway_attempts', 'self_harm_incidents']
    rate_cols  = ['progress_rate', 'concern_rate', 'referral_rate',
                  'safety_concern_rate', 'favorable_outcome_rate',
                  'plan_achievement_rate', 'checkup_completion',
                  'attendance_avg', 'attendance_rate_latest']
    score_cols = ['health_score_latest', 'nutrition_score_latest',
                  'sleep_score_latest', 'energy_score_latest', 'bmi_latest',
                  'avg_session_duration', 'avg_severity',
                  'family_cooperation_score', 'progress_percent_latest',
                  'progress_avg']

    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ── 9. Engineer targets ───────────────────────────────────────────────────
    risk_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}

    df['initial_risk_num'] = df['initial_risk_level'].map(risk_order)
    df['current_risk_num'] = df['current_risk_level'].map(risk_order)

    # risk_escalated: 0 positive cases in the original sample data.
    # Add synthetic escalation cases to residents.csv before using this target.
    # The function computes it correctly -- the data just needs representative examples.
    # Target: did current_risk_level get WORSE than initial_risk_level?
    df['risk_improved'] = (
        df['current_risk_num'] < df['initial_risk_num']
    ).astype(int)

    # reintegration_achieved: 31.7% positive rate
    df['reintegration_achieved'] = (
        df['reintegration_status'] == 'Completed'
    ).astype(int)

    print(f"[OK] Target 'reintegration_achieved': {df['reintegration_achieved'].mean():.1%} positive")
    print(f"[OK] Target 'current_risk_num': mean={df['current_risk_num'].mean():.2f} (0=Low, 3=Critical)")
    print(f"[OK] Target 'progress_percent_latest': mean={df['progress_percent_latest'].mean():.1f}")
    print(f"[OK] Target 'risk_improved': {df['risk_improved'].mean():.1%} positive")

    # ── 10. Feature lists ─────────────────────────────────────────────────────
    NUMERIC_FEATURES = [
        # Background
        'age_at_admission_days',
        'days_in_care',
        'length_of_stay_days',
        # Health (health_score_avg dropped: r=0.92 with latest)
        'health_score_latest',
        'nutrition_score_latest',
        'sleep_score_latest',
        'energy_score_latest',
        'bmi_latest',
        'checkup_completion',
        # Education (education_records kept; health_record_count dropped: r=1.0)
        'attendance_rate_latest',
        'attendance_avg',
        'progress_avg',        # NOTE: leaky for progress_percent_latest target
        'education_records',
        # Counseling
        'session_count',
        'avg_session_duration',
        'progress_rate',
        'concern_rate',
        'referral_rate',
        # Incidents
        'incident_count',
        'avg_severity',
        'high_severity_count',
        'runaway_attempts',
        'self_harm_incidents',
        # Visitations
        'visitation_count',
        'safety_concern_rate',
        'favorable_outcome_rate',
        'family_cooperation_score',
        # Interventions (achieved_plans dropped: r=1.0 with plan_achievement_rate)
        'total_plans',
        'plan_achievement_rate',
        'open_plans',
        'plan_categories',
        # Risk numerics
        'initial_risk_num',    # NOTE: leaky for current_risk_num target
    ]

    CATEGORICAL_FEATURES = [
        'case_status',
        'birth_status',
        'religion',
        'case_category',
        'referral_source',
        'initial_risk_level',  # NOTE: leaky for current_risk_num target
        'reintegration_type',
        'enrollment_status_latest',
        'education_level_latest',
        'completion_status_latest',
        'pwd_type',
        'special_needs_diagnosis',
        # Boolean flags -- encoded as 0/1 by OneHotEncoder
        'sub_cat_orphaned', 'sub_cat_trafficked', 'sub_cat_child_labor',
        'sub_cat_physical_abuse', 'sub_cat_sexual_abuse', 'sub_cat_osaec',
        'sub_cat_cicl', 'sub_cat_at_risk', 'sub_cat_street_child',
        'sub_cat_child_with_hiv',
        'is_pwd', 'has_special_needs',
        'family_is_4ps', 'family_solo_parent', 'family_indigenous',
        'family_parent_pwd', 'family_informal_settler',
    ]

    DROP_ALWAYS = {
        'reintegration_achieved': [
            # Leaky: status defines the target
            'reintegration_status',
            # Leaky: closed cases = completed cases, stay length encodes the outcome
            'length_of_stay_days',
            # Other targets
            'current_risk_num', 'current_risk_level',
            'initial_risk_num', 'risk_improved', 'risk_escalated',
            'progress_percent_latest',
        ],
        'progress_percent_latest': [
            # Leaky: historical avg of the same metric
            'progress_avg',
            # Leaky: closed cases = completed cases, stay length encodes the outcome
            'length_of_stay_days',
            # Other targets
            'reintegration_achieved', 'reintegration_status',
            'current_risk_num', 'current_risk_level',
            'initial_risk_num', 'risk_improved', 'risk_escalated',
            # Potentially leaky -- high correlation (r=0.49), monitor
            # 'attendance_rate_latest',  # uncomment if causing issues
        ],
        'current_risk_num': [
            # Leaky: these are or derive from the target
            'current_risk_level', 'initial_risk_num', 'initial_risk_level',
            'risk_improved', 'risk_escalated',
            # Other targets
            'reintegration_achieved', 'reintegration_status',
            'progress_percent_latest',
        ],
        'risk_improved': [
            # Leaky: computed from these
            'current_risk_num', 'current_risk_level',
            'initial_risk_num', 'risk_escalated',
            # Other targets
            'reintegration_achieved', 'reintegration_status',
            'progress_percent_latest',
        ],
        'risk_escalated': [
            # Leaky: computed from these two columns
            'current_risk_num', 'current_risk_level',
            'initial_risk_num', 'risk_improved',
            # Other targets
            'reintegration_achieved', 'reintegration_status',
            'progress_percent_latest',
            # NOTE: initial_risk_level is NOT leaky here -- it's the intake
            # snapshot that defines the baseline, not the outcome.
            # Keep it in features for this target.
        ],
    }

    print(f"\n[OK] prepare_residents() complete.")
    print(f"     Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"     Numeric ({len(NUMERIC_FEATURES)}), Categorical ({len(CATEGORICAL_FEATURES)})")
    print(f"     Targets: {list(DROP_ALWAYS.keys())}")

    return df, NUMERIC_FEATURES, CATEGORICAL_FEATURES, DROP_ALWAYS