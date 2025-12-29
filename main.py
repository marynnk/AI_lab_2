import pandas as pd
from pandas_profiling import ProfileReport
import pandas_profiling

days_in_year = 365

df = pd.read_csv("./cardio_train.csv", sep=";", index_col="id")

print("Top 10 records\n", df.head(10))
print("\n\nDescriptive stats\n", df.describe())

profile = pandas_profiling.ProfileReport(df)
profile.to_file(output_file="AAA data profiling.html")

group1_mean_height = df[df["gender"] == 1]["height"].mean()
group2_mean_height = df[df["gender"] == 2]["height"].mean()

male_gender = 1 if group1_mean_height > group2_mean_height else 2
female_gender = 1 if group1_mean_height < group2_mean_height else 2

print(f"\nMale gender {male_gender}, female gender {female_gender}")

gender_grouped_alco = df.groupby("gender")["alco"].mean()
print(
    f"\nMen consumes alcohol {gender_grouped_alco[male_gender]} but women consumes {gender_grouped_alco[female_gender]}")
man_alco_ratio = round(gender_grouped_alco[male_gender] / gender_grouped_alco[female_gender])
print(f"\nMen consumes in {man_alco_ratio} times more then women")

gender_grouped_smoking = df.groupby("gender")["smoke"].mean()
smoke_ratio = round(gender_grouped_smoking[male_gender] / gender_grouped_smoking[female_gender])
print(f"Men smokes {smoke_ratio} times more than women")

pressure_grouped_by_gender = df.groupby("gender")[["ap_hi", "ap_lo"]].mean()
men_systolic_pressure_ratio = round(
    pressure_grouped_by_gender.loc[male_gender, 'ap_hi'] / pressure_grouped_by_gender.loc[female_gender, 'ap_hi'],
    2) * 100 - 100
print(f"\nMen systolic blood pressure greater then women {men_systolic_pressure_ratio}%")

men_diastolic_pressure_ration = round(
    pressure_grouped_by_gender.loc[male_gender, 'ap_lo'] / pressure_grouped_by_gender.loc[female_gender, 'ap_lo'],
    2) * 100 - 100
print(f"Men diastolic blood pressure greater then women {men_diastolic_pressure_ration}%")

max_age = df["age"].max()
suggested_age_ratio = next((item for item in [days_in_year, 12, 52] if item < 100), None)
median_age_by_smoking = df.groupby("smoke")["age"].median()
smoke_ratio_live = abs(
    (median_age_by_smoking[1] * 12 / days_in_year) - (median_age_by_smoking[0] * suggested_age_ratio / days_in_year))
print(
    f"\nSmokers lives {round(smoke_ratio_live, 1)} month less than non-smokers or {round(smoke_ratio_live / suggested_age_ratio, 1)} years")


df['age_years'] = (df['age'] / days_in_year).round()

smokers = df[df['smoke'] == 1]

# Підгрупа 1: чоловіки 60-64 роки, тиск < 120, холестерин 1 (4 ммоль/л)
lower_pressure_and_cholesterol_mask = ((smokers["gender"] == male_gender) & (smokers['age_years'] >= 60) & (smokers['age_years'] <= 64) &
                 (smokers['ap_hi'] < 120) & (smokers['cholesterol'] == 1))
group1 = smokers[lower_pressure_and_cholesterol_mask]

# Підгрупа 2: чоловіки 60-64 роки, тиск [160, 180), холестерин 3 (8 ммоль/л)
higher_pressure_and_cholesterol_mask = ((smokers["gender"] == male_gender) & (smokers['age_years'] >= 60) & (smokers['age_years'] <= 64) &
                 (smokers['ap_hi'] >= 160) & (smokers['ap_hi'] < 180) &
                 (smokers['cholesterol'] == 3))
group2 = smokers[higher_pressure_and_cholesterol_mask]

# Рахуємо частку хворих (cardio=1) в кожній групі
risk1 = group1['cardio'].mean()
risk2 = group2['cardio'].mean()

if risk1 > 0:
    risk_ratio = round(risk2 / risk1)
    print(f"\n\nDisease ratio in 2nd group is greater then in 1st in {risk_ratio} times")
else:
    risk_ratio = round(risk1 / risk2)
    print(f"\n\nDisease ratio in 1st group is greater then in 2nd in {risk_ratio} times")


# BMI = weight (kg) / height^2 (m)
#height in dataset is in sm, so we need to convert it to m by dividing 100
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

print(f"\nMean BMI: {df['BMI'].median():.2f} (Regular: 18.5-25)")
print(f"Woman mean BMI: {df[df['gender'] == female_gender]['BMI'].mean():.2f}")
print(f"Men mean BMI: {df[df['gender'] == male_gender]['BMI'].mean():.2f}")
print(f"Healthy people mean BMI: {df[df['cardio'] == 0]['BMI'].mean():.2f}")
print(f"Chronically ill  people mean BMI: {df[df['cardio'] == 1]['BMI'].mean():.2f}")


#Data filter

initial_count = len(df)

# 1. ap_lo <= ap_hi
# 2. height in range [2.5%, 97.5%]
# 3. weight in range [2.5%, 97.5%]
height_low = df['height'].quantile(0.025)
height_high = df['height'].quantile(0.975)
weight_low = df['weight'].quantile(0.025)
weight_high = df['weight'].quantile(0.975)

df_cleaned = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= height_low) & (df['height'] <= height_high) &
    (df['weight'] >= weight_low) & (df['weight'] <= weight_high)
]

dropped_percent = round((initial_count - len(df_cleaned)) / initial_count * 100)
print(f"\nPercentage of skipped data: {dropped_percent}%")


#Overweight
overweight = df_cleaned[df_cleaned['BMI'] >= 25]
overweight_counts = overweight.groupby('gender').size()

# Припускаємо: 1 - жінки, 2 - чоловіки
print(f"\nFemale overweight quantity: {overweight_counts.get(female_gender, 0)}")
print(f"Male overweight quantity: {overweight_counts.get(male_gender, 0)}")