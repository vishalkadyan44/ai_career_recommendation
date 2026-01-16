import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# File path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FILE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "career_data_cleaned.csv")

if not os.path.exists(FILE_PATH):
    print("Error: Cleaned data file not found. Run data_cleaning.py first.")
    exit()

df = pd.read_csv(FILE_PATH)
print(f"Data Loaded for EDA. Shape: {df.shape}")


# Visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)


# correlation heatmap

numeric_df = df.select_dtypes(include=['number'])

if not numeric_df.empty:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Matrix (Numeric Features)")
    plt.tight_layout()
    plt.show()


#  gpa distribution

if 'gpa' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df['gpa'],
        kde=True,
        color="teal"
    )
    plt.title("Distribution of Student GPA")
    plt.xlabel("GPA")
    plt.ylabel("Frequency")
    plt.show()


#  gender count graph

if 'gender' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='gender', data=df, palette='pastel')
    plt.title("Distribution of Student Gender")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()


#  subjects distribution

if 'subjects' in df.columns:
    plt.figure(figsize=(10, 5))
    order = df['subjects'].value_counts().index
    sns.countplot(
        y='subjects',
        data=df,
        order=order,
        palette="viridis"
    )
    plt.title("Student Count by Subject")
    plt.xlabel("Count")
    plt.ylabel("Subject")
    plt.show()


#  Skills Analysis

if 'skills' in df.columns:
    all_skills = []

    for skills in df['skills'].dropna():
        skill_list = [s.strip() for s in str(skills).split(',')]
        all_skills.extend(skill_list)

    skill_counts = Counter(all_skills)
    skills_df = pd.DataFrame(
        skill_counts.items(),
        columns=['Skill', 'Count']
    ).sort_values(by='Count', ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x='Count',
        y='Skill',
        data=skills_df.head(10),
        palette="Blues_r"
    )
    plt.title("Top 10 Skills Distribution")
    plt.xlabel("Count")
    plt.ylabel("Skill")
    plt.show()


#  Extracurricular Activities Analysis

if 'extracurricularactivities' in df.columns:
    activities_df = df['extracurricularactivities'].value_counts().reset_index()
    activities_df.columns = ['Activity', 'Count']

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x='Count',
        y='Activity',
        data=activities_df,
        palette="Set2"
    )
    plt.title("Extracurricular Activities Distribution")
    plt.xlabel("Count")
    plt.ylabel("Activity")
    plt.show()

 
#   Skills vs Activities Comparison

# Normalization is ONLY for visual comparison
if 'skills' in df.columns and 'extracurricularactivities' in df.columns:
    skills_top = skills_df.head(5)
    activities_top = activities_df.head(5)

    activities_normalized = (
        activities_top['Count'] *
        (skills_top['Count'].max() / activities_top['Count'].max())
    )

    x = range(len(skills_top))

    plt.figure(figsize=(10, 5))
    plt.bar(
        x,
        skills_top['Count'],
        width=0.4,
        label="Skills",
        color="steelblue"
    )
    plt.bar(
        x,
        activities_normalized,
        width=0.4,
        label="Activities (normalized)",
        color="coral"
    )
    plt.xticks(x, skills_top['Skill'], rotation=45)
    plt.title("Skills vs Extracurricular Activities Comparison")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("EDA Process Completed Successfully.")
