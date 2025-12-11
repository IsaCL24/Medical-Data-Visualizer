import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1) Import data
# The file `medical_examination.csv` should be in the project root.
df = pd.read_csv('medical_examination.csv')

# 2) Add 'overweight' column
height_m = df['height'] / 100
bmi = df['weight'] / (height_m ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3) Normalize data: make 0 good, 1 bad for 'cholesterol' and 'gluc'
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    # 4) Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=[
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'
    ])

    # 5) Group and reformat the data to split it by 'cardio' and show counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 6) Draw the catplot with seaborn
    catplot = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar', height=5, aspect=1
    )

    # 7) Get the figure for the output
    fig = catplot.fig
    return fig


def draw_heat_map():
    # 1) Clean the data
    df_heat = df.copy()
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]
    df_heat = df_heat[
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    # 2) Calculate correlation matrix
    corr = df_heat.corr()

    # 3) Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 4) Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 5) Draw the heatmap with seaborn
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.1f', center=0, vmin=-1, vmax=1,
        square=True, linewidths=.5, cbar_kws={"shrink": .5}
    )

    return fig
