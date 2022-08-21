import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('nobel_prize_data.csv')

# challenge 1
df.shape  # how many rows and columns
df.info()  # column names and data type
df.sort_values('year').head(1)  # first year
df.sort_values('year', ascending=False).head(1)  # latest year in the dataset

# challenge 2
df.duplicated().values.any()  # Are there any duplicate values in the dataset?
df.isna().values.any()  # Are there any NaN values in the dataset?
df.isna().sum()  # Find a count of NaN's per column?

# Why are there so many NaN values?
col_subset = ['year', 'category', 'laureate_type', 'birth_date', 'full_name', 'organization_name']
df.loc[df.birth_date.isna()][col_subset]

df.birth_date = pd.to_datetime(df.birth_date)  # converting to datetime object

# adding a new column to show prize share as percentage
seperated_values = df.prize_share.str.split('/', expand=True)
numerator = pd.to_numeric(seperated_values[0])
denomenator = pd.to_numeric(seperated_values[1])
df['share_pct'] = numerator / denomenator
df.info()

biology = df.sex.value_counts()  # removes NaN's from count
fig = px.pie(df, values=biology.values, names=biology.index, title='Percentage of Male vs Female Winners',
             labels=biology.index, hole=0.4)
fig.update_traces(textposition='inside', textfont_size=15, textinfo='percent')
fig.show()

# First 3 Female Nobel Laureates
a = df.query("`sex` == 'Female'").sort_values('year', ascending=True)[:3]
a

# Did some people win a Nobel prize more than once? If so, who?
is_winner = df.duplicated(subset=['full_name'], keep=False)
multiple_winners = df[is_winner]
multiple_winners.nunique()  # 6 unique full names
colm_subset = ['year', 'category', 'laureate_type', 'full_name']
multiple_winners[colm_subset]

df.category.nunique()  # find number of unique categories

prizes_per_catg = df.category.value_counts()

v_bar = px.bar(y=prizes_per_catg.values, x=prizes_per_catg.index, color=prizes_per_catg.values,
               color_continuous_scale='Aggrnyl', title='Number of Prizes Awarded per Category')
v_bar.update_layout(xaxis_title='Noble Prize Category', coloraxis_showscale=False, yaxis_title='Number of Prizes')
v_bar.show()

prize_per_year = df.groupby(by='year').count().prize

moving_average = prize_per_year.rolling(window=5).mean()

plt.figure(figsize=(16, 8), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(ticks=np.arange(1900, 2021, step=5),
           fontsize=14,
           rotation=45)

ax = plt.gca()  # get current axis
ax.set_xlim(1900, 2020)

ax.scatter(x=prize_per_year.index,
           y=prize_per_year.values,
           c='dodgerblue',
           alpha=0.7,
           s=100, )

ax.plot(prize_per_year.index,
        moving_average.values,
        c='crimson',
        linewidth=3, )

plt.show()

yearly_avg_share = df.groupby(by='year').agg({'share_pct': pd.Series.mean})
share_moving_average = yearly_avg_share.rolling(window=5).mean()

plt.figure(figsize=(16, 8), dpi=200)
plt.title('Number of Nobel Prizes Awarded per Year', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(ticks=np.arange(1900, 2021, step=5),
           fontsize=14,
           rotation=45)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.set_xlim(1900, 2020)

# Can invert axis
ax2.invert_yaxis()

ax1.scatter(x=prize_per_year.index,
            y=prize_per_year.values,
            c='dodgerblue',
            alpha=0.7,
            s=100, )

ax1.plot(prize_per_year.index,
         moving_average.values,
         c='crimson',
         linewidth=3, )

ax2.plot(prize_per_year.index,
         share_moving_average.values,
         c='grey',
         linewidth=3, )

plt.show()

top_countries = df.groupby(['birth_country_current'], as_index=False).agg({'prize': pd.Series.count})
top_countries.sort_values(by='prize', ascending=False, inplace=True)
top20_countries = top_countries[:20]

h_bar = px.bar(
    x=top20_countries.prize,
    y=top20_countries.birth_country_current,
    orientation='h',
    color=top20_countries.prize,
    color_continuous_scale='Viridis',
    title='Top 20 Countries by Number of Prizes'
)

h_bar.update_layout(xaxis_title='Number of Prizes',
                    yaxis_title='Country',
                    coloraxis_showscale=False
                    )
h_bar.show()

df_countries = df.groupby(['birth_country_current', 'ISO'],
                          as_index=False).agg({'prize': pd.Series.count})
df_countries.sort_values('prize', ascending=False)

world_map = px.choropleth(df_countries,
                          locations='ISO',
                          color='prize',
                          hover_name='birth_country_current',
                          color_continuous_scale=px.colors.sequential.matter)

world_map.update_layout(coloraxis_showscale=True, )

world_map.show()

cat_country = df.groupby(['birth_country_current', 'category'], as_index=False).agg({'prize': pd.Series.count})
cat_country.sort_values(by='prize', ascending=False, inplace=True)

merged_df = pd.merge(cat_country, top20_countries, on='birth_country_current')
merged_df.columns = ['birth_country_current', 'category', 'cat_prize', 'total_prize']
merged_df.sort_values(by='total_prize', inplace=True, ascending=False)

cat_cntry_bar = px.bar(x=merged_df.cat_prize,
                       y=merged_df.birth_country_current,
                       color=merged_df.category,
                       orientation='h',
                       title='Top 20 Countries by Number of Prizes and Category')

cat_cntry_bar.update_layout(xaxis_title='Number of Prizes',
                            yaxis_title='Country')
cat_cntry_bar.show()

prize_by_year = df.groupby(['birth_country_current', 'year'], as_index=False).count()
prize_by_year = prize_by_year.sort_values('year')[['year', 'birth_country_current', 'prize']]

cumulative_prizes = prize_by_year.groupby(['birth_country_current', 'year']).sum().groupby(level=[0]).cumsum()
cumulative_prizes.reset_index(inplace=True)

l_chart = px.line(cumulative_prizes,
                  x='year',
                  y='prize',
                  color='birth_country_current',
                  hover_name='birth_country_current')

l_chart.update_layout(xaxis_title='Year',
                      yaxis_title='Number of Prizes')
l_chart.show()

top20_organizations = df.groupby(['organization_name'], as_index=False).agg({'prize': pd.Series.count})
top20_organizations.sort_values(by='prize', ascending=False, inplace=True)
top20_organizations = top20_organizations[:20]

hh_bar = px.bar(
    x=top20_organizations.prize,
    y=top20_organizations.organization_name,
    orientation='h',
    color=top20_organizations.prize,
    color_continuous_scale='Viridis',
    title='Top 20 Organizations by Number of Prizes'
)

hh_bar.update_layout(xaxis_title='Number of Prizes',
                     yaxis_title='Organization',
                     coloraxis_showscale=False
                     )
hh_bar.show()

top_cities = df.groupby(['organization_city'], as_index=False).agg({'prize': pd.Series.count})
top_cities.sort_values(by='prize', ascending=False, inplace=True)
top20_cities = top_cities[:20]

city_bar = px.bar(
    x=top20_cities.prize,
    y=top20_cities.organization_city,
    orientation='h',
    color=top20_cities.prize,
    color_continuous_scale='Plasma',
    title='Top 20 Org Cities by Number of Prizes'
)

city_bar.update_layout(xaxis_title='Number of Prizes',
                       yaxis_title='Org_City',
                       coloraxis_showscale=False
                       )
city_bar.show()

top_birth_cities = df.groupby(['birth_city'], as_index=False).agg({'prize': pd.Series.count})
top_birth_cities.sort_values(by='prize', ascending=False, inplace=True)
top20_birth_cities = top_birth_cities[:20]

city_bar2 = px.bar(
    x=top20_birth_cities.prize,
    y=top20_birth_cities.birth_city,
    orientation='h',
    color=top20_birth_cities.prize,
    color_continuous_scale='Plasma',
    title='Top 20 Birth Cities by Number of Prizes'
)

city_bar2.update_layout(xaxis_title='Number of Prizes',
                        yaxis_title='Birth City',
                        coloraxis_showscale=False
                        )
city_bar2.show()

country_city_org = df.groupby(['organization_country', 'organization_city', 'organization_name'], as_index=False).agg(
    {'prize': pd.Series.count})
country_city_org = country_city_org.sort_values('prize', ascending=False)

burst = px.sunburst(country_city_org,
                    path=['organization_country', 'organization_city', 'organization_name'],
                    values='prize',
                    title='Where do Discoveries Take Place?', )
burst.update_layout(xaxis_title='Number of Prizes',
                    yaxis_title='City',
                    coloraxis_showscale=False)
burst.show()

birth_year = df['birth_date'].dt.year
df['winning_age'] = df.year - birth_year

display(df.nlargest(n=1, columns='winning_age'))
display(df.nsmallest(n=1, columns='winning_age'))

df.winning_age.describe()

plt.figure(figsize=(8, 4), dpi=200)
sns.histplot(data=df,
             x=df.winning_age,
             bins=30)
plt.xlabel('Age')
plt.title('Distribution of Age on Receipt of Prize')
plt.show()

plt.figure(figsize=(8, 4), dpi=200)
with sns.axes_style("whitegrid"):
    sns.regplot(data=df,
                x='year',
                y='winning_age',
                lowess=True,
                scatter_kws={'alpha': 0.4},
                line_kws={'color': 'black'})
plt.show()

plt.figure(figsize=(8, 4), dpi=200)
with sns.axes_style("whitegrid"):
    sns.boxplot(data=df,
                x='category',
                y='winning_age',
                )
plt.show()

plt.figure(figsize=(8, 4), dpi=200)
with sns.axes_style("whitegrid"):
    sns.lmplot(data=df,
               x='year',
               y='winning_age',
               hue='category',
               lowess=True,
               aspect=2,
               scatter_kws={'alpha': 0.5},
               line_kws={'linewidth': 5})
plt.show()
