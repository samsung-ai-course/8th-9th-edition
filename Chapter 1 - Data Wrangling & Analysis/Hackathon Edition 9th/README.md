# Music Dataset Analysis Hackathon

[**Hackathon Notebook notebook**   ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsung-ai-course/8th-9th-edition/blob/main/Chapter%201%20-%20Data%20Wrangling%20%26%20Analysis/Hackathon%20Edition%209th/notebook.ipynb)

## Guia & Sugestão de como a equipa se organizar

[Guia](https://github.com/samsung-ai-course/8th-9th-edition/blob/main/Chapter%201%20-%20Data%20Wrangling%20&%20Analysis/Hackathon%20Edition%209th/team-guidelines.md)

## Overview

Welcome to the Music Dataset Analysis Hackathon! This challenge will test your data analysis skills using real-world music streaming and social media data. Your goal is to extract meaningful insights that could inform business decisions in the music industry.

## Dataset Description

You'll be working with a comprehensive dataset containing **26 variables** for songs from various artists worldwide, combining:
- Spotify streaming statistics and audio features
- YouTube video performance metrics

**Data Collection Date**: February 7th, 2023

### Available Data Categories

**Basic Information**: Track details, artist names, album information, and Spotify URIs

**Audio Features**: Danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and duration

**Performance Metrics**: Spotify streams, YouTube views, likes, and comments

## Your Mission

This hackathon is designed to help you develop essential data analysis skills:

1. **Ask the right questions** - Learn to formulate meaningful analytical questions
2. **Explore and understand** - Discover patterns and relationships in the data
3. **Generate insights** - Extract findings that have real business value
4. **Communicate effectively** - Present your discoveries in a clear, compelling way

## Getting Started: Understanding Your Data

Before diving into complex analysis, start with these fundamental questions:

### Data Quality & Structure
- What is the size of the dataset? How many songs and artists are included?
- Are there missing values? Which variables have incomplete data?
- What are the data types of each variable?
- Are there any obvious outliers or anomalies?

### Descriptive Statistics
- What are the distributions of key numerical variables (streams, views, likes)?
- What are the ranges and averages for audio features?
- Which artists and tracks appear most frequently?

## Suggested Analysis Directions

### 1. **Performance Analysis**

Explore what makes a song successful on different platforms:

- Which songs have the highest streams on Spotify? The most views on YouTube?
- Is there a correlation between Spotify streams and YouTube views?
- Do songs perform differently across platforms?
- What's the relationship between likes/comments and views on YouTube?

### 2. **Audio Features & Popularity**

Investigate which musical characteristics drive engagement:

- Do more danceable songs get more streams?
- Is there a relationship between energy levels and popularity?
- How does valence (positiveness) affect listener engagement?
- Are acoustic songs more or less popular than produced tracks?
- Does tempo influence streaming numbers?

### 3. **Genre & Style Patterns**

Identify trends in musical characteristics:

- What are the most common keys for popular songs?
- How do audio features cluster together? (e.g., high energy + high danceability)
- Can you identify distinct musical "profiles" or genres based on audio features?
- How do instrumentalness and speechiness vary across different types of content?

### 4. **Artist & Album Insights**

Understand artist performance and strategy:

- Which artists have the most songs in the dataset?
- Do certain artists consistently produce songs with similar audio features?
- Is there a difference in performance between singles and album tracks?
- Which artists have the best engagement rates (views per stream, likes per view)?

### 5. **Video Content Strategy**

Analyze YouTube-specific factors:

- Does having an official video impact streaming numbers?
- How does licensing status affect video performance?
- What's the relationship between video title/description and engagement?
- Which channels are most successful at promoting music?

### 6. **Temporal Patterns**

Consider timing and duration effects:

- Is there an optimal song duration for popularity?
- How does song length relate to streams and views?
- Are there patterns in when popular songs were released?

## Business-Relevant Questions to Explore

Think like a music industry professional. Your insights could help:

### For Artists & Producers
- What audio characteristics should new artists focus on to maximize reach?
- Should artists prioritize certain platforms over others?
- What's the optimal balance between artistic features (valence, acousticness) and commercial appeal?

### For Record Labels
- Which types of songs should be promoted as singles vs. album tracks?
- How important is the official music video in driving overall popularity?
- What factors predict a song's cross-platform success?

### For Marketing Teams
- What characteristics of high-performing songs can inform promotional strategies?
- How can we identify songs with viral potential early?
- What's the relationship between different engagement metrics?

### For Streaming Platforms
- What audio features do listeners engage with most?
- How can we improve recommendation algorithms based on these insights?
- Are there underserved musical niches with high engagement potential?

## Deliverables

Your analysis should include:

1. **Exploratory Data Analysis (EDA)**
   - Data cleaning and preparation steps
   - Summary statistics and visualizations
   - Initial observations about data quality and distributions

2. **In-Depth Analysis**
   - At least 3-5 focused analytical questions with answers
   - Appropriate statistical methods (correlations, regressions, clustering, etc.)
   - Clear visualizations supporting your findings

3. **Business Insights**
   - Actionable recommendations based on your analysis
   - Potential implications for different stakeholders
   - Suggestions for further investigation

4. **Presentation**
   - Clear narrative connecting your questions, analysis, and insights
   - Professional visualizations
   - Summary of key findings and recommendations

## Technical Recommendations

### Tools You Might Use
- pandas, matplotlib, seaborn (you haven't learnt this one yet but it's also a visualization library and chatgpt/claude can help you produce some visualization with it)

## Evaluation Criteria

Your work will be assessed on:

- **Question Quality** (25%): Are your questions meaningful, specific, and relevant?
- **Analytical Rigor** (25%): Did you use appropriate methods? Are your conclusions supported?
- **Insight Value** (25%): Do your findings provide actionable business value?
- **Communication** (25%): Are your findings clear, well-visualized, and compelling?

## Tips for Success

✅ **Start simple**: Begin with basic questions before moving to complex analyses

✅ **Visualize early and often**: Plots help you understand data and communicate findings

✅ **Think critically**: Correlation doesn't imply causation. Consider confounding factors

✅ **Be creative**: The best insights often come from unexpected questions

✅ **Tell a story**: Connect your analyses into a coherent narrative

✅ **Consider limitations**: This data is from February 2023. How might this affect your conclusions?

✅ **Iterate**: Findings from one analysis should inspire new questions

## Important Notes

⚠️ **Data Timestamp**: Remember that all metrics are from February 7th, 2023. Popular songs and trends may have evolved since then.

⚠️ **Causation vs. Correlation**: Be careful about claiming causal relationships. Most of your findings will be correlational.

⚠️ **Context Matters**: Consider external factors (marketing budgets, artist popularity, cultural moments) that aren't captured in the data.

## Resources

- Spotify Audio Features Documentation: Research what each audio feature means
- Music Industry Reports: Understanding current trends and business models
- Data Visualization Best Practices: Creating clear, impactful charts
- Statistical Methods Guides: Choosing the right analytical approach

---

**Good luck, and happy analyzing! We're excited to see what insights you discover.**

*Remember: The best data analysts don't just answer questions—they ask better ones.*

## The Dataset

A comprehensive dataset of songs from various artists worldwide, containing Spotify statistics and YouTube metrics for each track.

### Overview

This dataset combines music streaming data from Spotify with YouTube video performance metrics. Data was collected on **February 7th, 2023**.

### Dataset Contents

The dataset includes **26 variables** for each song:

#### Basic Information

- **Track**: Name of the song as it appears on Spotify
- **Artist**: Name of the artist
- **Url_spotify**: URL of the artist on Spotify
- **Album**: The album containing the song
- **Album_type**: Indicates if the song is released as a single or part of an album
- **Uri**: Spotify link used to find the song through the API

#### Audio Features

- **Danceability**: Suitability for dancing based on tempo, rhythm stability, beat strength, and regularity
  - Scale: 0.0 (least danceable) to 1.0 (most danceable)

- **Energy**: Perceptual measure of intensity and activity
  - Scale: 0.0 to 1.0
  - High energy: Fast, loud, and noisy (e.g., death metal)
  - Low energy: Calm and quiet (e.g., Bach prelude)
  - Factors: Dynamic range, perceived loudness, timbre, onset rate, and general entropy

- **Key**: The musical key of the track
  - Uses standard Pitch Class notation: 0 = C, 1 = C♯/D♭, 2 = D, etc.
  - Value of -1 indicates no key detected

- **Loudness**: Overall loudness of the track in decibels (dB)
  - Averaged across the entire track
  - Typical range: -60 to 0 dB

- **Speechiness**: Presence of spoken words in the track
  - **Above 0.66**: Probably made entirely of spoken words (e.g., talk shows, audiobooks, poetry)
  - **0.33 to 0.66**: May contain both music and speech (e.g., rap music)
  - **Below 0.33**: Most likely music and non-speech-like tracks

- **Acousticness**: Confidence measure of whether the track is acoustic
  - Scale: 0.0 to 1.0
  - 1.0 represents high confidence the track is acoustic

- **Instrumentalness**: Predicts whether a track contains no vocals
  - "Ooh" and "aah" sounds are treated as instrumental
  - **Above 0.5**: Likely instrumental tracks
  - **Closer to 1.0**: Higher confidence of no vocal content

- **Liveness**: Detects the presence of an audience in the recording
  - **Above 0.8**: Strong likelihood the track is live

- **Valence**: Musical positiveness conveyed by the track
  - Scale: 0.0 to 1.0
  - **High valence**: Positive emotions (happy, cheerful, euphoric)
  - **Low valence**: Negative emotions (sad, depressed, angry)

- **Tempo**: Overall estimated tempo in beats per minute (BPM)
  - Represents the speed or pace of the piece

- **Duration_ms**: Duration of the track in milliseconds

### Spotify Metrics

- **Stream**: Number of streams on Spotify

### YouTube Metrics

- **Url_youtube**: URL of the video linked to the song on YouTube (if available)
- **Title**: Title of the video clip on YouTube
- **Channel**: Name of the channel that published the video
- **Views**: Number of views
- **Likes**: Number of likes
- **Comments**: Number of comments
- **Description**: Description of the video on YouTube
- **Licensed**: Indicates whether the video represents licensed content uploaded to a channel linked to a YouTube content partner
- **official_video**: Boolean value indicating if the video found is the official music video

### Important Notes

⚠️ **Time-Dependent Data**: These statistics are heavily dependent on the collection date (**February 7th, 2023**). All metrics reflect values as of this date and will differ from current statistics.
