#CS 4412 Data Mining Project: 
#Project title: Discovering Music Patterns in Spotify Data
Author: Name: Vu Le
Email: vle28@students.ksu.edu
Course: CS 4412 - Data Mining
Semester: Spring 2026
Project Description: Thís project ís about using data mining techniques to find pattern in Spopify data music
Find song groups - Group similar songs by clutering
Discover feature patterns - Find which audio features appear together by association rules
Extract genre rules - Create simple rules that describe music genres by decision trees

#Dataset

Source
Spotify Tracks Dataset from Kaggle
DIRECT LINK: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Dataset Info

Size: 170,000 songs
Columns: 20 attributes
Format: CSV file
Content: Songs with audio features like energy, danceability, tempo

Important Features

Feature

energy How intense the song is (0-1)
danceability How easy to dance to (0-1)
tempo Speed in beats per minute
valence Happy vs sad (0-1)
loudness How loud in decibels
acousticness Acoustic vs electronic (0-1)
track_genreGenre label (rock, pop, etc.)

cs4412-project/
├── README.md                    # This file
├── docs/
│   └── proposal.pdf            # LaTeX proposal
├── data/
│   ├── raw/                    # Raw Spotify data (download from Kaggle)
│   └── processed/              # Cleaned data
├── notebooks/
│   ├── 1_explore_data.ipynb   # Look at the data
│   ├── 2_clustering.ipynb     # K-Means clustering
│   ├── 3_associations.ipynb   # Apriori rules
│   └── 4_classification.ipynb # Decision trees
├── src/
│   ├── clean_data.py          # Data cleaning
│   ├── clustering.py          # Clustering code
│   ├── associations.py        # Association rules code
│   └── classification.py      # Decision tree code
├── results/
│   ├── clusters.csv           # Cluster results
│   ├── rules.csv              # Association rules
│   └── figures/               # Charts and plots
└── requirements.txt           # Python packages needed

Installation
Python 3.8 or newer

Get the Data

Go to: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Click "Download" (you need a free Kaggle account)
Put the CSV file in data/raw/ folder

#Discovery Questions

1. What natural groups of songs exist?

Use K-Means clustering
Group songs by energy, danceability, tempo
Find 5-7 song types

2. Which audio features occur together?

Use Apriori 
Find patterns like "High Energy + High Loudness"

3. What rules describe different genres?

Use Decision Trees
Extract simple rules like "IF energy > 0.8 THEN Electronic"
Make rules easy to understand

#How to Run
Run Clustering: python src/clustering.py
Find Association Rules: python src/associations.py
Extract Genre Rules: python src/classification.py
