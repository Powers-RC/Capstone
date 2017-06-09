
Black-tailed Boundaries
=======================


 Table of contents
 -----------------
1. Project Motivation
2. Data
3. EDA
4. Data Collection
5. Data Processing
6. Model Selection


Motivation
----------
This project analyzes prairie dog colony data from the Boulder Open Data project in an attempt to identify prairie dog colonies via aerial images, clustering the different prairie dog mounds together and predicting the perimeter and areas based on these clusters. I have always been fascinated by Ecology and ecosystems so when presented with an opportunity to work with prairie dog data I could not pass up the opportunity. If you did not know prairie dogs are keystone species and play a vital role in maintaining North American grasslands ecosystems[1]. The goal of this project is to tackle a problem that could benefit Ecologist by eliminating the need to do lengthly ground surveys, saving time to allow for other areas of focus without the loss of important annual data.

Data
----
Each fall Ecologists with Open Space Mountain Parks map the perimeter of each colony which provides data on the extent of the colony. Both the CSV and KMZ files were use to obtain annual colony and spatial data.

The Boulder prairie dog data contains:
- 21 years of data from 1996 to 2016
- 146 different areas as of 2016
- Geospatial data for each of the areas
- Measurements of the perimeter and area for the associated colonies

![areas image](images/area_ss.png)


EDA
---
I did minimal exploratory data analysis on the data set because my project is more concerned with predicting the areas of these areas via aerial imagery. I did look at the growth of colony tracking by Boulder and OSMP over time and it appears that since this project began the areas being tracked have increase by over 700%. If I have enough time I would like to look at area/perimeter fluctuation over the 21 years.

![Annual area numbers](images/colony_growth.png)


Data Collection
---------------
In order to identify prairie dog mounds via aerial images....I need aerial imagery. So I developed a program using Selenium that would scrape Google Earth by navigating to a URL that contained specific coordinates and zoom level. Once the page was fully loaded the program would take a screen shot of the location, this process was repeated as necessary.  

![GE gif](images/GE_preview.gif)

[1] http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075229
