# Optimizing SLA Tier Selection in Online Services Through Legacy Lifecycle Profile and Support Analysis: A Quantitative Approach 

Companion repository

## Table of Contents
- [About](#about)
- [Quick start](#quick-start)
  + [Database](#database)
  + [Data Retrieval Scripts](#data-retrieval-scripts)
  + [Data Processing Scripts](#data-processing-scripts)
  + [Filtered Data and Graphs](#filtered-data-and-graphs)
  + [Data Aggregation Scripts](#data-aggregation-scripts)
  + [Final results and visualization Scripts](#final-results-and-visualization-scripts)
  

## About
This repository contains the software support for the scientific paper: Optimizing SLA Tier Selection in Online Services Through Legacy Lifecycle Profile and Support Analysis: A Quantitative Approach.

## Quick start

Needed tools: Python, Perl, MySQL databse

### Database
- The original database that forms the basis of this work is available here: https://doi.org/10.5281/zenodo.14497695
- The database folder contains the derived data from the original database. 

| Field     | Type       | Null | Key | Default | Extra          | Meaning 
|-----------|------------|------|-----|---------|----------------| ---------
| ag_id     | int(11)    | NO   | MUL | NULL    | auto_increment |
| ag_a_id   | int(11)    | YES  | MUL | NULL    |                | User-agent ID from agent table in the parent database
| ag_day    | date       | YES  |     | NULL    |                | Date for which the aggregate visit volume is presented
| ag_count  | int(11)    | YES  |     | NULL    |                | Aggregate visit count
| ag_daycnt | int(11)    | YES  |     | NULL    |                | Number of days elapsed since the first recorded data for the user-agent
| ag_nodata | tinyint(1) | YES  |     | NULL    |                | True if no data is available for the given date (non-valid 0)
|-----------|------------|------|-----|---------|----------------|

### Data Retrieval Scripts
* They are available in the similarly named folder

The Perl scripts connect to the database and retrieve data for file processing in the required format. Please adjust the output folder as per your requirements.


### Data Processing Scripts
* They are available in the similarly named folder

The Python scripts perform the model fitting and resampling the model at the same index points. The will save the data in a data file and provide charts for visualiyation.
Please adjust the input and output folders as needed.

### Filtered Data and Graphs
* They are available in the similarly named folder

The result charts and datasets created by the data processing scripts

### Data Aggregation Scripts
* They are available in the similarly named folder

The Perl scripts aggregate the fitted data and also create weights for the weighted normalization from the original data in the databse.

###  Final results and visualization Scripts
* They are available in the similarly named folder

The Python scripts will calculate the final statistics and analysis from the aggregated data. They will also create the visualization aid.


