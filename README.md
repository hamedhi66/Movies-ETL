# Movies-ETL

**ASSUMPTIONS:**

- Most of data is parsed to a correct format. Data cleaning was performed to the extent that made sense in regards to time. Further cleaning could be performed, but did not worth the time. We already considered more than 90% of the data. 

- Currencies are not converted to dollar. Therefor rows which has currency other than dollar are dropped. 

- "wmr", "kmf" and "rf" are considered to call for "wiki_file", "kaggle_metadata_file" and "ratings_file" respectively. 
- For data cleaning:
  - Null column are removed
  - Element-wise negation operator "~" is used to avoid value error  when comparing values
- For merging language data between Wikipedia and Kaggle Map, we used Kaggle Map as the data was more organized. Wikipedia had more information but took more time to clean and adjust.

