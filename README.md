# Decoding Bias: Exploring Sexism in Software Development through Online Narratives and AI Analysis

This repository contains the ongoing development of the source code for the paper "Decoding Bias: Exploring Sexism in Software Development through Online Narratives and AI Analysis"

The structure of the repository is as follows:

## Index
1. Archive
2. In Progress
3. Clean text data files
4. Data files
5. LICENSE
6. Subreddit Data Extraction
7. Semantic Similarity
   - Women Software Developers' Experiences of Sexism
   - Taxonomy Definitions

## Useage
To use the paper's source code, begin by modifying the Subreddit Data Extraction script with your Reddit account's information.
<pre>
  <code>reddit = praw.Reddit(
    client_id="",
    client_secret="",
    password="",
    user_agent="",
    username=""
    )</code> 
</pre>

Once the data extraction process is complete:
1. Run the Semantic Similarity - Women Software Developers' Experiences of Sexism script to locate potential data points containing narratives of sexist experiences.
2. Based on the cosine distribution of the previous script, set an appropriate cosine distance threshold in the Semantic Similarity - Taxonomy Definition script. Then, runt the script to classify the data points using the taxonomy definitions.
