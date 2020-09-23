# Assignment 1. Report.

**Team:** lagos
**Participants:** Maxim Evgrafov (*@maevgor*), Alexandr Grichshenko (*@AlexGr99*).
**GitHub repository:** https://github.com/MaEvGoR/BD_hadoop_ass1

## Introduction.

In this assigmnment we were tasked with designing a simple search engine using MapReduce paradigm. The project was implemented in Java programming language and tested on university hadoop cluster (Wikipedia files) as well as locally during development process. In the remaining part of the report we will describe the pipeline of MapReduce jobs we used to obtained relevant documents and demonstrate them in action.

## Search Engine implementation.

Generally, we followed the model presented in the assignment description with a few alternations to address specific implementation issues. 

![Alt text](ranker_dir/Screenshot from 2020-09-23 07-37-54.png?raw=true "Title")

We decided to split the workload the following way: Maxim was developing the *Indexing Engine*, while Alexandr worked on the *Ranking Engine*. We reckon that such approach is logical since both parts can be tested independently on a smaller text corpus. The final testing on Wikipedia files combining both parts was performed by Maxim.

### Indexing Engine implementation.

#### Inputs to the Indexing Engine
The inputs to the engine are:
* Directory with WikiDescription files in format
```python
{"id": wiki_id, "title": wiki_title, "url": wiki_url, "text": wiki_text}
```
* Path to the output directory (should not be already created)

> hadoop jar Indexer.jar Indexer EnWikiSmall wiki_small_out

#### Indexing Engine Implementation



### Ranking Engine implementation.

The diagram below presents an outline of the architecture of the ranking engine.

######### IMAGE #############

#### Inputs to the Ranking Engine.
The inputs to the engine are:
* Vocabulary with word IDs and words themselves
* Files storing TF/IDF weights for each word and document
* List of documents with document IDs titles, URL and length
* Query text
* Number of relevant files (N) to be returned by the engine

#### Jobs description.

As seen on the diagram above the operation of the engine relies on 4 distinct MapReduce jobs that compute the Okapi BM25 relevance function. 

1. *Query Vectorizor* refers to the Vocabulary and Query text to save only the IDs of the words present in the query. Only Mapper is used in this job.
2. *Document Lengths Reader* reads the list of documents and saves their lengths in memory alongside doc IDs to be used later to compute the relevance function. Only Mapper is used in this job.
3. *Relevance Function Calculator* considers the TF/IDF weights as well as document lengths to assign relevance to each document. Mapper calculates a part of the function related to a single word and Reducer sums up the result.
4. *Document Ranker* uses the output from the previous job and returns titles of N documents with the highest relevance.


## Results.

Screenshots of runs on the cluster and short descriptions.


## References.

[1] Stephen Robertson & Hugo Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond". Foundations and Trends in Information Retrieval.
