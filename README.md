
# imbed

Tools to work with embeddings, easily an flexibily.

Note: Work in progress...

To install:	```pip install imbed```

As we all know, though RAG (Retrieval Augumented Generation) is hyper-popular at the moment, the R part, though around for decades 
(mainly under the names "information retrieval" (IR), "search", "indexing",...), has a lot to contribute towards the success, or failure, of the effort.
The [many characteristics of the retrieval part](https://arxiv.org/abs/2312.10997) need to be tuned to align with the final generation and business objectives. 
There's still a lot of science to do. 

So the last thing we want is to be slowed down by pedestrian aspects of the process. 
We want to be agile in getting data prepared and analyzed, so we spend more time doign science, and iterate our models quickly.

There are two major aspects the `imbed` wishes to contribute two that.
* search: getting from raw data to an iterface where we can search the information effectively
* visualize: exploring the data visually (which requires yet another kind of embedding, to 2D or 3D vectors)

What we're looking for here is a setup where with minimal **configuration** (not code), we can make pipelines where we can point to the original data, enter a few parameters, 
wait, and get a "search controller" (that is, an object that has all the methods we need to do retrieval stuff). Here's an example of the kind of interface we'd like to target.

```python
raw_docs = mk_text_store(doc_src_uri)  # the store used will depend on the source and format of where the docs are stored
segments = mk_segments_store(raw_docs, ...)  # will not copy any data over, but will give a key-value view of chunked (split) docs
search_ctrl = mk_search_controller(vectorDB, embedder, ...)
search_ctrl.fit(segments, doc_src_uri, ...)
search_ctrl.save(...)
```



