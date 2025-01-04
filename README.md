This project involves creating an AI expert over a codebase using Retrieval-Augmented Generation (RAG). Implementing the ability to chat with a codebase so you can understand how they work and what can be improved.
ct, you will learn how to embed the contents of a codebase, insert them into a vector database called Pinecone, and then get answers to your queries based on the contents of the codebase using LLMs.

The submission for this project is a web app where you can chat with a codebase.

-build the web app only using Python, see the [Streamlit documentation](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) for a guide on how to build a chatbot.
-If you want to use React and Next.js, see the AI Chatbot template on [Vercel](https://vercel.com/templates/next.js/nextjs-ai-chatbot)
-See an example of a web app that does this [here](https://sage.storia.ai/) and check out a recording of our [RAG workshop](https://app.headstarter.co/content/accelerator/recordings/rag-workshop).

#### Here are some additional challenges for this project if you are finished early:
- Add support for image uploads when chatting with the codebase - this is called Multimodal RAG.
- Add a way to select different codebases to chat with.
- Add a way to update the Pinecone index when you push any new commits to your repo. This would be done through a webhook that's triggered on each commit, where the codebase is re-embedded and added to Pinecone.
- Add a way to chat with multiple codebases at the same time.

### References

  - [Inspiration](https://sage.storia.ai/)
  - [RAG on Codebase](https://blog.lancedb.com/rag-codebase-1/)
  - [Enbeddings](https://huggingface.co/blog/getting-started-with-embeddings)
  - [Embedding model leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
  - [How Greptile does Codebase RAG](https://huggingface.co/spaces/mteb/leaderboard)
  - [RAG for codebase with 10k Repos](https://www.qodo.ai/blog/rag-for-large-scale-code-repos/)
  - [How embeddings are generated?](https://arxiv.org/abs/1301.3781)
