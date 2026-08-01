Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge bases. Despite its widespread adoption, standard RAG architectures frequently fail when retrieved contexts are noisy, incomplete, or irrelevant.
When retrieval fails, conventional corrective strategies apply a static countermeasure (e.g., blanket web search or uniform re-querying) regardless of why the failure occurred. This uniform treatment incurs unnecessary computational overhead and frequently fails to resolve the underlying retrieval issue.
To address these limitations, we propose a diagnosis-driven retrieval framework that dynamically evaluates context sufficiency, classifies retrieval failure into discrete modes, and executes custom-tailored adaptive actions.

### System Architecture
<img width="896" height="1193" alt="Overall-architecture" src="https://github.com/user-attachments/assets/b376bd81-191a-46b7-b8ed-bbb0028437a3" />
