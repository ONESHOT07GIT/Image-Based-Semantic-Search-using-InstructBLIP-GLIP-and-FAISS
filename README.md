# Image-Based-Semantic-Search-using-InstructBLIP-GLIP-and-FAISS

# Content-Based Image Search System using CLIP and InstructBLIP

## Introduction
This report details a content-based image search system developed as part of a Computer Vision project, focusing on leveraging advanced models for image retrieval based on textual queries. The system was deployed using Streamlit, with code hosted on GitHub, and includes a detailed breakdown of member contributions.

## Project Objective
The primary goal is to develop a content-based image search system that enables users to retrieve images from a dataset using natural language queries. This system bridges the gap between text and image domains by leveraging state-of-the-art computer vision and natural language processing techniques.

## Methodology: Approaches Used
To achieve the objective, two distinct approaches were implemented:

### 1. BLIP-Based Approach
- **Model:** InstructBLIP (instruction-following vision-language model)
- **Steps:**
  - Generated image captions using InstructBLIP.
  - Converted captions to embeddings using `all-MiniLM-L6-v2` (SentenceTransformer).
  - Indexed the embeddings using FAISS (IndexFlatIP + IndexIVFFlat).
  - Text queries were encoded and matched to the top 5 image embeddings.
- **Advantage:** Leverages descriptive captioning for context-rich similarity search.

### 2. CLIP-Based Approach
- **Model:** CLIP (Contrastive Language-Image Pretraining)
- **Steps:**
  - Encoded both images and queries using CLIP’s embedding space.
  - Retrieved top 5 results based on cosine similarity.
- **Advantage:** Direct alignment of image and text in a shared space improves handling of nuanced queries.

## Results and Discussion
- **BLIP:** Good for descriptive queries but performance varied with caption quality.
- **CLIP:** Handled complex, nuanced queries more consistently due to direct text-image embedding comparison.
- **Evaluation:** Results were evaluated qualitatively. No formal metrics (e.g., mAP, precision) were computed.

## Deployment
- **Platform:** Streamlit (local deployment)
- **Instructions:**
  - Clone repo, install dependencies via `pip install -r requirements.txt`
  - Run: `streamlit run interface_blip.py` or `interface_clip.py`
  - BLIP requires precomputed `embeddings.pkl`
  - CLIP requires image folder path
- **Note:** Application is locally runnable; no public URL provided.

## Code Accessibility
- Source Code: (#) *https://github.com/ONESHOT07GIT/Image-Based-Semantic-Search-using-InstructBLIP-GLIP-and-FAISS*

## Conclusion and Future Work
The project demonstrates two effective content-based image search methods:
- CLIP for direct embedding comparison
- InstructBLIP+FAISS for caption-based similarity

### Future Improvements:
- Add benchmarking using standard datasets (e.g., COCO, Flickr)
- Expand dataset diversity
- Fine-tune InstructBLIP for domain-specific captioning
- Deploy publicly for broader access and feedback

---



