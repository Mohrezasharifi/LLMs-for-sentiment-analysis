# LLMs-for-sentiment-analysis
<img width="1505" height="693" alt="Screenshot 2025-08-06 at 01 02 06" src="https://github.com/user-attachments/assets/8bd52af3-73de-4607-bef7-c50674d34bcb" />


# Aim of the Project: 

This study aims to investigate the application of Large Language Models (LLMs) not only for sentiment analysis but also for detecting the underlying emotions and intentions expressed in text. While traditional research in this area has largely focused on identifying sentiment polarity—such as positive, negative, or neutral—this work seeks to explore deeper layers of meaning embedded in language. Understanding emotions (e.g., joy, anger, sadness) and intentions (e.g., persuasion, inquiry, command) provides a more nuanced perspective on human communication. By leveraging the contextual and generative capabilities of LLMs, this research opens new avenues for extracting richer and more actionable insights from textual data, ultimately contributing to more intelligent and empathetic human-computer interactions.

# Tech Stack:
* BART-MNLI in a ZeroShot setting for LLM annotation
* RoBERTa for sentiment analysis
* DistiledBERT
* GPT
* Tensorflow
* Pytorch
* Python Programming Language

# Datasets:
* GoEmotions

<img width="514" height="784" alt="dataused" src="https://github.com/user-attachments/assets/06f7df1f-c193-4f13-b01a-768a71ed97dd" />


* Ekman's Basic Emotions and Plutchik's wheel of emotions
<img width="327" height="154" alt="image" src="https://github.com/user-attachments/assets/dd7d91b5-724b-4726-a91b-c4f53cc88861" />


<img width="384" height="131" alt="image" src="https://github.com/user-attachments/assets/14194c45-638c-47e2-b3d9-58fe4d3cbf4e" />

source: https://www.researchgate.net/figure/Ekmans-six-basic-emotions-and-Plutchiks-wheel-of-emotions-the-middle-circle-contains-8_fig1_346179935

source: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/

# Large Language Models Applied: 

<img width="984" height="323" alt="Screenshot 2025-08-15 at 22 18 09" src="https://github.com/user-attachments/assets/17849ce5-ed28-4101-af34-2d082fd6d107" />


# Results: 

Finetuned RoBERTa on the data for Sentiment Analysis on five classes from Very Negative to Very Negative:


<img width="1166" height="381" alt="Screenshot 2025-08-15 at 22 14 24" src="https://github.com/user-attachments/assets/0db6fc07-1220-4a79-bee3-25c982bb41cc" />


Finetuned DistilBERT on GoEmotions Taxonomy:

<img width="953" height="702" alt="Screenshot 2025-08-15 at 22 16 19" src="https://github.com/user-attachments/assets/9cf5dfb7-cc15-4c5d-9db9-a6b69368e17d" />

Beyond quantitative gains, this research underscores the value of interdisciplinary annotation and careful methodology in producing reproducible results. Future work should focus on scaling annotation, exploring ordinal and multi-label classification, and applying active learning to improve efficiency. By combining advanced NLP techniques with domain expertise, this study offers a reproducible framework for extracting actionable insights from EV drivers’ experiences with public charging stations, with applications in infrastructure planning, customer service, and sustainable mobility policy.

# Demo: 
https://698e13ad37f2f10770.gradio.live

#Contact

For further clarification or inqueries, get in touch with through m.r.sharif2@ncl.ac.uk or rezatayeb2017@gmail.com
