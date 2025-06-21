# LLM.int8 vs Naive 8-bit Weight Quantization Using Pytorch

Large Language Models (LLMs) are compute-hungry beasts. Their size = number of parameters × precision. To reduce memory and accelerate inference, I explored quantization techniques, compressing weights from FP32 to INT8.

I ran two from-scratch methods:
• Absmax (symmetric) quantization
• Zeropoint (asymmetric) quantization

Both reduced memory significantly… but at a cost: higher perplexity, due to sensitivity to outliers.

🔍 𝗧𝗵𝗲 𝗢𝘂𝘁𝗹𝗶𝗲𝗿 𝗣𝗿𝗼𝗯𝗹𝗲𝗺
Outliers, extreme values (negative or positive), are common in transformer layers. Though rare, they skew quantization and can hurt precision. But removing them outright degrades performance.

✅ 𝗦𝗼𝗹𝘂𝘁𝗶𝗼𝗻: 𝗟𝗟𝗠.𝗶𝗻𝘁𝟴() (Bitsandbytes)
This method applies vector-wise quantization + mixed precision:
• Most weights → INT8
• Outliers (~0.1%) → FP16

The result: better accuracy with 2.9× smaller memory.

📦 𝗠𝗲𝗺𝗼𝗿𝘆 𝗙𝗼𝗼𝘁𝗽𝗿𝗶𝗻𝘁
• Original (FP16): 510 MB
• INT8: 176 MB

🧠 Pro tip: Mixed INT8+FP16 handles outliers without effecting model performance — ideal for real-world LLM deployments.

References: 
- T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. 2022](https://arxiv.org/abs/2208.07339).
- Y. Beldaka, and T. Dettmers, [A Gentle Introduction to 8-bit Matrix Multiplication](https://huggingface.co/blog/hf-bitsandbytes-integration), Hugging Face Blog (2022).
- A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, [A Survey of Quantization Methods for Efficient Neural Network Inference. 2021](https://arxiv.org/abs/2103.13630).
- H. Wu, P. Judd, X. Zhang, M. Isaev, and P. Micikevicius, [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation. 2020](https://arxiv.org/abs/2004.09602).
- Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/), Lil’Log (2023).
- Kamil Czarnogorski, [Local Large Language Models, Int8](https://int8.io/local-large-language-models-beginners-guide/) (2023).
