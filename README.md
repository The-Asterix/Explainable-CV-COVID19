# Explainable-CV-COVID19
Includes a ResNet-18 COVID-19 classifier and Grad-CAM explainability metrics.

# Explainable CV for COVID-19 X-Rays

### What this does
I built an image classification pipeline to sort chest X-rays into three categories: COVID-19, Normal, and Viral Pneumonia. I used a pre-trained ResNet-18 model and fine-tuned it on the COVID-19 Radiography Database from Kaggle. 

Since this is for medical imaging, raw accuracy isn't enough—the model's decisions need to make sense. To test this, I implemented a post-hoc explainability pipeline:
* **Grad-CAM:** Generates heatmaps to show exactly which parts of the lungs the model focuses on.
* **Shannon Entropy:** Measures how concentrated or scattered the model's attention is.
* **AOPC (Area Over Perturbation Curve):** Tests the faithfulness of the heatmaps by progressively masking important pixels and tracking how fast the model's confidence drops.

### Files in this repo
* `covid_explainability.ipynb`: The main Google Colab notebook with all my training loops, heatmap generation and math for the evaluation metrics.
* `covid_explainability.py`: The exported Python script version of the notebook for quick reading.

