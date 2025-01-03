# Semantic Segmentation of Aerial Images using U-Net

## Overview
This project demonstrates the application of **semantic segmentation** using deep learning to analyze aerial images for land cover classification. The developed model leverages the U-Net architecture to identify urban infrastructure elements (e.g., buildings, roads) and natural features (e.g., vegetation, land, water bodies) with a focus on precision and efficiency. 

The segmentation model aims to assist urban planners, environmentalists, disaster response teams, and other professionals engaged in resource management and city planning.

---

## Methodology
1. **Model Architecture**:  
   - **U-Net** with an encoder-decoder structure for pixel-wise classification.

2. **Data Preprocessing**:  
   - **Data Augmentation**: Techniques like flipping, rotation, and scaling.  
   - **Normalization**: Ensures consistency across the dataset.  
   - **Patch Creation**: Divides images into smaller segments for improved learning.  

3. **Training and Optimization**:  
   - Dropout layers to prevent overfitting.  
   - Learning rate adjustments using the `ReduceLROnPlateau` callback.  

4. **Evaluation Metrics**:  
   - **Intersection over Union (IoU)** for class-wise performance.  
   - Confusion Matrix for detailed classification insights.

---

## Results
- **Training Performance**:  
  - Training Loss: `0.9509`, Validation Loss: `0.9490`  
  - IoU: Training - `0.585`, Validation - `0.5736`

- **Testing Performance**:  
  - Notable accuracy for "buildings" and "land."  
  - Misclassifications observed for "water" due to color similarities and class imbalance.

- **Domain Adaptation Challenges**:  
  - Testing the model on a local dataset from Kathmandu (trained on Dubai data) highlighted poor generalization due to:  
    - **Geographical Variances**  
    - **Limited Data Diversity**  
    - **Overfitting**  
    - **Domain Shift**

---

## Key Learnings
- The U-Net model performs well for the dataset it was trained on but struggles with domain shifts.  
- Addressing class imbalance and increasing dataset diversity are crucial for robust generalization.

---

## Applications
This segmentation approach can support:  
- Urban planning and infrastructure development.  
- Environmental monitoring and disaster response.  

---

## Acknowledgments
This project was developed with deep learning techniques and tested extensively to provide actionable insights for real-world applications. Future work involves addressing domain generalization challenges through advanced architectures and diverse datasets.


1. **Clone the Project**: 

    ```bash
    git clone <project_url>
    ```

2. **Create a Virtual Environment**:

    ```bash
    python3 -m venv myenv
    ```

    This will create a virtual environment named `myenv`.

3. **Activate the Virtual Environment**:

    On Windows:
    ```bash
    myenv\Scripts\activate
    ```

    On Unix or MacOS:
    ```bash
    source myenv/bin/activate
    ```

4. **Install Dependencies from Pipfile**:

    ```bash
    pip install -r Pipfile
    ```

    This command will install all the packages listed in the `Pipfile` and its corresponding `Pipfile.lock`.

5. **Install Additional Packages**:

    ```bash
    python -m pip install Pillow
    pip install django-cors-headers
    ```

    These commands will install additional packages not listed in the `Pipfile`.
