# PII-Data-Detection
## Automated techniques to detect and remove PII from educational data.

### Introduction:
Welcome to this Jupyter notebook developed for The Learning Agency Lab - PII Data Detection! This notebook is designed to help you participate in the competition and to Develop automated techniques to detect and remove PII from educational data.



### Inspiration and Credits 🙌
This notebook is inspired by the work of Aleksandr Lavrikov, available at [this Kaggle project](https://www.kaggle.com/code/lavrikovav/0-968-to-onnx-30-200-speedup-pii-inference). I extend my gratitude to Aleksandr Lavrikov for sharing their insights and code publicly.


### How to Use This Notebook:
1. **Setup Environment**:
   - Ensure all required libraries are installed. (Refer to the import libraries cell in the notebook for details.)
   
2. **Data Preparation**:
   - Prepare your training and test datasets in JSON format. The paths to these datasets should be specified in the `config` class of the notebook.
   - Optionally, you can downsample the training data for faster processing, specify the percentage in the `downsample` variable of the `config` class.
   
3. **Training and Evaluation**:
   - Train the model using the provided training dataset by running the appropriate cells in the notebook. Adjust hyperparameters if necessary.
   - Evaluate the trained model's performance on the test dataset to assess its effectiveness in detecting PII.

4. **Inference**:
   - Use the trained model to perform inference on new data. The notebook provides functionalities to tokenize input data, predict PII labels, and extract PII entities.

5. **Export Results**:
   - Export the processed predictions, including identified PII entities such as phone numbers, email addresses, URLs, etc., to a CSV file for further analysis or usage.

**🌟 Explore my profile and other public projects, and don't forget to share your feedback!**

## 👉 [Visit my Profile]( https://www.kaggle.com/code/zulqarnainalipk) 👈

## How to Use 🛠️
To use this notebook effectively, please follow these steps:
1. Ensure you have the competition data and environment set up.
2. Execute each cell sequentially to perform data preparation, feature engineering, model training, and prediction submission.
3. Customize and adapt the code as needed to improve model performance or experiment with different approaches.
.

## Acknowledgments 🙏
I acknowledge The Learning Agency Lab organizers for providing the dataset and the competition platform.

Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
👉 [Visit my Profile](https://www.kaggle.com/zulqarnainalipk) 👈


## 📊 All datasets can be automatically imported by running them on Kaggle.
## 🚀 Using Accelerator GPU T4 x2 by Kaggle on Kaggle is recommended.
