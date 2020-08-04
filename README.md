# [**BlueBook for Bulldozers**](https://www.kaggle.com/c/bluebook-for-bulldozers/overview)

**Fastai Library or API**
- [Fast.ai](https://www.fast.ai/about/) is the first deep learning library to provide a single consistent interface to all the most commonly used deep learning applications for vision, text, tabular data, time series, and collaborative filtering.
- [Fast.ai](https://www.fast.ai/about/) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

**Objective**
- Predict the auction sale price for a piece of Heavy Equipment to create a Blue Book for Bulldozers.

**Preparing the Model**
- I have used [Fastai](https://www.fast.ai/about/) API to train the Model. It seems quite challenging to understand the code if you have never encountered with Fast.ai API before.
One important note for anyone who has never used Fastai API before is to go through [Fastai Documentation](https://docs.fast.ai/). And if you are using Fastai in Jupyter Notebook then you can use doc(function_name) to get the documentation instantly.

**Data**
- I had prepared the Data for this Project from [Kaggle](https://www.kaggle.com/c/bluebook-for-bulldozers/data)

**Random Forest Model**
- Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.
- I have used Random Forest to predict the SalePrice.

**Snapshot of using Random Forest**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596540051/Rm_hpzj9r.png)

**Interpretation of Random Forest**
- I have taken out the most important features from the Model and plotted it for better and effective understanding.

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596540324/Fea_d8uimx.png)

**Dendrogram**
- I have also plotted Dendrogram using the most Importatn Features.

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596540553/Den_lcq4et.png)

**PDP Plot**
- PDP Plot is so effective to show the Partial Dependence of one feature with whole other features. This plot is so effective, still it is unknown among many of us.

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596540819/PDP_bw0cho.png)
