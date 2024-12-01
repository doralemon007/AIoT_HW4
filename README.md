# Titanic Survival Prediction Project

## Overview

This project aims to predict survival on the Titanic dataset using **PyCaret** and **Optuna**. Through natural language programming, we performed feature engineering, model selection, ensemble model building, and hyperparameter optimization. The final goal was to create an optimized model with high accuracy.

___

## Steps

### Step 1: Feature Engineering

**Prompt**:

> Help me find the most suitable features or combinations of features for the Titanic dataset. Refer to the internet and other people's experience if necessary.

**Implementation**: The following features were engineered based on domain knowledge:

1.  `FamilySize`: Sum of `SibSp` and `Parch` plus one.
2.  `IsAlone`: Binary indicator for passengers traveling alone.
3.  `Title`: Extracted titles (Mr., Mrs., Miss, etc.) from the `Name` column and grouped rare titles.
4.  `AgeGroup`: Binned `Age` into categorical groups (`Child`, `Teenager`, etc.).
5.  `FareBand`: Binned `Fare` into quartiles.
6.  `HasCabin`: Binary indicator for the presence of cabin data.
7.  `Deck`: Extracted deck information from the `Cabin` column.

**Code**:

```
<div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary dark:bg-gray-950"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none">python</div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-sidebar-surface-primary px-2 font-sans text-xs text-token-text-secondary dark:bg-token-main-surface-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>複製程式碼</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-python">titanic_data[<span class="hljs-string">'FamilySize'</span>] = titanic_data[<span class="hljs-string">'SibSp'</span>] + titanic_data[<span class="hljs-string">'Parch'</span>] + <span class="hljs-number">1</span>
titanic_data[<span class="hljs-string">'IsAlone'</span>] = (titanic_data[<span class="hljs-string">'FamilySize'</span>] == <span class="hljs-number">1</span>).astype(<span class="hljs-built_in">int</span>)
titanic_data[<span class="hljs-string">'Title'</span>] = titanic_data[<span class="hljs-string">'Name'</span>].<span class="hljs-built_in">str</span>.extract(<span class="hljs-string">' ([A-Za-z]+)\.'</span>, expand=<span class="hljs-literal">False</span>)
titanic_data[<span class="hljs-string">'Title'</span>] = titanic_data[<span class="hljs-string">'Title'</span>].replace([<span class="hljs-string">'Mlle'</span>, <span class="hljs-string">'Ms'</span>], <span class="hljs-string">'Miss'</span>).replace(<span class="hljs-string">'Mme'</span>, <span class="hljs-string">'Mrs'</span>)
titanic_data[<span class="hljs-string">'Title'</span>] = titanic_data[<span class="hljs-string">'Title'</span>].replace([<span class="hljs-string">'Lady'</span>, <span class="hljs-string">'Countess'</span>, <span class="hljs-string">'Capt'</span>, <span class="hljs-string">'Col'</span>, <span class="hljs-string">'Don'</span>, <span class="hljs-string">'Dr'</span>, 
                                                       <span class="hljs-string">'Major'</span>, <span class="hljs-string">'Rev'</span>, <span class="hljs-string">'Sir'</span>, <span class="hljs-string">'Jonkheer'</span>, <span class="hljs-string">'Dona'</span>], <span class="hljs-string">'Rare'</span>)
titanic_data[<span class="hljs-string">'AgeGroup'</span>] = pd.cut(titanic_data[<span class="hljs-string">'Age'</span>], bins=[<span class="hljs-number">0</span>, <span class="hljs-number">12</span>, <span class="hljs-number">18</span>, <span class="hljs-number">35</span>, <span class="hljs-number">60</span>, <span class="hljs-number">80</span>], labels=[<span class="hljs-string">'Child'</span>, <span class="hljs-string">'Teenager'</span>, <span class="hljs-string">'Young Adult'</span>, <span class="hljs-string">'Middle Aged'</span>, <span class="hljs-string">'Senior'</span>])
titanic_data[<span class="hljs-string">'FareBand'</span>] = pd.qcut(titanic_data[<span class="hljs-string">'Fare'</span>], <span class="hljs-number">4</span>, labels=[<span class="hljs-string">'Low'</span>, <span class="hljs-string">'Medium'</span>, <span class="hljs-string">'High'</span>, <span class="hljs-string">'Very High'</span>])
titanic_data[<span class="hljs-string">'HasCabin'</span>] = titanic_data[<span class="hljs-string">'Cabin'</span>].notnull().astype(<span class="hljs-built_in">int</span>)
titanic_data[<span class="hljs-string">'Deck'</span>] = titanic_data[<span class="hljs-string">'Cabin'</span>].<span class="hljs-built_in">str</span>[<span class="hljs-number">0</span>].fillna(<span class="hljs-string">'Unknown'</span>)
</code></div></div>
```

___

### Step 2: Model Selection

**Prompt**:

> Use PyCaret to find the top 5 machine learning models for the Titanic dataset.

**Implementation**: PyCaret was used to evaluate and rank all classification algorithms. The top 5 models were selected based on their accuracy.

**Code**:

```
<div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary dark:bg-gray-950"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none">python</div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-sidebar-surface-primary px-2 font-sans text-xs text-token-text-secondary dark:bg-token-main-surface-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>複製程式碼</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-python"><span class="hljs-keyword">from</span> pycaret.classification <span class="hljs-keyword">import</span> setup, compare_models, pull

<span class="hljs-comment"># PyCaret Setup</span>
clf_setup = setup(data=titanic_data, target=<span class="hljs-string">'Survived'</span>, session_id=<span class="hljs-number">42</span>, verbose=<span class="hljs-literal">True</span>)

<span class="hljs-comment"># Compare models and select top 5</span>
top_models = compare_models(n_select=<span class="hljs-number">5</span>)

<span class="hljs-comment"># Display results</span>
top_models_results = pull()
<span class="hljs-built_in">print</span>(<span class="hljs-string">"Top 5 Models:"</span>)
<span class="hljs-built_in">print</span>(top_models_results.head(<span class="hljs-number">5</span>))
</code></div></div>
```

___

### Step 3: Ensemble Model Building

**Prompt**:

> Use the top 5 models to create ensemble models. Compare bagging, boosting, and blending approaches without detailed hyperparameter tuning.

**Implementation**: The top 5 models were used to create ensemble models:

1.  **Bagging**: Averaging predictions from multiple instances of the same model.
2.  **Boosting**: Sequentially improving errors of the previous model.
3.  **Blending**: Combining predictions of top models into a single output.

**Code**:

```
<div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary dark:bg-gray-950"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none">python</div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-sidebar-surface-primary px-2 font-sans text-xs text-token-text-secondary dark:bg-token-main-surface-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>複製程式碼</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-python"><span class="hljs-keyword">from</span> pycaret.classification <span class="hljs-keyword">import</span> ensemble_model, blend_models

<span class="hljs-comment"># Bagging</span>
bagged_model = ensemble_model(top_models[<span class="hljs-number">0</span>], method=<span class="hljs-string">'Bagging'</span>)

<span class="hljs-comment"># Boosting</span>
boosted_model = ensemble_model(top_models[<span class="hljs-number">0</span>], method=<span class="hljs-string">'Boosting'</span>)

<span class="hljs-comment"># Blending</span>
blended_model = blend_models(top_models)

<span class="hljs-comment"># Display Ensemble Results</span>
<span class="hljs-built_in">print</span>(<span class="hljs-string">"\nBagging Results:"</span>, bagged_model)
<span class="hljs-built_in">print</span>(<span class="hljs-string">"\nBoosting Results:"</span>, boosted_model)
<span class="hljs-built_in">print</span>(<span class="hljs-string">"\nBlending Results:"</span>, blended_model)
</code></div></div>
```

___

### Step 4: Hyperparameter Optimization

**Prompt**:

> Perform hyperparameter optimization on the best ensemble model (bagged\_model) using PyCaret and Optuna.

**Implementation**: Hyperparameter optimization was performed on the bagged model using Optuna. The `tune_model` function automatically optimized the parameters to maximize accuracy.

**Code**:

```
<div class="contain-inline-size rounded-md border-[0.5px] border-token-border-medium relative bg-token-sidebar-surface-primary dark:bg-gray-950"><div class="flex items-center text-token-text-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md h-9 bg-token-sidebar-surface-primary dark:bg-token-main-surface-secondary select-none">python</div><div class="sticky top-9 md:top-[5.75rem]"><div class="absolute bottom-0 right-2 flex h-9 items-center"><div class="flex items-center rounded bg-token-sidebar-surface-primary px-2 font-sans text-xs text-token-text-secondary dark:bg-token-main-surface-secondary"><span class="" data-state="closed"><button class="flex gap-1 items-center select-none py-1"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="icon-sm"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"></path></svg>複製程式碼</button></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="!whitespace-pre hljs language-python"><span class="hljs-keyword">from</span> pycaret.classification <span class="hljs-keyword">import</span> tune_model

<span class="hljs-comment"># Hyperparameter Optimization on Bagged Model</span>
optimized_bagged_model = tune_model(bagged_model, optimize=<span class="hljs-string">'Accuracy'</span>, n_iter=<span class="hljs-number">50</span>)

<span class="hljs-comment"># Display Optimized Model</span>
<span class="hljs-built_in">print</span>(<span class="hljs-string">"Optimized Bagged Model:"</span>)
<span class="hljs-built_in">print</span>(optimized_bagged_model)
</code></div></div>
```

___

## Final Results

### Cross-Validation Metrics for the Optimized Bagged Model:

| Metric | Mean | Std Deviation |
| --- | --- | --- |
| **Accuracy** | 0.8491 | 0.0397 |
| **AUC** | 0.8814 | 0.0408 |
| **Recall** | 0.7493 | 0.0636 |
| **Precision** | 0.8416 | 0.0603 |
| **F1 Score** | 0.7917 | 0.0557 |
| **Kappa** | 0.6742 | 0.0860 |
| **MCC** | 0.6779 | 0.0859 |

### Optimized Hyperparameters:

The optimized model uses a `BaggingClassifier` with:

-   Base Estimator: GradientBoostingClassifier
-   Hyperparameters:
    -   Max Features: 0.7
    -   Max Samples: 0.8
    -   Number of Estimators: 10

___

## Conclusion

Using natural language programming, the following goals were achieved:

1.  **Feature Engineering**: Derived meaningful features to improve predictions.
2.  **Model Selection**: Identified the top 5 models automatically using PyCaret.
3.  **Ensemble Learning**: Built and compared bagging, boosting, and blending ensemble models.
4.  **Hyperparameter Optimization**: Improved the performance of the best ensemble model using Optuna.

The final optimized bagged model achieved an **accuracy of 84.91%** and an **AUC of 88.14%**, demonstrating strong performance in predicting Titanic survival.