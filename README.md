# Customer Survival Analysis and Churn Prediction

App: https://churn-and-survival.herokuapp.com/

Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.

Telephone service companies, Internet service providers, pay TV companies, insurance firms, alarm monitoring services, etc. often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.

Predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn. Since these models generate a small prioritized list of potential defectors, they are effective at focusing customer retention marketing programs on the subset of the customer base who are most vulnerable to churn.

In this project, I aim to perform customer survival analysis and build a model which can predict customer churn. I also aim to build an app which can be used to understand why a specific customer would stop the service and to know their expected lifetime value. (CLV/LTV)

## Customer Churn Prediction App
<img src=https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/app_pic.png>

## Project Organization
```
.
├── Data/                               : contains project dataset
│   └── Telco_Customer_Churn.csv
├── static/                             : plots to show gauge chart, hazard and survival curve, shap values in Flask App 
│   └── images/
│       ├── app_pic.png
│       └── shap.png
├── templates/                          : contains html template for flask app
│   └── index.html
├── Customer Churn Prediction.ipynb     : Random Forest model to predict customer churn
├── Customer Survival Analysis.ipynb    : Survival Analysis kaplan-Meier curve, log-rank test and Cox-proportional Hazard model
├── Exploratory Data Analysis.ipynb     : Data Analysis to explore customer data
├── LICENSE.md                          : Apache License 2.0
├── Procfile                            : procfile for app deployment
└── README.md                           : Report
├── app.py                              : Flask App
├── churnmodel.pkl                      : Random Forest model for churn prediction
├── explainer.bz2                       : Shap Explainer
├── requirements.txt                    : requirements to run this model
└── survivemodel.pkl                    : Cox-proportional Hazard model

```
## Customer Survival Analysis

**Survival Analysis:** 
Survival analysis is generally defined as a set of methods for analyzing data where the outcome variable is the time until the occurrence of an event of interest. The event can be death, occurrence of a disease, marriage, divorce, etc. The time to event or survival time can be measured in days, weeks, years, etc.

For example, if the event of interest is heart attack, then the survival time can be the time in years until a person develops a heart attack.

**Objective:**
The objective of this analysis is to utilize non-parametric and semi-parametric methods of survival analysis to answer the following questions.
- How the likelihood of the customer churn changes over time?
- How we can model the relationship between customer churn, time, and other customer characteristics?
- What are the significant factors that drive customer churn?
- What is the survival and Hazard curve of a specific customer?
- What is the expected lifetime value of a customer?

**Kaplan-Meier Survival Curve:**

<p align="center">
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/survival_curve.png" width="400" height="300">
</p>

From above graph, we can say that
- AS expected, for telcom, churn is relatively low. The company was able to retain more than 60% of its customers even after 72 months.
- There is a constant decrease in survival probability probability between 3-60 months.
- After 60 months or 5 years, survival probability decreases with a higher rate. 

**Log-Rank Test:** 

Log-rank test is carried out to analyze churning probabilities group wise and to find if there is statistical significance between groups. The plots show survival curve group wise.

<p align="center">
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/gender.png" width="250" height="200"/> 
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/senior_citizen.png" width="250" height="200"/>
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/partner.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/dependents.png" width="250" height="200"/> 
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/phone_service.png" width="250" height="200"/>
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/multiple_lines.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/internet_service.png" width="250" height="200"/> 
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/online_security.png" width="250" height="200"/> 
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/online_backup.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/device_protection.png" width="250" height="200"/> 
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/tech_support.png" width="250" height="200"/>
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/contract.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/streaming_movies.png" width="250" height="200"/>
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/payment_method.png" width="250" height="200"/> 
<img src="https://github.com/jaysmitjadhav/Churn-Prediction-and-Survival-Analysis/blob/main/static/images/paperless_billing.png" width="250" height="200"/>
</p>

From above graphs we can conclude following:
- Customer's Gender and the phone service type are not indictive features and their p value of log rank test is above threshold value 0.05.
- If customer is young and has a family, he or she is less likely to churn. The reason might be the busy life, more money or other factors.
- If customer is not enrolled in services like online backup, online security, device protection, tech support, streaming TV and streaming movies even though having active internet service, the survival probability is lower.
- The company should traget customers who opt for internet service as their survival probability constantly descreases. Also, Fiber Optilc type of Internet Service is costly and fast compared to DSL and this might be the reason of higher customer churning. 
- More offers should be given to customers who opt for month-to-month contract and company should target customers to subscribe for long-term service. 
- If a customer's payment method is automatic, he or she is less likely to churn. The reason is in the case of electronic check and mailed check, a customer has to make an effort to pay and it takes time.
