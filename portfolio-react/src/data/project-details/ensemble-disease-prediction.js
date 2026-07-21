export default {
  architecture: [
    'Data Preprocessing: The training dataset is loaded from CSV and symptom features are extracted while the prognosis column is label encoded into numerical classes. The preprocessing pipeline validates feature consistency between training and testing datasets, ensuring identical symptom ordering before model training. Symptom metadata and label encoders are serialized for consistent inference.',

    'Ensemble Model Training: Three complementary machine learning classifiers—Gaussian Naïve Bayes, Decision Tree, and Random Forest (200 estimators)—are independently trained on the complete symptom dataset. Each model captures different decision boundaries, with Naïve Bayes modeling probabilistic feature relationships, Decision Tree learning interpretable rule-based splits, and Random Forest improving robustness through ensemble bagging. Models are evaluated on an external testing dataset using Accuracy, Precision, Recall, and F1-score before being saved for deployment.',

    'Ensemble Inference: During prediction, the user symptom vector is processed by all three trained classifiers simultaneously. Each model produces class probability distributions, which are combined using Soft Voting (probability averaging). The disease with the highest aggregated probability is returned as the final prediction, along with confidence scores and the Top-3 most probable diseases, providing a more reliable and interpretable diagnosis than relying on a single classifier.'
  ],

  result:
    'Developed a complete disease prediction pipeline capable of training, validating, and deploying multiple machine learning models through a unified ensemble framework. The system performs automated preprocessing, evaluates model performance on unseen testing data using standard classification metrics, and generates confidence-based disease predictions with ranked Top-3 recommendations for improved clinical decision support.',

  novelty:
    'Instead of depending on a single classifier, the proposed framework combines probabilistic (Gaussian Naïve Bayes), rule-based (Decision Tree), and ensemble (Random Forest) learning through Soft Voting to improve prediction robustness and reduce model-specific bias. The architecture also provides prediction confidence scores, Top-3 disease recommendations, and reusable preprocessing metadata, making it suitable for scalable clinical decision-support applications.'
};