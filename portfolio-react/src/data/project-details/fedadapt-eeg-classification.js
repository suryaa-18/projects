export default {
  architecture: [
    'Federated Learning Pipeline: EEG signals from the BCI Competition IV-2a/2b datasets are distributed across multiple clients, where each subject acts as an independent federated participant. A central server initializes the global model and coordinates multiple communication rounds by broadcasting model weights, collecting locally trained updates, and aggregating them using the FedAvg algorithm. This decentralized training strategy preserves data privacy while learning generalized representations across heterogeneous EEG distributions.',

    'Quantum-Enhanced Feature Representation: Raw EEG trials are normalized and encoded into parameterized quantum circuits to project spatial-spectral EEG features into a higher-dimensional quantum feature space. The quantum-enhanced representations improve feature separability, suppress redundant information, and provide richer nonlinear feature embeddings before deep learning, increasing robustness under limited-data and subject-independent scenarios.',

    'Deep EEG Classification: Quantum-enhanced EEG features are processed using a lightweight CNN to learn spatial-spectral representations across EEG channels. Local models are optimized using cross-entropy loss and gradient-based optimization during each federated round. After multiple rounds of global aggregation, the final federated model is evaluated under both subject-dependent and subject-independent settings using Accuracy and Cohen’s Kappa, measuring its generalization capability for motor imagery classification.'
  ],

  result:
    'Developed a privacy-preserving EEG classification framework capable of collaborative learning without sharing raw brain signal data. The proposed system demonstrated improved robustness and subject-independent generalization by combining federated learning, quantum-enhanced feature mapping, and CNN-based spatial-spectral learning. Performance was evaluated on the BCI Competition IV-2a and IV-2b benchmark datasets using Accuracy and Cohen’s Kappa.',

  novelty:
    'Proposed a hybrid quantum-federated learning framework that integrates quantum-assisted feature representation with decentralized EEG classification. Unlike conventional federated approaches that directly train on raw features, the framework enhances feature separability through quantum encoding before collaborative learning, enabling better privacy preservation, improved robustness to inter-subject variability, and efficient deployment for brain-computer interface applications.'
};