const projectsData = [
  {
    title: 'Noise-Aware Hybrid Quantum Transfer Learning for Breast Cancer Diagnosis',
    description: 'Built a hybrid quantum-classical model for breast cancer classification on the BreakHis histopathology dataset, achieving 90% accuracy under noise-robust evaluation.',
    tech: ['Python', 'PyTorch', 'PennyLane', 'OpenCV', 'NumPy'],
    image: '/assets/1.svg',
      github: 'https://github.com/suryaa-18/projects/tree/main/QTL-with-noise-mitigation-system-for-breast-cancer-classification',
  },
  {
    title: 'FedAdapt: Federated Learning with Riemannian Manifold & ATCNet for EEG Classification',
    description: 'Developed a privacy-preserving federated learning framework for motor-imagery EEG classification, achieving 78.8% cross-subject accuracy on the BCI Competition IV 2a dataset.',
    tech: ['Python', 'PyTorch', 'BCI', 'Flower', 'MOABB', 'ATCNet'],
    image: '/assets/2.svg',
      github: 'https://github.com/suryaa-18/projects/tree/main/FedAdapt%20FL%20Motor%20Imagery%20EEG',
  },
  {
    title: 'AI-Powered Expense Tracker & Financial Planning System',
    description: 'Built a full-stack expense management application with token-based authentication, REST APIs, NLP expense parsing, and interactive financial analytics dashboards.',
    tech: ['Django REST Framework', 'React', 'SQLite', 'REST API', 'NLP'],
    image: '/assets/3.svg',
      github: 'https://github.com/suryaa-18/projects/tree/main/Expense%20Tracker%20-%20Django%20%26%20AI',
  },
  {
  title: 'Tri-Stage Automated Dengue Detection System',
  description: 'Developed an end-to-end AI diagnostic pipeline using YOLOv8s for blood cell detection, Sequential CNN for WBC classification, and Random Forest with CBC features for automated dengue prediction.',
  tech: ['Python', 'YOLOv8', 'CNN', 'Random Forest', 'OpenCV', 'Scikit-learn'],
  image: '/assets/4.svg',
  github: 'https://github.com/suryaa-18/projects/tree/main/Tri-State-Dengue-Automation',
},
{
  title: 'LSTM-Based Stock Price Prediction System',
  description: 'Built a time-series forecasting system using stacked LSTM networks with historical stock data, MinMax normalization, dropout regularization, and predictive visualization for technology stocks.',
  tech: ['Python', 'TensorFlow', 'Keras', 'LSTM', 'Scikit-learn', 'yFinance'],
  image: '/assets/5.svg',
  github: 'https://github.com/suryaa-18/projects/tree/main/Stock-Prediction-using-Machine-Learning',
},
{
  title: 'Ensemble-Based Disease Prediction System',
  description: 'Developed a symptom-based disease prediction system using an ensemble of Naive Bayes, Decision Tree, and Random Forest models with soft voting to provide disease predictions, confidence scores, and top-ranked diagnoses.',
  tech: ['Python', 'Scikit-learn', 'Random Forest', 'Decision Tree', 'Naive Bayes', 'Pandas'],
  image: '/assets/6.svg',
  github: 'https://github.com/suryaa-18/projects/tree/main/Disease-Prediction-by-Model-Ensembling',
},
];

const GithubIcon = () => (
  <svg className="icon" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.305-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/>
  </svg>
);

export default function Projects() {
  return (
    <section className="projects section" id="projects">
      <div className="container">
        <h2 className="section__title">Projects</h2>
        <div className="projects__container grid">
          {projectsData.map((project, index) => (
            <article key={index} className="project__card">
              <div className="project__image">
                <img
                  src={project.image}
                  alt={project.title}
                  className="project__img"
                />
              </div>
              <div className="project__content">
                <h3 className="project__title">{project.title}</h3>
                <p className="project__date">{project.date}</p>
                <p className="project__description">{project.description}</p>
                <div className="project__tech">
                  {project.tech.map((tech, i) => (
                    <span key={i} className="project__tech-tag">
                      {tech}
                    </span>
                  ))}
                </div>
                {project.github && (
                  <div className="project__links">
                    <a
                      href={project.github}
                      className="project__link"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <GithubIcon />
                      Code
                    </a>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
