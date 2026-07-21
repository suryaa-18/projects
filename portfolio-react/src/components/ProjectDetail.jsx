import { ArrowLeft, ExternalLink } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import { projectsData } from '../data/projects';
import quantumTransferLearning from '../data/project-details/quantum-transfer-learning-breast-cancer';
import fedAdaptEeg from '../data/project-details/fedadapt-eeg-classification';
import expenseTracker from '../data/project-details/ai-expense-tracker';
import dengueDetection from '../data/project-details/automated-dengue-detection';
import stockPrediction from '../data/project-details/lstm-stock-price-prediction';
import diseasePrediction from '../data/project-details/ensemble-disease-prediction';

const projectDetails = {
  'quantum-transfer-learning-breast-cancer': quantumTransferLearning,
  'fedadapt-eeg-classification': fedAdaptEeg,
  'ai-expense-tracker': expenseTracker,
  'automated-dengue-detection': dengueDetection,
  'lstm-stock-price-prediction': stockPrediction,
  'ensemble-disease-prediction': diseasePrediction,
};

const GithubIcon = () => (
  <svg className="icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.305-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
  </svg>
);

export default function ProjectDetail() {
  const { slug } = useParams();
  const project = projectsData.find((item) => item.slug === slug);
  const detail = projectDetails[slug];

  if (!project || !detail) {
    return (
      <main className="project-detail project-detail--missing">
        <div className="container">
          <p className="project-detail__eyebrow">Project not found</p>
          <h1>That case study does not exist.</h1>
          <Link className="btn btn--primary" to="/#projects">Return to projects</Link>
        </div>
      </main>
    );
  }

  return (
    <main className="project-detail">
      <div className="container project-detail__container">
        <Link className="project-detail__back" to="/#projects">
          <ArrowLeft size={18} />
          All projects
        </Link>

        <header className="project-detail__hero">
          <div className="project-detail__hero-copy">
            <p className="project-detail__eyebrow">Project case study</p>
            <h1>{project.title}</h1>
            <p>{project.description}</p>
            <div className="project-detail__tech" aria-label="Technologies used">
              {project.tech.map((tech) => <span key={tech}>{tech}</span>)}
            </div>
            <a className="btn btn--primary" href={project.github} target="_blank" rel="noopener noreferrer">
              <GithubIcon />
              View source code
              <ExternalLink size={16} />
            </a>
          </div>
          <div className="project-detail__image-wrap">
            <img src={project.image} alt="" className="project-detail__image" />
          </div>
        </header>

        {detail.modules ? (
          <>
            <section className="project-detail__section" aria-labelledby="modules-heading">
              <p className="project-detail__eyebrow">01</p>
              <h2 id="modules-heading">Modules Used</h2>
              <ul className="project-detail__list">
                {detail.modules.map((module) => (
                  <li key={module.name}>
                    <h3>{module.name}</h3>
                    <p>{module.description}</p>
                  </li>
                ))}
              </ul>
            </section>

            <div className="project-detail__outcomes">
              <section className="project-detail__section project-detail__section--result" aria-labelledby="technology-heading">
                <p className="project-detail__eyebrow">02</p>
                <h2 id="technology-heading">Technologies Used</h2>
                <ul className="project-detail__technology">
                  {detail.technologies.map((technology) => (
                    <li key={technology.name}>
                      <strong>{technology.name}</strong>
                      <span>{technology.purpose}</span>
                    </li>
                  ))}
                </ul>
              </section>
              <section className="project-detail__section project-detail__section--novelty" aria-labelledby="functionality-heading">
                <p className="project-detail__eyebrow">03</p>
                <h2 id="functionality-heading">Functionality</h2>
                <ul className="project-detail__functionality">
                  {detail.functionality.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </section>
            </div>
          </>
        ) : (
          <>
            <section className="project-detail__section" aria-labelledby="architecture-heading">
              <p className="project-detail__eyebrow">01</p>
              <h2 id="architecture-heading">Architecture</h2>
              <ol className="project-detail__steps">
                {detail.architecture.map((step, index) => (
                  <li key={step}>
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    <p>{step}</p>
                  </li>
                ))}
              </ol>
            </section>

            <div className="project-detail__outcomes">
              <section className="project-detail__section project-detail__section--result" aria-labelledby="result-heading">
                <p className="project-detail__eyebrow">02</p>
                <h2 id="result-heading">Result</h2>
                <p>{detail.result}</p>
              </section>
              <section className="project-detail__section project-detail__section--novelty" aria-labelledby="novelty-heading">
                <p className="project-detail__eyebrow">03</p>
                <h2 id="novelty-heading">Novelty</h2>
                <p>{detail.novelty}</p>
              </section>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
