import { Check } from 'lucide-react';

const certifications = [
  {
    name: 'Data Foundations Associate',
    issuer: 'Oracle',
    description: 'Oracle credential focused on foundational data knowledge for data-driven work.',
    highlights: [
      'Data fundamentals and core concepts',
      'Data-oriented problem solving',
    ],
  },
  {
    name: 'AI Engineer for Developers Associate',
    issuer: 'DataCamp',
    description: 'DataCamp credential focused on applying AI engineering concepts in developer workflows.',
    highlights: [
      'AI engineering fundamentals',
      'Developer-focused AI applications',
    ],
  },
];

export default function Certifications() {
  return (
    <section className="certifications section" id="certifications">
      <div className="container">
        <h2 className="section__title">Certifications</h2>
        <div className="achievements__container certifications__container">
          {certifications.map((certification) => (
            <article key={certification.name} className="achievement__card">
              <div className="achievement__icon">
                <Check className="icon" size={28} />
              </div>
              <h3 className="achievement__title">{certification.name}</h3>
              <p className="achievement__meta">{certification.issuer}</p>
              <p className="achievement__description">{certification.description}</p>
              <ul className="achievement__highlights">
                {certification.highlights.map((highlight) => (
                  <li key={highlight}>{highlight}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
