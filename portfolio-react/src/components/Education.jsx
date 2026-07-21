import { GraduationCap } from 'lucide-react';

const educationData = [
  {
    degree: 'Master of Computer Applications',
    institute: 'Vellore Institute of Technology, Chennai',
    duration: 'Jul 2024 - May 2026',
    score: 'CGPA: 9.10 / 10.0',
    highlights: [],
    icon: GraduationCap,
  },
  {
    degree: 'Bachelor of Computer Systems and Design',
    institute: 'PSG College of Technology, Coimbatore',
    duration: 'Sep 2021 - May 2024',
    score: 'CGPA: 8.23 / 10.0',
    highlights: [],
    icon: GraduationCap,
  },
  {
    degree: 'Grade XII, Higher Secondary Education (HSE)',
    institute: 'Venkatalakshmi Matriculation Higher Secondary School, Coimbatore',
    duration: 'May 2021',
    score: 'Percentage: 93.16%',
    highlights: ['State Board of Tamil Nadu'],
    icon: GraduationCap,
  },
  {
    degree: 'Grade X, Secondary School Education (SSE)',
    institute: 'Venkatalakshmi Matriculation Higher Secondary School, Coimbatore',
    duration: 'May 2019',
    score: 'Percentage: 88.8%',
    highlights: ['State Board of Tamil Nadu'],
    icon: GraduationCap,
  },
];

export default function Education() {
  return (
    <section className="education section" id="education">
      <div className="container">
        <h2 className="section__title">Education</h2>
        <div className="education__container grid">
          {educationData.map((edu, index) => (
            <article key={index} className="education__card">
              <div className="education__icon">
                <edu.icon className="icon" size={24} />
              </div>
              <div className="education__content">
                <h3 className="education__degree">{edu.degree}</h3>
                <p className="education__institute">{edu.institute}</p>
                <p className="education__duration">{edu.duration}</p>
                <p className="education__score">{edu.score}</p>
                <p className="education__highlights">{edu.highlights.join(', ')}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
