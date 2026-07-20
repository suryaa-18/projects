import { Check, Users, Star } from 'lucide-react';

const achievementsData = [
  {
    title: 'Placement Representative',
    organization: 'PSG College of Technology',
    duration: '2021 - 2024',
    description: 'Supported the B.Sc. CSD batch throughout the placement process.',
    highlights: [
      'Coordinated between 60+ students and the placement cell',
      'Facilitated campus recruitment drives',
      'Organized career readiness sessions',
    ],
    icon: Check,
  },
  {
    title: 'Convenor & Core Member, Animal Welfare Club',
    organization: 'PSG College of Technology',
    duration: '2022 - 2024',
    description: 'Led student-led animal welfare initiatives on campus and in the community.',
    highlights: [
      'Led a team of 20+ volunteers',
      'Organized campus awareness campaigns',
      'Coordinated community outreach events',
    ],
    icon: Users,
  },
  {
    title: 'ERM Lead',
    organization: 'PSG College of Technology',
    duration: '2023 - 2024',
    description: 'Managed cross-functional teams for departmental fest operations.',
    highlights: [
      'Oversaw logistics and scheduling',
      'Coordinated on-ground event execution',
      'Managed eight events during the year',
    ],
    icon: Star,
  },
];

export default function Achievements() {
  return (
    <section className="achievements section" id="achievements">
      <div className="container">
        <h2 className="section__title">Leadership & Activities</h2>
        <div className="achievements__container grid">
          {achievementsData.map((achievement, index) => (
            <article key={index} className="achievement__card">
              <div className="achievement__icon">
                <achievement.icon className="icon" size={28} />
              </div>
              <h3 className="achievement__title">{achievement.title}</h3>
              <p className="achievement__meta">
                {achievement.organization} | {achievement.duration}
              </p>
              <p className="achievement__description">{achievement.description}</p>
              <ul className="achievement__highlights">
                {achievement.highlights.map((highlight) => (
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
