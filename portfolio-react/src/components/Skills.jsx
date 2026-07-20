const skillsData = [
  {
    category: 'Programming & Data',
    skills: [
      'Python', 'Java', 'C++', 'SQL', 'Power BI', 'Excel', 'Data Analysis',
      'EDA', 'Matplotlib', 'Seaborn',
    ],
  },
  {
    category: 'AI & Machine Learning',
    skills: [
      'Machine Learning', 'Deep Learning', 'Generative AI', 'Agentic AI',
      'Federated Learning', 'Transfer Learning', 'Predictive Modeling',
      'Feature Engineering', 'PyTorch', 'TensorFlow', 'Scikit-learn',
      'Pandas', 'NumPy', 'OpenCV', 'PennyLane',
    ],
  },
  {
    category: 'Web Development & Tools',
    skills: [
      'Django REST Framework', 'React', 'Node.js', 'REST APIs', 'Git',
      'GitHub', 'Jupyter Notebook', 'AWS'
    ],
  },
];

export default function Skills() {
  return (
    <section className="skills section" id="skills">
      <div className="container">
        <h2 className="section__title">Skills</h2>
        <div className="skills__container grid">
          {skillsData.map((category, index) => (
            <div key={index} className="skills__category">
              <h3 className="skills__category-title">{category.category}</h3>
              <div className="skills__list">
                {category.skills.map((skill, i) => (
                  <span key={i} className="skill__tag">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
