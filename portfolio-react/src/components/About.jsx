export default function About() {
  return (
    <section className="about section" id="about">
      <div className="container">
        <h2 className="section__title">About </h2>
        <div className="about__container">
          <div className="about__content">
            <p className="about__description">
              I'm an MCA Graduate at VIT Chennai with expertise in software development, Artificial Intelligence, and data-driven application development. Proficient in Python, Java, C++, SQL, Django REST Framework, React, and modern AI frameworks including PyTorch and TensorFlow. I have hands-on experience building scalable REST APIs, full-stack web applications, machine learning models, federated learning systems, and data analytics solutions, with a strong focus on creating practical, user-centric software.
              I enjoy exploring emerging technologies and continuously expanding my technical skills through research-driven projects and self-learning. My work spans AI-powered applications, quantum machine learning, federated learning, and data analytics, where I focus on solving real-world problems with efficient and scalable solutions. Beyond academics, serving as a Placement Representative and leading student initiatives has strengthened my communication, leadership, teamwork, and project management skills, enabling me to collaborate effectively in diverse environments.
            </p>
            <div className="about__info grid">
              <div className="about__info-item">
                <span className="about__info-label">Name:</span>
                <span className="about__info-value">Surya K</span>
              </div>
              <div className="about__info-item">
                <span className="about__info-label">Email:</span>
                <span className="about__info-value">surya.k7880@gmail.com</span>
              </div>
              <div className="about__info-item">
                <span className="about__info-label">Phone:</span>
                <span className="about__info-value">+91-9585737125</span>
              </div>
              <div className="about__info-item">
                <span className="about__info-label">Location:</span>
                <span className="about__info-value">Coimbatore, Tamil Nadu</span>
              </div>
              <div className="about__info-item">
                <span className="about__info-label">Highest Qualification:</span>
                <span className="about__info-value">MCA</span>
              </div>
              <div className="about__info-item">
                <span className="about__info-label">Graduation year:</span>
                <span className="about__info-value">2026</span>
              </div>
            </div>
            <a href="/assets/resume.pdf" className="btn btn--primary about__btn" download>
              Download Resume
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}