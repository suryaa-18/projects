import { useState } from 'react';
import { Mail, Phone, MapPin, Send } from 'lucide-react';

export default function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });
  const [formStatus, setFormStatus] = useState({ type: '', message: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const validateEmail = (email) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.name || !formData.email || !formData.subject || !formData.message) {
      setFormStatus({ type: 'error', message: 'Please fill in all fields' });
      return;
    }

    if (!validateEmail(formData.email)) {
      setFormStatus({ type: 'error', message: 'Please enter a valid email address' });
      return;
    }

    setIsSubmitting(true);
    setFormStatus({ type: '', message: '' });

    try {
      const response = await fetch('https://formsubmit.co/ajax/surya.k7880@gmail.com', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          _subject: `Portfolio contact: ${formData.subject}`,
          _replyto: formData.email,
          _template: 'table',
        }),
      });

      if (!response.ok) {
        throw new Error('Message delivery failed');
      }

      setFormStatus({
        type: 'success',
        message: "Thank you for your message! I'll get back to you soon.",
      });
      setFormData({ name: '', email: '', subject: '', message: '' });
    } catch {
      setFormStatus({
        type: 'error',
        message: 'Unable to send your message. Please email me directly instead.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="contact section" id="contact">
      <div className="container">
        <h2 className="section__title">Get In Touch</h2>
        <div className="contact__container grid">
          <div className="contact__info">
            <p className="contact__description">
              I'm currently looking for <strong>Internship / Full-time</strong> opportunities.
              Feel free to reach out if you have any opportunities or just want to connect!
            </p>
            <div className="contact__details">
              <div className="contact__item">
                <div className="contact__icon">
                  <Mail className="icon" size={22} />
                </div>
                <div className="contact__item-info">
                  <span className="contact__item-label">Email</span>
                  <a href="mailto:surya.k7880@gmail.com" className="contact__item-value">
                    surya.k7880@gmail.com
                  </a>
                </div>
              </div>
              <div className="contact__item">
                <div className="contact__icon">
                  <Phone className="icon" size={22} />
                </div>
                <div className="contact__item-info">
                  <span className="contact__item-label">Phone</span>
                  <a href="tel:+919585737125" className="contact__item-value">
                    +91-9585737125
                  </a>
                </div>
              </div>
              <div className="contact__item">
                <div className="contact__icon">
                  <MapPin className="icon" size={22} />
                </div>
                <div className="contact__item-info">
                  <span className="contact__item-label">Location</span>
                  <span className="contact__item-value">Coimbatore, Tamil Nadu</span>
                </div>
              </div>
            </div>
          </div>
          <form className="contact__form" onSubmit={handleSubmit} id="contact-form">
            <div className="form__group">
              <label htmlFor="name" className="form__label">Name</label>
              <input
                type="text"
                id="name"
                name="name"
                className="form__input"
                placeholder="Your Name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form__group">
              <label htmlFor="email" className="form__label">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                className="form__input"
                placeholder="your.email@example.com"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form__group">
              <label htmlFor="subject" className="form__label">Subject</label>
              <input
                type="text"
                id="subject"
                name="subject"
                className="form__input"
                placeholder="Subject"
                value={formData.subject}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form__group">
              <label htmlFor="message" className="form__label">Message</label>
              <textarea
                id="message"
                name="message"
                className="form__input form__textarea"
                placeholder="Your message..."
                rows={5}
                value={formData.message}
                onChange={handleChange}
                required
              />
            </div>
            <button type="submit" className="btn btn--primary form__submit" disabled={isSubmitting}>
              {isSubmitting ? 'Sending...' : (
                <>
                  Send Message
                  <Send className="icon" size={18} />
                </>
              )}
            </button>
            {formStatus.message && (
              <p className={`form__message ${formStatus.type}`} id="form-message" aria-live="polite">
                {formStatus.message}
              </p>
            )}
          </form>
        </div>
      </div>
    </section>
  );
}
